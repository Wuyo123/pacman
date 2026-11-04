"""Final-epoch supervised M0 training from game-separated observation episodes."""

import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .model import BeliefMiTA


def split_games(paths):
    paths = sorted(map(Path, paths), key=lambda path: path.name)
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("at least two distinct game files are required")
    cut = 180 if len(paths) == 200 else min(len(paths) - 1, max(1, int(len(paths) * 0.9)))
    return paths[:cut], paths[cut:]


def graph_gaussian(distances, enemy, sigma=1.0):
    """Normalized graph-distance Gaussian; negative/nonfinite distances are unreachable."""
    rows = torch.as_tensor(distances)[torch.as_tensor(enemy).long()].float()
    reachable = (rows >= 0) & torch.isfinite(rows)
    weights = torch.exp(-rows.square() / (2 * sigma * sigma)).masked_fill(~reachable, 0)
    return weights / weights.sum(-1, keepdim=True)


class EpisodeDataset(Dataset):
    """Keep raw arrays once; construct eight-frame histories only when sampled."""

    def __init__(self, paths):
        self.paths = list(map(Path, paths))
        self.episodes = []
        self.offsets = [0]
        for path in self.paths:
            with np.load(path, allow_pickle=False) as archive:
                episode = {key: torch.from_numpy(archive[key]) for key in
                           ("frames", "agents", "labels", "enemy", "visible", "legal", "values",
                            "landmarks", "distances", "cells")}
            frames = episode["frames"]
            if frames.ndim != 3 or frames.shape[-1] != 18 or len(frames) == 0:
                raise ValueError(f"invalid observation frames: {path}")
            self.episodes.append(episode)
            self.offsets.append(self.offsets[-1] + len(frames))

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        game = bisect_right(self.offsets, index) - 1
        episode = self.episodes[game]
        step = index - self.offsets[game]
        history = torch.arange(step - 7, step + 1).clamp_min(0)
        enemy = episode["enemy"][step].long()
        return {
            "x": episode["frames"][history].float(),
            "agents": episode["agents"][step].long(),
            "landmarks": episode["landmarks"].long(),
            "labels": episode["labels"][step].long(),
            "enemy": enemy,
            "visible": episode["visible"][step].bool(),
            "legal": episode["legal"][step].bool(),
            "value": episode["values"][step].float(),
            "belief_target": graph_gaussian(episode["distances"], enemy),
            "cells": episode["cells"],
        }


def training_loss(output, batch):
    labels, legal = batch["labels"], batch["legal"]
    if not legal.gather(-1, labels[..., None]).all():
        raise ValueError("expert labels must be legal for both own agents")
    first = output["action_logits"][:, 0].masked_fill(~legal[:, 0], -torch.inf)
    second = output["conditional_action_logits"][torch.arange(len(labels), device=labels.device), labels[:, 0]]
    second = second.masked_fill(~legal[:, 1], -torch.inf)
    marginal = output["action_logits"][:, 1].detach().masked_fill(~legal[:, 1], -torch.inf)
    action = F.cross_entropy(first, labels[:, 0]) + F.cross_entropy(second, labels[:, 1])
    marginal_loss = F.cross_entropy(marginal, labels[:, 1])
    belief = F.kl_div(output["belief_logits"].log_softmax(-1), batch["belief_target"],
                      reduction="none").sum(-1).mean()
    value = F.mse_loss(output["value"], batch["value"])
    loss = action + belief + 0.5 * value
    return loss, {"loss": loss.detach().item(), "action_loss": action.detach().item(),
                  "marginal_loss": marginal_loss.detach().item(), "belief_kl": belief.detach().item(),
                  "value_mse": value.detach().item()}


def _move(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _evaluate(model, loader, device, check_deadline):
    model.eval()
    sums, counts = {}, 0
    action_hits = torch.zeros(2, device=device)
    blind_count, blind_nll, blind_hit3, blind_l1 = 0, 0., 0., 0.
    for batch in loader:
        check_deadline()
        batch = _move(batch, device)
        output = model(batch["x"], batch["landmarks"], batch["agents"])
        _, parts = training_loss(output, batch)
        size = len(batch["x"])
        counts += size
        for key, number in parts.items():
            sums[key] = sums.get(key, 0.) + number * size
        legal, labels = batch["legal"], batch["labels"]
        first = output["action_logits"][:, 0].masked_fill(~legal[:, 0], -torch.inf)
        chosen_first = first.argmax(-1)
        second = output["conditional_action_logits"][torch.arange(size, device=device), chosen_first]
        action_scores = torch.stack((first, second.masked_fill(~legal[:, 1], -torch.inf)), dim=1)
        top3 = action_scores.topk(3, dim=-1).indices
        hits = (top3 == labels[..., None]).any(-1)
        action_hits += hits.sum(0)
        log_belief = output["belief_logits"].log_softmax(-1)
        blind = ~batch["visible"]
        nll = -log_belief.gather(-1, batch["enemy"][..., None]).squeeze(-1)
        belief_hits = (log_belief.topk(min(3, log_belief.shape[-1]), dim=-1).indices
                       == batch["enemy"][..., None]).any(-1)
        cells = batch["cells"].float()
        true_cells = cells.gather(1, batch["enemy"][..., None].expand(-1, -1, 2))
        l1_distance = (cells[:, None] - true_cells[:, :, None]).abs().sum(-1)
        expected_l1 = (log_belief.exp() * l1_distance).sum(-1)
        blind_count += blind.sum().item()
        blind_nll += nll[blind].sum().item()
        blind_hit3 += belief_hits[blind].sum().item()
        blind_l1 += expected_l1[blind].sum().item()
    return {**{key: value / counts for key, value in sums.items()},
            "samples": counts, "action_top3_per_agent": (action_hits / counts).tolist(),
            "blind_count": blind_count,
            "blind_nll": blind_nll / blind_count if blind_count else None,
            "blind_hit3": blind_hit3 / blind_count if blind_count else None,
            "blind_expected_l1": blind_l1 / blind_count if blind_count else None}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(data_dir, out_dir, epochs=20, d=64, batch_size=64, seed=20261104, deadline=None):
    """Train every requested epoch. deadline, if set, is an absolute monotonic time."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("positive epochs and batch_size are required")
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "epochs.jsonl").exists() or (out_dir / "model.pt").exists():
        raise FileExistsError("use a fresh output directory to preserve training provenance")
    train_paths, val_paths = split_games(Path(data_dir).glob("*.npz"))
    dataset = EpisodeDataset(train_paths + val_paths)
    cut = dataset.offsets[len(train_paths)]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(dataset, range(cut)), batch_size=batch_size,
                              shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(Subset(dataset, range(cut, len(dataset))), batch_size=batch_size, num_workers=0)
    model = BeliefMiTA(d=d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    provenance = {
        "seed": seed, "device": str(device), "torch_version": str(torch.__version__),
        "batch_size": batch_size, "requested_epochs": epochs, "lr": 3e-4,
        "loss_weights": {"first_ce": 1., "conditional_second_ce": 1., "marginal_second_ce": 0.,
                         "belief_kl": 1., "value_mse": 0.5},
        "belief_target": "normalized graph Gaussian sigma=1 over walkable cells",
        "action_top3_second": "conditional policy given predicted first action",
        "blind_expected_l1": "posterior expected Manhattan distance to true enemy cell",
        "train_games": [path.name for path in train_paths], "val_games": [path.name for path in val_paths],
        "train_samples": cut, "val_samples": len(dataset) - cut,
        "files": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in dataset.paths],
        "model_source_sha256": _sha256(Path(__file__).with_name("model.py")),
        "training_source_sha256": _sha256(Path(__file__)),
    }
    summary = {"status": "running", "epochs_completed": 0, "epoch_in_progress": 0,
               "batches_completed_in_epoch": 0, "provenance": provenance}
    started = time.monotonic()

    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("training monotonic deadline reached")

    def save():
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        temporary = out_dir / "model.pt.tmp"
        torch.save({"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "d": d, "feature_dim": 18, "epochs": summary["epochs_completed"],
                    "status": summary["status"], "epoch_in_progress": summary["epoch_in_progress"],
                    "batches_completed_in_epoch": summary["batches_completed_in_epoch"],
                    "provenance": provenance}, temporary)
        temporary.replace(out_dir / "model.pt")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    try:
        check_deadline()
        with (out_dir / "epochs.jsonl").open("w", encoding="utf-8") as log:
            for epoch in range(1, epochs + 1):
                check_deadline()
                summary["epoch_in_progress"] = epoch
                summary["batches_completed_in_epoch"] = 0
                model.train()
                sums, count = {}, 0
                epoch_started = time.monotonic()
                for batch in train_loader:
                    check_deadline()
                    batch = _move(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(batch["x"], batch["landmarks"], batch["agents"])
                    loss, parts = training_loss(output, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                    size = len(batch["x"])
                    count += size
                    for key, number in parts.items():
                        sums[key] = sums.get(key, 0.) + number * size
                    summary["batches_completed_in_epoch"] += 1
                validation = _evaluate(model, val_loader, device, check_deadline)
                record = {"epoch": epoch, "train": {key: value / count for key, value in sums.items()},
                          "validation": validation, "seconds": time.monotonic() - epoch_started}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                summary.update(epochs_completed=epoch, epoch_in_progress=0, batches_completed_in_epoch=0,
                               validation=validation)
                save()
                print(json.dumps({"epoch": epoch, "train_loss": record["train"]["loss"],
                                  "val_loss": validation["loss"], "seconds": record["seconds"]}), flush=True)
        summary["status"] = "completed"
        save()
        return summary
    except (TimeoutError, RuntimeError, FloatingPointError) as error:
        summary["status"] = ("timeout" if isinstance(error, TimeoutError) else
                             "cuda_oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "error")
        summary["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--deadline-seconds", type=float)
    args = parser.parse_args()
    deadline = time.monotonic() + args.deadline_seconds if args.deadline_seconds is not None else None
    train(args.data_dir, args.out_dir, args.epochs, args.d, args.batch_size, args.seed, deadline)
