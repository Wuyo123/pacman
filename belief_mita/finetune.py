"""One fixed DAgger aggregation round and its equal-budget old-data control."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .model import BeliefMiTA
from .training import EpisodeDataset, _evaluate, _move, _sha256, split_games, training_loss


def epoch_indices(old_count, recovery_count, samples, use_recovery, generator):
    """Uniform replacement sampling with an exact arm-specific source budget."""
    if samples < 1 or old_count < 1 or (use_recovery and (recovery_count < 1 or samples % 2)):
        raise ValueError("positive source counts and an even balanced sample budget are required")
    old_samples = samples // 2 if use_recovery else samples
    indices = torch.randint(old_count, (old_samples,), generator=generator)
    if use_recovery:
        new_indices = torch.randint(recovery_count, (samples // 2,), generator=generator) + old_count
        indices = torch.cat((indices, new_indices))
    return indices[torch.randperm(samples, generator=generator)]


def finetune(old_data_dir, recovery_data_dir, base_checkpoint, out_dir, use_recovery,
             epochs=5, samples_per_epoch=24000, deadline=None, *, device=None):
    """Append fixed epochs from the same v0 weights; deadline uses time.monotonic()."""
    started = time.monotonic()
    if epochs < 1 or samples_per_epoch < 1 or (use_recovery and samples_per_epoch % 2):
        raise ValueError("positive epochs/sample budget and an even recovery budget are required")
    torch.set_num_threads(1)
    seed, batch_size = 20261105, 64
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if device is None else device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official finetuning; CPU is an explicit test override")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    old_paths = sorted(Path(old_data_dir).glob("*.npz"))
    if len(old_paths) != 200:
        raise ValueError("the frozen old dataset must contain exactly 200 games")
    old_train_paths, old_val_paths = split_games(old_paths)
    recovery_paths = sorted(Path(recovery_data_dir).glob("*.npz")) if use_recovery else []
    if use_recovery and len(recovery_paths) != 20:
        raise ValueError("the recovery dataset must contain exactly 20 games")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / name).exists() for name in ("model.pt", "epochs.jsonl", "summary.json")):
        raise FileExistsError("use a fresh output directory to preserve both experiment arms")
    base_checkpoint = Path(base_checkpoint)
    base_hash = _sha256(base_checkpoint)
    base = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    if base.get("status") != "completed" or base.get("epochs") != 20:
        raise ValueError("base must be the completed twenty-epoch v0 checkpoint")
    if base.get("d") != 64 or base.get("feature_dim") != 18:
        raise ValueError("the fixed recovery protocol requires the original d64, feature_dim18 model")
    model = BeliefMiTA(d=64)
    model.load_state_dict(base["state_dict"], strict=True)
    del base
    optimizer = None
    provenance = {
        "arm": "recovery" if use_recovery else "old_extra", "seed": seed, "device": str(device),
        "base_checkpoint": str(base_checkpoint.resolve()), "base_checkpoint_sha256": base_hash,
        "old_data_dir": str(Path(old_data_dir).resolve()),
        "recovery_data_dir": str(Path(recovery_data_dir).resolve()),
        "old_train_games": [str(path.resolve()) for path in old_train_paths],
        "old_val_games": [str(path.resolve()) for path in old_val_paths],
        "recovery_games": [str(path.resolve()) for path in recovery_paths],
        "requested_epochs": epochs, "samples_per_epoch": samples_per_epoch,
        "mix_per_epoch": {"old": samples_per_epoch // 2 if use_recovery else samples_per_epoch,
                          "recovery": samples_per_epoch // 2 if use_recovery else 0},
        "sampling": "uniform with replacement per source, followed by a seeded shuffle",
        "batch_size": batch_size, "lr": 3e-4, "fresh_optimizer": True,
        "loss": "first CE + conditional second CE + graph-Gaussian KL + 0.5 value MSE",
        "torch_version": str(torch.__version__),
        "source_sha256": {name: _sha256(Path(__file__).with_name(name))
                          for name in ("finetune.py", "training.py", "model.py")},
    }
    summary = {"status": "running", "epochs_completed": 0, "epoch_in_progress": 0,
               "batches_completed_in_epoch": 0, "provenance": provenance}

    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("fixed-budget finetuning monotonic deadline reached")

    def save():
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["peak_gpu_memory_bytes"] = (torch.cuda.max_memory_allocated(device)
                                             if device.type == "cuda" else 0)
        checkpoint = {"state_dict": model.state_dict(), "d": 64, "feature_dim": 18,
                      "epochs": summary["epochs_completed"], "status": summary["status"],
                      "epoch_in_progress": summary["epoch_in_progress"],
                      "batches_completed_in_epoch": summary["batches_completed_in_epoch"],
                      "base_checkpoint_sha256": base_hash, "provenance": provenance}
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        temporary = out_dir / "model.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(out_dir / "model.pt")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    try:
        check_deadline()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        old_data = EpisodeDataset(old_train_paths + old_val_paths)
        check_deadline()
        old_count = old_data.offsets[180]
        train_data = Subset(old_data, range(old_count))
        val_data = Subset(old_data, range(old_count, len(old_data)))
        recovery_count = 0
        if use_recovery:
            recovery_data = EpisodeDataset(recovery_paths)
            recovery_count = len(recovery_data)
            train_data = ConcatDataset((train_data, recovery_data))
        provenance.update(old_train_samples=old_count, old_val_samples=len(val_data),
                          recovery_samples=recovery_count)
        provenance["data_sha256"] = []
        for path in old_paths + recovery_paths:
            check_deadline()
            provenance["data_sha256"].append({"path": str(path.resolve()), "sha256": _sha256(path)})
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0)
        generator = torch.Generator().manual_seed(seed)
        with (out_dir / "epochs.jsonl").open("w", encoding="utf-8") as log:
            for epoch in range(1, epochs + 1):
                check_deadline()
                epoch_started = time.monotonic()
                summary.update(epoch_in_progress=epoch, batches_completed_in_epoch=0)
                indices = epoch_indices(old_count, recovery_count, samples_per_epoch, use_recovery, generator)
                sample_counts = {"old": int((indices < old_count).sum()),
                                 "recovery": int((indices >= old_count).sum())}
                loader = DataLoader(Subset(train_data, indices.tolist()), batch_size=batch_size, num_workers=0)
                model.train()
                sums, count = {}, 0
                for batch in loader:
                    check_deadline()
                    batch = _move(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(batch["x"], batch["landmarks"], batch["agents"])
                    loss, parts = training_loss(output, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite finetuning loss")
                    loss.backward()
                    optimizer.step()
                    size = len(batch["x"])
                    count += size
                    for key, value in parts.items():
                        sums[key] = sums.get(key, 0.) + value * size
                    summary["batches_completed_in_epoch"] += 1
                validation = _evaluate(model, val_loader, device, check_deadline)
                record = {"epoch": epoch, "sample_counts": sample_counts,
                          "steps": summary["batches_completed_in_epoch"],
                          "train": {key: value / count for key, value in sums.items()},
                          "validation": validation, "seconds": time.monotonic() - epoch_started}
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                summary.update(epochs_completed=epoch, epoch_in_progress=0, batches_completed_in_epoch=0,
                               validation=validation)
                save()
                print(json.dumps({"arm": provenance["arm"], "epoch": epoch,
                                  "train_loss": record["train"]["loss"],
                                  "val_loss": validation["loss"], "seconds": record["seconds"]}), flush=True)
        summary["status"] = "completed"
        save()
        return summary
    except (Exception, KeyboardInterrupt) as error:
        summary["status"] = ("timeout" if isinstance(error, TimeoutError) else
                             "cuda_oom" if isinstance(error, torch.cuda.OutOfMemoryError) else
                             "interrupted" if isinstance(error, KeyboardInterrupt) else "error")
        summary["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("old_data_dir")
    parser.add_argument("recovery_data_dir")
    parser.add_argument("base_checkpoint")
    parser.add_argument("out_dir")
    parser.add_argument("--use-recovery", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-epoch", type=int, default=24000)
    parser.add_argument("--deadline-seconds", type=float)
    args = parser.parse_args()
    deadline = time.monotonic() + args.deadline_seconds if args.deadline_seconds is not None else None
    finetune(args.old_data_dir, args.recovery_data_dir, args.base_checkpoint, args.out_dir, args.use_recovery,
             epochs=args.epochs, samples_per_epoch=args.samples_per_epoch, deadline=deadline)
