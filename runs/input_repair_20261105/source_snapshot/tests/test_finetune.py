import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from belief_mita.model import BeliefMiTA


class FinetuneTests(unittest.TestCase):
    def module(self):
        path = Path(__file__).parents[1] / "belief_mita" / "finetune.py"
        self.assertTrue(path.is_file(), "fixed-budget finetuning has not been implemented")
        return importlib.import_module("belief_mita.finetune")

    def fixture(self, root):
        old, new = root / "old", root / "new"
        old.mkdir()
        new.mkdir()
        rng = np.random.default_rng(5)
        for directory, games in ((old, 200), (new, 20)):
            for game in range(games):
                np.savez_compressed(directory / f"game_{game:04d}.npz",
                                    frames=rng.normal(size=(1, 4, 18)).astype(np.float16),
                                    agents=np.array([[0, 1]], dtype=np.int64),
                                    labels=np.array([[2, 3]], dtype=np.int64),
                                    enemy=np.array([[0, 3]], dtype=np.int64),
                                    visible=np.array([[False, True]]),
                                    legal=np.ones((1, 2, 5), dtype=bool),
                                    values=np.array([1.], dtype=np.float32),
                                    landmarks=np.arange(16, dtype=np.int64) % 4,
                                    distances=np.abs(np.arange(4)[:, None] - np.arange(4)[None]).astype(np.int32),
                                    cells=np.stack((np.arange(4), np.zeros(4, dtype=int)), axis=1),
                                    student_action=np.array([4]), actor_slot=np.array([0]))
        torch.manual_seed(29)
        base = BeliefMiTA()
        with torch.no_grad():
            base.action_head.weight.fill_(0.123)
        checkpoint = root / "base.pt"
        torch.save({"state_dict": base.state_dict(), "d": 64, "feature_dim": 18,
                    "epochs": 20, "status": "completed"}, checkpoint)
        return old, new, checkpoint

    def test_balanced_sampling_has_exact_counts_and_fixed_seed(self):
        module = self.module()
        balanced = module.epoch_indices(180, 20, 24000, True, torch.Generator().manual_seed(20261105))
        self.assertEqual(len(balanced), 24000)
        self.assertEqual((balanced < 180).sum().item(), 12000)
        self.assertEqual((balanced >= 180).sum().item(), 12000)
        self.assertTrue(((balanced >= 0) & (balanced < 200)).all())
        again = module.epoch_indices(180, 20, 24000, True, torch.Generator().manual_seed(20261105))
        torch.testing.assert_close(balanced, again)
        old = module.epoch_indices(180, 0, 24000, False, torch.Generator().manual_seed(20261105))
        self.assertEqual(len(old), 24000)
        self.assertTrue((old < 180).all())

    def test_one_cpu_minibatch_updates_base_and_keeps_validation_games_out(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new, base = self.fixture(root)
            result = module.finetune(old, new, base, root / "out", True,
                                     epochs=1, samples_per_epoch=8, device="cpu")
            saved = torch.load(root / "out" / "model.pt", map_location="cpu", weights_only=True)
            self.assertEqual((result["status"], saved["epochs"]), ("completed", 1))
            self.assertTrue(np.isfinite(result["validation"]["loss"]))
            self.assertFalse(torch.equal(saved["state_dict"]["action_head.weight"],
                                         torch.full_like(saved["state_dict"]["action_head.weight"], 0.123)))
            provenance = saved["provenance"]
            self.assertEqual(len(provenance["old_train_games"]), 180)
            self.assertEqual(len(provenance["old_val_games"]), 20)
            self.assertFalse(set(provenance["old_train_games"]) & set(provenance["old_val_games"]))
            self.assertTrue(provenance["old_val_games"][0].endswith("game_0180.npz"))
            self.assertEqual(saved["base_checkpoint_sha256"], hashlib.sha256(base.read_bytes()).hexdigest())
            record = json.loads((root / "out" / "epochs.jsonl").read_text())
            self.assertEqual(record["sample_counts"], {"old": 4, "recovery": 4})
            self.assertEqual(record["steps"], 1)

    def test_expired_deadline_preserves_exact_loaded_base(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new, base = self.fixture(root)
            with self.assertRaises(TimeoutError):
                module.finetune(old, new, base, root / "out", False, epochs=1,
                                samples_per_epoch=8, deadline=time.monotonic() - 1, device="cpu")
            saved = torch.load(root / "out" / "model.pt", map_location="cpu", weights_only=True)
            original = torch.load(base, map_location="cpu", weights_only=True)
            self.assertEqual((saved["status"], saved["epochs"]), ("timeout", 0))
            for key, value in original["state_dict"].items():
                torch.testing.assert_close(saved["state_dict"][key], value, atol=0, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
