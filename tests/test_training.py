import importlib
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch


class TrainingTests(unittest.TestCase):
    def training(self):
        path = Path(__file__).parents[1] / "belief_mita" / "training.py"
        self.assertTrue(path.is_file(), "M0 training has not been implemented")
        return importlib.import_module("belief_mita.training")

    def episode(self, directory, game=0):
        path = Path(directory) / f"game_{game:04d}.npz"
        frames = np.broadcast_to(np.arange(3, dtype=np.float16)[:, None, None], (3, 4, 18)).copy()
        frames += game * 10
        np.savez_compressed(path, frames=frames, agents=np.tile([0, 1], (3, 1)),
                            labels=np.tile([2, 3], (3, 1)), enemy=np.tile([0, 3], (3, 1)),
                            visible=np.tile([False, True], (3, 1)),
                            legal=np.ones((3, 2, 5), dtype=bool), values=np.ones(3, dtype=np.float32),
                            cells=np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.int64),
                            landmarks=np.arange(16, dtype=np.int64) % 4,
                            distances=np.array([[0, 1, 2, -1], [1, 0, 1, -1],
                                                [2, 1, 0, -1], [-1, -1, -1, 0]], dtype=np.int32))
        return path

    def test_histories_repeat_first_frame_and_never_cross_games(self):
        training = self.training()
        with tempfile.TemporaryDirectory() as directory:
            dataset = training.EpisodeDataset([self.episode(directory), self.episode(directory, 1)])
            self.assertEqual(len(dataset), 6)
            torch.testing.assert_close(dataset[0]["x"], torch.zeros(8, 4, 18))
            torch.testing.assert_close(dataset[2]["x"][:, 0, 0],
                                       torch.tensor([0., 0., 0., 0., 0., 0., 1., 2.]))
            torch.testing.assert_close(dataset[3]["x"], torch.full((8, 4, 18), 10.))
            self.assertEqual(tuple(dataset[2]["belief_target"].shape), (2, 4))

    def test_game_split_is_exact_and_disjoint(self):
        training = self.training()
        games = [Path(f"game_{i:04d}.npz") for i in reversed(range(200))]
        train, val = training.split_games(games)
        self.assertEqual((len(train), len(val)), (180, 20))
        self.assertEqual(train[0].name, "game_0000.npz")
        self.assertEqual(val[0].name, "game_0180.npz")
        self.assertFalse(set(train) & set(val))
        self.assertEqual(tuple(map(len, training.split_games(games[:2]))), (1, 1))

    def test_graph_gaussian_uses_shortest_path_distance_and_excludes_unreachable(self):
        training = self.training()
        distances = torch.tensor([[0, 1, 2, -1], [1, 0, 1, -1],
                                  [2, 1, 0, -1], [-1, -1, -1, 0]])
        q = training.graph_gaussian(distances, torch.tensor([0, 3]))
        expected = torch.tensor([1., np.exp(-0.5), np.exp(-2.), 0.], dtype=torch.float32)
        expected /= expected.sum()
        torch.testing.assert_close(q[0], expected)
        torch.testing.assert_close(q[1], torch.tensor([0., 0., 0., 1.]))

    def test_cpu_loss_updates_parameters_with_a_real_minibatch(self):
        training = self.training()
        from belief_mita.model import BeliefMiTA
        with tempfile.TemporaryDirectory() as directory:
            dataset = training.EpisodeDataset([self.episode(directory)])
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
            model = BeliefMiTA()
            before = model.action_head.weight.detach().clone()
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            loss, parts = training.training_loss(model(batch["x"], batch["landmarks"], batch["agents"]), batch)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(np.isfinite(v) for v in parts.values()))
            loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.action_head.weight))

    def test_total_loss_matches_original_joint_action_belief_and_value_objective(self):
        training = self.training()
        torch.manual_seed(17)
        output = {"action_logits": torch.randn(2, 2, 5, requires_grad=True),
                  "conditional_action_logits": torch.randn(2, 5, 5, requires_grad=True),
                  "belief_logits": torch.randn(2, 2, 4, requires_grad=True),
                  "value": torch.randn(2, requires_grad=True)}
        batch = {"labels": torch.tensor([[0, 1], [2, 3]]),
                 "legal": torch.ones(2, 2, 5, dtype=torch.bool),
                 "belief_target": torch.randn(2, 2, 4).softmax(-1),
                 "value": torch.tensor([1., -1.])}
        loss, _ = training.training_loss(output, batch)
        functional = torch.nn.functional
        expected = (functional.cross_entropy(output["action_logits"][:, 0], batch["labels"][:, 0])
                    + functional.cross_entropy(output["conditional_action_logits"][torch.arange(2),
                                                                                     batch["labels"][:, 0]],
                                               batch["labels"][:, 1])
                    + functional.kl_div(output["belief_logits"].log_softmax(-1), batch["belief_target"],
                                        reduction="none").sum(-1).mean()
                    + 0.5 * functional.mse_loss(output["value"], batch["value"]))
        torch.testing.assert_close(loss, expected, atol=0, rtol=0)
        loss.backward()
        torch.testing.assert_close(output["action_logits"].grad[:, 1], torch.zeros(2, 5), atol=0, rtol=0)

    def test_expired_deadline_records_checkpoint_and_raises(self):
        training = self.training()
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            data.mkdir()
            self.episode(data)
            self.episode(data, 1)
            out = Path(directory) / "out"
            with self.assertRaises(TimeoutError):
                training.train(data, out, epochs=1, batch_size=2, deadline=time.monotonic() - 1)
            self.assertTrue((out / "model.pt").is_file())
            self.assertIn('"status": "timeout"', (out / "summary.json").read_text())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
