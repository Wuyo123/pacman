import importlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch
from unittest.mock import patch


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

    def versioned_episode(self, directory, game=0):
        path = self.episode(directory, game)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays['frames_v1'] = np.concatenate(
            [arrays['frames'], np.ones((*arrays['frames'].shape[:2], 19), dtype=np.float16)], axis=-1)
        arrays['observation_version'] = np.asarray(1, dtype=np.int64)
        np.savez_compressed(path, **arrays)
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

    def test_versioned_dataset_selects_matched_feature_array(self):
        training = self.training()
        with tempfile.TemporaryDirectory() as directory:
            path = self.episode(directory)
            with np.load(path) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["frames_v1"] = np.concatenate(
                [arrays["frames"], np.ones((*arrays["frames"].shape[:2], 19), dtype=np.float16)], axis=-1)
            arrays["observation_version"] = np.asarray(1, dtype=np.int64)
            np.savez_compressed(path, **arrays)
            old = training.EpisodeDataset([path], observation_version=0)
            new = training.EpisodeDataset([path], observation_version=1)
            self.assertEqual(old[0]["x"].shape[-1], 18)
            self.assertEqual(new[0]["x"].shape[-1], 37)
            torch.testing.assert_close(old[0]["x"], new[0]["x"][..., :18])

    def test_v1_initialization_copies_v0_weights_and_zeros_added_input_columns(self):
        training = self.training()
        old, new = training.initialize_paired_models(seed=23)
        old_state, new_state = old.state_dict(), new.state_dict()
        for key in old_state:
            if key == "temporal_encoder.weight_ih_l0":
                torch.testing.assert_close(new_state[key][:, :18], old_state[key], atol=0, rtol=0)
                torch.testing.assert_close(new_state[key][:, 18:], torch.zeros_like(new_state[key][:, 18:]), atol=0, rtol=0)
            else:
                torch.testing.assert_close(new_state[key], old_state[key], atol=0, rtol=0)
        old_hash, old_shared = training.model_initialization_hashes(old)
        new_hash, new_shared = training.model_initialization_hashes(new)
        self.assertNotEqual(old_hash, new_hash)
        self.assertEqual(old_shared, new_shared)
        cells = 4
        old_x = torch.randn(2, 8, cells, 18)
        new_x = torch.cat((old_x, torch.randn(2, 8, cells, 19)), dim=-1)
        landmarks = torch.arange(16).remainder(cells).expand(2, -1)
        agents = torch.tensor([[0, 1], [2, 3]])
        old.eval()
        new.eval()
        with torch.no_grad():
            old_output = old(old_x, landmarks, agents)
            new_output = new(new_x, landmarks, agents)
        for key in old_output:
            torch.testing.assert_close(old_output[key], new_output[key], atol=1e-6, rtol=1e-6)

    def test_resume_shuffle_matches_uninterrupted_epoch_and_skips_completed_batches(self):
        training = self.training()
        from torch.utils.data import DataLoader, TensorDataset
        values = torch.arange(10)
        generator = torch.Generator().manual_seed(20261105)
        uninterrupted = DataLoader(TensorDataset(values), batch_size=3, shuffle=True,
                                   generator=generator, num_workers=0)
        list(uninterrupted)
        expected = [batch[0].tolist() for batch in uninterrupted]
        resumed_generator = torch.Generator().manual_seed(20261105)
        resumed = DataLoader(TensorDataset(values), batch_size=3, shuffle=True,
                             generator=resumed_generator, num_workers=0)
        iterator = training.restore_train_iterator(resumed, completed_epochs=1,
                                                   batches_completed_in_epoch=1)
        actual = [batch[0].tolist() for batch in iterator]
        self.assertEqual(actual, expected[1:])

    def _assert_optimizer_states_equal(self, expected, actual):
        first, second = expected['optimizer_state_dict']['state'], actual['optimizer_state_dict']['state']
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            self.assertEqual(first[key].keys(), second[key].keys())
            for name, value in first[key].items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, second[key][name], atol=0, rtol=0)
                else:
                    self.assertEqual(value, second[key][name])

    def _assert_resumed_checkpoint_equals_full(self, full_path, resumed_path, expected_updates):
        full = torch.load(full_path, map_location='cpu', weights_only=False)
        resumed = torch.load(resumed_path, map_location='cpu', weights_only=False)
        self.assertEqual(full['epochs'], resumed['epochs'])
        self.assertEqual(full['observation_version'], resumed['observation_version'])
        for key, value in full['state_dict'].items():
            torch.testing.assert_close(value, resumed['state_dict'][key], atol=0, rtol=0)
        self._assert_optimizer_states_equal(full, resumed)
        steps = {int(state['step']) for state in resumed['optimizer_state_dict']['state'].values()}
        self.assertEqual(steps, {expected_updates})

    def test_mid_epoch_resume_matches_uninterrupted_model_and_adam_updates(self):
        training = self.training()
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.is_available', return_value=False):
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            self.versioned_episode(data, 0)
            self.versioned_episode(data, 1)
            full = training.train(data, root / 'full', epochs=3, d=8, batch_size=2,
                                  seed=20261105, observation_version=1)
            original_getitem = training.EpisodeDataset.__getitem__
            calls = 0
            def interrupt_after_one_batch(dataset, index):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise FloatingPointError('injected mid-epoch interruption')
                return original_getitem(dataset, index)
            with patch.object(training.EpisodeDataset, '__getitem__', interrupt_after_one_batch):
                with self.assertRaises(FloatingPointError):
                    training.train(data, root / 'partial', epochs=3, d=8, batch_size=2,
                                   seed=20261105, observation_version=1)
            partial = torch.load(root / 'partial/model.pt', map_location='cpu', weights_only=False)
            self.assertEqual(partial['epoch_in_progress'], 1)
            self.assertEqual(partial['batches_completed_in_epoch'], 1)
            resumed = training.train(data, root / 'resumed', epochs=3, d=8, batch_size=2,
                                     seed=20261105, observation_version=1,
                                     resume_from=root / 'partial/model.pt')
            self.assertEqual(resumed['epochs_completed'], 3)
            with (root / 'resumed/epochs.jsonl').open() as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual(records[0]['train_metrics_scope'], 'remaining_batches_after_resume')
            self._assert_resumed_checkpoint_equals_full(root / 'full/model.pt',
                                                       root / 'resumed/model.pt', 6)

    def test_validation_interruption_retries_validation_without_duplicate_updates(self):
        training = self.training()
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.is_available', return_value=False):
            root = Path(directory)
            data = root / 'data'
            data.mkdir()
            self.versioned_episode(data, 0)
            self.versioned_episode(data, 1)
            full = training.train(data, root / 'full', epochs=3, d=8, batch_size=2,
                                  seed=20261105, observation_version=1)
            with patch.object(training, '_evaluate', side_effect=FloatingPointError('injected validation interruption')):
                with self.assertRaises(FloatingPointError):
                    training.train(data, root / 'partial', epochs=3, d=8, batch_size=2,
                                   seed=20261105, observation_version=1)
            partial = torch.load(root / 'partial/model.pt', map_location='cpu', weights_only=False)
            self.assertEqual(partial['epoch_in_progress'], 1)
            self.assertEqual(partial['batches_completed_in_epoch'], 2)
            resumed = training.train(data, root / 'resumed', epochs=3, d=8, batch_size=2,
                                     seed=20261105, observation_version=1,
                                     resume_from=root / 'partial/model.pt')
            self.assertEqual(resumed['epochs_completed'], 3)
            with (root / 'resumed/epochs.jsonl').open() as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual(records[0]['train_metrics_scope'], 'validation_only')
            self.assertEqual(records[0]['train'], {})
            self._assert_resumed_checkpoint_equals_full(root / 'full/model.pt',
                                                       root / 'resumed/model.pt', 6)

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
