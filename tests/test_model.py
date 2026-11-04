import importlib.util
from pathlib import Path
import unittest

import torch


class ModelTests(unittest.TestCase):
    def model(self, **kwargs):
        path = Path(__file__).parents[1] / "belief_mita" / "model.py"
        self.assertTrue(path.is_file(), "BeliefMiTA model has not been implemented")
        spec = importlib.util.spec_from_file_location("belief_mita_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(7)
        return module.BeliefMiTA(**kwargs)

    def inputs(self, n=23, batch=2):
        torch.manual_seed(11)
        x = torch.randn(batch, 8, n, 18)
        landmarks = torch.arange(16).remainder(n).expand(batch, -1)
        agents = torch.tensor([0, 1]).expand(batch, -1)
        return x, landmarks, agents

    def test_contract_and_finite_outputs_on_different_map_sizes(self):
        model = self.model()
        for n in (7, 23, 51):
            out = model(*self.inputs(n))
            expected = {"belief_logits": (2, 2, n), "action_logits": (2, 2, 5),
                        "conditional_action_logits": (2, 5, 5), "value": (2,)}
            self.assertEqual(set(out), set(expected))
            for key, shape in expected.items():
                self.assertEqual(tuple(out[key].shape), shape)
                self.assertTrue(torch.isfinite(out[key]).all(), key)
            self.assertTrue((out["value"].abs() <= 1).all())

    def test_sparse_routes_and_legal_belief_distributions(self):
        model = self.model()
        x, landmarks, agents = self.inputs()
        mask = torch.arange(23)[None, :] < torch.tensor([[20], [5]])
        landmarks = landmarks.remainder(torch.tensor([[20], [5]]))
        out = model(x, landmarks, agents, mask, return_attention=True)
        weights = out["attention"]["landmark_weights"]
        self.assertEqual(tuple(weights.shape), (2, 4, 16, 23))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 16))
        self.assertTrue(((weights > 0).sum(-1) <= 12).all())
        self.assertEqual(weights.masked_select(~mask[:, None, None, :]).abs().sum(), 0)
        agent_weights = out["attention"]["agent_weights"]
        self.assertEqual(tuple(agent_weights.shape), (2, 4, 2, 20))
        torch.testing.assert_close(agent_weights.sum(-1), torch.ones(2, 4, 2))
        belief = out["belief_logits"].softmax(-1)
        self.assertEqual(belief.masked_select(~mask[:, None, :]).sum(), 0)
        torch.testing.assert_close(belief.sum(-1), torch.ones(2, 2))
        torch.testing.assert_close(out["conditional_action_logits"].softmax(-1).sum(-1),
                                   torch.ones(2, 5))

    def test_padding_does_not_change_real_cell_outputs(self):
        model = self.model().eval()
        x, landmarks, agents = self.inputs(19, batch=1)
        out = model(x, landmarks, agents)
        padded = torch.cat((x, torch.randn(1, 8, 9, 18) * 1000), dim=2)
        mask = torch.arange(28)[None, :] < 19
        padded_out = model(padded, landmarks, agents, mask)
        for key in ("action_logits", "conditional_action_logits", "value"):
            torch.testing.assert_close(out[key], padded_out[key], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(out["belief_logits"], padded_out["belief_logits"][..., :19],
                                   atol=2e-6, rtol=2e-5)

    def test_gradients_reach_learned_routes_and_temporal_history(self):
        model = self.model()
        x, landmarks, agents = self.inputs()
        x.requires_grad_()
        mask = torch.arange(23)[None, :].expand(2, -1) < 21
        out = model(x, landmarks, agents, mask, return_attention=True)
        loss = (out["value"].square().sum() + out["action_logits"].square().mean()
                + out["conditional_action_logits"].square().mean()
                + out["belief_logits"][..., :21].square().mean())
        loss.backward()
        for layer in (model.landmark_encoder, model.key_projection, model.temporal_encoder):
            grads = [p.grad for p in layer.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
        self.assertGreater(x.grad[:, 0, :21].abs().sum().item(), 0)
        self.assertEqual(x.grad[:, :, 21:].abs().sum().item(), 0)

    def test_global_path_reads_cells_excluded_from_all_sparse_routes(self):
        model = self.model(top_k=1).eval()
        x, landmarks, agents = self.inputs(97, batch=1)
        sparse, global_tokens = [], []
        sparse_hook = model.expert_norm.register_forward_hook(
            lambda module, args, output: sparse.append(output.detach().clone()))
        global_hook = model.compress_up.register_forward_hook(
            lambda module, args, output: global_tokens.append(output.detach().clone()))
        try:
            with torch.no_grad():
                baseline = model(x, landmarks, agents, return_attention=True)
                excluded = set(baseline["attention"]["top_indices"].flatten().tolist())
                excluded.update(landmarks.flatten().tolist())
                excluded.update(agents.flatten().tolist())
                cell = next(i for i in range(97) if i not in excluded)
                changed = x.clone()
                changed[:, :, cell] += 0.01
                perturbed = model(changed, landmarks, agents, return_attention=True)
            torch.testing.assert_close(baseline["attention"]["top_indices"],
                                       perturbed["attention"]["top_indices"])
            torch.testing.assert_close(sparse[0], sparse[1], atol=0, rtol=0)
            self.assertFalse(torch.equal(global_tokens[0], global_tokens[1]),
                             "global tokens must include cells outside every sparse route")
            self.assertFalse(torch.equal(baseline["action_logits"], perturbed["action_logits"]))
        finally:
            sparse_hook.remove()
            global_hook.remove()

    def test_invalid_selected_cells_and_empty_maps_are_rejected(self):
        model = self.model()
        x, landmarks, agents = self.inputs()
        with self.assertRaisesRegex(ValueError, "legal cell"):
            model(x, landmarks, agents, torch.zeros(2, 23, dtype=torch.bool))
        mask = torch.ones(2, 23, dtype=torch.bool)
        mask[:, 0] = False
        with self.assertRaisesRegex(ValueError, "landmark|agent"):
            model(x, landmarks, agents, mask)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
