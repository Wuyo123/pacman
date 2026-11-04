import unittest
import numpy as np
try:
    from belief_mita.search import joint_top3, risky_cells, sampled_state
except ImportError:
    joint_top3 = risky_cells = sampled_state = None


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(joint_top3, 'shielded cooperative PUCT not implemented')

    def test_pruning_returns_three_legal_joint_plans(self):
        first = np.array([0.4, 0.1, 0.1, 0.1, 0.3])
        second = np.full((5, 5), 0.2)
        legal = np.array([[True, False, True, False, True], [False, True, False, True, True]])
        plans = joint_top3(first, second, legal)
        self.assertEqual(len(plans), 3)
        for (a, b), prior in plans:
            self.assertTrue(legal[0, a] and legal[1, b])
            self.assertGreater(prior, 0)
        self.assertAlmostEqual(sum(p for _, p in plans), 1.0)

    def test_literal_threshold_shield(self):
        from types import SimpleNamespace
        topology = SimpleNamespace(gate=np.array([-1, 0, 0]), escape_depth=np.array([0, 1, 2]),
                                   distances=np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]))
        belief = np.array([[0., 0., 1.], [1., 0., 0.]])
        risky = risky_cells(topology, belief, [True, False], tau=0.01, delta=2)
        self.assertEqual(risky.tolist(), [False, True, True])
        self.assertFalse(risky_cells(topology, belief, [False, False]).any())

    def test_sampled_opponents_obey_visibility_and_role(self):
        from belief_mita.runtime import initial_state, topology_for
        state = initial_state()
        observation = state.makeObservation(0)
        topology = topology_for('defaultCapture')
        belief = np.full((2, len(topology.cells)), 1 / len(topology.cells))
        sample = sampled_state(observation, belief, topology, [0, 2], True, np.random.default_rng(9))
        for enemy in [1, 3]:
            position = sample.getAgentPosition(enemy)
            self.assertIn(position, topology.index)
            self.assertFalse(position[0] < topology.width / 2)
            self.assertTrue(all(abs(position[0]-observation.getAgentPosition(i)[0]) +
                                abs(position[1]-observation.getAgentPosition(i)[1]) > 5 for i in [0, 2]))
        self.assertIsNone(observation.getAgentPosition(1))


if __name__ == '__main__':
    unittest.main()
