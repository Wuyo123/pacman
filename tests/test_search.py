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

    def test_current_actor_receives_distinct_actions_instead_of_duplicate_plans(self):
        import inspect
        self.assertIn('acting_slot', inspect.signature(joint_top3).parameters,
                      'pruning has no awareness of the current actor')
        first = np.array([0.96, 0.01, 0.01, 0.01, 0.01])
        conditional = np.full((5,5), 0.2)
        legal = np.ones((2,5), dtype=bool)
        original = joint_top3(first, conditional, legal)
        self.assertEqual(len({plan[0] for plan,_ in original}), 1)
        for slot in [0,1]:
            improved = joint_top3(first, conditional, legal, acting_slot=slot)
            self.assertEqual(len({plan[slot] for plan,_ in improved}), 3)
            self.assertAlmostEqual(sum(prior for _,prior in improved), 1.0)
            self.assertTrue(all(legal[0,plan[0]] and legal[1,plan[1]] for plan,_ in improved))

    def test_diverse_pruning_retains_only_available_current_actions(self):
        import inspect
        self.assertIn('acting_slot', inspect.signature(joint_top3).parameters)
        legal = np.array([[False,True,False,False,True],[True,False,False,False,False]])
        plans = joint_top3(np.ones(5)/5,np.ones((5,5))/5,legal,acting_slot=0)
        self.assertEqual({plan[0] for plan,_ in plans}, {1,4})
        self.assertTrue(all(plan[1] == 0 for plan,_ in plans))

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
