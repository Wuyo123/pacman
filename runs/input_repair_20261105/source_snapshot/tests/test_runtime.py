import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from belief_mita import runtime
except ImportError:
    runtime = None


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runtime, 'CTF runtime has not been implemented')
        self.state = runtime.initial_state('defaultCapture')
        self.topology = runtime.MapTopology(self.state.getWalls())

    def test_encoder_cannot_see_hidden_oracle_positions(self):
        import numpy as np
        observation = self.state.makeObservation(0)
        self.assertIsNone(observation.getAgentPosition(1))
        changed = observation.deepCopy()
        # Mutating an unexposed oracle state must never enter the encoder.
        full = self.state.deepCopy()
        full.data.agentStates[1].configuration.pos = self.topology.cells[-4]
        encoder1 = runtime.ObservationEncoder(self.topology, [0, 2], True)
        encoder2 = runtime.ObservationEncoder(self.topology, [0, 2], True)
        np.testing.assert_array_equal(encoder1.update(observation, 0), encoder2.update(changed, 0))
        self.assertEqual(encoder1.history_array().shape, (8, len(self.topology.cells), 18))
        self.assertEqual(float(encoder1.frames[-1][:, 9].sum()), 0.0)

    def test_actual_observer_sonar_ring(self):
        observation = self.state.makeObservation(2)
        encoder = runtime.ObservationEncoder(self.topology, [0, 2], True)
        frame = encoder.update(observation, 2)
        pos = observation.getAgentPosition(2)
        distance = observation.getAgentDistances()[1]
        expected = [abs(abs(x-pos[0])+abs(y-pos[1])-distance) <= 6 for x,y in self.topology.cells]
        self.assertEqual(frame[:, 11].tolist(), [float(v) for v in expected])

    def test_v1_preserves_v0_channels_and_encodes_public_stats_by_agent_and_observer(self):
        import numpy as np
        observation = self.state.makeObservation(2)
        for agent, state in enumerate(observation.data.agentStates):
            state.numCarrying = agent + 1
            state.numReturned = 2 * (agent + 1)
            state.isPacman = agent % 2 == 1
            state.scaredTimer = agent * 10
        observation.data.score = 6
        v0 = runtime.ObservationEncoder(self.topology, [0, 2], True, version=0)
        v1 = runtime.ObservationEncoder(self.topology, [0, 2], True, version=1)
        frame0 = v0.update(observation, 2)
        frame1 = v1.update(observation, 2)
        self.assertEqual(frame0.shape[-1], 18)
        self.assertEqual(frame1.shape[-1], 37)
        np.testing.assert_array_equal(frame1[:, :18], frame0)
        # V1 order is team slots followed by enemies: agents [0, 2, 1, 3].
        for slot, agent in enumerate([0, 2, 1, 3]):
            np.testing.assert_array_equal(frame1[:, 18 + slot], (agent + 1) / 30)
            np.testing.assert_array_equal(frame1[:, 22 + slot], (2 * (agent + 1)) / 60)
            np.testing.assert_array_equal(frame1[:, 27 + slot], float(agent % 2 == 1))
            np.testing.assert_array_equal(frame1[:, 31 + slot], (agent * 10) / 40)
        np.testing.assert_array_equal(frame1[:, 26], 6 / 60)
        np.testing.assert_array_equal(frame1[:, 35], 0.)
        np.testing.assert_array_equal(frame1[:, 36], 1.)

    def test_v1_uses_only_observation_for_hidden_positions(self):
        import numpy as np
        from belief_mita.runtime import Configuration
        first_state = self.state.deepCopy()
        second_state = self.state.deepCopy()
        hidden_cells = [point for point in self.topology.cells
                        if point[0] >= self.topology.width / 2
                        and all(abs(point[0] - first_state.getAgentPosition(i)[0])
                                + abs(point[1] - first_state.getAgentPosition(i)[1]) > 5 for i in [0, 2])]
        second_state.data.agentStates[1].configuration = Configuration(hidden_cells[-1], 'Stop')
        observation = first_state.makeObservation(0)
        changed_hidden = second_state.makeObservation(0)
        self.assertIsNone(observation.getAgentPosition(1))
        self.assertIsNone(changed_hidden.getAgentPosition(1))
        changed_hidden.agentDistances = observation.agentDistances[:]
        first = runtime.ObservationEncoder(self.topology, [0, 2], True, version=1)
        second = runtime.ObservationEncoder(self.topology, [0, 2], True, version=1)
        np.testing.assert_array_equal(first.update(observation, 0), second.update(changed_hidden, 0))
        self.assertEqual(first.history_array().shape, (8, len(self.topology.cells), 37))

    def test_v1_separates_carrying_at_shared_positions_and_public_timer(self):
        import numpy as np
        observation = self.state.makeObservation(0)
        shared = observation.getAgentPosition(0)
        observation.data.agentStates[2].configuration.pos = shared
        observation.data.agentStates[0].numCarrying = 3
        observation.data.agentStates[2].numCarrying = 11
        first = runtime.ObservationEncoder(self.topology, [0, 2], True, version=1).update(observation, 0)
        changed = observation.deepCopy()
        changed.data.agentStates[0].numCarrying = 4
        changed.data.agentStates[3].scaredTimer = 7
        second = runtime.ObservationEncoder(self.topology, [0, 2], True, version=1).update(changed, 0)
        np.testing.assert_array_equal(first[:, :18], second[:, :18])
        np.testing.assert_array_equal(first[:, 18], 3 / 30)
        np.testing.assert_array_equal(second[:, 18], 4 / 30)
        np.testing.assert_array_equal(first[:, 19], second[:, 19])
        np.testing.assert_array_equal(first[:, 34], 0.)
        np.testing.assert_array_equal(second[:, 34], 7 / 40)

    def test_v1_score_is_team_signed_and_observer_slot_is_ordered_for_blue(self):
        import numpy as np
        observation = self.state.makeObservation(3)
        observation.data.score = 6
        encoder = runtime.ObservationEncoder(self.topology, [1, 3], False, version=1)
        frame = encoder.update(observation, 3)
        np.testing.assert_array_equal(frame[:, 26], -6 / 60)
        np.testing.assert_array_equal(frame[:, 35], 0.)
        np.testing.assert_array_equal(frame[:, 36], 1.)
        self.assertEqual(encoder.team + encoder.enemies, [1, 3, 0, 2])

    def test_transition_validation_rejects_illegal_action(self):
        action = self.state.getLegalActions(0)[0]
        successor = self.state.generateSuccessor(0, action)
        runtime.validate_transition(self.state, successor, 0, action)
        with self.assertRaises(AssertionError):
            runtime.validate_transition(self.state, successor, 0, 'INVALID')

    def test_oracle_joint_plan_is_legal(self):
        oracle = runtime.OraclePlanner(self.topology, [0, 2], True, depth=3)
        plan = oracle.plan(self.state, 0)
        self.assertEqual(len(plan), 2)
        for slot, index in enumerate([0, 2]):
            self.assertIn(runtime.ACTIONS[plan[slot]], self.state.getLegalActions(index))

    def test_oracle_does_not_plan_after_official_time_expiry(self):
        oracle = runtime.OraclePlanner(self.topology, [0, 2], True, depth=3)
        self.state.data.timeleft = 1
        seen_times = []
        original = oracle.utility
        def observed_utility(state, focus, targets=None):
            seen_times.append(state.data.timeleft)
            return original(state, focus, targets)
        oracle.utility = observed_utility
        oracle.choose(self.state, 0)
        self.assertGreaterEqual(min(seen_times), 0, 'expert used states beyond game expiry')

    def test_student_rollout_collects_expert_labels_without_exposing_truth(self):
        class StopStudent:
            def start(self, observation, topology, team, red):
                self.team = team
            def choose(self, observation, actor, encoder, landmarks, started):
                for enemy in [i for i in range(4) if i not in self.team]:
                    assert observation.getAgentPosition(enemy) is None
                return 'Stop'
        result, episode = runtime.play_game(8000012, length=8, collect=True, controller=StopStudent())
        self.assertTrue(result['validated'])
        self.assertEqual(episode['student_action'].tolist(), [4]*len(episode['frames']))
        self.assertEqual(episode['labels'].shape, (len(episode['frames']),2))
        self.assertTrue((episode['frames'][:,:,9:11] == 0).all())

    def test_v1_collection_keeps_matched_v0_frames_from_same_observation(self):
        result, episode = runtime.play_game(8000013, length=8, collect=True, observation_version=1)
        self.assertTrue(result['validated'])
        self.assertEqual(episode['frames'].shape[-1], 18)
        self.assertEqual(episode['frames_v1'].shape[-1], 37)
        import numpy as np
        np.testing.assert_array_equal(episode['frames_v1'][:, :, :18], episode['frames'])


if __name__ == '__main__':
    unittest.main()
