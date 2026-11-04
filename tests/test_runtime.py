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


if __name__ == '__main__':
    unittest.main()
