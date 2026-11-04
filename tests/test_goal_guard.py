import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from belief_mita.runtime import ACTIONS, Actions
from belief_mita.topology import MapTopology


class Walls:
    width, height = 10, 5

    def __getitem__(self, x):
        return [x in (0, 9) or y in (0, 4) for y in range(self.height)]


class Food:
    def __init__(self, cells):
        self.cells = list(cells)

    def asList(self):
        return list(self.cells)


class PublicObservation:
    def __init__(self, topology, actor=0, position=(6, 2), carry=0, returned=0, timeleft=1200):
        self.topology = topology
        self.data = SimpleNamespace(timeleft=timeleft)
        self.positions = {0: (2, 1), 1: None, 2: (2, 3), 3: None}
        self.positions[actor] = position
        self.agents = {i: SimpleNamespace(numCarrying=0, numReturned=0,
                                         scaredTimer=0, isPacman=False) for i in range(4)}
        self.agents[actor].numCarrying = carry
        self.agents[actor].numReturned = returned
        self.blue_food = [(7, 2), (8, 1), (8, 3)]
        self.red_food = [(2, 2), (1, 1), (1, 3)]

    def getAgentPosition(self, actor):
        return self.positions[actor]

    def getAgentState(self, actor):
        return self.agents[actor]

    def getLegalActions(self, actor):
        x, y = self.positions[actor]
        result = []
        for action in ACTIONS:
            dx, dy = Actions.directionToVector(action)
            if (int(x + dx), int(y + dy)) in self.topology.index:
                result.append(action)
        return result

    def getBlueFood(self):
        return Food(self.blue_food)

    def getRedFood(self):
        return Food(self.red_food)

    def getBlueTeamIndices(self):
        return [1, 3]

    def getRedTeamIndices(self):
        return [0, 2]


class GoalGuardTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / 'belief_mita' / 'goal_guard.py'
        self.assertTrue(path.is_file(), 'public goal-progress controller has not been implemented')
        spec = importlib.util.spec_from_file_location('belief_mita.goal_guard', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.topology = MapTopology(Walls())

    def choose(self, obs, red=True, actor=0, allowed=None):
        return self.module.choose_goal_action(obs, actor, self.topology, red, allowed)

    def test_carry_three_returns_home_instead_of_eating_nearby_food(self):
        observation = PublicObservation(self.topology, carry=3)
        self.assertEqual(self.choose(observation), 'West')

    def test_no_carry_pursues_food_even_at_low_time(self):
        observation = PublicObservation(self.topology, timeleft=99)
        self.assertEqual(self.choose(observation), 'East')

    def test_low_time_or_last_food_returns_only_when_carrying(self):
        observation = PublicObservation(self.topology, carry=1, timeleft=99)
        self.assertEqual(self.choose(observation), 'West')
        observation.data.timeleft = 1200
        observation.blue_food = [(7, 2), (8, 2)]
        self.assertEqual(self.choose(observation), 'West')
        observation.agents[0].numCarrying = 0
        self.assertEqual(self.choose(observation), 'East')

    def test_blue_return_direction_uses_blue_home_boundary(self):
        observation = PublicObservation(self.topology, actor=1, position=(3, 2), carry=3)
        observation.positions[0] = observation.positions[2] = None
        self.assertEqual(self.choose(observation, red=False, actor=1), 'East')
        observation.agents[1].numCarrying = 0
        self.assertEqual(self.choose(observation, red=False, actor=1), 'West')

    def test_mask_is_respected_even_when_other_action_has_better_goal_distance(self):
        observation = PublicObservation(self.topology)
        mask = np.array([True, False, False, False, False])
        self.assertEqual(self.choose(observation, allowed=mask), 'North')
        with self.assertRaises(ValueError):
            self.choose(observation, allowed=np.zeros(5, dtype=bool))
        observation.positions[0] = (1, 1)
        with self.assertRaises(ValueError):
            self.choose(observation, allowed=[False, True, False, False, False])

    def test_visible_unscared_ghost_blocks_dangerous_food_approach(self):
        observation = PublicObservation(self.topology)
        observation.positions[1] = (8, 2)
        observation.blue_food = [(8, 2), (8, 1), (8, 3)]
        self.assertEqual(self.choose(observation), 'North')
        observation.agents[1].scaredTimer = 20
        self.assertEqual(self.choose(observation), 'East')

    def test_visible_ghost_can_block_pocket_via_gate_before_direct_contact(self):
        observation = PublicObservation(self.topology)
        observation.positions[1] = (5, 3)
        pocket = self.topology.index[7, 2]
        self.topology.gate[pocket] = self.topology.index[5, 2]
        self.topology.escape_depth[pocket] = 2
        self.assertEqual(self.choose(observation), 'South')

    def test_hidden_enemy_internal_position_is_never_read(self):
        observation = PublicObservation(self.topology)
        def forbidden():
            raise AssertionError('hidden-state position accessed')
        for enemy in (1, 3):
            observation.agents[enemy].getPosition = forbidden
        self.assertEqual(self.choose(observation), 'East')

    def test_guard_triggers_only_after_eight_persistent_stall_observations(self):
        guard = self.module.ProgressGuard(window=8)
        observation = PublicObservation(self.topology)
        for step in range(7):
            observation.positions[0] = ((6, 2), (7, 2))[step % 2]
            self.assertFalse(guard.should_override(observation, 0))
        self.assertTrue(guard.should_override(observation, 0))
        guard.reset()
        self.assertFalse(guard.should_override(observation, 0))

    def test_guard_progress_and_distinct_motion_prevent_override(self):
        guard = self.module.ProgressGuard()
        observation = PublicObservation(self.topology)
        for _ in range(8):
            guard.should_override(observation, 0)
        observation.agents[0].numCarrying = 1
        self.assertFalse(guard.should_override(observation, 0))
        for _ in range(7):
            guard.should_override(observation, 0)
        observation.agents[0].numCarrying = 0
        observation.agents[0].numReturned = 1
        self.assertFalse(guard.should_override(observation, 0))
        guard.reset()
        for x in range(1, 9):
            observation.positions[0] = (x, 2)
            self.assertFalse(guard.should_override(observation, 0))

    def test_guard_tracks_actors_independently(self):
        guard = self.module.ProgressGuard()
        observation = PublicObservation(self.topology)
        for _ in range(7):
            guard.should_override(observation, 0)
        self.assertFalse(guard.should_override(observation, 2))
        self.assertTrue(guard.should_override(observation, 0))

    def test_heuristic_controller_uses_only_goal_rule_and_compatible_statistics(self):
        controller = self.module.HeuristicController()
        observation = PublicObservation(self.topology, carry=3)
        controller.start(observation, self.topology, [0, 2], True)
        action = controller.choose(observation, 0, None, None, 0)
        self.assertEqual(action, 'West')
        self.assertEqual(controller.statistics['decisions'], 1)
        self.assertEqual(controller.statistics['simulations'], 0)
        self.assertEqual(controller.statistics['forward_ms'], [])
        self.assertEqual(controller.statistics['root_action_choices'], [0, 0, 0, 0])
        observation.agents[0].numCarrying = 0
        observation.blue_food = []
        self.assertEqual(controller.choose(observation, 0, None, None, 0), 'Stop')
        self.assertEqual(controller.statistics['decisions'], 2)
        self.assertEqual(controller.statistics['chosen_stop'], 1)


if __name__ == '__main__':
    unittest.main()
