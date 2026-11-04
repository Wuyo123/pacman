"""Official CTF transitions, legal observation encoding, and full-state search expert."""

from collections import deque
from functools import lru_cache
import importlib.util
from pathlib import Path
import random
import sys
import time

import numpy as np

from .topology import MapTopology

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'vendor' / 'capture'
sys.path.insert(0, str(ENGINE))
import capture
import layout
from game import Actions, Configuration, Directions

ACTIONS = ('North', 'South', 'East', 'West', 'Stop')


def initial_state(name='defaultCapture', length=1200):
    if name.startswith('RANDOM'):
        maze = layout.Layout(capture.randomLayout(int(name[6:])).splitlines())
    else:
        path = ENGINE / 'layouts' / (name + '.lay')
        maze = layout.Layout(path.read_text().splitlines())
    state = capture.GameState()
    state.initialize(maze, 4)
    state.data.timeleft = length
    return state


@lru_cache(maxsize=32)
def topology_for(name):
    return MapTopology(initial_state(name).getWalls())


def seed_distances(state, topology):
    # Exact BFS distances replace only the framework's repeated UCS computation.
    import distanceCalculator
    walls = state.getWalls()
    if walls not in distanceCalculator.distanceMap:
        distanceCalculator.distanceMap[walls] = {
            (p, q): int(topology.distances[i, j])
            for i, p in enumerate(topology.cells) for j, q in enumerate(topology.cells)}


def load_team(name, team, red):
    files = {'baseline': ENGINE / 'baselineTeam.py', 'champion': ENGINE / 'myTeam.py',
             'mcts': ROOT / 'vendor' / 'mcts' / 'myTeam(1).py'}
    if name not in files:
        raise ValueError(f'unknown opponent {name}')
    spec = importlib.util.spec_from_file_location('ctf_' + name, files[name])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.createTeam(*team, red)


def validate_transition(before, after, actor, action):
    """Check action, geometry, time, respawn exceptions, and independent score accounting."""
    assert action in before.getLegalActions(actor), ('illegal_action', actor, action)
    assert after.data.timeleft == before.data.timeleft - 1, 'time did not advance once'
    for index in range(4):
        old, new = before.getAgentPosition(index), after.getAgentPosition(index)
        assert new is not None and not after.getWalls()[new[0]][new[1]], ('wall', index, new)
        spawn = after.getInitialAgentPosition(index)
        if index == actor:
            dx, dy = Actions.directionToVector(action)
            expected = (int(old[0] + dx), int(old[1] + dy))
            assert new in (expected, spawn), ('motion', index, old, new, action)
        else:
            assert new in (old, spawn), ('non_actor_motion', index, old, new)
    returned = sum(after.getAgentState(i).numReturned * (1 if i % 2 == 0 else -1)
                   for i in range(4))
    assert after.getScore() == returned, ('score', after.getScore(), returned)


class ObservationEncoder:
    def __init__(self, topology, team, red):
        self.topology, self.team, self.red = topology, list(team), red
        self.enemies = [i for i in range(4) if i not in team]
        self.frames = deque(maxlen=8)
        self.previous_defended = None
        self.xy = np.asarray(topology.cells, dtype=np.float32)
        self.static = np.zeros((len(topology.cells), 18), np.float32)
        self.static[:, 0:2] = self.xy / [max(1, topology.width - 1), max(1, topology.height - 1)]
        self.static[:, 2] = ((self.xy[:, 0] < topology.width / 2) == red)
        self.static[list(topology.articulation), 14] = 1
        self.static[:, 15] = topology.escape_depth / max(topology.width, topology.height)

    def update(self, observation, observer):
        self.last_observation = observation
        frame = self.static.copy()
        index = self.topology.index
        attack = observation.getBlueFood() if self.red else observation.getRedFood()
        defense = observation.getRedFood() if self.red else observation.getBlueFood()
        attack_caps = observation.getBlueCapsules() if self.red else observation.getRedCapsules()
        defense_caps = observation.getRedCapsules() if self.red else observation.getBlueCapsules()
        defended = set(defense.asList())
        for channel, points in ((3, attack.asList()), (4, defended), (5, attack_caps), (6, defense_caps)):
            for point in points:
                frame[index[point], channel] = 1
        for slot, own in enumerate(self.team):
            point = observation.getAgentPosition(own)
            frame[index[point], 7 + slot] = 1
            frame[index[point], 16] = observation.getAgentState(own).numCarrying / 30
        origin = observation.getAgentPosition(observer)
        sonar = observation.getAgentDistances()
        for slot, enemy in enumerate(self.enemies):
            point = observation.getAgentPosition(enemy)
            if point is not None:
                frame[index[point], 9 + slot] = 1
            if sonar:
                distance = np.abs(self.xy - origin).sum(1)
                frame[:, 11 + slot] = np.abs(distance - sonar[enemy]) <= 6
        if self.previous_defended is not None:
            for point in self.previous_defended - defended:
                frame[index[point], 13] = 1
        self.previous_defended = defended
        frame[:, 17] = observation.data.timeleft / 1200
        self.frames.append(frame)
        return frame

    def history_array(self):
        if not self.frames:
            raise ValueError('history needs an observation')
        frames = list(self.frames)
        return np.stack([frames[0]] * (8 - len(frames)) + frames)


class OraclePlanner:
    """Depth-three alpha-beta expert with full-state interception/escape evaluation."""
    def __init__(self, topology, team, red, depth=3):
        self.topology, self.team, self.red, self.depth = topology, list(team), red, depth
        self.enemies = [i for i in range(4) if i not in team]
        self.home_boundary = [i for i in topology.boundary
                              if (topology.cells[i][0] < topology.width / 2) == red]
        if not self.home_boundary:
            self.home_boundary = list(range(len(topology.cells)))
        self.home_distance = topology.distances[:, self.home_boundary].min(1)

    def _distance(self, first, second):
        return int(self.topology.distances[self.topology.index[first], self.topology.index[second]])

    def targets(self, state):
        invaders = [e for e in self.enemies if state.getAgentState(e).isPacman]
        targets = {}
        if invaders:
            # Assign distinct gates/targets to the closest available defenders.
            candidates = []
            for enemy in invaders:
                point = state.getAgentPosition(enemy)
                enemy_id = self.topology.index[point]
                gate = int(self.topology.gate[enemy_id])
                candidates.append(self.topology.cells[gate] if gate >= 0 else point)
            for own in self.team:
                pos = state.getAgentPosition(own)
                target = min(candidates, key=lambda point: self._distance(pos, point))
                targets[own] = target
                if len(candidates) > 1:
                    candidates.remove(target)
        return targets

    def utility(self, state, focus, targets=None):
        sign = 1 if self.red else -1
        score = 140 * sign * state.getScore()
        attack = state.getBlueFood() if self.red else state.getRedFood()
        food_ids = [self.topology.index[p] for p in attack.asList()]
        own_state = state.getAgentState(focus)
        pos = own_state.getPosition()
        cell = self.topology.index[(int(pos[0]), int(pos[1]))]
        carry = own_state.numCarrying
        score += 10 * sum(state.getAgentState(i).numCarrying for i in self.team)
        score -= 12 * sum(state.getAgentState(i).numCarrying for i in self.enemies)
        target = (targets or {}).get(focus)
        if focus == self.team[1] and target is not None and own_state.scaredTimer == 0:
            score -= 5 * self._distance(self.topology.cells[cell], target)
            score -= 25 * own_state.isPacman
        elif focus == self.team[1] and not own_state.isPacman:
            # A second attacker is allowed if there is no invading opponent.
            if food_ids:
                score -= 2 * int(self.topology.distances[cell, food_ids].min())
        elif carry >= 3 or state.data.timeleft < 100:
            score -= 5 * int(self.home_distance[cell])
        elif food_ids:
            score -= 3 * int(self.topology.distances[cell, food_ids].min())
        if own_state.isPacman:
            for enemy in self.enemies:
                enemy_state = state.getAgentState(enemy)
                if not enemy_state.isPacman and enemy_state.scaredTimer == 0:
                    enemy_cell = self.topology.index[state.getAgentPosition(enemy)]
                    distance = int(self.topology.distances[cell, enemy_cell])
                    score -= 70 * max(0, 3 - distance)
                    gate = int(self.topology.gate[cell])
                    if gate >= 0 and self.topology.escape_depth[cell] + 2 >= self.topology.distances[enemy_cell, gate]:
                        score -= 25 + 3 * int(self.topology.escape_depth[cell])
        return float(score)

    def _search(self, state, actor, depth, focus, targets, alpha, beta):
        if depth == 0 or state.isOver() or state.data.timeleft <= 0:
            return self.utility(state, focus, targets)
        maximizing = actor in self.team
        candidates = [(action, state.generateSuccessor(actor, action))
                      for action in state.getLegalActions(actor)]
        candidates.sort(key=lambda item: self.utility(item[1], focus, targets), reverse=maximizing)
        best = -float('inf') if maximizing else float('inf')
        for action, successor in candidates:
            value = self._search(successor, (actor + 1) % 4, depth - 1, focus, targets, alpha, beta)
            if maximizing:
                best, alpha = max(best, value), max(alpha, value)
            else:
                best, beta = min(best, value), min(beta, value)
            if beta <= alpha:
                break
        return best

    def choose(self, state, actor):
        targets = self.targets(state)
        best_value, best_action = -float('inf'), 'Stop'
        for action in state.getLegalActions(actor):
            successor = state.generateSuccessor(actor, action)
            value = self._search(successor, (actor + 1) % 4, self.depth - 1,
                                 actor, targets, best_value, float('inf'))
            # Prefer motion on ties, without making oracle targets stochastic.
            value -= 0.15 * (action == 'Stop')
            if value > best_value:
                best_value, best_action = value, action
        return best_action

    def plan(self, state, actor):
        return np.asarray([ACTIONS.index(self.choose(state, own)) for own in self.team], dtype=np.int64)


def play_game(seed, red=True, opponent='baseline', name='defaultCapture', length=1200,
              collect=False, controller=None, deadline=None):
    random.seed(seed)
    np.random.seed(seed)
    state = initial_state(name, length)
    topology = topology_for(name)
    seed_distances(state, topology)
    team = [0, 2] if red else [1, 3]
    other = [i for i in range(4) if i not in team]
    opponents = dict(zip(other, load_team(opponent, other, not red)))
    for agent in opponents.values():
        agent.registerInitialState(state.deepCopy())
    encoder = ObservationEncoder(topology, team, red)
    oracle = OraclePlanner(topology, team, red)
    if controller is not None:
        controller.start(state.makeObservation(team[0]), topology, team, red)
    landmarks = topology.landmark_indices(state.getRedFood().asList() + state.getBlueFood().asList(),
                                         state.getCapsules(), 16)
    rows = {key: [] for key in ('frames', 'agents', 'labels', 'enemy', 'visible', 'legal')}
    times, deaths, moves = [], [0, 0, 0, 0], []
    actor = random.randint(0, 1)
    started = time.perf_counter()
    for turn in range(length):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError('M0 wall-clock deadline during game')
        if state.isOver():
            break
        observation = state.makeObservation(actor)
        begin = time.perf_counter()
        if actor in team:
            frame = encoder.update(observation, actor)
            if controller is None:
                joint_plan = oracle.plan(state, actor)
                action = ACTIONS[int(joint_plan[team.index(actor)])]
            else:
                environment_rng = random.getstate()
                try:
                    action = controller.choose(observation, actor, encoder, landmarks, begin)
                finally:
                    # Simulated sonar draws must not change actual opponent/noise randomness.
                    random.setstate(environment_rng)
            if collect:
                assert controller is None, 'teacher labels must come from oracle'
                rows['frames'].append(frame)
                rows['agents'].append([topology.index[observation.getAgentPosition(i)] for i in team])
                rows['labels'].append(joint_plan)
                rows['enemy'].append([topology.index[state.getAgentPosition(i)] for i in other])
                rows['visible'].append([observation.getAgentPosition(i) is not None for i in other])
                rows['legal'].append([[a in observation.getLegalActions(i) for a in ACTIONS] for i in team])
        else:
            action = opponents[actor].getAction(observation)
        elapsed = (time.perf_counter() - begin) * 1000
        if actor in team:
            times.append(elapsed)
        before = state
        state = state.generateSuccessor(actor, action)
        validate_transition(before, state, actor, action)
        for index in range(4):
            old, new = before.getAgentPosition(index), state.getAgentPosition(index)
            if new == state.getInitialAgentPosition(index) and old != new:
                # Ordinary arrival at spawn is not a death.
                dx, dy = Actions.directionToVector(action) if index == actor else (0, 0)
                if new != (int(old[0] + dx), int(old[1] + dy)):
                    deaths[index] += 1
        moves.append([actor, ACTIONS.index(action)])
        actor = (actor + 1) % 4
    signed_score = state.getScore() * (1 if red else -1)
    result = {'seed': seed, 'red': red, 'opponent': opponent, 'layout': name,
              'score': signed_score, 'outcome': int(np.sign(signed_score)), 'moves': len(moves),
              'own_decisions': len(times), 'seconds': time.perf_counter() - started,
              'own_ms_p50': float(np.percentile(times, 50)), 'own_ms_p95': float(np.percentile(times, 95)),
              'own_ms_max': float(max(times)), 'over_50ms': sum(t > 50 for t in times),
              'deaths': [deaths[i] for i in team], 'validated': True,
              'action_trace': moves}
    if collect:
        episode = {key: np.asarray(value, dtype=np.float16 if key == 'frames' else
                    np.bool_ if key in ('visible', 'legal') else np.int64) for key, value in rows.items()}
        episode.update(values=np.full(len(rows['frames']), result['outcome'], np.float32),
                       landmarks=landmarks, distances=topology.distances, cells=np.asarray(topology.cells))
        return result, episode
    return result
