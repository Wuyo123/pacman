"""Belief-sampled cooperative PUCT with the original dead-end threshold shield."""

from collections import deque
import copy
from dataclasses import dataclass, field
import math
from pathlib import Path
import time

import numpy as np
import torch

from .model import BeliefMiTA
from .runtime import ACTIONS, Actions, Configuration, ObservationEncoder, OraclePlanner
from .schema import validate_checkpoint_schema


def joint_top3(first, conditional_second, legal, acting_slot=None):
    joint = first[:, None] * conditional_second
    joint *= legal[0, :, None] & legal[1, None, :]
    if acting_slot is None:
        choices = np.argsort(-joint.ravel(), kind='stable')[:3]
    elif acting_slot in (0, 1):
        candidates = []
        for action in range(5):
            if acting_slot == 0:
                partner = int(joint[action].argmax())
                candidates.append(5*action + partner)
            else:
                partner = int(joint[:,action].argmax())
                candidates.append(5*partner + action)
        choices = sorted(candidates, key=lambda i: (-float(joint.ravel()[i]), i))[:3]
    else:
        raise ValueError('acting_slot must be0,1 or None')
    choices = [int(i) for i in choices if joint.ravel()[i] > 0]
    total = sum(joint.ravel()[i] for i in choices)
    return [((i // 5, i % 5), float(joint.ravel()[i] / total)) for i in choices]


def risky_cells(topology, belief, threatening, tau=0.01, delta=2):
    risky = np.zeros(len(topology.gate), dtype=bool)
    for cell, gate in enumerate(topology.gate):
        if gate < 0:
            continue
        fastest = math.inf
        for slot, active in enumerate(threatening):
            if active:
                support = np.flatnonzero(belief[slot] > tau)
                if len(support):
                    fastest = min(fastest, float(topology.distances[support, gate].min()))
        risky[cell] = topology.escape_depth[cell] + delta >= fastest
    return risky


def evidence_belief(observation, belief, topology, team, red):
    """Restrict sampling to public roles and exact sighting/nonvisibility evidence."""
    result = belief.copy()
    enemies = [i for i in range(4) if i not in team]
    xy = np.asarray(topology.cells)
    own_positions = [observation.getAgentPosition(i) for i in team]
    visible_region = np.zeros(len(xy), bool)
    for point in own_positions:
        visible_region |= np.abs(xy - point).sum(1) <= 5
    for slot, enemy in enumerate(enemies):
        point = observation.getAgentPosition(enemy)
        if point is not None:
            result[slot] = 0
            result[slot, topology.index[point]] = 1
            continue
        enemy_invades = observation.getAgentState(enemy).isPacman
        on_our_side = (xy[:, 0] < topology.width / 2) == red
        legal = ~visible_region & (on_our_side == enemy_invades)
        result[slot] *= legal
        if result[slot].sum() <= 0:
            result[slot] = legal.astype(float)
        if result[slot].sum() <= 0:
            raise ValueError('no hidden location satisfies public role and nonvisibility')
        result[slot] /= result[slot].sum()
    return result


def sampled_state(observation, belief, topology, team, red, rng):
    belief = evidence_belief(observation, belief, topology, team, red)
    state = observation.deepCopy()
    for slot, enemy in enumerate(i for i in range(4) if i not in team):
        if observation.getAgentPosition(enemy) is None:
            cell = int(rng.choice(len(topology.cells), p=belief[slot]))
            state.data.agentStates[enemy].configuration = Configuration(topology.cells[cell], 'Stop')
    return state


@dataclass
class Edge:
    plan: tuple
    prior: float
    visits: int = 0
    total: float = 0.
    observations: dict = field(default_factory=dict)


class NeuralController:
    def __init__(self, checkpoint, seed, budget_ms=50, device='cpu', mode='original'):
        if mode not in ('original', 'diverse', 'policy', 'guard'):
            raise ValueError('unknown controller mode')
        torch.set_num_threads(1)
        checkpoint = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
        if not isinstance(checkpoint.get('epochs'),int) or checkpoint['epochs'] < 1 or checkpoint['status'] != 'completed':
            raise ValueError('evaluation requires a completed training checkpoint')
        self.observation_version = validate_checkpoint_schema(checkpoint)
        self.model = BeliefMiTA(feature_dim=checkpoint.get('feature_dim', 18), d=checkpoint['d']).to(device).eval()
        self.model.load_state_dict(checkpoint['state_dict'])
        self.device, self.budget = torch.device(device), budget_ms / 1000
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.statistics = {'decisions': 0, 'simulations': 0, 'shield_removed': 0,
                           'shield_fallbacks': 0, 'forward_ms': [], 'deadline_stops': 0,
                           'root_action_choices': [0,0,0,0], 'forced_despite_alternatives': 0,
                           'search_changed_policy': 0, 'chosen_stop': 0, 'goal_guard_overrides': 0,
                           'root_visited_edges': [0, 0, 0, 0], 'simulation_depth': [0, 0, 0, 0],
                           'observation_hits': 0, 'observation_misses': 0}

    def start(self, observation, topology, team, red):
        self.topology, self.team, self.red = topology, list(team), red
        self.enemies = [i for i in range(4) if i not in team]
        self.opponent = OraclePlanner(topology, self.enemies, not red, depth=1)
        if self.mode == 'guard':
            from .goal_guard import ProgressGuard
            self.progress_guard = ProgressGuard(window=8)
        # Warm up without accessing any hidden state or targets.
        encoder = ObservationEncoder(topology, team, red, version=self.observation_version)
        encoder.update(observation, team[0])
        landmarks = topology.landmark_indices(observation.getRedFood().asList() +
                    observation.getBlueFood().asList(), observation.getCapsules())
        for _ in range(3):
            self.infer(observation, encoder, landmarks)
        self.statistics['forward_ms'].clear()

    @torch.inference_mode()
    def infer(self, observation, encoder, landmarks):
        started = time.perf_counter()
        x = torch.from_numpy(encoder.history_array()[None]).to(self.device)
        ids = torch.tensor([[self.topology.index[observation.getAgentPosition(i)]
                             for i in self.team]], device=self.device)
        lands = torch.as_tensor(landmarks[None], dtype=torch.long, device=self.device)
        output = self.model(x, lands, ids)
        belief = output['belief_logits'].softmax(-1)[0].cpu().numpy().astype(np.float64)
        first = output['action_logits'][0, 0].softmax(-1).cpu().numpy()
        second = output['conditional_action_logits'][0].softmax(-1).cpu().numpy()
        value = float(output['value'][0].cpu())
        self.statistics['forward_ms'].append((time.perf_counter() - started) * 1000)
        return evidence_belief(observation, belief, self.topology, self.team, self.red), first, second, value

    def plans(self, observation, belief, first, second, count_statistics=False, actor=None):
        legal = np.asarray([[a in observation.getLegalActions(i) for a in ACTIONS] for i in self.team])
        raw = legal.copy()
        threatening = [not observation.getAgentState(i).isPacman and
                       observation.getAgentState(i).scaredTimer == 0 for i in self.enemies]
        risks = risky_cells(self.topology, belief, threatening)
        for slot, own in enumerate(self.team):
            origin = observation.getAgentPosition(own)
            for action_id in np.flatnonzero(legal[slot]):
                dx, dy = Actions.directionToVector(ACTIONS[action_id])
                point = (int(origin[0] + dx), int(origin[1] + dy))
                invading = (point[0] < self.topology.width / 2) != self.red
                if invading and risks[self.topology.index[point]]:
                    legal[slot, action_id] = False
            if not legal[slot].any():
                # No safe action exists under this literal shield; retreat toward its exit.
                choices = np.flatnonzero(raw[slot])
                ranked = []
                for candidate in choices:
                    dx, dy = Actions.directionToVector(ACTIONS[int(candidate)])
                    point = (int(origin[0] + dx), int(origin[1] + dy))
                    ranked.append((self.topology.escape_depth[self.topology.index[point]], int(candidate)))
                action_id = min(ranked)[1]
                legal[slot, action_id] = True
                if count_statistics:
                    self.statistics['shield_fallbacks'] += 1
        if count_statistics:
            self.statistics['shield_removed'] += int((raw & ~legal).sum())
            self.root_legal = legal.copy()
        acting_slot = self.team.index(actor) if self.mode == 'diverse' else None
        return [Edge(plan, prior) for plan, prior in joint_top3(first, second, legal, acting_slot)]

    def macro_step(self, state, actor, plan, encoder):
        encoder = copy.copy(encoder)
        encoder.frames = deque(encoder.frames, maxlen=8)
        for step in range(4):
            if state.isOver() or state.data.timeleft <= 0:
                break
            mover = (actor + step) % 4
            if mover in self.team:
                action = ACTIONS[plan[self.team.index(mover)]]
                if action not in state.getLegalActions(mover):
                    action = 'Stop'
            else:
                action = self.opponent.choose(state, mover)
            state = state.generateSuccessor(mover, action)
            following = (mover + 1) % 4
            if following in self.team and not state.isOver() and state.data.timeleft > 0:
                encoder.update(state.makeObservation(following), following)
        return state, encoder

    def choose(self, observation, actor, encoder, landmarks, started):
        deadline = started + self.budget
        belief, first, second, _ = self.infer(observation, encoder, landmarks)
        root = self.plans(observation, belief, first, second, count_statistics=True, actor=actor)
        if not root:
            raise RuntimeError('no legal joint plan')
        slot = self.team.index(actor)
        count = len({edge.plan[slot] for edge in root})
        self.statistics['root_action_choices'][count] += 1
        self.statistics['forced_despite_alternatives'] += count == 1 and len(observation.getLegalActions(actor)) > 1
        policy_best = max(root, key=lambda edge: edge.prior)
        if self.mode == 'policy':
            self.statistics['decisions'] += 1
            self.statistics['root_visited_edges'][0] += 1
            self.statistics['chosen_stop'] += policy_best.plan[slot] == 4
            return ACTIONS[policy_best.plan[slot]]
        simulations = 0
        while time.perf_counter() < deadline - 0.012:
            state = sampled_state(observation, belief, self.topology, self.team, self.red, self.rng)
            history, edges, path, discount = encoder, root, [], 1.
            value = 0.
            initial_score = state.getScore()
            for depth in range(3):
                if time.perf_counter() >= deadline - 0.008:
                    break
                total_visits = sum(edge.visits for edge in edges)
                edge = max(edges, key=lambda item: (item.total / item.visits if item.visits else 0.) +
                           1.5 * item.prior * math.sqrt(total_visits + 1) / (1 + item.visits))
                path.append(edge)
                state, history = self.macro_step(state, actor, edge.plan, history)
                if state.isOver() or state.data.timeleft <= 0:
                    value = float(np.sign(state.getScore() * (1 if self.red else -1)))
                    break
                future = history.last_observation
                if time.perf_counter() >= deadline - 0.006:
                    break
                next_belief, next_first, next_second, value = self.infer(future, history, landmarks)
                # Branch on observable histories, not on inaccessible sampled coordinates.
                signature = (tuple(future.getAgentPosition(i) for i in range(4)),
                             tuple(future.getAgentDistances()), future.getScore(),
                             tuple(future.getRedFood().asList()), tuple(future.getBlueFood().asList()))
                if signature not in edge.observations:
                    self.statistics['observation_misses'] += 1
                    edge.observations[signature] = self.plans(future, next_belief, next_first, next_second, actor=actor)
                    break
                self.statistics['observation_hits'] += 1
                edges = edge.observations[signature]
                if not edges:
                    break
                discount *= 0.99
            score_reward = (state.getScore() - initial_score) * (1 if self.red else -1) / 10
            value = float(np.clip(score_reward + discount * value, -1, 1))
            for edge in path:
                edge.visits += 1
                edge.total += value
            simulations += 1
            self.statistics['simulation_depth'][len(path)] += 1
        self.statistics['decisions'] += 1
        self.statistics['simulations'] += simulations
        self.statistics['root_visited_edges'][sum(edge.visits > 0 for edge in root)] += 1
        if simulations == 0:
            self.statistics['deadline_stops'] += 1
        best = max(root, key=lambda edge: (edge.visits, edge.total / max(1, edge.visits), edge.prior))
        self.statistics['search_changed_policy'] += best.plan[slot] != policy_best.plan[slot]
        chosen = ACTIONS[best.plan[slot]]
        if self.mode == 'guard' and self.progress_guard.should_override(observation,actor):
            from .goal_guard import choose_goal_action
            alternative = choose_goal_action(observation,actor,self.topology,self.red,allowed=self.root_legal[slot])
            self.statistics['goal_guard_overrides'] += alternative != chosen
            chosen = alternative
        self.statistics['chosen_stop'] += chosen == 'Stop'
        return chosen
