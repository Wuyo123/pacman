"""Public-information goal progress for the bounded M0 follow-up."""

from collections import deque

from .runtime import ACTIONS, Actions


def choose_goal_action(observation, actor, topology, red, allowed=None):
    """Rank actual legal, allowed actions by goal distance and visible danger.

    A mask with no actual legal action raises ValueError; no masked action is
    restored. With no objective, staying still is preferred when permitted.
    """
    legal = observation.getLegalActions(actor)
    candidates = [action for i, action in enumerate(ACTIONS)
                  if action in legal and (allowed is None or allowed[i])]
    if not candidates:
        raise ValueError('allowed mask contains no actual legal action')
    own = observation.getAgentState(actor)
    food = (observation.getBlueFood() if red else observation.getRedFood()).asList()
    carry = own.numCarrying
    returning = carry >= 3 or (carry > 0 and
                              (observation.data.timeleft < 100 or len(food) <= 2))
    if returning:
        targets = sorted(i for i in topology.boundary
                         if (topology.cells[i][0] < topology.width // 2) == red)
    else:
        targets = [topology.index[point] for point in food]
    enemies = observation.getBlueTeamIndices() if red else observation.getRedTeamIndices()
    ghosts = []
    for enemy in enemies:
        point = observation.getAgentPosition(enemy)
        enemy_state = observation.getAgentState(enemy)
        if point is not None and not enemy_state.isPacman and enemy_state.scaredTimer == 0:
            ghosts.append(topology.index[point])
    origin = observation.getAgentPosition(actor)

    def score(action):
        dx, dy = Actions.directionToVector(action)
        point = (int(origin[0] + dx), int(origin[1] + dy))
        cell = topology.index[point]
        distance = int(topology.distances[cell, targets].min()) if targets else (action != 'Stop')
        value = -distance - 0.25 * (action == 'Stop')
        invading = (point[0] < topology.width // 2) != red
        if invading:
            if any(topology.distances[cell, ghost] <= 2 for ghost in ghosts):
                value -= 1000
            gate = int(topology.gate[cell])
            if gate >= 0 and any(topology.escape_depth[cell] + 2 >=
                                 topology.distances[ghost, gate] for ghost in ghosts):
                value -= 1000
        return value

    return max(candidates, key=score)


class ProgressGuard:
    def __init__(self, window=8):
        if window < 1:
            raise ValueError('window must be positive')
        self.window = window
        self.histories = {}

    def reset(self):
        self.histories.clear()

    def should_override(self, observation, actor):
        history = self.histories.setdefault(actor, deque(maxlen=self.window))
        own = observation.getAgentState(actor)
        history.append((observation.getAgentPosition(actor), own.numCarrying, own.numReturned))
        return (len(history) == self.window and
                len({row[0] for row in history}) <= 4 and
                len({row[1:] for row in history}) == 1)


class HeuristicController:
    def __init__(self):
        self.statistics = {'decisions': 0, 'simulations': 0, 'shield_removed': 0,
                           'shield_fallbacks': 0, 'forward_ms': [], 'deadline_stops': 0,
                           'root_action_choices': [0, 0, 0, 0], 'forced_despite_alternatives': 0,
                           'search_changed_policy': 0, 'chosen_stop': 0}

    def start(self, observation, topology, team, red):
        self.topology, self.team, self.red = topology, list(team), red

    def choose(self, observation, actor, encoder, landmarks, started):
        action = choose_goal_action(observation, actor, self.topology, self.red)
        self.statistics['decisions'] += 1
        self.statistics['chosen_stop'] += action == 'Stop'
        return action
