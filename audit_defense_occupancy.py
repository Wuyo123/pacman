"""Read-only occupancy audit of already sealed goal-control action traces."""

import json
from belief_mita.runtime import ACTIONS, Actions, ROOT, initial_state, validate_transition


def audit(record):
    state = initial_state(record['layout'])
    team = [0, 2] if record['red'] else [1, 3]
    enemies = [i for i in range(4) if i not in team]
    counts = dict(defender_decisions=0, defender_home=0, defender_scared=0,
                  invasion_decisions=0, defender_home_during_invasion=0,
                  both_away_during_invasion=0, enemy_respawns=0)
    for actor, action_id in record['action_trace']:
        if actor == team[1]:
            own = state.getAgentState(actor)
            invaders = [i for i in enemies if state.getAgentState(i).isPacman]
            counts['defender_decisions'] += 1
            counts['defender_home'] += not own.isPacman
            counts['defender_scared'] += own.scaredTimer > 0
            if invaders:
                counts['invasion_decisions'] += 1
                counts['defender_home_during_invasion'] += not own.isPacman
                counts['both_away_during_invasion'] += all(state.getAgentState(i).isPacman for i in team)
        after = state.generateSuccessor(actor, ACTIONS[action_id])
        validate_transition(state, after, actor, ACTIONS[action_id])
        for enemy in enemies:
            old, new = state.getAgentPosition(enemy), after.getAgentPosition(enemy)
            if new == state.getInitialAgentPosition(enemy) and new != old:
                dx, dy = Actions.directionToVector(ACTIONS[action_id]) if actor == enemy else (0, 0)
                counts['enemy_respawns'] += new != (int(old[0] + dx), int(old[1] + dy))
        state = after
    assert state.getScore() * (1 if record['red'] else -1) == record['score']
    return counts


def main():
    source = ROOT / 'runs/m0_goal_20261105/development'
    output = {'definition': 'second team slot; decision-weighted occupancy, not causal attribution',
              'new_world_rollouts': 0, 'arms': {}}
    for arm in ('original', 'guard', 'heuristic'):
        paths = sorted((source / arm).glob('game_*.json'))
        assert len(paths) == 20
        rows = [audit(json.loads(path.read_text())) for path in paths]
        totals = {key: sum(row[key] for row in rows) for key in rows[0]}
        output['arms'][arm] = {'games': len(rows), **totals,
            'defender_home_fraction': totals['defender_home'] / totals['defender_decisions'],
            'home_fraction_during_invasion': (totals['defender_home_during_invasion'] /
                                            totals['invasion_decisions']) if totals['invasion_decisions'] else None}
    target = ROOT / 'runs/root_cause_20261105/defense_occupancy.json'
    target.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
