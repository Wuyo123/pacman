"""Replay sealed case 9700004 red; record events, without new policy decisions."""

import json

from belief_mita.runtime import ACTIONS, Actions, OraclePlanner, ROOT, initial_state, topology_for, validate_transition


def snapshot(state):
    agents = []
    for agent in range(4):
        data = state.getAgentState(agent)
        pos = state.getAgentPosition(agent)
        visible = agent in (0, 2) or any(
            abs(pos[0] - state.getAgentPosition(own)[0]) +
            abs(pos[1] - state.getAgentPosition(own)[1]) <= 5 for own in (0, 2))
        agents.append({'agent': agent, 'offline_position': pos, 'visible_to_red': visible,
                       'invading': data.isPacman, 'carry': data.numCarrying,
                       'returned': data.numReturned, 'scared': data.scaredTimer})
    return agents


def audit(arm):
    path = ROOT / 'runs/input_repair_continuation_20261105' / arm / 'evaluation/game_08.json'
    record = json.loads(path.read_text())
    assert record['seed'] == 9700004 and record['red']
    state = initial_state(record['layout'])
    topology = topology_for(record['layout'])
    teacher = OraclePlanner(topology, [0, 2], True)
    opponent_model = OraclePlanner(topology, [1, 3], False, depth=1)
    events, decisions = [], []
    for ply, (actor, action_id) in enumerate(record['action_trace'], 1):
        action = ACTIONS[action_id]
        before = snapshot(state)
        if actor in (0, 2):
            invaders = [row for row in before if row['agent'] in (1, 3) and row['invading']]
            decisions.append({'ply': ply, 'actor': actor, 'action': action, 'agents': before,
                'visible_invaders': [row['agent'] for row in invaders if row['visible_to_red']],
                'offline_invader_distances': {str(row['agent']): int(topology.distances[
                    topology.index[state.getAgentPosition(actor)],
                    topology.index[state.getAgentPosition(row['agent'])]]) for row in invaders}})
            if arm == 'v1' and ply in (207, 211, 215, 319):
                decisions[-1]['full_state_teacher_reference'] = teacher.choose(state, actor)
            if arm == 'v1' and ply == 211:
                probes = []
                for candidate in ('East', 'Stop'):
                    probe = state.generateSuccessor(actor, candidate)
                    reply = opponent_model.choose(probe, 3)
                    probe = probe.generateSuccessor(3, reply)
                    probes.append({'own_action': candidate, 'model_reply': reply,
                        'own_position_after_reply': probe.getAgentPosition(actor),
                        'score_delta': probe.getScore() - state.getScore()})
                decisions[-1]['offline_full_state_opponent_probes'] = probes
        after_state = state.generateSuccessor(actor, action)
        validate_transition(state, after_state, actor, action)
        after = snapshot(after_state)
        tags = []
        for old, new in zip(before, after):
            agent = old['agent']
            if old['invading'] != new['invading']:
                tags.append(f'{agent}:role->{int(new["invading"])}')
            if new['carry'] > old['carry']:
                tags.append(f'{agent}:collect+{new["carry"] - old["carry"]}')
            if new['returned'] > old['returned']:
                tags.append(f'{agent}:return+{new["returned"] - old["returned"]}')
            if new['scared'] > old['scared']:
                tags.append(f'{agent}:scared->{new["scared"]}')
            point, target = old['offline_position'], new['offline_position']
            dx, dy = Actions.directionToVector(action) if agent == actor else (0, 0)
            if tuple(target) != (int(point[0] + dx), int(point[1] + dy)):
                tags.append(f'{agent}:respawn')
        if tags:
            events.append({'ply': ply, 'actor': actor, 'action': action, 'tags': tags,
                           'score': after_state.getScore(), 'before': before, 'after': after})
        state = after_state
    assert state.getScore() == record['score']
    assert len(record['action_trace']) == record['moves']
    return {'seed': record['seed'], 'score': record['score'], 'moves': record['moves'],
            'events': events, 'own_decisions': decisions, 'final': snapshot(state)}


def main():
    output = {'new_world_rollouts': 0, 'positions_are_offline_truth': True,
              'visibility_definition': 'official shared Manhattan sight <=5',
              'arms': {arm: audit(arm) for arm in ('v0', 'v1')}}
    target = ROOT / 'runs/root_cause_20261105/case9700004_timeline.json'
    target.write_text(json.dumps(output, indent=2), encoding='utf-8')
    for arm, data in output['arms'].items():
        print(json.dumps({'arm': arm, 'score': data['score'],
            'events': [{'ply': row['ply'], 'tags': row['tags']} for row in data['events']],
            'privileged_teacher_queries': [{'ply': row['ply'], 'actual': row['action'],
                'teacher': row['full_state_teacher_reference']} for row in data['own_decisions']
                if 'full_state_teacher_reference' in row],
            'opponent_probes': [row['offline_full_state_opponent_probes'] for row in data['own_decisions']
                                if 'offline_full_state_opponent_probes' in row]}))


if __name__ == '__main__':
    main()
