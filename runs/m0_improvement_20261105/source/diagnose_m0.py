"""Read-only replay/validation diagnostics for the frozen M0 checkpoint."""

from collections import Counter, deque
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from belief_mita.model import BeliefMiTA
from belief_mita.runtime import ACTIONS, ROOT, initial_state, validate_transition
from belief_mita.training import EpisodeDataset


def replay(record):
    random.seed(record['seed'])
    state = initial_state(record['layout'])
    team = [0, 2] if record['red'] else [1, 3]
    stops, decisions, collected, crossings = [0, 0], [0, 0], [0, 0], [0, 0]
    max_carry, stagnant_carry = [0, 0], [0, 0]
    trails = [deque(maxlen=16), deque(maxlen=16)]
    for actor, action_id in record['action_trace']:
        after = state.generateSuccessor(actor, ACTIONS[action_id])
        validate_transition(state, after, actor, ACTIONS[action_id])
        if actor in team:
            slot = team.index(actor)
            before_agent, after_agent = state.getAgentState(actor), after.getAgentState(actor)
            decisions[slot] += 1
            stops[slot] += action_id == 4
            collected[slot] += max(0, after_agent.numCarrying-before_agent.numCarrying)
            crossings[slot] += not before_agent.isPacman and after_agent.isPacman
            max_carry[slot] = max(max_carry[slot], after_agent.numCarrying)
            trails[slot].append(after.getAgentPosition(actor))
            stagnant_carry[slot] += (after_agent.numCarrying > 0 and len(trails[slot]) == 16
                                      and len(set(trails[slot])) <= 4)
        state = after
    assert state.getScore() * (1 if record['red'] else -1) == record['score']
    return {'seed': record['seed'], 'red': record['red'], 'outcome': record['outcome'],
            'score': record['score'], 'collected': collected,
            'returned': [state.getAgentState(i).numReturned for i in team],
            'opponent_returned': sum(state.getAgentState(i).numReturned for i in range(4) if i not in team),
            'final_carry': [state.getAgentState(i).numCarrying for i in team],
            'max_carry': max_carry, 'crossings': crossings, 'decisions': decisions,
            'stop_rate': [s/n for s,n in zip(stops,decisions)], 'stagnant_carry_moves': stagnant_carry}


@torch.inference_mode()
def validation_routing():
    torch.set_num_threads(1)
    run = ROOT/'runs/m0_original'
    paths = sorted((run/'data').glob('*.npz'))[-20:]
    dataset = EpisodeDataset(paths)
    actors = []
    for path in paths:
        record = json.loads(path.with_suffix('.json').read_text())
        team = [0, 2] if record['red'] else [1, 3]
        actors.extend(team.index(actor) for actor, action in record['action_trace'] if actor in team)
    assert len(actors) == len(dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(run/'training/model.pt', map_location='cpu', weights_only=False)
    model = BeliefMiTA(d=checkpoint['d']).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    histograms = [Counter(), Counter()]
    totals = {'samples': 0, 'legal_top3_trivial': 0, 'current_action_top1_correct': 0,
              'only_one_current_action_despite_alternatives': 0, 'expert_current_action_pruned': 0,
              'stop_expert': 0, 'stop_prediction': 0}
    cursor = 0
    for batch in DataLoader(dataset, batch_size=64):
        x, landmarks, agents = [batch[key].to(device) for key in ('x','landmarks','agents')]
        output = model(x, landmarks, agents)
        legal = batch['legal'].to(device)
        first = output['action_logits'][:, 0].masked_fill(~legal[:,0], -torch.inf).softmax(-1)
        second = output['conditional_action_logits'].masked_fill(~legal[:,1,None,:], -torch.inf).softmax(-1)
        joint = first[:,:,None]*second
        ids = joint.flatten(1).topk(3,dim=1).indices.cpu().numpy()
        for row in range(len(ids)):
            slot = actors[cursor]
            current = ids[row]//5 if slot == 0 else ids[row]%5
            count = int(batch['legal'][row,slot].sum())
            truth = int(batch['labels'][row,slot])
            distinct = len(set(current.tolist()))
            histograms[slot][distinct] += 1
            totals['samples'] += 1
            totals['legal_top3_trivial'] += count <= 3
            totals['current_action_top1_correct'] += int(current[0] == truth)
            totals['only_one_current_action_despite_alternatives'] += distinct == 1 and count > 1
            totals['expert_current_action_pruned'] += truth not in current
            totals['stop_expert'] += truth == 4
            totals['stop_prediction'] += int(current[0] == 4)
            cursor += 1
    return {**totals, 'distinct_current_action_histogram_by_slot': [dict(h) for h in histograms],
            'scope': '12000 validation decisions, unshielded top3joint; not actual online root logs'}


def main():
    out = ROOT/'runs/m0_improvement_20261105'
    out.mkdir(exist_ok=True)
    records = [json.loads(path.read_text()) for path in sorted((ROOT/'runs/m0_original/evaluation_gcfix').glob('game_*.json'))]
    games = [replay(record) for record in records]
    draws = [game for game in games if game['outcome'] == 0]
    report = {'source': 'frozen M0 GC-fixed20games and180/20 game validation split',
              'games': games, 'draw_summary': {
                  'draws': len(draws), 'zero_return_draws': sum(sum(g['returned']) == 0 for g in draws),
                  'draws_with_food_collected': sum(sum(g['collected']) > 0 for g in draws),
                  'draws_with_stagnant_carrier': sum(sum(g['stagnant_carry_moves']) > 0 for g in draws),
                  'draws_ending_with_carried_food': sum(sum(g['final_carry']) > 0 for g in draws)},
              'validation_routing': validation_routing()}
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key != 'games'}),flush=True)


if __name__ == '__main__':
    main()
