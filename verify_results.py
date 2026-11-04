"""Replay recorded official actions and verify input/checkpoint provenance once."""

import hashlib
import json
from pathlib import Path
import random

import numpy as np

from belief_mita.runtime import ACTIONS, ROOT, initial_state, validate_transition


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    run = ROOT/'runs/m0_original'
    training = json.loads((run/'training/summary.json').read_text())
    assert training['status'] == 'completed' and training['epochs_completed'] == 20
    assert training['provenance']['device'] == 'cuda'
    assert len(training['provenance']['train_games']) == 180
    assert len(training['provenance']['val_games']) == 20
    assert not set(training['provenance']['train_games']) & set(training['provenance']['val_games'])
    for source in training['provenance']['files']:
        assert sha(source['path']) == source['sha256'], source['path']
    manifest = json.loads((run/'manifest.json').read_text())
    for relative, expected in manifest['source_sha256'].items():
        assert sha(run/'original_source'/relative) == expected, relative
    report = {'training_epochs': 20, 'train_games': 180, 'validation_games': 20,
              'input_hashes_verified': len(training['provenance']['files']),
              'original_source_snapshot_verified': True, 'evaluations': {}}
    for name in ('evaluation', 'evaluation_gcfix'):
        directory = run/name
        gate = json.loads((directory/'gate.json').read_text())
        assert gate['completed_games'] == 20 and gate['failures'] == 0
        assert gate['checkpoint_sha256'] == sha(run/'training/model.pt')
        scores, decisions = [], 0
        for index in range(20):
            record = json.loads((directory/f'game_{index:02d}.json').read_text())
            assert record['validated']
            random.seed(record['seed'])
            np.random.seed(record['seed'])
            state = initial_state(record['layout'])
            trace = record['action_trace']
            assert len(trace) == record['moves'] and 0 < len(trace) <= 1200
            for step, (actor, action) in enumerate(trace):
                assert actor == (trace[0][0] + step) % 4
                successor = state.generateSuccessor(actor, ACTIONS[action])
                validate_transition(state, successor, actor, ACTIONS[action])
                state = successor
            score = state.getScore() * (1 if record['red'] else -1)
            assert score == record['score']
            assert int(np.sign(score)) == record['outcome']
            scores.append(record['outcome'])
            decisions += record['own_decisions']
        assert sum(value > 0 for value in scores) == gate['wins']
        assert sum(value == 0 for value in scores) == gate['draws']
        assert (gate['wins'] + 0.5 * gate['draws']) / 20 == gate['score_rate']
        report['evaluations'][name] = {'replayed_games': 20, 'own_decisions': decisions,
                                       'all_scores_and_actions_verified': True,
                                       'same_checkpoint': True, 'decision': gate['decision']}
    (run/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    verify()
