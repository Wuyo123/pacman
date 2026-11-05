"""Independent read-only acceptance of the completed input repair continuation."""

import hashlib
import json
from collections import Counter

import torch

from belief_mita.runtime import ROOT
from diagnose_m0 import replay


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = ROOT / 'runs/input_repair_20261105'
    run = ROOT / 'runs/input_repair_continuation_20261105'
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'NO_IMPROVEMENT_ESTABLISHED'
    assert json.loads((original / 'manifest.json').read_text())['status'] == 'INCONCLUSIVE_RESOURCE'
    assert sha(original / 'manifest.json') == manifest['original_90m_manifest_sha256']
    assert sha(original / 'v1/training/model.pt') == manifest['original_checkpoint_sha256']['v1_partial']
    for name, digest in manifest['source_sha256'].items():
        assert sha(run / 'source_snapshot' / name) == digest, name
    for item in manifest['data_audit']['files']:
        assert sha(original / 'data' / item['path']) == item['sha256']
    assert len(manifest['data_audit']['files']) == 200
    report = {'data_hashes_verified': 200, 'source_snapshot_verified': True,
              'original_resource_result_unchanged': True, 'arms': {}, 'replayed': 0}
    outcomes = {}
    for arm, path in [('v0', original / 'v0/training/model.pt'),
                      ('v1', run / 'v1/training/model.pt')]:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['status'] == 'completed' and checkpoint['epochs'] == 20
        if arm == 'v1':
            assert checkpoint['continuation']['source_checkpoint_sha256'] == manifest['original_checkpoint_sha256']['v1_partial']
        steps = {int(state['step']) for state in checkpoint['optimizer_state_dict']['state'].values()}
        assert steps == {33760}
        expected = (manifest['original_checkpoint_sha256']['v0'] if arm == 'v0'
                    else manifest['continuation_checkpoint_sha256'])
        assert sha(path) == expected
        gate = json.loads((run / arm / 'evaluation/gate.json').read_text())
        assert gate['checkpoint_sha256'] == expected
        results, depth, roots = [], [0] * 4, [0] * 4
        diagnostics = Counter()
        for index in range(20):
            record = json.loads((run / arm / 'evaluation' / f'game_{index:02d}.json').read_text())
            assert record['seed'] == 9700000 + index // 2 and record['red'] == (index % 2 == 0)
            assert record['validated'] and record['over_50ms'] == 0 and record['own_ms_max'] <= 50
            assert len(record['action_trace']) == record['moves']
            replay(record)
            report['replayed'] += 1
            results.append(record['outcome'])
            stats = record['search']
            for key in ('decisions', 'simulations', 'observation_hits', 'observation_misses',
                        'search_changed_policy', 'deadline_stops'):
                diagnostics[key] += stats[key]
            diagnostics['own_decisions'] += record['own_decisions']
            depth = [x + y for x, y in zip(depth, stats['simulation_depth'])]
            roots = [x + y for x, y in zip(roots, stats['root_visited_edges'])]
        counts = Counter(results)
        rate = (counts[1] + .5 * counts[0]) / 20
        assert (counts[1], counts[0], counts[-1]) == (gate['wins'], gate['draws'], gate['losses'])
        assert rate == gate['score_rate']
        assert gate['completed_games'] == 20 and gate['failures'] == gate['moves_over_50ms'] == 0
        assert gate['forward_p95_ms'] <= 8
        outcomes[arm] = results
        report['arms'][arm] = {'epochs': 20, 'optimizer_updates': 33760, 'wins': counts[1],
            'draws': counts[0], 'losses': counts[-1], 'score_rate': rate,
            'forward_p95_ms': gate['forward_p95_ms'], 'search': dict(diagnostics),
            'simulation_depth_histogram': depth, 'root_visited_edges_histogram': roots}
    delta = sum(b - a for a, b in zip(outcomes['v0'], outcomes['v1'])) / 40
    assert abs(delta - manifest['comparison']['v1_minus_v0_score_rate']) < 1e-12
    report['v1_minus_v0_score_rate'] = delta
    differences = [b - a for a, b in zip(outcomes['v0'], outcomes['v1'])]
    report['paired_better_same_worse'] = [sum(x > 0 for x in differences),
                                        sum(x == 0 for x in differences), sum(x < 0 for x in differences)]
    assert report['replayed'] == 40
    report['status'] = 'VERIFIED_NO_IMPROVEMENT_ESTABLISHED'
    (run / 'independent_verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
