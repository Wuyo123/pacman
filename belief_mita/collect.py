"""One bounded, resumable M0 oracle data batch; no neural policy in the collector."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .runtime import ROOT, play_game


def game_spec(index):
    return {'id': index, 'seed': 1000000 + index // 2, 'red': index % 2 == 0,
            'layout': 'defaultCapture', 'opponent': 'baseline'}


def collect_one(index, out, deadline):
    spec = game_spec(index)
    target = Path(out) / f'game_{index:04d}.npz'
    result_path = target.with_suffix('.json')
    if target.exists() and result_path.exists():
        result = json.loads(result_path.read_text())
        assert result['validated'] and result['seed'] == spec['seed'] and result['red'] == spec['red']
        assert hashlib.sha256(target.read_bytes()).hexdigest() == result['data_sha256']
        return result
    result, episode = play_game(spec['seed'], spec['red'], name=spec['layout'],
                                opponent=spec['opponent'], collect=True, deadline=deadline)
    assert len(episode['frames']) == result['own_decisions']
    assert all(episode['legal'][np.arange(len(episode['labels'])), slot,
                                episode['labels'][:, slot]].all() for slot in range(2))
    for slot in range(2):
        hidden = ~episode['visible'][:, slot]
        assert (episode['frames'][hidden, :, 9 + slot].sum(axis=1) == 0).all()
    partial = target.with_suffix('.partial.npz')
    np.savez_compressed(partial, **episode)
    partial.replace(target)
    result.update(id=index, data_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    result_path.write_text(json.dumps(result), encoding='utf-8')
    return result


def collect(out, games=200, workers=4, deadline=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(collect_one, i, str(out), deadline): i for i in range(games)}
        for job in as_completed(jobs):
            index = jobs[job]
            try:
                result = job.result()
                results.append(result)
            except Exception as exc:
                failures.append({'id': index, 'error': repr(exc)})
            if (len(results) + len(failures)) % 20 == 0 or failures:
                print(json.dumps({'stage': 'collect', 'completed': len(results), 'failed': len(failures),
                                  'planned': games, 'seconds': time.perf_counter() - started}), flush=True)
    summary = {'status': 'PASS' if len(results) == games and not failures else 'FAILED',
               'planned_games': games, 'completed_games': len(results), 'failures': failures,
               'steps': sum(r['own_decisions'] for r in results),
               'wins': sum(r['outcome'] > 0 for r in results), 'draws': sum(r['outcome'] == 0 for r in results),
               'losses': sum(r['outcome'] < 0 for r in results),
               'seconds': time.perf_counter() - started,
               'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT/'belief_mita').glob('*.py')}}
    (out.parent/'collection_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    if summary['status'] != 'PASS':
        raise RuntimeError('collection incomplete; inspect collection_summary.json')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/m0_original/data')
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + max(0, args.deadline_epoch - time.time())
    collect(args.out, args.games, args.workers, deadline)
