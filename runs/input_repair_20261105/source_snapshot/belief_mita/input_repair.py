"""Run the fixed paired V0/V1 input repair experiment once."""

import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .collect import collect
from .evaluate import evaluate
from .runtime import ROOT
from .training import train
from .training import split_games

PROTOCOL = ROOT / 'INPUT_REPAIR_PROTOCOL_20261105.md'
OUT = ROOT / 'runs' / 'input_repair_20261105'
SEED_START = 1_100_000
EVAL_SEED_START = 9_700_000
TRAIN_SEED = 20_261_105
WALL_SECONDS = 90 * 60


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_summary(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    provenance = checkpoint['provenance']
    return {'epochs': checkpoint['epochs'], 'feature_dim': checkpoint['feature_dim'],
            'observation_version': checkpoint['observation_version'],
            'initialization_sha256': provenance['initialization_sha256'],
            'shared_initialization_sha256': provenance['shared_initialization_sha256'],
            'data_sha256': [item['sha256'] for item in provenance['files']]}


def compare(gates):
    old, new = gates['v0']['results'], gates['v1']['results']
    paired = []
    for first, second in zip(old, new):
        assert (first['seed'], first['red']) == (second['seed'], second['red'])
        paired.append({'seed': first['seed'], 'red': first['red'],
                       'v0_outcome': first['outcome'], 'v1_outcome': second['outcome'],
                       'v0_score': first.get('score'), 'v1_score': second.get('score'),
                       'outcome_delta': second['outcome'] - first['outcome']})
    delta = gates['v1']['score_rate'] - gates['v0']['score_rate']
    complete_pair = all(gate['completed_games'] == 20 and len(gate['results']) == 20
                        for gate in gates.values()) and len(paired) == 20
    paired_delta = sum(item['outcome_delta'] for item in paired) / 40 if complete_pair else None
    if complete_pair:
        assert abs(delta - paired_delta) < 1e-12, 'paired outcomes disagree with arm score rates'
    if not complete_pair:
        decision = 'INCONCLUSIVE_RESOURCE'
    elif any(gate['failures'] or gate['moves_over_50ms'] or gate['forward_p95_ms'] > 8
             for gate in gates.values()):
        decision = 'INVALID_EXECUTION'
    else:
        decision = None
    valid_v1 = (gates['v1']['completed_games'] == 20 and gates['v1']['failures'] == 0
                and gates['v1']['moves_over_50ms'] == 0 and gates['v1']['forward_p95_ms'] <= 8
                and gates['v1']['score_rate'] >= 0.75)
    return {'paired_games': len(paired), 'v1_minus_v0_score_rate': delta,
            'v0_m0_gate_met': (gates['v0']['completed_games'] == 20 and gates['v0']['failures'] == 0
                               and gates['v0']['moves_over_50ms'] == 0
                               and gates['v0']['forward_p95_ms'] <= 8
                               and gates['v0']['score_rate'] >= 0.75),
            'v1_m0_gate_met': valid_v1,
            'paired': paired,
            'decision': decision or ('INPUT_REPAIR_IMPROVEMENT_SCREEN_MET'
                                     if valid_v1 and delta >= 0.05 - 1e-9 else 'NO_IMPROVEMENT_ESTABLISHED'),
            'paired_delta_check': paired_delta}


def snapshot_sources():
    target = OUT / 'source_snapshot'
    target.mkdir(parents=True)
    shutil.copy2(PROTOCOL, target / PROTOCOL.name)
    (target / 'belief_mita').mkdir(exist_ok=True)
    for source in sorted((ROOT / 'belief_mita').glob('*.py')):
        shutil.copy2(source, target / 'belief_mita' / source.name)
    (target / 'tests').mkdir(exist_ok=True)
    for source in sorted((ROOT / 'tests').glob('*.py')):
        shutil.copy2(source, target / 'tests' / source.name)
    shutil.copy2(ROOT / 'diagnose_m0.py', target / 'diagnose_m0.py')
    return {str(path.relative_to(target)): sha256(path) for path in sorted(target.rglob('*')) if path.is_file()}


def audit_data(data_dir, deadline):
    from .schema import FEATURE_DIMS
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob('game_*.npz'))
    if len(paths) != 200:
        raise RuntimeError(f'expected 200 collected files, got {len(paths)}')
    hashes, steps = [], 0
    for index, path in enumerate(paths):
        if time.monotonic() >= deadline:
            raise TimeoutError('deadline during input archive audit')
        record = json.loads(path.with_suffix('.json').read_text())
        digest = sha256(path)
        if record.get('id') != index or record.get('seed') != SEED_START + index // 2:
            raise ValueError(f'collection index/seed mismatch: {path.name}')
        if record.get('red') != (index % 2 == 0) or record.get('observation_version') != 1:
            raise ValueError(f'collection side/schema mismatch: {path.name}')
        if record.get('data_sha256') != digest:
            raise ValueError(f'collection hash mismatch: {path.name}')
        with np.load(path, allow_pickle=False) as archive:
            version = archive['observation_version']
            if version.shape != () or version.dtype.kind not in 'iu' or int(version) != 1:
                raise ValueError(f'archive schema mismatch: {path.name}')
            frames, expanded = archive['frames'], archive['frames_v1']
            if (frames.ndim != 3 or frames.shape[-1] != FEATURE_DIMS[0]
                    or expanded.shape != (*frames.shape[:2], FEATURE_DIMS[1])
                    or not np.array_equal(expanded[..., :18], frames)):
                raise ValueError(f'paired V0/V1 inputs differ: {path.name}')
            labels, legal = archive['labels'], archive['legal']
            if not np.all(legal[np.arange(len(labels))[:, None], np.arange(2)[None, :], labels]):
                raise ValueError(f'illegal teacher label: {path.name}')
            visible = archive['visible']
            for slot in range(2):
                hidden = ~visible[:, slot]
                if not np.all(frames[hidden, :, 9 + slot].sum(axis=1) == 0):
                    raise ValueError(f'hidden enemy channel leak: {path.name}')
            steps += len(frames)
        hashes.append({'path': path.name, 'sha256': digest, 'steps': len(frames)})
    train_paths, val_paths = split_games(paths)
    if [path.name for path in train_paths] != [f'game_{i:04d}.npz' for i in range(180)]:
        raise ValueError('training split is not the frozen first 180 games')
    if [path.name for path in val_paths] != [f'game_{i:04d}.npz' for i in range(180, 200)]:
        raise ValueError('validation split is not the frozen final 20 games')
    return {'files': hashes, 'games': len(paths), 'steps': steps,
            'train_games': len(train_paths), 'validation_games': len(val_paths),
            'all_v0_v1_frames_match': True, 'all_labels_legal': True,
            'hidden_channels_clear': True}


def replay_evaluations(deadline):
    from diagnose_m0 import replay
    records, failures = [], []
    resource_timeout = False
    for label in ('v0', 'v1'):
        for index in range(20):
            if time.monotonic() >= deadline:
                resource_timeout = True
                failures.extend({'arm': rest_label, 'index': rest_index,
                                 'error': 'deadline before replay'}
                                for rest_label in ('v0', 'v1')
                                for rest_index in range(20)
                                if (rest_label, rest_index) not in {
                                    (item['arm'], item['index']) for item in records + failures})
                break
            path = OUT / label / 'evaluation' / f'game_{index:02d}.json'
            try:
                record = json.loads(path.read_text())
                if not record.get('validated') or len(record.get('action_trace', [])) != record.get('moves'):
                    raise ValueError('missing or incomplete action trace')
                result = replay(record)
                records.append({'arm': label, 'index': index, **result, 'validated': True})
            except Exception as error:
                failures.append({'arm': label, 'index': index, 'error': repr(error)})
        if resource_timeout:
            break
    audit = {'planned_games': 40, 'replayed_games': len(records), 'failures': failures,
             'attempted_games': len(records) + sum(item.get('error') != 'deadline before replay'
                                                   for item in failures),
             'resource_timeout': resource_timeout,
             'all_valid_traces_replayed': len(records) == 40 and not failures, 'results': records}
    (OUT / 'replay_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


def run():
    if OUT.exists():
        raise FileExistsError(f'preserve existing experiment directory: {OUT}')
    OUT.mkdir(parents=True)
    source_snapshot = snapshot_sources()
    started_wall = time.time()
    deadline = time.monotonic() + WALL_SECONDS
    manifest = {'status': 'RUNNING', 'started_epoch': started_wall,
                'wall_budget_seconds': WALL_SECONDS, 'planned_records': 240,
                'collection_games': 200, 'evaluation_games_per_arm': 20,
                'collection_seed_start': SEED_START, 'evaluation_seed_start': EVAL_SEED_START,
                'training_seed': TRAIN_SEED, 'epochs': 20, 'batch_size': 64, 'd': 64,
                'protocol_sha256': sha256(PROTOCOL), 'source_snapshot_sha256': source_snapshot,
                'arms': {}, 'evaluations': {}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    try:
        collection = collect(OUT / 'data', games=200, workers=4, deadline=deadline,
                             seed_start=SEED_START, observation_version=1)
        manifest['collection_summary'] = collection
        manifest['completed_records'] = collection['completed_games']
        manifest['data_audit'] = audit_data(OUT / 'data', deadline)
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        for version, label in ((0, 'v0'), (1, 'v1')):
            training = train(OUT / 'data', OUT / label / 'training', epochs=20, d=64,
                             batch_size=64, seed=TRAIN_SEED, deadline=deadline,
                             observation_version=version)
            if training['status'] != 'completed' or training['epochs_completed'] != 20:
                raise TimeoutError(f'{label} training incomplete')
            manifest['arms'][label] = checkpoint_summary(OUT / label / 'training' / 'model.pt')
            (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if (manifest['arms']['v0']['data_sha256'] != manifest['arms']['v1']['data_sha256']
                or manifest['arms']['v0']['shared_initialization_sha256']
                != manifest['arms']['v1']['shared_initialization_sha256']):
            raise RuntimeError('paired training data or shared initialization mismatch')
        gates = {}
        for label in ('v0', 'v1'):
            gates[label] = evaluate(OUT / label / 'training' / 'model.pt', OUT / 'data',
                                    OUT / label / 'evaluation', deadline=deadline,
                                    defer_gc=True, seed_start=EVAL_SEED_START,
                                    device_override='cpu')
            manifest['evaluations'][label] = gates[label]
            manifest['completed_records'] = collection['completed_games'] + sum(
                gate['completed_games'] for gate in manifest['evaluations'].values())
            (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        audit = replay_evaluations(deadline)
        manifest['replay_audit'] = audit
        comparison = compare(gates)
        if audit['resource_timeout']:
            comparison['decision'] = 'INCONCLUSIVE_RESOURCE'
        elif not audit['all_valid_traces_replayed'] and comparison['decision'] != 'INCONCLUSIVE_RESOURCE':
            comparison['decision'] = 'INVALID_EXECUTION'
        manifest['comparison'] = comparison
        manifest['status'] = comparison['decision']
        manifest['attempted_records'] = (collection['completed_games'] + len(collection['failures'])
                                         + sum(gate['completed_games'] for gate in gates.values()))
        manifest['completed_valid_records'] = (
            collection['completed_games'] + sum(gate['completed_games'] - gate['failures']
                                                for gate in gates.values()))
    except TimeoutError as error:
        manifest.update(status='INCONCLUSIVE_RESOURCE', error=repr(error))
    except Exception as error:
        summary_path = OUT / 'collection_summary.json'
        if summary_path.exists():
            manifest['collection_summary'] = json.loads(summary_path.read_text())
            manifest['completed_records'] = manifest['collection_summary']['completed_games']
            manifest['attempted_records'] = (manifest['completed_records']
                                             + len(manifest['collection_summary']['failures']))
        collection_expired = time.monotonic() >= deadline
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            collection_expired |= any('TimeoutError' in item.get('error', '')
                                      for item in summary.get('failures', []))
        manifest.update(status='INCONCLUSIVE_RESOURCE' if collection_expired else 'EXECUTION_ERROR',
                        error=repr(error))
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - (deadline - WALL_SECONDS)
        manifest['deadline_reached'] = time.monotonic() >= deadline
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'elapsed_seconds': manifest['elapsed_seconds'],
                      'planned_records': manifest['planned_records'],
                      'collection': manifest.get('collection_summary'),
                      'comparison': manifest.get('comparison')}, default=str), flush=True)
    return manifest


if __name__ == '__main__':
    run()
