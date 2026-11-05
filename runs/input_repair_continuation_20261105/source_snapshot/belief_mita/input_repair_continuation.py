"""Continue the frozen V1 checkpoint and run the remaining paired evaluation once."""

import json
import math
from pathlib import Path
import shutil
import time

import torch

from .evaluate import evaluate
from .input_repair import audit_data, compare, sha256
from .resume_training import optimizer_steps, resume_train
from .runtime import ROOT

ORIGINAL = ROOT / 'runs' / 'input_repair_20261105'
OUT = ROOT / 'runs' / 'input_repair_continuation_20261105'
PROTOCOL = ROOT / 'INPUT_REPAIR_CONTINUATION_PROTOCOL_20261105.md'
V0_CHECKPOINT = ORIGINAL / 'v0' / 'training' / 'model.pt'
V1_CHECKPOINT = ORIGINAL / 'v1' / 'training' / 'model.pt'
DATA = ORIGINAL / 'data'
EVAL_SEED_START = 9_700_000
TRAIN_SEED = 20_261_105
WALL_SECONDS = 45 * 60


def snapshot_sources():
    sources = [PROTOCOL, Path(__file__), Path(__file__).with_name('training.py'),
               Path(__file__).with_name('resume_training.py'),
               Path(__file__).with_name('evaluate.py'), Path(__file__).with_name('search.py'),
               Path(__file__).with_name('runtime.py'), Path(__file__).with_name('model.py'),
               Path(__file__).with_name('schema.py'), Path(__file__).with_name('topology.py'),
               Path(__file__).with_name('input_repair.py'), ROOT / 'diagnose_m0.py',
               ROOT / 'tests' / 'test_training.py', ROOT / 'tests' / 'test_input_repair_continuation.py']
    target = OUT / 'source_snapshot'
    target.mkdir(parents=True)
    for source in sources:
        destination = target / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {str(path.relative_to(target)): sha256(path) for path in target.rglob('*') if path.is_file()}


def replay_paths():
    return [('v0', OUT / 'v0' / 'evaluation'), ('v1', OUT / 'v1' / 'evaluation')]


def replay_all(deadline):
    from diagnose_m0 import replay
    records, failures = [], []
    jobs = replay_paths()
    resource_timeout = False
    for label, directory in jobs:
        for index in range(20):
            if time.monotonic() >= deadline:
                resource_timeout = True
                break
            path = directory / f'game_{index:02d}.json'
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
                if not record.get('validated') or len(record.get('action_trace', [])) != record.get('moves'):
                    raise ValueError('missing or incomplete action trace')
                result = replay(record)
                records.append({'arm': label, 'index': index, **result, 'validated': True})
            except Exception as error:
                failures.append({'arm': label, 'index': index, 'error': repr(error)})
        if resource_timeout:
            break
    attempted = {(item['arm'], item['index']) for item in records + failures}
    for label, _directory in jobs:
        for index in range(20):
            if (label, index) not in attempted:
                failures.append({'arm': label, 'index': index, 'error': 'deadline before replay'})
    report = {'planned_games': 40, 'replayed_games': len(records),
              'attempted_games': len(records) + sum(item['error'] != 'deadline before replay'
                                                    for item in failures),
              'failures': failures, 'resource_timeout': resource_timeout,
              'all_valid_traces_replayed': len(records) == 40 and not failures,
              'results': records}
    (OUT / 'replay_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def run():
    if OUT.exists():
        raise FileExistsError(f'preserve existing continuation output: {OUT}')
    OUT.mkdir(parents=True)
    original_manifest = json.loads((ORIGINAL / 'manifest.json').read_text(encoding='utf-8'))
    if original_manifest.get('status') != 'INCONCLUSIVE_RESOURCE':
        raise ValueError('original 90-minute result must remain sealed as resource-inconclusive')
    checkpoint_hashes = {'v0': sha256(V0_CHECKPOINT), 'v1_partial': sha256(V1_CHECKPOINT)}
    v1_partial = torch.load(V1_CHECKPOINT, map_location='cpu', weights_only=False)
    v0 = torch.load(V0_CHECKPOINT, map_location='cpu', weights_only=False)
    if v0.get('epochs') != 20 or v0.get('status') != 'completed':
        raise ValueError('V0 reference is not the completed original-input-repair 20-epoch model')
    completed_epochs = v1_partial.get('epochs')
    in_progress_epoch = v1_partial.get('epoch_in_progress')
    partial_batches = v1_partial.get('batches_completed_in_epoch')
    if (completed_epochs, in_progress_epoch, partial_batches) != (15, 16, 2):
        raise ValueError('sealed V1 partial checkpoint is not at the audited 15/16/2 progress point')
    source_provenance, control_provenance = v1_partial['provenance'], v0['provenance']
    common_keys = ('files', 'train_games', 'val_games', 'seed', 'shared_initialization_sha256',
                   'model_source_sha256')
    if any(source_provenance.get(key) != control_provenance.get(key) for key in common_keys):
        raise ValueError('V0 and V1 checkpoints differ in paired training provenance')
    if source_provenance.get('model_source_sha256') != sha256(Path(__file__).with_name('model.py')):
        raise ValueError('current model.py differs from the source used by the original checkpoints')
    train_samples = source_provenance.get('train_samples')
    batch_size = v1_partial.get('provenance', {}).get('batch_size')
    batch_count = math.ceil(train_samples / batch_size)
    previous_updates = completed_epochs * batch_count + partial_batches
    expected_final_updates = 20 * batch_count
    if optimizer_steps(v0['optimizer_state_dict']) != expected_final_updates:
        raise ValueError('V0 control does not contain the expected 20-epoch Adam updates')
    resources = (ROOT / 'runs' / 'input_repair_continuation_preflight.json')
    resource_preflight = json.loads(resources.read_text(encoding='utf-8')) if resources.exists() else None
    source_hashes = snapshot_sources()
    manifest = {'status': 'PREPARED', 'original_90m_status': original_manifest.get('status'),
                'original_90m_manifest_sha256': sha256(ORIGINAL / 'manifest.json'),
                'original_checkpoint_sha256': checkpoint_hashes,
                'original_training_source_sha256': v1_partial.get('provenance', {}).get('training_source_sha256'),
                'resume_training_source_sha256': source_hashes[str(Path(__file__).with_name('training.py').relative_to(ROOT))],
                'driver_source_sha256': source_hashes[str(Path(__file__).relative_to(ROOT))],
                'resume_helper_source_sha256': source_hashes[str(Path(__file__).with_name('resume_training.py').relative_to(ROOT))],
                'protocol_sha256': sha256(PROTOCOL), 'source_sha256': source_hashes,
                'resource_preflight': resource_preflight,
                'data_directory': str(DATA), 'data_audit': None,
                'partial_progress': {'completed_epochs': completed_epochs,
                                     'in_progress_epoch': in_progress_epoch,
                                     'batches_completed_in_epoch': partial_batches,
                                     'train_samples': train_samples, 'batch_size': batch_size,
                                     'batches_per_epoch': batch_count,
                                     'optimizer_updates_at_source': previous_updates,
                                     'expected_final_optimizer_updates': expected_final_updates},
                'max_extension_seconds': WALL_SECONDS, 'evaluation_seed_start': EVAL_SEED_START,
                'training_seed': TRAIN_SEED, 'evaluations': {}, 'planned_evaluation_games': 40}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    started = time.monotonic()
    deadline = started + WALL_SECONDS
    manifest['status'] = 'RUNNING'
    manifest['started_epoch'] = time.time()
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    try:
        if (resource_preflight is None or not resource_preflight.get('game_processes_absent')
                or not resource_preflight.get('resource_ok')):
            raise RuntimeError('required read-only resource preflight is missing or insufficient')
        manifest['data_audit'] = audit_data(DATA, deadline)
        resumed = resume_train(DATA, OUT / 'v1' / 'training', V1_CHECKPOINT,
                               epochs=20, d=64, batch_size=64, seed=TRAIN_SEED,
                               deadline=deadline, observation_version=1,
                               protocol_sha256=manifest['protocol_sha256'])
        if resumed['status'] != 'completed' or resumed['epochs_completed'] != 20:
            raise TimeoutError('V1 continuation did not complete all 20 epochs')
        final_checkpoint = torch.load(OUT / 'v1' / 'training' / 'model.pt',
                                      map_location='cpu', weights_only=False)
        final_steps = optimizer_steps(final_checkpoint['optimizer_state_dict'])
        if final_steps != expected_final_updates:
            raise RuntimeError('V1 optimizer step count does not match 20 complete epochs')
        manifest['continuation_checkpoint_sha256'] = sha256(OUT / 'v1' / 'training' / 'model.pt')
        manifest['final_optimizer_updates'] = final_steps
        gates = {}
        for label, checkpoint in (('v0', V0_CHECKPOINT),
                                  ('v1', OUT / 'v1' / 'training' / 'model.pt')):
            gates[label] = evaluate(checkpoint, DATA, OUT / label / 'evaluation',
                                    deadline=deadline, defer_gc=True,
                                    seed_start=EVAL_SEED_START, device_override='cpu')
            manifest['evaluations'][label] = gates[label]
            (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        manifest['comparison'] = compare(gates)
        replay_audit = replay_all(deadline)
        manifest['replay_audit'] = replay_audit
        if replay_audit['resource_timeout']:
            manifest['comparison']['decision'] = 'INCONCLUSIVE_RESOURCE'
        elif (not replay_audit['all_valid_traces_replayed']
              and manifest['comparison']['decision'] != 'INCONCLUSIVE_RESOURCE'):
            manifest['comparison']['decision'] = 'INVALID_EXECUTION'
        manifest['status'] = manifest['comparison']['decision']
    except TimeoutError as error:
        manifest.update(status='INCONCLUSIVE_RESOURCE', error=repr(error))
    except Exception as error:
        manifest.update(status='EXECUTION_ERROR', error=repr(error))
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - started
        manifest['deadline_reached'] = time.monotonic() >= deadline
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'elapsed_seconds': manifest['elapsed_seconds'],
                      'partial_updates': previous_updates,
                      'expected_final_updates': expected_final_updates,
                      'comparison': manifest.get('comparison')}, default=str), flush=True)
    return manifest


if __name__ == '__main__':
    run()
