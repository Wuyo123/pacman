"""Run the original training/evaluation sequence once after complete oracle collection."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import torch

from .evaluate import evaluate
from .runtime import ROOT
from .training import train


def run(deadline_epoch):
    out = ROOT/'runs'/'m0_original'
    collection = json.loads((out/'collection_summary.json').read_text())
    assert collection['status'] == 'PASS' and collection['completed_games'] == 200
    assert len(list((out/'data').glob('*.npz'))) == 200
    assert torch.cuda.is_available(), 'original GPU training requires actual CUDA access'
    deadline = time.monotonic() + max(0, deadline_epoch - time.time())
    manifest = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'deadline_epoch': deadline_epoch,
                'engine_commit': subprocess.check_output(['git', '-C', str(ROOT/'vendor'/'capture'),
                                                          'rev-parse', 'HEAD'], text=True).strip(),
                'cuda': torch.version.cuda, 'torch': str(torch.__version__),
                'gpu': torch.cuda.get_device_name(0),
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT/'belief_mita').glob('*.py'))},
                'original_plan_sha256': hashlib.sha256((ROOT/'ORIGINAL_PLAN_20261104.md').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    ledger = {'protocol': 'original plan M0', 'M0': 'TRAINING', 'M1': 'NOT_RUN',
              'M2': 'NOT_RUN', 'M3': 'NOT_RUN', 'collection': collection,
              'manifest': 'runs/m0_original/manifest.json'}
    def update():
        ledger['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (ROOT/'RUN_LEDGER.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    update()
    try:
        training = train(out/'data', out/'training', epochs=20, d=64, batch_size=64,
                         deadline=deadline)
        assert training['status'] == 'completed' and training['epochs_completed'] == 20
        ledger['M0'] = 'EVALUATING'
        update()
        result = evaluate(out/'training'/'model.pt', out/'data', out/'evaluation', deadline=deadline)
        ledger['M0'] = result['decision']
        ledger['gate'] = {k:v for k,v in result.items() if k != 'results'}
        if result['decision'] == 'RETRY_D48':
            ledger['M0'] = 'ORIGINAL_D48_RETRY'
            update()
            train(out/'data', out/'training_d48', epochs=20, d=48, batch_size=64, deadline=deadline)
            result = evaluate(out/'training_d48'/'model.pt', out/'data', out/'evaluation_d48', deadline=deadline)
            ledger['M0'] = result['decision']
            ledger['gate_d48'] = {k:v for k,v in result.items() if k != 'results'}
        if result['decision'] == 'GO':
            ledger['M1'] = 'READY'
        update()
        print(json.dumps({'M0': ledger['M0'], 'M1': ledger['M1'], 'gate': ledger.get('gate')}), flush=True)
    except Exception as error:
        ledger['M0'] = 'INCONCLUSIVE_RESOURCE' if isinstance(error, (TimeoutError, torch.cuda.OutOfMemoryError)) else 'EXECUTION_ERROR'
        ledger['error'] = repr(error)
        update()
        traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    run(args.deadline_epoch)
