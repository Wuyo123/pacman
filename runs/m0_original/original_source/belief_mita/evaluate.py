"""Frozen original M0 latency and twenty-game gate with complete denominators."""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .model import BeliefMiTA
from .runtime import play_game
from .search import NeuralController
from .training import EpisodeDataset


def evaluation_spec(index):
    return {'seed': 9000000 + index // 2, 'red': index % 2 == 0}


def gate_result(outcomes, forward_p95_ms, failures=0, over_budget=0):
    wins = sum(value > 0 for value in outcomes)
    draws = sum(value == 0 for value in outcomes)
    score_rate = (wins + 0.5 * draws) / 20
    if len(outcomes) != 20:
        decision = 'INCOMPLETE'
    elif failures or over_budget:
        decision = 'INVALID_EXECUTION'
    elif forward_p95_ms > 15:
        decision = 'RETRY_D48'
    elif forward_p95_ms <= 8 and score_rate >= 0.75:
        decision = 'GO'
    else:
        decision = 'M0_NOT_PASSED'
    return {'decision': decision, 'wins': wins, 'draws': draws, 'losses': len(outcomes)-wins-draws,
            'planned_games': 20, 'completed_games': len(outcomes), 'failures': failures,
            'score_rate': score_rate, 'forward_p95_ms': forward_p95_ms,
            'moves_over_50ms': over_budget, 'original_forward_gate_ms': 8,
            'original_score_gate': 0.75}


def benchmark(checkpoint, data_dir, out):
    checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
    paths = sorted(Path(data_dir).glob('*.npz'))[-20:]
    dataset = EpisodeDataset(paths)
    samples = [dataset[int(i)] for i in np.linspace(0, len(dataset)-1, 100)]
    timings = {}
    torch.set_num_threads(1)
    for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        model = BeliefMiTA(d=checkpoint['d']).to(device).eval()
        model.load_state_dict(checkpoint['state_dict'])
        inputs = [(sample['x'][None].to(device), sample['landmarks'][None].to(device),
                   sample['agents'][None].to(device)) for sample in samples]
        measured = []
        with torch.inference_mode():
            for x in inputs[:20]:
                model(*x)
            if device == 'cuda':
                torch.cuda.synchronize()
            for x in inputs:
                start = time.perf_counter()
                model(*x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                measured.append((time.perf_counter() - start)*1000)
        timings[device] = {'p50': float(np.percentile(measured, 50)),
                           'p95': float(np.percentile(measured, 95)),
                           'p99': float(np.percentile(measured, 99)), 'max': max(measured),
                           'samples': len(measured), 'warmup': 20, 'raw_ms': measured}
        del model, inputs
    selected = min(timings, key=lambda device: timings[device]['p95'])
    result = {'selected_device': selected, 'timings': timings,
              'definition': 'batch1 model forward, tensors resident on device, synchronized CUDA',
              'dataset': 'last20 validation games;100 uniformly spaced real histories'}
    Path(out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return selected, timings[selected]['p95']


class DecisionDeadline(Exception):
    pass


class TimedController(NeuralController):
    def choose(self, observation, actor, encoder, landmarks, started):
        def timeout(signum, frame):
            raise DecisionDeadline()
        previous = signal.signal(signal.SIGALRM, timeout)
        remaining = started + self.budget - time.perf_counter() - 0.001
        signal.setitimer(signal.ITIMER_REAL, max(0.0001, remaining))
        try:
            return super().choose(observation, actor, encoder, landmarks, started)
        except DecisionDeadline:
            self.statistics['deadline_stops'] += 1
            legal = observation.getLegalActions(actor)
            return 'Stop' if 'Stop' in legal else legal[0]
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)


def evaluate(checkpoint, data_dir, out_dir, deadline=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'gate.json').exists():
        raise FileExistsError('do not overwrite an existing M0 gate')
    device, forward_ms = benchmark(checkpoint, data_dir, out/'latency.json')
    results, outcomes, failures = [], [], 0
    over_budget = 0
    for index in range(20):
        if deadline is not None and time.monotonic() >= deadline:
            break
        spec = evaluation_spec(index)
        controller = TimedController(checkpoint, spec['seed'], device=device)
        try:
            result = play_game(**spec, controller=controller, deadline=deadline)
            statistics = dict(controller.statistics)
            forwards = statistics.pop('forward_ms')
            statistics['calls'] = len(forwards)
            statistics['forward_with_transfer_p95_ms'] = float(np.percentile(forwards, 95)) if forwards else None
            result['search'] = statistics
            outcomes.append(result['outcome'])
            over_budget += result['over_50ms']
        except Exception as exc:
            result = {**spec, 'outcome': -1, 'error': repr(exc), 'validated': False}
            outcomes.append(-1)
            failures += 1
        results.append(result)
        (out/f'game_{index:02d}.json').write_text(json.dumps(result), encoding='utf-8')
        print(json.dumps({'stage': 'evaluate', 'completed': len(results), 'score': result.get('score'),
                          'outcome': result['outcome'], 'failures': failures,
                          'over_50ms': result.get('over_50ms')}), flush=True)
    gate = gate_result(outcomes, forward_ms, failures, over_budget)
    gate['checkpoint_sha256'] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    gate['inference_device'] = device
    gate['results'] = [{k:v for k,v in result.items() if k != 'action_trace'} for result in results]
    (out/'gate.json').write_text(json.dumps(gate, indent=2), encoding='utf-8')
    print(json.dumps(gate), flush=True)
    return gate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('data_dir')
    parser.add_argument('out_dir')
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.out_dir,
             time.monotonic() + max(0, args.deadline_epoch - time.time()))
