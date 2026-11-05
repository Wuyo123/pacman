"""Frozen original M0 latency and twenty-game gate with complete denominators."""

import argparse
from contextlib import contextmanager, nullcontext
import gc
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
from .schema import FEATURE_DIMS, validate_checkpoint_schema


def evaluation_spec(index, seed_start=9000000):
    return {'seed': seed_start + index // 2, 'red': index % 2 == 0}


def gate_result(outcomes, forward_p95_ms, failures=0, over_budget=0):
    wins = sum(value > 0 for value in outcomes)
    draws = sum(value == 0 for value in outcomes)
    score_rate = (wins + 0.5 * draws) / 20
    if len(outcomes) != 20:
        decision = 'INCOMPLETE'
    elif failures:
        decision = 'INVALID_EXECUTION'
    elif forward_p95_ms > 15:
        decision = 'RETRY_D48'
    elif over_budget:
        decision = 'INVALID_EXECUTION'
    elif forward_p95_ms <= 8 and score_rate >= 0.75:
        decision = 'GO'
    else:
        decision = 'M0_NOT_PASSED'
    return {'decision': decision, 'wins': wins, 'draws': draws, 'losses': len(outcomes)-wins-draws,
            'planned_games': 20, 'completed_games': len(outcomes), 'failures': failures,
            'score_rate': score_rate, 'forward_p95_ms': forward_p95_ms,
            'moves_over_50ms': over_budget, 'original_forward_gate_ms': 8,
            'original_score_gate': 0.75}


def benchmark(checkpoint, data_dir, out, deadline=None, device_override=None):
    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('M0 budget expired during latency benchmark')
    check_deadline()
    checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
    observation_version = validate_checkpoint_schema(checkpoint)
    paths = sorted(Path(data_dir).glob('*.npz'))[-20:]
    dataset = EpisodeDataset(paths, observation_version=observation_version)
    samples = [dataset[int(i)] for i in np.linspace(0, len(dataset)-1, 100)]
    timings = {}
    torch.set_num_threads(1)
    devices = [device_override] if device_override is not None else (
        ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
    if any(device not in ('cpu', 'cuda') for device in devices):
        raise ValueError('benchmark device must be cpu or cuda')
    if 'cuda' in devices and not torch.cuda.is_available():
        raise ValueError('CUDA benchmark requested but unavailable')
    for device in devices:
        check_deadline()
        model = BeliefMiTA(feature_dim=FEATURE_DIMS[observation_version], d=checkpoint['d']).to(device).eval()
        model.load_state_dict(checkpoint['state_dict'])
        inputs = [(sample['x'][None].to(device), sample['landmarks'][None].to(device),
                   sample['agents'][None].to(device)) for sample in samples]
        measured = []
        with torch.inference_mode():
            for x in inputs[:20]:
                check_deadline()
                model(*x)
            if device == 'cuda':
                torch.cuda.synchronize()
            for x in inputs:
                check_deadline()
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


@contextmanager
def defer_cyclic_gc():
    """Move nondeterministic cyclic-collection pauses outside the action callback."""
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


class TimedController(NeuralController):
    def choose(self, observation, actor, encoder, landmarks, started):
        def timeout(signum, frame):
            raise DecisionDeadline()
        previous = signal.getsignal(signal.SIGALRM)
        try:
            try:
                signal.signal(signal.SIGALRM, timeout)
                remaining = started + self.budget - time.perf_counter() - 0.001
                signal.setitimer(signal.ITIMER_REAL, max(0.0001, remaining))
                return super().choose(observation, actor, encoder, landmarks, started)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
        except DecisionDeadline:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            self.statistics['deadline_stops'] += 1
            legal = observation.getLegalActions(actor)
            return 'Stop' if 'Stop' in legal else legal[0]


def evaluate(checkpoint, data_dir, out_dir, deadline=None, defer_gc=False, seed_start=9000000,
             device_override=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'gate.json').exists():
        raise FileExistsError('do not overwrite an existing M0 gate')
    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    observation_version = validate_checkpoint_schema(checkpoint_data)
    device, forward_ms = benchmark(checkpoint, data_dir, out/'latency.json', deadline, device_override)
    results, outcomes, failures = [], [], 0
    over_budget = 0
    for index in range(20):
        if deadline is not None and time.monotonic() >= deadline:
            break
        spec = evaluation_spec(index, seed_start=seed_start)
        controller = TimedController(checkpoint, spec['seed'], device=device)
        try:
            with defer_cyclic_gc() if defer_gc else nullcontext():
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
    gate['deferred_cyclic_gc'] = defer_gc
    gate['observation_version'] = observation_version
    gate['feature_dim'] = FEATURE_DIMS[observation_version]
    gate['source_sha256'] = {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in Path(__file__).parent.glob('*.py')}
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
    parser.add_argument('--defer-gc', action='store_true')
    parser.add_argument('--seed-start', type=int, default=9000000)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.out_dir,
             time.monotonic() + max(0, args.deadline_epoch - time.time()), args.defer_gc,
             args.seed_start)
