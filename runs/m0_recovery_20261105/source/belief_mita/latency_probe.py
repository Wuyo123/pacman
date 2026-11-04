"""One diagnostic game correlating Python collection pauses with measured late moves."""

import gc
import json
from pathlib import Path
import signal
import time

from .evaluate import TimedController
from .runtime import ROOT, play_game
from .search import NeuralController


def run():
    pauses, active, late = [], {}, []
    def callback(phase, info):
        generation = info['generation']
        now = time.perf_counter()
        if phase == 'start':
            active[generation] = now
        elif generation in active:
            pauses.append((active.pop(generation), now, generation))
    class ProbeController(TimedController):
        def infer(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                return super().infer(*args, **kwargs)
            finally:
                if hasattr(self, 'events'):
                    self.events.append(['infer', started, time.perf_counter()])

        def macro_step(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                return super().macro_step(*args, **kwargs)
            finally:
                self.events.append(['macro', started, time.perf_counter()])

        def choose(self, observation, actor, encoder, landmarks, started):
            self.events = []
            alarm_at, alarm_stack = [], []
            def timeout(signum, frame):
                alarm_at.append(time.perf_counter())
                for _ in range(8):
                    if frame is None:
                        break
                    alarm_stack.append([frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno])
                    frame = frame.f_back
                raise TimeoutError('probe decision deadline')
            previous = signal.signal(signal.SIGALRM, timeout)
            # Diagnostic only: let collection finish so its stop callback is observable.
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                action = NeuralController.choose(self, observation, actor, encoder, landmarks, started)
            except TimeoutError:
                self.statistics['deadline_stops'] += 1
                legal = observation.getLegalActions(actor)
                action = 'Stop' if 'Stop' in legal else legal[0]
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            ended = time.perf_counter()
            if ended-started > 0.050:
                overlaps = [(end-start)*1000 for start,end,generation in pauses
                            if start <= ended and end >= started]
                late.append({'elapsed_ms': (ended-started)*1000, 'gc_pause_ms': overlaps,
                             'alarm_at_ms': [(point-started)*1000 for point in alarm_at],
                             'alarm_stack': alarm_stack,
                             'phases': [[name, (begin-started)*1000, (end-begin)*1000]
                                        for name,begin,end in self.events]})
            return action
    controller = ProbeController(ROOT/'runs/m0_original/training/model.pt', seed=8000000)
    gc.callbacks.append(callback)
    try:
        result = play_game(seed=8000000, controller=controller)
    finally:
        gc.callbacks.remove(callback)
    report = {'seed': 8000000, 'purpose': 'engineering diagnostic, excluded from M0 twenty games',
              'late_decisions': late, 'total_own_decisions': result['own_decisions'],
              'over_50ms': result['over_50ms'],
              'gc_pauses_over_10ms': [{'ms':(end-start)*1000, 'generation':generation}
                                    for start,end,generation in pauses if end-start>0.010]}
    report['asynchronous_alarm_disabled_for_diagnostic'] = True
    (ROOT/'runs/m0_original/latency_noalarm.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    run()
