"""Predeclared frozen-weight M0 controls, candidate selection, and fresh confirmation."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

from belief_mita.evaluate import TimedController, defer_cyclic_gc
from belief_mita.runtime import ROOT, play_game
from diagnose_m0 import replay

OUT = ROOT/'runs/m0_improvement_20261105'
CHECKPOINT = ROOT/'runs/m0_original/training/model.pt'
MODES = ('original','diverse','policy')


def select_candidate(summaries):
    baseline = summaries['original']
    if baseline['completed'] != 20 or baseline['failures'] or baseline['over_50ms']:
        return None
    control = baseline['score_rate']
    eligible = [mode for mode in ('diverse','policy') if summaries[mode]['completed'] == 20
                and summaries[mode]['failures'] == 0 and summaries[mode]['over_50ms'] == 0
                and summaries[mode]['score_rate'] >= .75
                and summaries[mode]['score_rate']-control >= .05-1e-9]
    return max(eligible, key=lambda mode:(summaries[mode]['score_rate'],mode=='policy')) if eligible else None


def one_game(stage, mode, index, deadline):
    spec = {'seed':(9100000 if stage == 'development' else 9200000)+index//2,
            'red':index%2 == 0}
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError('stage budget expired before controller setup')
        controller = TimedController(CHECKPOINT,spec['seed'],device='cpu',mode=mode)
        with defer_cyclic_gc():
            if time.monotonic() >= deadline:
                raise TimeoutError('stage budget expired before game startup')
            result = play_game(**spec, controller=controller, deadline=deadline)
        statistics = dict(controller.statistics)
        forwards = statistics.pop('forward_ms')
        statistics['forward_p95_ms'] = float(np.percentile(forwards,95)) if forwards else None
        result['search'] = statistics
    except Exception as error:
        result = {**spec,'outcome':-1,'validated':False,'error':repr(error)}
    result.update(stage=stage,mode=mode,index=index)
    path = OUT/stage/mode/f'game_{index:02d}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result),encoding='utf-8')
    return result


def summarize(records):
    results = {}
    for mode in sorted(set(record['mode'] for record in records)):
        games = [r for r in records if r['mode'] == mode]
        wins = sum(r['outcome'] > 0 for r in games)
        draws = sum(r['outcome'] == 0 for r in games)
        verified = [r for r in games if r['validated']]
        histogram = [sum(r['search']['root_action_choices'][i] for r in verified) for i in range(4)]
        results[mode] = {'completed':len(verified),'records':len(games),'planned':20,'wins':wins,'draws':draws,
                         'losses':len(games)-wins-draws,'score_rate':(wins+.5*draws)/20,
                         'failures':len(games)-len(verified),
                         'over_50ms':sum(r['over_50ms'] for r in verified),
                         'own_decisions':sum(r['own_decisions'] for r in verified),
                         'max_ms':max((r['own_ms_max'] for r in verified),default=None),
                         'forward_p95_ms_max':max((r['search']['forward_p95_ms'] for r in verified),default=None),
                         'root_action_choices':histogram,
                         'search_changed_policy':sum(r['search']['search_changed_policy'] for r in verified),
                         'simulations':sum(r['search']['simulations'] for r in verified),
                         'chosen_stop':sum(r['search']['chosen_stop'] for r in verified)}
    return results


def paired_comparison(records, candidate):
    outcomes = {(r['mode'],r['seed'],r['red']):(r['outcome']+1)/2 for r in records}
    seeds = sorted(set(r['seed'] for r in records))
    differences = [np.mean([outcomes[candidate,seed,red]-outcomes['original',seed,red]
                           for red in (True,False)]) for seed in seeds]
    rng = np.random.default_rng(20261105)
    boot = np.asarray(differences)[rng.integers(0,len(seeds),size=(10000,len(seeds)))].mean(1)
    positives = sum(value>0 for value in differences)
    negatives = sum(value<0 for value in differences)
    n = positives+negatives
    p = min(1.,2*sum(math.comb(n,k) for k in range(min(positives,negatives)+1))/2**n) if n else 1.
    return {'seed_blocks':len(seeds),'mean_score_gain':float(np.mean(differences)),
             'bootstrap95_seed_block_gain':[float(v) for v in np.quantile(boot,[.025,.975])],
             'positive_blocks':positives,'negative_blocks':negatives,'ties':len(seeds)-n,
             'paired_sign_two_sided_p':p,'differences':differences}


def run_stage(stage,modes,deadline):
    stage_dir = OUT/stage
    if stage_dir.exists():
        raise FileExistsError('refusing to overwrite an experiment stage')
    stage_dir.mkdir()
    jobs = []
    for index in range(20):
        offset = index%len(modes)
        rotated = modes[offset:]+modes[:offset]
        jobs.extend((mode,index) for mode in rotated)
    records = []
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(one_game,stage,mode,index,deadline):(mode,index) for mode,index in jobs}
        for future in as_completed(futures):
            mode,index = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {'mode':mode,'index':index,'outcome':-1,'validated':False,
                          'seed':(9100000 if stage=='development' else 9200000)+index//2,
                          'red':index%2==0,'error':repr(error)}
                path = stage_dir/mode/f'game_{index:02d}.json'
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(result))
            records.append(result)
            if len(records)%10 == 0:
                print(json.dumps({'stage':stage,'completed':len(records),'planned':len(jobs)}),flush=True)
    summary = summarize(records)
    comparisons = {mode:paired_comparison(records,mode) for mode in modes if mode!='original'}
    replays, audit_complete = {}, True
    for mode in modes:
        games = [r for r in records if r['mode'] == mode and r['validated']]
        replays[mode] = []
        for record in games:
            if time.monotonic() >= deadline:
                audit_complete = False
                break
            replays[mode].append(replay(record))
    report = {'summary':summary,'comparisons':comparisons,'replay_diagnostics':replays,
              'planned_games':len(jobs),'completed_records':len(records),
              'all_valid_traces_replayed':audit_complete,
              'resource_status':'COMPLETE' if audit_complete and time.monotonic()<deadline else 'INCONCLUSIVE_RESOURCE'}
    (stage_dir/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'stage':stage,'summary':summary,'comparisons':comparisons}),flush=True)
    return report


def main():
    if (OUT/'manifest.json').exists():
        raise FileExistsError('experiment already exists')
    started = time.monotonic()
    manifest = {'started_utc':datetime.now(timezone.utc).isoformat(),'base_commit':'7bf5a308bf05757bbe1f82def8183cc0d50db6e2',
                'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                'workers':3,'device':'cpu','action_budget_ms':50,'development_minutes':45,'total_minutes':90,
                'source_sha256':{},'protocol_sha256':hashlib.sha256((ROOT/'IMPROVEMENT_PROTOCOL_20261105.md').read_bytes()).hexdigest()}
    for path in list((ROOT/'belief_mita').glob('*.py'))+[Path(__file__).resolve(),ROOT/'diagnose_m0.py']:
        relative = path.relative_to(ROOT)
        manifest['source_sha256'][str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot = OUT/'source'/relative
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    development = run_stage('development',list(MODES),started+45*60)
    selected = select_candidate(development['summary']) if development['all_valid_traces_replayed'] else None
    result = {'selected':selected,'development':development['summary'],
              'confirmation_status':'NOT_RUN_PREREGISTERED_STOP','base_model_unchanged':True}
    (OUT/'selection.json').write_text(json.dumps(result,indent=2))
    if selected is not None:
        confirmation = run_stage('confirmation',['original',selected],started+90*60)
        candidate = confirmation['summary'][selected]
        control = confirmation['summary']['original']
        success = (candidate['completed']==20 and candidate['failures']==0 and candidate['over_50ms']==0
                   and candidate['score_rate']>=.75 and candidate['score_rate']>control['score_rate']
                   and control['failures']==0 and control['over_50ms']==0
                   and confirmation['all_valid_traces_replayed'])
        result.update(confirmation_status='GO_EXPLORATORY' if success else 'NOT_CONFIRMED',
                      confirmation=confirmation['summary'],paired=confirmation['comparisons'][selected])
    result['elapsed_seconds'] = time.monotonic()-started
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
