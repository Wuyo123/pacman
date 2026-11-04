"""Final bounded goal-progress trial with a rule-only attribution control."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from belief_mita.evaluate import TimedController, defer_cyclic_gc
from belief_mita.goal_guard import HeuristicController
from belief_mita.runtime import ROOT, play_game
from improve_m0 import CHECKPOINT, finalize_stage

OUT = ROOT/'runs/m0_goal_20261105'


def selection(summaries):
    base = summaries['original']
    if base['completed']!=20 or base['failures'] or base['over_50ms']:
        return None
    eligible = [arm for arm in ('guard','heuristic') if summaries[arm]['completed']==20
                and not summaries[arm]['failures'] and not summaries[arm]['over_50ms']
                and summaries[arm]['score_rate']>=.75
                and summaries[arm]['score_rate']-base['score_rate']>=.05-1e-9]
    if len(eligible)==2 and abs(summaries['guard']['score_rate']-summaries['heuristic']['score_rate'])<.05-1e-9:
        return 'heuristic'
    return max(eligible,key=lambda arm:summaries[arm]['score_rate']) if eligible else None


def job(stage,arm,index,deadline):
    spec = {'seed':(9600000 if stage=='development' else 9200000)+index//2,'red':index%2==0}
    try:
        if time.monotonic()>=deadline:
            raise TimeoutError('shared90minute budget expired')
        controller = (HeuristicController() if arm=='heuristic' else
                      TimedController(CHECKPOINT,spec['seed'],mode='guard' if arm=='guard' else 'original'))
        with defer_cyclic_gc():
            result = play_game(**spec,controller=controller,deadline=deadline)
        statistics = dict(controller.statistics)
        forwards = statistics.pop('forward_ms')
        statistics['forward_p95_ms'] = float(np.percentile(forwards,95)) if forwards else None
        result['search'] = statistics
    except Exception as error:
        result = {**spec,'validated':False,'outcome':-1,'error':repr(error)}
    result.update(stage=stage,mode=arm,index=index)
    target = OUT/stage/arm/f'game_{index:02d}.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result),encoding='utf-8')
    return result


def run_stage(stage,arms,deadline):
    if (OUT/stage).exists():
        raise FileExistsError('stage already exists')
    records = []
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(job,stage,arm,index,deadline) for index in range(20)
                   for arm in arms[index%len(arms):]+arms[:index%len(arms)]]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%10==0:
                print(json.dumps({'stage':stage,'completed_records':len(records),'planned':20*len(arms)}),flush=True)
    report = finalize_stage(stage,arms,records,deadline,out=OUT)
    for arm in arms:
        report['summary'][arm]['goal_guard_overrides'] = sum(
            r.get('search',{}).get('goal_guard_overrides',0) for r in records if r['mode']==arm)
    (OUT/stage/'summary.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    if (OUT/'manifest.json').exists():
        raise FileExistsError('final trial already exists')
    OUT.mkdir(exist_ok=True)
    first = json.loads((ROOT/'runs/m0_improvement_20261105/manifest.json').read_text())
    remaining = datetime.fromisoformat(first['started_utc']).timestamp()+90*60-time.time()
    deadline = time.monotonic()+remaining
    manifest = {'started_utc':datetime.now(timezone.utc).isoformat(),'shared_budget_start':first['started_utc'],
                'remaining_seconds':remaining,'checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                'heuristic_has_no_network':True,'source_sha256':{},
                'protocol_sha256':hashlib.sha256((ROOT/'GOAL_PROTOCOL_20261105.md').read_bytes()).hexdigest()}
    for path in list((ROOT/'belief_mita').glob('*.py'))+[Path(__file__).resolve(),ROOT/'improve_m0.py']:
        relative = path.relative_to(ROOT)
        manifest['source_sha256'][str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot = OUT/'source'/relative
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    result = {'status':'RUNNING'}
    try:
        development = run_stage('development',['original','guard','heuristic'],deadline)
        selected = selection(development['summary']) if development['all_valid_traces_replayed'] else None
        result.update(status='NOT_ESTABLISHED',selected=selected,development=development['summary'])
        (OUT/'result.json').write_text(json.dumps(result,indent=2))
        if selected:
            confirmation = run_stage('confirmation',['original',selected],deadline)
            candidate,control = confirmation['summary'][selected],confirmation['summary']['original']
            success = (candidate['completed']==20 and not candidate['failures'] and not candidate['over_50ms']
                       and candidate['score_rate']>=.75 and candidate['score_rate']>control['score_rate']
                       and control['completed']==20 and not control['failures'] and not control['over_50ms']
                       and confirmation['all_valid_traces_replayed'])
            result.update(status='GO_EXPLORATORY' if success else 'NOT_CONFIRMED',
                          confirmation=confirmation['summary'],paired=confirmation['comparisons'][selected])
        (OUT/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    except Exception as error:
        result.update(status='INCONCLUSIVE_RESOURCE' if isinstance(error,TimeoutError) else 'EXECUTION_ERROR',error=repr(error))
        (OUT/'result.json').write_text(json.dumps(result,indent=2))
        raise


if __name__ == '__main__':
    main()
