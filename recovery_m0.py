"""One bounded student-state labeling round and equal-update training controls."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from belief_mita.evaluate import TimedController, defer_cyclic_gc
from belief_mita.runtime import ROOT, play_game
from diagnose_m0 import replay
from improve_m0 import CHECKPOINT, summarize, paired_comparison

OUT = ROOT/'runs/m0_recovery_20261105'


def selection(summary):
    if any(s['completed']!=20 or s['failures'] or s['over_50ms'] for s in summary.values()):
        return None
    base,extra,new = [summary[arm]['score_rate'] for arm in ('original','old_extra','recovery')]
    if new>=.75 and new-base>=.05-1e-9 and new-extra>=.05-1e-9:
        return 'recovery'
    if extra>=.75 and extra-base>=.05-1e-9 and extra>=new:
        return 'old_extra'
    return None


def job(stage,arm,index,deadline):
    seed_base = {'collection':9300000,'development':9400000,'confirmation':9200000}[stage]
    spec = {'seed':seed_base+index//2,'red':index%2 == 0}
    output = OUT/stage/arm
    output.mkdir(parents=True,exist_ok=True)
    try:
        if time.monotonic()>=deadline:
            raise TimeoutError('shared90minute budget expired')
        checkpoint = CHECKPOINT if arm=='original' else OUT/arm/'model.pt'
        controller = TimedController(checkpoint,spec['seed'],mode='original',device='cpu')
        with defer_cyclic_gc():
            result = play_game(**spec,controller=controller,collect=stage=='collection',deadline=deadline)
        if stage=='collection':
            result,episode = result
            slots = episode['actor_slot']
            truth = episode['labels'][np.arange(len(slots)),slots]
            assert episode['legal'][np.arange(len(slots)),slots,truth].all()
            result['samples'] = len(slots)
            result['teacher_disagreements'] = int((truth!=episode['student_action']).sum())
            result['teacher_stop'] = int((truth==4).sum())
            result['student_stop'] = int((episode['student_action']==4).sum())
            dataset_file = OUT/'data'/f'game_{index:04d}.npz'
            dataset_file.parent.mkdir(exist_ok=True)
            np.savez_compressed(dataset_file,**episode)
            result['data_sha256'] = hashlib.sha256(dataset_file.read_bytes()).hexdigest()
        statistics = dict(controller.statistics)
        forwards = statistics.pop('forward_ms')
        statistics['forward_p95_ms'] = float(np.percentile(forwards,95)) if forwards else None
        result['search'] = statistics
    except Exception as error:
        result = {**spec,'validated':False,'outcome':-1,'error':repr(error)}
    result.update(stage=stage,mode=arm,index=index)
    (output/f'game_{index:02d}.json').write_text(json.dumps(result),encoding='utf-8')
    return result


def stage_run(stage,arms,deadline):
    if (OUT/stage).exists():
        raise FileExistsError('stage already exists; preserve it')
    records = []
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(job,stage,arm,index,deadline) for index in range(20)
                   for arm in arms[index%len(arms):]+arms[:index%len(arms)]]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%10==0:
                print(json.dumps({'stage':stage,'completed_records':len(records),'planned':20*len(arms)}),flush=True)
    summary = summarize(records)
    audits = {}
    for arm in arms:
        audits[arm] = []
        for record in records:
            if record['mode']==arm and record['validated']:
                if time.monotonic()>=deadline:
                    raise TimeoutError('shared budget expired before trace audit')
                audits[arm].append(replay(record))
    report = {'summary':summary,'replay_diagnostics':audits,'all_valid_traces_replayed':True}
    if stage!='collection':
        report['comparisons'] = {arm:paired_comparison(records,arm) for arm in arms if arm!='original'}
    else:
        valid = [r for r in records if r['validated']]
        report['teacher_labels'] = {key:sum(r[key] for r in valid) for key in
                                   ('samples','teacher_disagreements','teacher_stop','student_stop')}
    (OUT/stage/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'stage':stage,**{key:value for key,value in report.items() if key!='replay_diagnostics'}}),flush=True)
    return report


def main():
    from belief_mita.finetune import finetune
    if (OUT/'manifest.json').exists():
        raise FileExistsError('recovery experiment already exists')
    OUT.mkdir(exist_ok=True)
    first = json.loads((ROOT/'runs/m0_improvement_20261105/manifest.json').read_text())
    remaining = datetime.fromisoformat(first['started_utc']).timestamp()+90*60-time.time()
    deadline = time.monotonic()+remaining
    manifest = {'started_utc':datetime.now(timezone.utc).isoformat(),'remaining_seconds':remaining,
                'shared_budget_start':first['started_utc'],'source_sha256':{},
                'base_checkpoint_sha256':hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                'protocol_sha256':hashlib.sha256((ROOT/'RECOVERY_PROTOCOL_20261105.md').read_bytes()).hexdigest()}
    for path in list((ROOT/'belief_mita').glob('*.py'))+[Path(__file__).resolve(),ROOT/'improve_m0.py']:
        relative = path.relative_to(ROOT)
        manifest['source_sha256'][str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        target = OUT/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    result = {'status':'RUNNING','stages':{}}
    def save():
        (OUT/'result.json').write_text(json.dumps(result,indent=2))
    try:
        collection = stage_run('collection',['original'],deadline)
        result['collection'] = collection['teacher_labels']
        if collection['summary']['original']['failures']:
            result['status'] = 'COLLECTION_EXECUTION_ERROR'
            save()
            return
        for arm in ('old_extra','recovery'):
            result['stages'][arm] = 'TRAINING'
            save()
            training = finetune(ROOT/'runs/m0_original/data',OUT/'data',CHECKPOINT,OUT/arm,
                                use_recovery=arm=='recovery',epochs=5,samples_per_epoch=24000,deadline=deadline)
            assert training['status']=='completed'
            result['stages'][arm] = 'COMPLETED'
        development = stage_run('development',['original','old_extra','recovery'],deadline)
        selected = selection(development['summary'])
        result.update(selected=selected,development=development['summary'],status='NOT_ESTABLISHED')
        save()
        if selected:
            confirmation = stage_run('confirmation',['original',selected],deadline)
            candidate = confirmation['summary'][selected]
            control = confirmation['summary']['original']
            success = (candidate['completed']==20 and not candidate['failures'] and not candidate['over_50ms']
                       and candidate['score_rate']>=.75 and candidate['score_rate']>control['score_rate']
                       and control['completed']==20 and not control['failures'] and not control['over_50ms'])
            result.update(status='GO_EXPLORATORY' if success else 'NOT_CONFIRMED',
                          confirmation=confirmation['summary'],paired=confirmation['comparisons'][selected])
        save()
        print(json.dumps(result),flush=True)
    except Exception as error:
        result.update(status='INCONCLUSIVE_RESOURCE' if isinstance(error,TimeoutError) else 'EXECUTION_ERROR',error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
