"""Independent pilots and confirmation for both fixed P104 refit seeds."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from .checkpointing import sha256_file
from .evaluation_stats import estimate, sample_plan
from .fo2_setup import verify_build
from .learning_evaluate import initialise, collect, validate_design
from .p104_protocol import POLICY_SEEDS
from .policy_replay_artifacts import safe_path
from .train import write_json


def verified_exports(experiment, policy):
    root=Path(experiment); fit=json.loads((root/'experiment.json').read_text())
    if (fit['status']!='COMPLETE' or fit['protocol']!='p104_policy_export_v1'
            or fit['fit_seeds']!=list(POLICY_SEEDS) or sha256_file(policy)!=fit['source_policy_sha256']):
        raise ValueError('Completed paired exports and matching source required')
    specs={}
    for seed in POLICY_SEEDS:
        budget=json.loads((root/f'budget_seed{seed}.json').read_text())
        history=budget['validation_history']
        if budget['seed']!=seed or [r['update'] for r in history]!=fit['budgets']:
            raise ValueError('Budget selection grid or seed differs from protocol')
        if budget['selected_update']!=min(history,key=lambda r:(r['weighted_kl'],r['update']))['update']:
            raise ValueError('Budget selection differs from validation rule')
        pair=[]
        for arm in ('control','all_replay'):
            rows=[r for r in fit['results'] if r['arm']==arm and r['fit_seed']==seed]
            if len(rows)!=1: raise ValueError('Exactly one export per matched arm and seed required')
            row=rows[0]; path=safe_path(root,row['checkpoint'])
            if (sha256_file(path)!=row['sha256'] or row['fit']['selected_update']!=budget['selected_update']
                    or row['initial_model_sha256']!=budget['initial_model_sha256']):
                raise ValueError('Export checksum, initialization or budget mismatch')
            if arm=='all_replay' and row['fit']['validation_history']:
                raise ValueError('All-replay export cannot select on included validation data')
            if arm=='all_replay' and (row['fit']['validation_groups']!=0
                    or row['fit']['updates_completed']!=budget['selected_update']
                    or row['fit']['training_groups']!=sum(fit['partition_sizes'].values())):
                raise ValueError('All-replay fit did not use the frozen budget and complete replay')
            if arm=='control' and row['fit']['validation_history']!=history:
                raise ValueError('Control validation history differs from budget receipt')
            pair.append(row)
            specs[f'{arm}_{seed}']=dict(kind='checkpoint',checkpoint=str(path))
        if pair[0]['learning_rate']!=pair[1]['learning_rate']: raise ValueError('Unmatched learning rates')
    return specs,fit


def observations(rows, kind):
    per_seed={}
    for seed in POLICY_SEEDS:
        per_seed[str(seed)]=np.asarray([r[f'head_{seed}']['mean_bb'] if kind=='head' else
            r[f'all_replay_{seed}']['mean_bb']-r[f'control_{seed}']['mean_bb'] for r in rows])
    # Seeds within a deal are correlated: ONE observation per independent deal group.
    return np.mean(list(per_seed.values()),axis=0),per_seed


def evaluate(experiment, policy, output, *, kind, assets=None, workers=8, max_seconds=5400,
             smoke=False, primary_report=None, opponent='fo2'):
    if kind not in ('head','fo2'): raise ValueError('Unknown evaluation stage')
    if opponent not in ('fo2','check_call') or (not smoke and opponent!='fo2'):
        raise ValueError('Production benchmark requires FO2')
    specs,fit=verified_exports(experiment,policy)
    if bool(fit['smoke'])!=bool(smoke): raise ValueError('Smoke flag must match exports')
    pilot_n,minimum,maximum=(4,8,8) if smoke else (10000,400000 if kind=='head' else 100000,1000000)
    effect=1. if kind=='head' else 2.
    seed=1048701 if kind=='head' else 1048801
    validate_design(seed,workers,pilot_n,minimum,maximum,max_seconds,effect)
    gate=False
    if kind=='fo2':
        primary=json.loads(Path(primary_report).read_text())
        if (primary['kind']!='head' or primary['fit_sha256']!=sha256_file(Path(experiment)/'experiment.json')):
            raise ValueError('Secondary gate belongs to another primary experiment')
        gate=(primary['status']=='COMPLETE' and primary.get('result',{}).get('confirmatory',False)
              and primary['result']['conclusion']=='improved')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    build=verify_build(assets) if kind=='fo2' and opponent=='fo2' else None
    if kind=='head':
        matchups=[(f'head_{s}',f'all_replay_{s}',f'control_{s}',100+i) for i,s in enumerate(POLICY_SEEDS)]
    else:
        specs.update(P104=dict(kind='checkpoint',checkpoint=str(Path(policy).resolve())),
                     opponent=dict(kind=opponent,assets=str(Path(assets or '.').resolve())))
        matchups=[(name,name,'opponent',100+i) for i,name in enumerate(
            [f'{arm}_{s}' for s in POLICY_SEEDS for arm in ('control','all_replay')]+['P104'])]
    report=dict(status='STARTING',kind=kind,smoke=smoke,fit_sha256=sha256_file(Path(experiment)/'experiment.json'),
        primary_gate_passed=gate,completed_groups=0,
        inference_scope='One frozen replay, two fixed fit seeds. Independent unit is a shared deal group; seeds/hands are not independent replicates.')
    write_json(output/'manifest.json',dict(**report,policies=specs,opponent_build=build,matchups=matchups,
        pilot_seed=seed,confirmation_seed=seed+1,pilot_excluded=True,pilot_groups=pilot_n,
        minimum=minimum,maximum=maximum,target_effect_bb100=effect,power=.8,alpha=.05,
        secondary_rule='FO2 improvement uses fixed-sequence gate: confirmatory only after primary improvement.',
        workers=workers,max_seconds=max_seconds))
    deadline=time.monotonic()+max_seconds; main=[]
    try:
        with mp.get_context('spawn').Pool(workers,initializer=initialise,initargs=(specs,)) as pool:
            tick=time.monotonic()
            pilot=collect(pool,count=pilot_n,seed=seed,matchups=matchups,path=output/'pilot.jsonl',deadline=deadline,workers=workers)
            if len(pilot)!=pilot_n: raise TimeoutError('Incomplete pilot')
            plan=sample_plan(observations(pilot,kind)[0],minimum=minimum,maximum=maximum,effect_bb100=effect)
            rate=pilot_n/(time.monotonic()-tick)
            affordable=int(.8*rate*max(0,deadline-time.monotonic()))
            selected=max(minimum,min(maximum,affordable,max(minimum,plan['required'])))
            old_n=plan['selected']
            plan.update(selected=selected,capped=selected<plan['required'],affordable=affordable,
                minimum_fits_budget=affordable>=minimum,pilot_groups_per_second=rate,
                estimated_detectable_effect_bb100=plan['estimated_detectable_effect_bb100']*np.sqrt(old_n/selected))
            report['plan']=plan;write_json(output/'plan.json',plan)
            if affordable<minimum: report['status']='INSUFFICIENT_TIME_FOR_MINIMUM'
            else:
                collect(pool,count=selected,seed=seed+1,matchups=matchups,path=output/'confirmatory.jsonl',deadline=deadline,workers=workers,rows=main)
                report['status']='COMPLETE' if len(main)==selected else 'INCOMPLETE_TIME_LIMIT'
        if build is not None and verify_build(assets)!=build: raise ValueError('FO2 build changed during evaluation')
    except TimeoutError as exc: report.update(status='INCOMPLETE_TIME_LIMIT',reason=str(exc))
    except BaseException as exc:
        report.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}');raise
    finally:
        if len(main)>=2:
            aggregate,seeds=observations(main,kind)
            result=estimate(aggregate,complete=report['status']=='COMPLETE' and not smoke)
            if result['confirmatory'] and kind=='fo2' and not gate:
                result.update(confirmatory=False,conclusion='descriptive_primary_gate_not_passed')
            report['result']=result
            report['per_seed']={s:dict(estimate(x,complete=False),conclusion='descriptive') for s,x in seeds.items()}
            if kind=='fo2':
                names=[m[0] for m in matchups]
                report['absolute_fo2']={n:dict(estimate([r[n]['mean_bb'] for r in main],complete=False),conclusion='descriptive') for n in names}
        report.update(completed_groups=len(main),completed_hands=len(main)*2*len(matchups))
        write_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('experiment','policy','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--kind',choices=('head','fo2'),required=True)
    p.add_argument('--assets',type=Path);p.add_argument('--primary-report',type=Path)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--max-seconds',type=float,default=5400)
    p.add_argument('--smoke',action='store_true');p.add_argument('--opponent',default='fo2')
    if evaluate(**vars(p.parse_args()))['status']!='COMPLETE':raise SystemExit(2)

if __name__=='__main__':main()
