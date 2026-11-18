"""Fresh paired deals for one validation-selected candidate, P32 and FO2."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
from .checkpointing import sha256_file
from .evaluation_stats import estimate, sample_plan
from .learning_evaluate import collect, initialise, validate_design
from .fo2_setup import DEFAULT_ASSETS, verify_build
from .policy_replay_artifacts import safe_path
from .train import write_json


def observations(rows):
    return dict(head_to_head=np.array([r['head']['mean_bb'] for r in rows]),
                fo2_improvement=np.array([r['candidate_fo2']['mean_bb']-r['p32_fo2']['mean_bb'] for r in rows]))


def verified_candidate(experiment, policy):
    root=Path(experiment); report=json.loads((root/'experiment.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    if report['status']!='COMPLETE' or selection!=report['selection']:
        raise ValueError('Completed refits and frozen selection are required')
    from .bottleneck_refit import select_arm
    arm,scores=select_arm(report['results'])
    if selection['arm']!=arm or selection['scores']!=scores or selection['fit_seed']!=report['fit_seeds'][0]:
        raise ValueError('Selection violates predeclared validation/first-seed rule')
    candidate=safe_path(root,selection['checkpoint'])
    if sha256_file(candidate)!=selection['sha256'] or sha256_file(policy)!=report['source_policy_sha256']:
        raise ValueError('Selected policy or baseline checksum differs')
    row=next(r for r in report['results'] if r['arm']==arm and r['fit_seed']==selection['fit_seed'])
    if row['checkpoint']!=selection['checkpoint'] or row['sha256']!=selection['sha256']:
        raise ValueError('Selected export does not match fitting result')
    return candidate,report


def evaluate(experiment,policy,output,*,assets=DEFAULT_ASSETS,workers=8,max_seconds=3600,
             pilot_pairs=5000,minimum=100000,maximum=250000,seed=871901,opponent='fo2'):
    validate_design(seed,workers,pilot_pairs,minimum,maximum,max_seconds,2.)
    if opponent not in ('fo2','check_call'): raise ValueError('Invalid diagnostic opponent')
    candidate,fit=verified_candidate(experiment,policy)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    smoke=fit['smoke'] or opponent!='fo2'
    build=verify_build(assets) if opponent=='fo2' else None
    specs=dict(candidate=dict(kind='checkpoint',checkpoint=str(candidate.resolve())),
        p32=dict(kind='checkpoint',checkpoint=str(Path(policy).resolve())),
        fo2=dict(kind=opponent,assets=str(Path(assets).resolve())))
    matchups=[('head','candidate','p32',10),('candidate_fo2','candidate','fo2',20),('p32_fo2','p32','fo2',20)]
    manifest=dict(experiment_sha256=sha256_file(Path(experiment)/'experiment.json'),selection=fit['selection'],
        source_policy_sha256=sha256_file(policy),pilot_seed=seed,confirmation_seed=seed+1,
        alpha_per_test=.025,tests=['head_to_head','fo2_improvement'],pilot_excluded=True,
        pilot_pairs=pilot_pairs,minimum=minimum,maximum=maximum,max_seconds=max_seconds,workers=workers,
        unit='Shared independent deal group, seat averaged. One predeclared fit seed.',
        opponent_variant='FO2 free-fold-to-check adaptation',opponent_build=build,smoke=smoke)
    write_json(output/'manifest.json',manifest)
    report=dict(status='EVALUATING',smoke=smoke,results={},completed_groups=0)
    deadline=time.monotonic()+max_seconds; main=[]
    try:
        with mp.get_context('spawn').Pool(workers,initializer=initialise,initargs=(specs,)) as pool:
            tick=time.monotonic()
            pilot=collect(pool,count=pilot_pairs,seed=seed,matchups=matchups,path=output/'pilot.jsonl',deadline=deadline,workers=workers)
            if len(pilot)!=pilot_pairs: report['status']='INCOMPLETE_PILOT'
            else:
                plans={k:sample_plan(v,minimum=minimum,maximum=maximum,effect_bb100=2.,alpha=.025) for k,v in observations(pilot).items()}
                rate=pilot_pairs/(time.monotonic()-tick); affordable=int(.8*rate*max(0,deadline-time.monotonic()))
                required=max(p['required'] for p in plans.values()); selected=min(maximum,affordable,max(minimum,required))
                report['plan']=dict(tests=plans,selected=selected,required=required,capped=selected<required,affordable=affordable,pilot_groups_per_second=rate)
                for plan in plans.values():
                    plan.update(selected=selected,capped=selected<plan['required'],
                        estimated_detectable_effect_bb100=plan['estimated_detectable_effect_bb100']*np.sqrt(plan['selected']/max(1,selected)))
                write_json(output/'plan.json',report['plan'])
                if selected<minimum: report['status']='INSUFFICIENT_TIME_FOR_MINIMUM'
                else:
                    collect(pool,count=selected,seed=seed+1,matchups=matchups,path=output/'confirmatory.jsonl',deadline=deadline,workers=workers,rows=main)
                    report['status']='COMPLETE' if len(main)==selected else 'INCOMPLETE_TIME_LIMIT'
        if opponent=='fo2' and verify_build(assets)!=build: report['status']='INVALID_OPPONENT_VERSION_CHANGE'
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    finally:
        if len(main)>=2:
            report['results']={k:estimate(v,alpha=.025,complete=report['status']=='COMPLETE' and not smoke) for k,v in observations(main).items()}
            report['absolute_fo2_descriptive']={k:estimate([r[k]['mean_bb'] for r in main],complete=False) for k in ('candidate_fo2','p32_fo2')}
        report.update(completed_groups=len(main),completed_hands=6*len(main),inference_scope='Conditional on one P32 replay and the fixed selected policy; not independent training seeds or exploitability')
        write_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('experiment','policy','output','assets'): p.add_argument('--'+n,type=Path,required=True)
    result=evaluate(**vars(p.parse_args()))
    if result['status']!='COMPLETE': raise SystemExit(1)

if __name__=='__main__': main()
