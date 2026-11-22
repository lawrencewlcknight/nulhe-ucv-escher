"""P104 all-row target decomposition and conditional noise-reduced fitting."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import regret_diagnostic as diag
from .checkpointing import LoadedHULHEPolicy, sha256_file
from .diagnostic_probes import regret_models
from .game import load_hulhe_game, serialisable_game_definition
from .p104_protocol import REGRET_SEEDS, ITERATION
from .policy_replay_artifacts import read_metadata
from .position_investigation.common import deadline_after
from .train import write_json


def conditional_gate(decompositions):
    base=diag.fit_gate(decompositions)
    cells=[x['after']['preflop']['repeated'] for x in decompositions]
    excess=[r['mean_target_mse'] is not None and r.get('iid_mean_noise_scale') is not None
            and r['mean_target_mse']>1.25*r['iid_mean_noise_scale'] for r in cells]
    return dict(run_fits=base['run_fits'] and any(excess), repetition_gate=base,
        mean_error_above_noise_sensitivity=excess,
        rule='Original repetition/error-share gate AND at least one player mean error >1.25 times IID mean-noise scale.',
        caution='Exploratory spending gate only. Unknown trajectory correlation prevents treating IID sensitivity as a significance test.')


def group_targets(data, ids):
    """Exact weighted group means for the row-MSE objective, within training only."""
    x,y,m=[data[k][ids] for k in ('features','target','mask')]
    first,inv,counts=diag.groups(x,m)
    means=np.zeros((len(first),3),np.float64);np.add.at(means,inv,y);means/=counts[:,None]
    return x[first],means.astype(np.float32),m[first],counts/counts.sum()


def fit(source,data,encoder,arm,seed,check,updates=(100,250,500)):
    torch.manual_seed(seed)
    models=[diag.ExpandedRegret(m,encoder.policy_layout) if arm=='expanded_group_mean' else deepcopy(m) for m in source]
    for a,b,d in zip(models,source,data):
        if not np.array_equal(diag.predict(a,d),diag.predict(b,d)):raise ValueError('Initial predictions changed')
    opts=[torch.optim.Adam(m.parameters(),lr=.0003) for m in models]
    rng=np.random.default_rng(seed); started=time.monotonic()
    training=[]
    for d in data:
        ids=d['partitions'][0]
        if arm=='row_current':
            training.append((*[d[k][ids] for k in ('features','target','mask')],None))
        else: training.append(group_targets(d,ids))
    baseline=diag.scores(models,data,1)['preflop_mean_target_mse']
    if baseline is None or not np.isfinite(baseline):raise ValueError('No preflop validation score')
    best_score,best_step,best=baseline,0,deepcopy(models)
    history=[dict(update=0,validation_preflop_mean_mse=baseline)]
    for step in range(1,max(updates)+1):
        check()
        for model,opt,(x,y,mask,prob),d in zip(models,opts,training,data):
            batch=min(1024,len(d['partitions'][0]))
            idx=rng.choice(len(x),batch,replace=True,p=prob)
            xx,yy,mm=[torch.from_numpy(a[idx]).float() for a in (x,y,mask)]
            model.train();opt.zero_grad(set_to_none=True)
            loss=((model(xx)-yy)*mm).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite regret fit')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        if step in updates:
            for m in models:m.eval()
            score=diag.scores(models,data,1)['preflop_mean_target_mse']
            if score is None or not np.isfinite(score):raise ValueError('Nonfinite regret validation')
            history.append(dict(update=step,validation_preflop_mean_mse=score))
            if score<best_score:best_score,best_step,best=score,step,deepcopy(models)
    return best,dict(arm=arm,seed=seed,selected_update=best_step,validation_history=history,
        validation_preflop_mean_mse=best_score,fit_seconds=time.monotonic()-started,
        parameters_per_player=sum(p.numel() for p in best[0].parameters()),
        training=diag.scores(best,data,0),validation=diag.scores(best,data,1),
        optimizer='Fresh Adam, learning rate 0.0003, clip 10; source P104 weights. Diagnostic refit, not production continuation.')


def run(captures,state,policy,output,*,threads=8,max_seconds=1200,smoke=False):
    check=deadline_after(max_seconds);torch.set_num_threads(threads)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(status='ANALYSING',protocol='p104_regret_v1',smoke=smoke,results=[],
        fit_seeds=list(REGRET_SEEDS),source_policy_sha256=sha256_file(policy),
        limitations=['One isolated iteration; not a full training experiment.',
            'All rows of this iteration are captured, not accumulated history.',
            'IID mean-noise scales are sensitivity diagnostics; trajectory identifiers are unavailable in archived replay.',
            'Original production post-fit errors are in-sample; diagnostic splits hold out new-iteration groups from these refits only.',
            'Outputs are instantaneous regret strategies, not average policies or resumable production states.'])
    started=time.monotonic()
    try:
        loaded=LoadedHULHEPolicy(load_hulhe_game(),policy)
        capture,data=diag.load_captures(captures,loaded.feature_encoder)
        _,metadata=read_metadata(state)
        if (capture['source_policy_sha256']!=sha256_file(policy)
            or capture['source_manifest_sha256']!=sha256_file(Path(state)/'manifest.json')
            or metadata['solver']['num_iteration']!=capture['source_iteration']
            or metadata['encoder']!=loaded.checkpoint['feature_encoder']
            or metadata['config']!=loaded.checkpoint['training_config']
            or metadata['game']!=serialisable_game_definition()):raise ValueError('P104 source identities differ')
        if not smoke and (capture['source_iteration']!=ITERATION or not capture.get('capture_all_regrets')):
            raise ValueError('Requires all-row P104 capture')
        for record in capture['captures']:
            if record['name'].startswith('regret_') and not smoke and record['sampled_rows']!=record['rows']:
                raise ValueError('Truncated regret capture')
        source=regret_models(metadata,loaded.feature_encoder)
        for m,d in zip(source,data):
            if not np.allclose(diag.predict(m,d)*d['mask'],d['before'],rtol=3e-5,atol=3e-5):
                raise ValueError('P104 weights do not reproduce pre-fit predictions')
        report.update(source_iteration=capture['source_iteration'],capture_iteration=capture['iteration'],
            source_manifest_sha256=sha256_file(Path(state)/'manifest.json'),
            capture_report_sha256=sha256_file(Path(captures)/'report.json'),
            decomposition=[{phase:diag.by_street(d,d[phase]) for phase in ('before','after')} for d in data],
            partitions=[{n:len(ids) for n,ids in zip(('train','validation','audit'),d['partitions'])} for d in data])
        gate=conditional_gate(report['decomposition']);report['fit_gate']=gate
        adequate=all(all(len(ids) and np.any(d['street'][ids]==0) for ids in d['partitions']) for d in data)
        gate['partitions_adequate']=adequate;gate['run_fits'] &= adequate
        candidates={};selected_arm='source';scores={}
        if gate['run_fits']:
            updates=(1,2) if smoke else (100,250,500)
            baseline=diag.scores(source,data,1)['preflop_mean_target_mse']
            def fit_arm(arm):
                for seed in REGRET_SEEDS:
                    models,row=fit(source,data,loaded.feature_encoder,arm,seed,check,updates)
                    candidates[(arm,seed)]=models;report['results'].append(row)
                    write_json(output/'report.json',report)
                scores[arm]=float(np.mean([r['validation_preflop_mean_mse'] for r in report['results'] if r['arm']==arm]))
            fit_arm('row_current');fit_arm('group_mean_current')
            # Spend on capacity only if the cheaper same-architecture fits fail
            # to remove at least 10% of validation mean-target error.
            expand=min(scores.values())>.9*baseline
            report['expansion_gate']=dict(run=expand,source_validation=baseline,
                best_same_architecture_validation=min(scores.values()),
                rule='Expand only if same-architecture validation error reduction is less than 10%; audit and rivers remain unseen.')
            if expand:fit_arm('expanded_group_mean')
            ordered=['row_current','group_mean_current','expanded_group_mean']
            winner=min(scores,key=lambda a:(scores[a],ordered.index(a)))
            if scores[winner]<baseline:selected_arm=winner
        report['selection']=dict(arm=selected_arm,seeds=list(REGRET_SEEDS),validation_scores=scores,
            rule='Best two-seed mean validation error, source wins ties; retain both fixed seeds; no audit/river selection.')
        write_json(output/'selection.json',report['selection'])
        # Freeze selection before any audit scoring or playing diagnostics.
        for row in report['results']:
            row['audit']=diag.scores(candidates[(row['arm'],row['seed'])],data,2)
        report['source_audit']=diag.scores(source,data,2) if adequate else None
        report['exports']=[]
        for seed in REGRET_SEEDS:
            selected=source if selected_arm=='source' else candidates[(selected_arm,seed)]
            path=output/f'regret_{seed}.pt'
            torch.save(dict(kind='p104_diagnostic_regret_only',source_policy_sha256=sha256_file(policy),
                arm=selected_arm,seed=seed,models=[m.state_dict() for m in selected],
                architectures=[m.checkpoint_metadata() for m in selected]),path)
            report['exports'].append(dict(seed=seed,path=path.name,sha256=sha256_file(path)))
        report['status']='COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED',reason=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;write_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('captures','state','policy','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--max-seconds',type=float,default=1200)
    p.add_argument('--smoke',action='store_true');run(**vars(p.parse_args()))

if __name__=='__main__':main()
