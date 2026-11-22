"""P104 paired fits, exact objective identities, frozen selection and Batch contracts."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_policy_sampling_experiment import frozen_source
from test_regret_diagnostic import diagnostic_inputs
from hulhe_escher import p104_policy_refit as refit
from hulhe_escher import p104_policy_evaluate as evaluate
from hulhe_escher import p104_regret as regret
from hulhe_escher import p104_rivers as rivers
from hulhe_escher.checkpointing import sha256_file,LoadedHULHEPolicy
from hulhe_escher.game import load_hulhe_game
from hulhe_escher.p104_protocol import POLICY_SEEDS,REGRET_SEEDS
from hulhe_escher.strategic_audit.library import digest


@pytest.fixture
def fitted(frozen_source,tmp_path):
    state,policy=frozen_source
    output=tmp_path/'fit'
    before={p:sha256_file(p) for p in (policy,state/'manifest.json')}
    report=refit.run(state,policy,output,threads=1,smoke=True,max_seconds=90)
    assert all(sha256_file(p)==h for p,h in before.items())
    return state,policy,output,report


def test_real_paired_refits_and_offline_evaluation(fitted,tmp_path):
    state,policy,output,report=fitted
    assert report['status']=='COMPLETE' and len(report['results'])==6
    for seed in POLICY_SEEDS:
        rows={r['arm']:r for r in report['results'] if r['fit_seed']==seed}
        a,b=rows['control'],rows['all_replay']
        assert a['initial_model_sha256']==b['initial_model_sha256']
        assert a['fit']['selected_update']==b['fit']['selected_update']
        assert a['fit']['validation_groups']>0 and b['fit']['validation_groups']==0
        assert not b['fit']['validation_history']
        assert b['fit']['training_groups']==sum(report['partition_sizes'].values())
        assert rows['legacy_reference']['fit']['selected_update'] in (4,8)
    specs,_=evaluate.verified_exports(output,policy)
    for spec in specs.values():
        loaded=LoadedHULHEPolicy(load_hulhe_game(),spec['checkpoint'])
        assert loaded.checkpoint['policy_refit']['source_sha256']==sha256_file(policy)
    head=evaluate.evaluate(output,policy,tmp_path/'head',kind='head',workers=1,smoke=True,max_seconds=90)
    assert head['status']=='COMPLETE' and head['completed_groups']==8 and head['completed_hands']==32
    assert not head['result']['confirmatory']
    fo2=evaluate.evaluate(output,policy,tmp_path/'fo2',kind='fo2',workers=1,smoke=True,max_seconds=90,
        opponent='check_call',primary_report=tmp_path/'head/report.json')
    assert fo2['status']=='COMPLETE' and fo2['completed_hands']==80
    assert len(fo2['per_seed'])==2 and len(fo2['absolute_fo2'])==5


def test_refit_export_tampering_rejected(fitted):
    _,policy,root,report=fitted
    budget=root/f'budget_seed{POLICY_SEEDS[0]}.json'
    row=json.loads(budget.read_text());row['selected_update']=999;budget.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='selection'):evaluate.verified_exports(root,policy)


def test_deal_group_aggregation_preserves_seed_covariance():
    rows=[]
    for x in (1.,2.,3.):
        rows.append({f'head_{s}':dict(mean_bb=x*(i+1)) for i,s in enumerate(POLICY_SEEDS)})
    means,seeds=evaluate.observations(rows,'head')
    np.testing.assert_array_equal(means,[1.5,3.,4.5])
    assert len(means)==3 and len(seeds)==2  # Not six independent observations.


def test_weighted_group_mean_gradient_matches_raw_rows():
    x=np.array([[1.,2.],[1.,2.],[3.,1.]],np.float32)
    y=np.array([[1,99,3],[3,-99,1],[4,1,2]],np.float32)
    masks=np.array([[1,0,1],[1,0,1],[1,1,1]],np.float32)
    data=dict(features=x,target=y,mask=masks)
    xx,yy,mm,prob=regret.group_targets(data,np.arange(3))
    model=torch.nn.Linear(2,3);a=deepcopy(model);b=deepcopy(model)
    raw=((a(torch.from_numpy(x))-torch.from_numpy(y))*torch.from_numpy(masks)).square().mean()
    grouped=(((b(torch.from_numpy(xx))-torch.from_numpy(yy))*torch.from_numpy(mm)).square().mean(-1)*torch.tensor(prob)).sum()
    raw.backward();grouped.backward()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad)
    assert raw.item()>grouped.item()  # Difference is prediction-independent within-state variation.


def test_regret_gate_is_conditional_on_noise_and_repetition():
    def d(mse,noise):return dict(after=dict(preflop=dict(repeated=dict(
        groups=40,rows=120,mean_error_fraction=.2,mean_target_mse=mse,iid_mean_noise_scale=noise))))
    assert not regret.conditional_gate([d(.1,.1),d(.1,.1)])['run_fits']
    assert regret.conditional_gate([d(.2,.1),d(.1,.1)])['run_fits']


def test_regret_fit_selection_and_river_panels(diagnostic_inputs,tmp_path):
    captures,state,policy,encounters=diagnostic_inputs
    # Make a clear, deterministic mean-prediction error to exercise conditional fitting.
    capture=json.loads((captures/'report.json').read_text())
    for record in capture['captures']:
        path=captures/record['artifact']
        with np.load(path) as f:data={k:f[k].copy() for k in f.files}
        data['target']+=1.
        np.savez_compressed(path,**data)
        record['sha256']=sha256_file(path)
        record['before_mse']=record['after_mse']=float(((data['before'].astype(float)-data['target'])**2).mean())
    (captures/'report.json').write_text(json.dumps(capture))
    source_hash=sha256_file(policy)
    result=regret.run(captures,state,policy,tmp_path/'diagnostic',threads=1,smoke=True,max_seconds=120)
    assert result['status']=='COMPLETE' and result['fit_gate']['run_fits']
    assert {r['arm'] for r in result['results']} >= {'row_current','group_mean_current'}
    assert len(result['exports'])==2 and result['selection']['seeds']==list(REGRET_SEEDS)
    assert all('audit' in r for r in result['results'])
    assert sha256_file(policy)==source_hash
    library=json.loads((encounters/'library.json').read_text())
    river=rivers.run(tmp_path/'diagnostic',policy,policy,encounters,tmp_path/'rivers',mode='regret',
        state=state,workers=1,max_seconds=120,smoke=True,expected_frozen_sha256=source_hash,
        expected_library_sha256=digest(library))
    assert river['status']=='COMPLETE' and river['completed_positions']==1 and not river['assessment_unlocked']
    assert len(river['comparisons'])==2
    for r in river['positions']:
        for candidate in r['policies'].values():assert len(candidate['panels'])==10
    from hulhe_escher.p104_cloud import summary
    summary(tmp_path,'regret')
    assert json.loads((tmp_path/'summary.json').read_text())['diagnostic']['status']=='COMPLETE'


def test_policy_river_uses_fixed_opponent_and_locked_positions(diagnostic_inputs,tmp_path):
    _,state,policy,encounters=diagnostic_inputs
    refit.run(state,policy,tmp_path/'fit',threads=1,smoke=True,max_seconds=90)
    library=json.loads((encounters/'library.json').read_text())
    r=rivers.run(tmp_path/'fit',policy,policy,encounters,tmp_path/'rivers',mode='policy',workers=2,
        max_seconds=120,smoke=True,expected_frozen_sha256=sha256_file(policy),expected_library_sha256=digest(library))
    assert r['status']=='COMPLETE' and r['completed_positions']==1
    assert len(r['positions'][0]['policies'])==5 and len(r['comparisons'])==6
    for c in r['comparisons'].values():
        assert len([s for s in c['summaries'] if s['scope']=='all'])==10
        assert set(c['action_probability_changes']['all'])=={'fold','call_check','raise_bet'}
    for name in ('head','fo2'):
        folder=tmp_path/name;folder.mkdir()
        (folder/'report.json').write_text(json.dumps(dict(status='COMPLETE',result=dict(
            bb_per_100=0.,confidence=.95,interval=[-1.,1.],conclusion='inconclusive'))))
    from hulhe_escher.p104_cloud import summary
    summary(tmp_path,'policy')
    saved=json.loads((tmp_path/'summary.json').read_text())
    assert 'source_metrics' in saved['policy_refit'] and len(saved['river_comparisons'])==6


@pytest.mark.parametrize('mode,hours,iterations',[('policy',4,0),('regret',2,1)])
def test_batch_contract_and_no_submission_by_default(tmp_path,monkeypatch,mode,hours,iterations):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    import launch_p104_diagnostic as launch
    calls=[];monkeypatch.setattr(launch.subprocess,'run',lambda cmd,**kw:calls.append(cmd))
    launch.main(['--mode',mode,'--run-name','hulhe-p104-'+mode+'-test','--output',str(tmp_path/'bundle')])
    assert not calls
    plan=json.loads((tmp_path/'bundle/launch.json').read_text())
    job=json.loads((tmp_path/'bundle/job.json').read_text())
    assert plan['source_iteration']==3489 and plan['collection_iterations']==iterations
    assert plan['max_task_hours']==hours and plan['disk_gb']==200
    spec=job['taskGroups'][0]['taskSpec']
    assert spec['maxRetryCount']==0 and spec['maxRunDuration']==str(hours*3600)+'s'
    # Use the real executable without the patched subprocess.run.
    proc=subprocess.Popen(['bash','-n',str(tmp_path/'bundle/batch.sh')]);assert proc.wait()==0
    launch.main(['--mode',mode,'--run-name','hulhe-p104-'+mode+'-test2','--output',str(tmp_path/'submit'),'--execute'])
    assert len(calls)==4 and calls[-1][:4]==['gcloud','batch','jobs','submit']


@pytest.mark.parametrize('mode',['policy','regret'])
def test_cloud_failure_keeps_results_and_no_source_upload(tmp_path,monkeypatch,mode):
    from hulhe_escher import p104_cloud as cloud
    transfers=[];commands=[]
    monkeypatch.setattr(cloud,'storage',lambda *args:transfers.append(args))
    def inputs(root,mode):
        return root/'state',root/'P104.pkl',root/'archived-trainer',root/'contract.json',root/'P56.pkl',root/'encounters',{}
    monkeypatch.setattr(cloud,'prepare_inputs',inputs)
    monkeypatch.setattr(cloud,'prepare_fo2',lambda _:None)
    def child(cmd,*args,**kwargs):
        commands.append(cmd);return dict(returncode=1,timed_out=False)
    monkeypatch.setattr(cloud,'run_child',child)
    with pytest.raises(RuntimeError):cloud.run_cloud(tmp_path/'cloud','gs://test/diagnostic',mode=mode)
    status=json.loads((tmp_path/'cloud/results/cloud_status.json').read_text())
    assert status['status']=='FAILED'
    assert transfers[-1][0]=='cp' and str(transfers[-1][1]).endswith('cloud_status.json')
    assert all('/source' not in str(t) for t in transfers)
    if mode=='regret':assert '--capture-all-regrets' in commands[0] and '--save-regret-models' in commands[0]


def test_cloud_and_launcher_reject_source_overlap(tmp_path):
    from hulhe_escher import p104_cloud as cloud
    from hulhe_escher.p104_protocol import SOURCE_URI
    with pytest.raises(ValueError):cloud.run_cloud(tmp_path/'nope',SOURCE_URI+'/child',mode='policy')
    assert not (tmp_path/'nope').exists()


def test_production_minimum_is_not_reduced_when_time_is_insufficient(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    root=tmp_path/'fit';root.mkdir();(root/'experiment.json').write_text('{}')
    monkeypatch.setattr(evaluate,'verified_exports',lambda *a:({},dict(smoke=False)))
    monkeypatch.setattr(evaluate.mp,'get_context',lambda _:SimpleNamespace(Pool=lambda *a,**k:nullcontext(None)))
    clock=[0.]
    monkeypatch.setattr(evaluate.time,'monotonic',lambda:clock[0])
    def collect(pool,**kw):
        assert 'rows' not in kw  # Confirmation must never begin below the registered minimum.
        clock[0]+=100
        return [{f'head_{s}':dict(mean_bb=(-1.)**i) for s in POLICY_SEEDS} for i in range(kw['count'])]
    monkeypatch.setattr(evaluate,'collect',collect)
    report=evaluate.evaluate(root,tmp_path/'unused',tmp_path/'evaluation',kind='head',max_seconds=10)
    assert report['status']=='INSUFFICIENT_TIME_FOR_MINIMUM'
    assert report['plan']['selected']>=400000 and report['completed_groups']==0
    assert not report['plan']['minimum_fits_budget']


def test_completed_fo2_secondary_is_descriptive_when_primary_did_not_pass(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    root=tmp_path/'fit';root.mkdir();(root/'experiment.json').write_text('{}')
    primary=tmp_path/'primary.json'
    primary.write_text(json.dumps(dict(kind='head',fit_sha256=sha256_file(root/'experiment.json'),
        status='COMPLETE',result=dict(confirmatory=True,conclusion='inconclusive'))))
    monkeypatch.setattr(evaluate,'verified_exports',lambda *a:({},dict(smoke=False)))
    monkeypatch.setattr(evaluate,'verify_build',lambda _:'fixed-build')
    monkeypatch.setattr(evaluate.mp,'get_context',lambda _:SimpleNamespace(Pool=lambda *a,**k:nullcontext(None)))
    clock=[0.];monkeypatch.setattr(evaluate.time,'monotonic',lambda:clock[0])
    def collect(pool,**kw):
        clock[0]+=1
        examples=[{**{f'control_{s}':dict(mean_bb=0.) for s in POLICY_SEEDS},
            **{f'all_replay_{s}':dict(mean_bb=x) for s in POLICY_SEEDS},'P104':dict(mean_bb=0.)}
            for x in (-.01,.02)]
        data=(examples*(kw['count']//2+1))[:kw['count']]
        if 'rows' in kw:kw['rows'].extend(data)
        return data
    monkeypatch.setattr(evaluate,'collect',collect)
    report=evaluate.evaluate(root,tmp_path/'unused',tmp_path/'evaluation',kind='fo2',
        max_seconds=100,primary_report=primary)
    assert report['status']=='COMPLETE' and report['completed_groups']==100000
    assert report['result']['conclusion']=='descriptive_primary_gate_not_passed'
    assert not report['result']['confirmatory']


@pytest.mark.parametrize('mode,names',[('policy',['fit','head','fo2','rivers']),('regret',['capture','diagnostic','rivers'])])
def test_cloud_stage_contract_and_final_publication(tmp_path,monkeypatch,mode,names):
    from hulhe_escher import p104_cloud as cloud
    transfers=[];stages=[]
    monkeypatch.setattr(cloud,'storage',lambda *args:transfers.append(args))
    monkeypatch.setattr(cloud,'prepare_inputs',lambda root,mode:
        (root/'state',root/'P104.pkl',root/'archived-trainer',root/'contract.json',root/'P56.pkl',root/'encounters',{}))
    monkeypatch.setattr(cloud,'prepare_fo2',lambda _:None)
    monkeypatch.setattr(cloud,'summary',lambda root,mode:(root/'summary.json').write_text('{"status":"COMPLETE"}'))
    def child(cmd,log,**kw):
        output=Path(cmd[cmd.index('--output')+1]);output.mkdir()
        stages.append((output.name,cmd,kw))
        (output/('experiment.json' if output.name=='fit' else 'report.json')).write_text('{"status":"COMPLETE"}')
        return dict(returncode=0,timed_out=False)
    monkeypatch.setattr(cloud,'run_child',child)
    r=cloud.run_cloud(tmp_path/'cloud','gs://test/run',mode=mode)
    assert r['status']=='COMPLETE' and [s[0] for s in stages]==names
    assert transfers[-1][0]=='cp' and str(transfers[-1][1]).endswith('cloud_status.json')
    for name,cmd,kw in stages:
        assert kw['timeout']<=5400
        if name=='capture':assert kw['cwd'].name=='archived-trainer' and '--skip-control-probe' in cmd
        else:assert '--max-seconds' in cmd
    assert (tmp_path/'cloud/results/summary.json').exists()
