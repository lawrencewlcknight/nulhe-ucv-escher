"""Analytical identities, leakage protection and bounded real fitting/river checks."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from test_policy_sampling_experiment import frozen_source
from test_bottleneck import add_regrets
from hulhe_escher import regret_diagnostic as diag
from hulhe_escher.checkpointing import LoadedHULHEPolicy, sha256_file
from hulhe_escher.diagnostic_probes import regret_models
from hulhe_escher.game import load_hulhe_game
from hulhe_escher.policy_replay_artifacts import read_metadata
from hulhe_escher.strategic_audit.library import build_library, digest


def test_variance_identity_known_components_and_singletons():
    x=np.array([[0],[0],[1]],np.float32)
    y=np.array([[0,100,0],[2,-100,2],[8,999,8]],np.float32)
    pred=np.array([[2,0,2],[2,0,2],[5,0,5]],np.float32)
    mask=np.tile([1,0,1],(3,1))
    result=diag.decomposition(x,y,pred,mask)
    assert result['all']['total_mse']==pytest.approx(13/3)
    assert result['all']['within_state_variance']==pytest.approx(2/3)
    assert result['all']['mean_target_mse']==pytest.approx(11/3)
    assert result['repeated']['within_state_variance']==1
    assert result['repeated']['mean_target_mse']==1
    assert result['repeated']['iid_mean_noise_scale']==1
    assert result['singletons']['rows']==1
    assert abs(result['identity_residual'])<1e-12


def test_zero_sample_mean_error_does_not_imply_zero_target_variance():
    x=np.ones((4,2),np.float32);y=np.tile([-1.,1.,-1.,1.],(3,1)).T
    r=diag.decomposition(x,y,np.zeros_like(y),np.ones_like(y))
    assert r['repeated']['mean_target_mse']==0
    assert r['repeated']['within_state_variance']==1
    assert r['singletons']['mean_target_mse'] is None


def test_reject_inconsistent_duplicate_predictions_and_masks():
    x=np.ones((2,2),np.float32);y=np.zeros((2,3));mask=np.ones_like(y)
    with pytest.raises(ValueError,match='predictions differ'):
        diag.decomposition(x,y,np.array([[0,0,0],[1,1,1]]),mask)
    mask[1,0]=0
    with pytest.raises(ValueError,match='legal masks'):
        diag.decomposition(x,y,y,mask)


def test_split_keeps_duplicates_and_signed_zero_together():
    x=np.random.default_rng(4).normal(size=(1000,4)).astype(np.float32)
    x[1]=x[0];x[0,0]=0.;x[1,0]=-0.
    parts=diag.partition(x,np.ones((len(x),3)))
    assert sorted(np.concatenate(parts))==list(range(len(x)))
    assert all((0 in p)==(1 in p) for p in parts)
    assert all(np.array_equal(a,b) for a,b in zip(parts,diag.partition(x,np.ones((len(x),3)))))


def test_fit_gate_requires_repetition_and_material_error():
    def row(n,share):return {'after':{'preflop':{'repeated':{'groups':n,'rows':3*n,'mean_error_fraction':share}}}}
    assert diag.fit_gate([row(40,.2),row(40,.01)])['run_fits']
    assert not diag.fit_gate([row(10,.9),row(40,.2)])['run_fits']
    assert not diag.fit_gate([row(40,.01),row(40,.01)])['run_fits']


@pytest.fixture
def diagnostic_inputs(frozen_source,tmp_path):
    state,policy=frozen_source;add_regrets(state)
    _,metadata=read_metadata(state)
    loaded=LoadedHULHEPolicy(load_hulhe_game(),policy);enc=loaded.feature_encoder
    models=regret_models(metadata,enc)
    root=tmp_path/'captures';root.mkdir()
    report=dict(status='COMPLETE',source_iteration=metadata['solver']['num_iteration'],
        iteration=metadata['solver']['num_iteration']+1,source_policy_sha256=sha256_file(policy),
        source_manifest_sha256=sha256_file(state/'manifest.json'),captures=[])
    for player,model in enumerate(models):
        rng=np.random.default_rng(40+player)
        x=np.repeat(rng.normal(size=(240,enc.policy_layout.total_size)).astype(np.float32),3,axis=0)
        pc,sc=enc.replay_context_columns();x[:,pc]=np.eye(2)[player];x[:,sc]=np.eye(4)[np.repeat(np.arange(240)%4,3)]
        with torch.no_grad():before=model(torch.from_numpy(x)).numpy()
        target=(before+rng.normal(0,.1,size=before.shape)).astype(np.float32);mask=np.ones_like(target)
        name=f'regret_{player}'
        np.savez_compressed(root/f'{name}_before.npz',features=x,mask=mask,target=target,before=before,row_indices=np.arange(len(x)))
        np.savez_compressed(root/f'{name}_after.npz',prediction=before)
        mse=float(((before.astype(float)-target)**2).mean())
        report['captures'].append(dict(name=name,artifact=f'{name}_before.npz',sha256=sha256_file(root/f'{name}_before.npz'),
            sampled_rows=len(x),before_mse=mse,after_mse=mse))
    (root/'report.json').write_text(json.dumps(report))
    encounters=tmp_path/'encounters';encounters.mkdir()
    lib=build_library()
    lib['positions']=[next(p for p in lib['positions'] if p['partition']==part and len(p['board'])==5)
                      for part in ('development','assessment')]
    for p in lib['positions']:p.update(source='self',encounter_weight=1.)
    (encounters/'library.json').write_text(json.dumps(lib))
    (encounters/'report.json').write_text(json.dumps(dict(status='COMPLETE',source_policy_sha256=sha256_file(policy),library_sha256=digest(lib))))
    return root,state,policy,encounters


def test_actual_fits_and_river_pipeline_preserve_inputs(diagnostic_inputs,tmp_path,monkeypatch):
    captures,state,policy,encounters=diagnostic_inputs
    original={p:sha256_file(p) for p in (policy,state/'manifest.json',captures/'regret_0_before.npz')}
    original_fit=diag.fit_candidate
    def bounded(*a,**kw):return original_fit(*a,**kw,updates=(1,2))
    monkeypatch.setattr(diag,'fit_candidate',bounded)
    residual=diag.ResidualPolicy
    monkeypatch.setattr(diag,'ResidualPolicy',lambda layout:residual(layout,width=16,blocks=1,head=8,branch_width=4))
    out=tmp_path/'results'
    r=diag.run(captures,policy,encounters,out,state=state,fits='force',threads=1,max_seconds=120)
    assert r['status']=='COMPLETE' and len(r['results'])==4
    assert r['selection']['seed']==3901
    assert len(r['river']['positions'])==1  # assessment position remains locked
    assert r['river']['positions'][0]['baseline']['status']=='COMPLETE'
    assert all('audit' in x for x in r['results'])
    assert all(sha256_file(p)==sha for p,sha in original.items())
    assert (out/'selected_regret.pt').exists() and not (out/'policy.pkl').exists()
    assert json.loads((out/'selection.json').read_text())==r['selection']


def test_capture_corruption_rejected(diagnostic_inputs,tmp_path):
    captures,state,policy,encounters=diagnostic_inputs
    with (captures/'regret_0_before.npz').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        diag.run(captures,policy,encounters,tmp_path/'out',fits='off')
    assert json.loads((tmp_path/'out/report.json').read_text())['status']=='FAILED'


def test_decomposition_only_requires_no_metadata(diagnostic_inputs,tmp_path):
    captures,_,policy,encounters=diagnostic_inputs
    r=diag.run(captures,policy,encounters,tmp_path/'out',fits='off')
    assert r['status']=='COMPLETE' and r['results']==[]


def test_batch_bundle_no_collection_and_no_replay_download(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    try:from launch_regret_diagnostic import build_bundle
    finally:sys.path.pop(0)
    plan=build_bundle(tmp_path/'bundle',run_name='hulhe-regret-test')
    job=json.loads((tmp_path/'bundle/job.json').read_text())
    assert plan['collection_workers']==0 and not plan['full_replay_download']
    spec=job['taskGroups'][0]['taskSpec']
    assert spec['maxRetryCount']==0 and spec['maxRunDuration']=='7200s'
    assert '--expected-capture-report-sha256' in (tmp_path/'bundle/batch.sh').read_text()
    subprocess.run(['bash','-n',str(tmp_path/'bundle/batch.sh')],check=True)


def test_river_identity_shortcut_and_changed_strategy_oracle(diagnostic_inputs):
    from copy import deepcopy
    from hulhe_escher.strategic_audit.workflow import RegretPolicy
    from hulhe_escher.position_investigation.common import deadline_after
    captures,state,policy,encounters=diagnostic_inputs
    library=json.loads((encounters/'library.json').read_text())
    a=RegretPolicy(state);b=deepcopy(a)
    loaded=LoadedHULHEPolicy(load_hulhe_game(),policy)
    same=diag.audit_rivers(library,a,b,loaded,deadline_after(30))
    assert same['by_encounter_source']['self']['value_change_bb']==0
    assert 'identity' in same['method']
    with torch.no_grad():
        # Change the linear output bias without changing the architecture.
        bias=next(p for n,p in b.models[0].named_parameters() if n.startswith('output') and p.ndim==1)
        bias[1]+=1
        bias=next(p for n,p in b.models[1].named_parameters() if n.startswith('output') and p.ndim==1)
        bias[1]+=1
    assert not diag.same_regret_models(a,b)
    changed=diag.audit_rivers(library,a,b,loaded,deadline_after(30))
    assert changed['positions'][0]['candidate']['status']=='COMPLETE'
    assert 'candidate_value_bb' in changed['positions'][0]['candidate']
    assert np.isfinite(changed['by_encounter_source']['self']['value_change_bb'])
