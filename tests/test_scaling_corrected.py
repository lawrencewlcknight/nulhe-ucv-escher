"""Pinned mixed architectures, correlated deals, multiplicity and Batch guards."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pytest
import torch

from test_policy_sampling_experiment import frozen_source
from hulhe_escher import scaling_corrected_comparison as comparison
from hulhe_escher import scaling_corrected_cloud as cloud
from hulhe_escher.checkpointing import load_checkpoint_payload, sha256_file
from hulhe_escher.features import make_feature_encoder, StructuredHULHEMLP
from hulhe_escher.deep_model import DeepResidualHULHEMLP
from hulhe_escher.scaling_config import scaling_config,deep_scaling_config


@pytest.fixture
def pinned(frozen_source,tmp_path,monkeypatch):
    torch.set_num_threads(1)
    _,policy = frozen_source
    base = load_checkpoint_payload(policy)
    root = tmp_path/'inputs'
    root.mkdir()
    files = {}
    fit = dict(status='COMPLETE',smoke=False,source_iteration=1132,results=[])
    for label in (*comparison.CANDIDATES,*(f'P32-corrected-{s}' for s in comparison.SEEDS)):
        payload = deepcopy(base)
        payload['outer_iteration'] = 1132
        if label in comparison.CANDIDATES:
            deep = label.startswith('Deep')
            config = deep_scaling_config(small=True) if deep else scaling_config(small=True)
            encoder = make_feature_encoder(config['feature_encoder_id'])
            cls = DeepResidualHULHEMLP if deep else StructuredHULHEMLP
            model = cls(encoder.policy_layout,config['average_policy_network_layers'],3,
                        branch_width=config['structured_branch_width'])
            payload.update(training_config=config,outer_iteration=868 if deep else 953,
                policy_state_dict=model.state_dict(),policy_model=model.checkpoint_metadata(),
                policy_network_layers=config['average_policy_network_layers'])
        with (root/(label+'.pkl')).open('wb') as f:
            pickle.dump(payload,f)
        sha = sha256_file(root/(label+'.pkl'))
        if label in comparison.CANDIDATES:
            plan = dict(config=payload['training_config'],seed=payload['seed'])
            record = dict(outer_iteration=payload['outer_iteration'],smoke=False,active_seconds=115200,
                sha256=sha,metrics=dict(policy_validation=dict(group_sampling='objective_mass')))
            training = dict(config=plan['config'],seed=plan['seed'],status='TRAINING_COMPLETE',smoke=False,
                fresh_initialization=True,final_recovery_verified=True,checkpoints=dict(P32=record))
            for name,value in (('launch',plan),('scaling',training)):
                (root/(label+'-'+name+'.json')).write_text(json.dumps(value))
        else:
            fit['results'].append(dict(arm='objective_mass',fit_seed=int(label.rsplit('-',1)[-1]),
                fit=dict(group_sampling='objective_mass'),sha256=sha))
    (root/'P32-fit.json').write_text(json.dumps(fit))
    for path in root.iterdir():
        files[path.name] = dict(uri='gs://test/source/'+path.name,sha256=sha256_file(path))
    sources = tmp_path/'sources.json'
    sources.write_text(json.dumps(dict(files=files,fit_seeds=list(comparison.SEEDS))))
    monkeypatch.setattr(comparison,'SOURCES',sources)
    monkeypatch.setattr(cloud,'SOURCES',sources)
    return root


def test_verified_mixed_architectures_and_checksum_guard(pinned,monkeypatch):
    specs,_ = comparison.verify_inputs(pinned)
    assert len(specs)==4
    models = [load_checkpoint_payload(pinned/(name+'.pkl'))['policy_model'] for name in specs]
    assert models[0]!=models[1] and models[0]!=models[2]
    (pinned/'Deep-P32.pkl').write_bytes(b'wrong')
    monkeypatch.setattr(comparison,'load_checkpoint_payload',lambda p:pytest.fail('hash check first'))
    with pytest.raises(ValueError,match='checksum'):
        comparison.verify_inputs(pinned)


def test_game_contract_and_corrected_metadata_rejected(pinned,monkeypatch):
    original = comparison.load_checkpoint_payload
    def wrong(path):
        p = original(path)
        if path.name.startswith('Deep'):
            p['feature_encoder']={'wrong':True}
        return p
    monkeypatch.setattr(comparison,'load_checkpoint_payload',wrong)
    with pytest.raises(ValueError,match='feature_encoder'):
        comparison.verify_inputs(pinned)


def rows(values):
    return [{f'{name}/{seed}':dict(mean_bb=v+j+2*i) for i,name in enumerate(comparison.CANDIDATES)
             for j,seed in enumerate(comparison.SEEDS)} for v in values]


def test_correlated_refits_averaged_and_two_test_power_plan():
    data=rows([1.,2.,3.,4.])
    np.testing.assert_array_equal(comparison.observations(data,'Wide-P32'),[1.5,2.5,3.5,4.5])
    plan=comparison.common_plan(data,100.,10000.,minimum=100000,maximum=1000000)
    assert plan['selected']==max(p['selected'] for p in plan['by_candidate'].values())
    assert plan['alpha_per_test']==.025 and plan['familywise_alpha']==.05
    assert all(p['alpha']==.025 and p['pilot_excluded'] for p in plan['by_candidate'].values())
    assert not comparison.common_plan(data,1.,1.,minimum=100000,maximum=1000000)['minimum_fits_budget']
    matchups=comparison.matchups()
    assert len(matchups)==4 and len({x[3] for x in matchups})==4


def test_actual_spawned_four_policy_smoke(pinned,tmp_path):
    before={p:sha256_file(p) for p in pinned.iterdir()}
    result=comparison.run_evaluation(pinned,tmp_path/'evaluation',workers=2,max_seconds=90,smoke=True)
    assert result['status']=='COMPLETE' and result['completed_groups']==8 and result['completed_hands']==64
    for r in result['results'].values():
        assert r['observations']==[8] and not r['confirmatory'] and r['confidence']==.975
    assert all(not r['confirmatory'] for seed in result['by_fit_seed'].values() for r in seed.values())
    assert all(sha256_file(p)==sha for p,sha in before.items())


class FakePool:
    def __init__(self,*a,**kw): pass
    def __enter__(self): return self
    def __exit__(self,*a): pass


@pytest.mark.parametrize('mode',['complete','partial','error','pilot','budget'])
def test_pilot_exclusion_and_no_partial_significance(tmp_path,monkeypatch,mode):
    monkeypatch.setattr(comparison,'verify_inputs',lambda p:({},{}))
    monkeypatch.setattr(comparison.mp,'get_context',lambda _:type('Context',(),{'Pool':FakePool}))
    monkeypatch.setattr(comparison,'common_plan',lambda *a,**kw:dict(selected=4,
        by_candidate={},minimum_fits_budget=mode!='budget'))
    seeds=[]
    def collect(pool,*,seed,count,rows=None,**kw):
        seeds.append(seed)
        if rows is None:
            return globals()['rows']([999.]*(1 if mode=='pilot' else count))
        rows.extend(globals()['rows']([1.,2.,3.,4.][:2 if mode in ('error','partial') else 4]))
        if mode=='error': raise RuntimeError('failure')
        return rows
    monkeypatch.setattr(comparison,'collect',collect)
    report=comparison.run_evaluation(tmp_path,tmp_path/'out')
    if mode in ('pilot','budget'):
        assert not report['results'] and len(seeds)==1
    else:
        assert seeds==[comparison.DEAL_SEED,comparison.DEAL_SEED+1]
        r=report['results']['Wide-P32']
        assert r['bb_per_100']==(200 if mode in ('partial','error') else 300)
        assert r['confirmatory']==(mode=='complete') and r['confidence']==.975


@pytest.mark.parametrize('failure',[False,True])
def test_cloud_inputs_status_last_and_failure(pinned,tmp_path,monkeypatch,failure):
    calls=[]
    def storage(*args):
        calls.append(args)
        if args[0]=='cp' and str(args[1]).startswith('gs://test/source/'):
            shutil.copyfile(pinned/str(args[1]).rsplit('/',1)[1],args[2])
    def child(command,log,**kw):
        assert kw['timeout']==11100
        output=Path(command[command.index('--output')+1]);output.mkdir()
        (output/'report.json').write_text(json.dumps(dict(status='COMPLETE' if not failure else 'INCOMPLETE_TIME_LIMIT')))
        return dict(returncode=int(failure),timed_out=False)
    monkeypatch.setattr(cloud,'storage',storage)
    monkeypatch.setattr(cloud,'run_child',child)
    if failure:
        with pytest.raises(RuntimeError,match='did not complete'):
            cloud.run_cloud(tmp_path/'cloud','gs://test/results')
    else:
        assert cloud.run_cloud(tmp_path/'cloud','gs://test/results')['status']=='COMPLETE'
    assert len([c for c in calls if c[0]=='cp' and str(c[1]).startswith('gs://test/source/')])==9
    assert calls[-1][-1]=='gs://test/results/cloud_status.json'
    with pytest.raises(ValueError,match='overlaps'):
        cloud.validate_output('gs://test/source')


def test_launcher_pins_and_archived_modules(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
    import launch_scaling_corrected as launcher
    plan=launcher.build_bundle(tmp_path/'launch',run_name='hulhe-scaled-vs-p32c-test')
    assert plan['alpha_per_test']==.025 and plan['fit_seeds']==[1701,1702]
    assert plan['max_task_hours']==4 and plan['workers']==8 and not plan['retraining']
    assert len(plan['sources']['files'])==9
    job=json.loads((tmp_path/'launch/job.json').read_text())
    assert job['taskGroups'][0]['taskSpec']['maxRunDuration']=='14400s'
    assert job['taskGroups'][0]['taskSpec']['maxRetryCount']==0
    subprocess.run(['bash','-n',str(tmp_path/'launch/batch.sh')],check=True)
    with tarfile.open(tmp_path/'launch/source.tar.gz') as archive:
        names=archive.getnames()
    assert 'hulhe_escher/data/scaling_corrected_sources.json' in names
    assert 'hulhe_escher/scaling_corrected_comparison.py' in names
    assert not any('.env' in n or n.startswith('outputs/') for n in names)
    with pytest.raises(ValueError,match='overlaps'):
        launcher.build_bundle(tmp_path/'bad',run_name='hulhe-deep32-batch-20261122')
