"""Pinned Wide P104 recovery, eight continuation stages and fresh evaluations."""
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import pytest

from hulhe_escher import continuation_cloud as cloud, scaled_continuation as scaled
from hulhe_escher import continuation_evaluate as evaluation
from hulhe_escher.checkpointing import sha256_file
from hulhe_escher.scaling_config import scaling_config

REPO = Path(__file__).resolve().parents[1]
LABELS = tuple(f'P{104+6*i}' for i in range(9))


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    source, trainer = tmp_path/'source', tmp_path/'trainer'
    state = source/'training/state_003028'; state.mkdir(parents=True); trainer.mkdir()
    config = scaling_config()
    contract = dict(config=config, source_fingerprint=scaled.fingerprint(trainer),
        smoke=False, workers=8, threads=8, seed=20261115, policy_every=0, checkpoint_every=0,
        recovery_seconds=21600., policy_seconds=1e9)
    payload = dict(training_config=config, seed=20261115, outer_iteration=3028,
        training_elapsed_seconds=374601.65730819444)
    (source/'P104.pkl').write_bytes(pickle.dumps(payload))
    (source/'training_source.tar.gz').write_bytes(b'original trainer')
    (state/'manifest.json').write_text('{}')
    marker = dict(label='P104', outer_iteration=3028, recovery=state.name,
        active_seconds=payload['training_elapsed_seconds'], sha256=sha256_file(source/'P104.pkl'),
        smoke=False, metrics=dict(policy_validation=dict(group_sampling='objective_mass',
            policy_sampling_seed=[20261115,3028])))
    for path, data in [('training/manifest.json',contract), ('P104.json',marker),
                       ('training/latest.json',dict(iteration=3028,state=state.name))]:
        (source/path).write_text(json.dumps(data))
    spec = dict(source_uri='gs://test/original', policy_sha256=marker['sha256'], iteration=3028,
        source_archive_sha256=sha256_file(source/'training_source.tar.gz'),
        active_seconds=marker['active_seconds'], source_fingerprint=contract['source_fingerprint'],
        state_manifest_sha256=sha256_file(state/'manifest.json'), evaluation_seed_offset=7000000)
    monkeypatch.setattr(scaled,'source_spec',lambda variant, baseline_hours=32:spec)
    monkeypatch.setattr(scaled,'load_checkpoint_payload',lambda p:pickle.loads(p.read_bytes()))
    return source, trainer, spec


def test_p104_batch_plan_and_protected_sources(tmp_path):
    sys.path.insert(0,str(REPO/'tools'))
    import launch_wide_p104_continuation_batch as launcher
    output = tmp_path/'plan'
    plan = launcher.build_bundle(output,run_name='hulhe-wide-p104-plus48-test')
    assert plan['baseline'] == scaled.source_spec('wide',104)
    assert plan['baseline']['iteration'] == 3028 and plan['evaluation_seed_offset'] == 7000000
    assert plan['labels'] == list(LABELS) and plan['stage_count'] == 8
    assert plan['additional_active_hours'] == 48 and plan['stage_hours'] == 6
    assert plan['primary'] == 'Wide-P152 versus its own P104; alpha .05'
    assert plan['verify_each_stage'] and not plan['source_modified'] and not plan['config_changed']
    assert plan['policy_group_sampling'] == 'objective_mass' and plan['policy_replay_capacity'] == 12000000
    job = json.loads((output/'job.json').read_text())
    task = job['taskGroups'][0]['taskSpec']
    assert task['maxRunDuration'] == '266400s' and task['maxRetryCount'] == 0
    policy = job['allocationPolicy']['instances'][0]['policy']
    assert policy['machineType'] == 'c4-standard-16' and policy['bootDisk']['sizeGb'] == 200
    script = (output/'batch.sh').read_text()
    for flag in ('--baseline-hours 104','--stage-count 8','--training-wall-seconds 216000',
                 '--require-corrected-sampler','--scaled-variant wide'):
        assert flag in script
    subprocess.run(['bash','-n',str(output/'batch.sh')],check=True)
    for variant,hours in [('wide',32),('deep',32),('wide',56),('deep',56),('wide',104)]:
        name = scaled.source_spec(variant,hours)['source_uri'].split('/')[-1]
        with pytest.raises(ValueError,match='independent'):
            launcher.build_bundle(tmp_path/'bad',run_name=name)


def test_p104_source_and_exact_active_time(pinned):
    source,trainer,spec = pinned
    assert scaled.verify_scaled_source(source,trainer,'wide',104)['status'] == 'VERIFIED'
    path=source/'P104.json';marker=json.loads(path.read_text())
    marker['active_seconds'] += 1;path.write_text(json.dumps(marker))
    with pytest.raises(ValueError,match='P104 recovery'):
        scaled.verify_scaled_source(source,trainer,'wide',104)


def test_cloud_routes_p104_to_original_trainer(pinned,tmp_path,monkeypatch):
    source,trainer,spec = pinned
    def download(*args,**kwargs):
        assert args[-1] == 104 and kwargs['expected_iteration'] == 3028
        assert kwargs['expected_archive_sha256'] == spec['source_archive_sha256']
        assert kwargs['expected_state_manifest_sha256'] == spec['state_manifest_sha256']
        return source,trainer
    monkeypatch.setattr(cloud,'download_source',download)
    monkeypatch.setattr(cloud,'storage',lambda *a:None)
    monkeypatch.setattr(cloud,'run_transfer',lambda *a,**kw:None)
    monkeypatch.setattr(cloud,'prepare',lambda p:dict(fingerprint='fo2'))
    def train(src,archived,output,**kwargs):
        assert src == source and archived == trainer
        assert kwargs['baseline_hours'] == 104 and kwargs['stage_count'] == 8
        assert kwargs['wall_seconds'] == 216000 and kwargs['verify_each_stage'] and kwargs['require_objective_mass']
        output.mkdir(parents=True)
        return dict(training_source_fingerprint=spec['source_fingerprint'])
    def evaluate(*args,**kwargs):
        assert kwargs['extra_seed_offset'] == 7000000
        return dict(status='COMPLETE')
    monkeypatch.setattr(cloud,'run_continuation',train)
    monkeypatch.setattr(cloud,'run_suite',evaluate)
    root=tmp_path/'cloud';root.mkdir()
    cloud.run_cloud(root,spec['source_uri'],'gs://test/new',spec['policy_sha256'],baseline_hours=104,
        stage_count=8,training_wall_seconds=216000,require_corrected_sampler=True,scaled_variant='wide')
    provenance=json.loads((root/'run/provenance.json').read_text())
    assert provenance['additional_active_hours'] == 48 and provenance['baseline'] == 'P104'


@pytest.mark.parametrize('change',['stage_count','sampler','digest','old_source','deep'])
def test_bad_cloud_request_fails_before_download(tmp_path,monkeypatch,change):
    spec=scaled.source_spec('wide',104)
    kwargs=dict(baseline_hours=104,stage_count=8,scaled_variant='wide',require_corrected_sampler=True)
    output,digest='gs://test/new',spec['policy_sha256']
    if change=='stage_count':kwargs['stage_count']=4
    elif change=='sampler':kwargs['require_corrected_sampler']=False
    elif change=='digest':digest='bad'
    elif change=='deep':kwargs['scaled_variant']='deep'
    else:output=scaled.source_spec('wide',56)['source_uri']
    monkeypatch.setattr(cloud,'download_source',lambda *a,**kw:pytest.fail('Do not download'))
    with pytest.raises(ValueError):
        cloud.run_cloud(tmp_path,spec['source_uri'],output,digest,**kwargs)


def test_manifest_pin_before_replay(pinned,tmp_path,monkeypatch):
    source,trainer,spec=pinned
    def transfer(*args):
        assert args[0]=='cp','Reject before replay rsync'
        shutil.copyfile(source/args[1].removeprefix(spec['source_uri']+'/'),args[-1])
    monkeypatch.setattr(cloud,'storage',transfer)
    with pytest.raises(ValueError,match='manifest mismatch'):
        cloud.download_source(spec['source_uri'],tmp_path/'download',spec['policy_sha256'],104,
            expected_archive_sha256=spec['source_archive_sha256'],expected_iteration=3028,
            expected_state_manifest_sha256='wrong')


def test_p152_evaluation_and_fresh_seeds(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir()
    (root/'continuation.json').write_text(json.dumps(dict(status='TRAINING_COMPLETE',
        final_recovery_verified=True,smoke=False,baseline_hours=104,stage_count=8,labels=list(LABELS),
        checkpoints={label:dict(sha256=label) for label in LABELS})))
    monkeypatch.setattr(evaluation,'wait_policy',lambda r,l,d:(r/f'{l}.pkl',dict(sha256=l)))
    matches=[];opponents=[]
    def match(a,b,output,**kwargs):
        matches.append((Path(a).stem,Path(b).stem,kwargs))
        return dict(status='COMPLETE',result=None)
    def fo2(*args,**kwargs):
        opponents.append(kwargs);return dict(status='COMPLETE')
    monkeypatch.setattr(evaluation,'run_evaluation',match)
    monkeypatch.setattr(evaluation,'run_fo2',fo2)
    report=evaluation.run_suite(root,tmp_path/'eval',extra_seed_offset=7000000)
    assert report['status']=='COMPLETE' and report['primary']=='P152-vs-P104'
    assert report['secondary_family_size']==16 and not report['primary_passed']
    assert [(a,b) for a,b,_ in matches]==[('P152','P104'),*zip(LABELS[1:],LABELS[:-1])]
    assert [kw['alpha'] for _,_,kw in matches]==[.05]+[.05/16]*8
    assert [kw['seed'] for _,_,kw in matches]==[7124301+10*i for i in range(9)]
    assert opponents[0]['labels']==LABELS and opponents[0]['seed']==7125501
