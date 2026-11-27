"""Sequential P8 identity pins, unchanged training and resumable six-hour stages."""
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import pytest

from hulhe_escher import continuation_cloud as cloud, sequential_continuation as sequential
from hulhe_escher.continuation import run_continuation
from hulhe_escher.checkpointing import sha256_file
from hulhe_escher.experiment import publish_policy
from hulhe_escher.sequential_training import sequential_config

REPO = Path(__file__).resolve().parents[1]


def launcher():
    sys.path.insert(0, str(REPO/'tools'))
    import launch_sequential_continuation_batch
    return launch_sequential_continuation_batch


def test_plan_pins_p8_and_preserves_recipe(tmp_path):
    plan = launcher().build_bundle(tmp_path/'plan', run_name='hulhe-sequential-p8-plus24-test')
    assert plan['baseline'] == sequential.source_spec()
    assert plan['baseline']['iteration'] == 170
    assert plan['labels'] == ['P8','P14','P20','P26','P32']
    assert plan['per_network_parameters'] == 1496067 and plan['protection_coefficient'] == .3
    assert plan['verify_each_stage'] and not plan['config_changed'] and not plan['source_modified']
    job = json.loads((tmp_path/'plan/job.json').read_text())
    task = job['taskGroups'][0]['taskSpec']
    assert task['maxRunDuration'] == '158400s' and task['maxRetryCount'] == 0
    assert task['computeResource'] == dict(cpuMilli=16000, memoryMib=60000)
    script = (tmp_path/'plan/batch.sh').read_text()
    assert '--sequential-source' in script and '--require-corrected-sampler' in script
    assert '--baseline-hours 8' in script and '--corrected-sampler ' not in script
    subprocess.run(['bash','-n',str(tmp_path/'plan/batch.sh')], check=True)
    with pytest.raises(ValueError, match='independent'):
        launcher().build_bundle(tmp_path/'bad', run_name=plan['baseline']['source_uri'].split('/')[-1])


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    source, trainer = tmp_path/'original', tmp_path/'trainer'
    state = source/'training/state_000170'
    state.mkdir(parents=True); trainer.mkdir()
    config = sequential_config()
    contract = dict(config=config, source_fingerprint=sequential.fingerprint(trainer),
        smoke=False, workers=8, threads=8, seed=20261115, policy_every=0, checkpoint_every=0,
        recovery_seconds=21600., policy_seconds=1e9)
    payload = dict(training_config=config, seed=20261115, outer_iteration=170,
        training_elapsed_seconds=29035., policy_model=dict(type='structured_hulhe_protected_sequential_v1'))
    with (source/'P8.pkl').open('wb') as f: pickle.dump(payload,f)
    (source/'source.tar.gz').write_bytes(b'fixture archive')
    (state/'manifest.json').write_text('{}')
    marker = dict(label='P8', outer_iteration=170, recovery=state.name, active_seconds=29035.,
        sha256=sha256_file(source/'P8.pkl'), smoke=False,
        metrics=dict(policy_validation=dict(group_sampling='objective_mass', policy_sampling_seed=[20261115,170],
            protected_training=dict(protocol='protected_selective_v1', coefficient=.3))))
    for path, value in [('training/manifest.json',contract), ('P8.json',marker),
                        ('training/latest.json',dict(iteration=170,state=state.name))]:
        (source/path).write_text(json.dumps(value))
    spec = dict(source_uri='gs://test/original', policy_sha256=marker['sha256'], iteration=170,
        source_archive_sha256=sha256_file(source/'source.tar.gz'), active_seconds=29035.,
        source_fingerprint=contract['source_fingerprint'],
        state_manifest_sha256=sha256_file(state/'manifest.json'), evaluation_seed_offset=3000000)
    monkeypatch.setattr(sequential, 'source_spec', lambda baseline_hours=8:spec)
    monkeypatch.setattr(sequential, 'load_checkpoint_payload', lambda p:pickle.loads(p.read_bytes()))
    return source, trainer, spec


def test_verified_source(pinned):
    source, trainer, spec = pinned
    receipt = sequential.verify_sequential_source(source, trainer)
    assert receipt['status'] == 'VERIFIED' and receipt['protection_coefficient'] == .3
    assert not receipt['source_modified'] and not receipt['config_changed']


@pytest.mark.parametrize('what', ['archive','config','recovery','source','smoke','protection','sampler','seed','hours'])
def test_source_drift_rejected(pinned, what):
    source, trainer, spec = pinned
    if what == 'archive': (source/'source.tar.gz').write_bytes(b'changed')
    elif what == 'source': spec['source_fingerprint'] = 'wrong'
    elif what == 'recovery': (source/'training/state_000170/manifest.json').write_text('{"changed":true}')
    elif what in ('config','smoke'):
        path = source/'training/manifest.json'; d = json.loads(path.read_text())
        if what == 'config': d['config']['average_policy_learning_rate'] = .003
        else: d['smoke'] = True
        path.write_text(json.dumps(d))
    else:
        path = source/'P8.json'; d=json.loads(path.read_text()); fit=d['metrics']['policy_validation']
        if what == 'protection': fit['protected_training']['coefficient'] = .1
        elif what == 'sampler': fit['group_sampling'] = 'uniform'
        elif what == 'seed': fit['policy_sampling_seed'] = [0,170]
        else: d['active_seconds'] += 1
        path.write_text(json.dumps(d))
    with pytest.raises(ValueError): sequential.verify_sequential_source(source, trainer)


def test_cloud_uses_archived_trainer_and_fresh_evaluation(pinned, tmp_path, monkeypatch):
    source, trainer, spec = pinned
    def download(*a, **kw):
        assert a[-1] == 8
        assert kw == dict(expected_archive_sha256=spec['source_archive_sha256'],
            expected_iteration=170, expected_state_manifest_sha256=spec['state_manifest_sha256'])
        return source, trainer
    monkeypatch.setattr(cloud,'download_source',download)
    calls=[]
    monkeypatch.setattr(cloud,'storage',lambda *a:calls.append(a))
    monkeypatch.setattr(cloud,'run_transfer',lambda *a,**k:None)
    monkeypatch.setattr(cloud,'prepare',lambda p:dict(fingerprint='fo2'))
    def train(src, archived, output, **kw):
        assert src == source and archived == trainer
        assert kw['verify_each_stage'] and kw['require_protected_sequential'] and kw['require_objective_mass']
        output.mkdir(parents=True)
        return dict(training_source_fingerprint=spec['source_fingerprint'])
    def evaluate(*a,**kw):
        assert kw['extra_seed_offset'] == 3000000
        return dict(status='COMPLETE')
    monkeypatch.setattr(cloud,'run_continuation',train)
    monkeypatch.setattr(cloud,'run_suite',evaluate)
    root=tmp_path/'cloud';root.mkdir()
    cloud.run_cloud(root,spec['source_uri'],'gs://test/continued',spec['policy_sha256'],
        require_corrected_sampler=True,sequential_source=True)
    assert any(a[-1]=='gs://test/continued/training_source.tar.gz' for a in calls)
    assert json.loads((root/'sampler-verification.json').read_text())['status']=='VERIFIED'


def test_wrong_source_rejected_before_download(tmp_path,monkeypatch):
    monkeypatch.setattr(cloud,'download_source',lambda *a,**kw:pytest.fail('Do not download'))
    spec=sequential.source_spec()
    with pytest.raises(ValueError,match='pinned corrected P8'):
        cloud.run_cloud(tmp_path,spec['source_uri'],'gs://test/new','bad',
            require_corrected_sampler=True,sequential_source=True)
    with pytest.raises(ValueError,match='sibling'):
        cloud.run_cloud(tmp_path,spec['source_uri'],spec['source_uri']+'/child',spec['policy_sha256'],
            require_corrected_sampler=True,sequential_source=True)


def test_manifest_pin_before_replay_download(pinned,tmp_path,monkeypatch):
    source,trainer,spec=pinned
    calls=[]
    def transfer(*a):
        calls.append(a)
        assert a[0]=='cp','Must reject before downloading full replay'
        shutil.copyfile(source/a[1].split('gs://test/original/')[1],a[-1])
    monkeypatch.setattr(cloud,'storage',transfer)
    with pytest.raises(ValueError,match='manifest mismatch'):
        cloud.download_source(spec['source_uri'],tmp_path/'cloud',spec['policy_sha256'],8,
            expected_archive_sha256=spec['source_archive_sha256'], expected_iteration=170,
            expected_state_manifest_sha256='wrong')


@pytest.mark.parametrize('baseline_hours', [8, 32])
def test_real_archived_sequential_continuation(tmp_path,monkeypatch,baseline_hours):
    labels = [f'P{baseline_hours+6*i}' for i in range(5)]
    archived=tmp_path/'archived'
    # Optional downloaded original archive exercises the precise production code.
    old=os.environ.get('HULHE_SEQUENTIAL_ARCHIVE')
    archive=Path(old) if old else tmp_path/'source.tar.gz'
    if not old: launcher().source_archive(REPO,archive)
    cloud.extract_source(archive,archived)
    baseline=tmp_path/'baseline';baseline.mkdir()
    config=sequential_config(small=True)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    env={**os.environ,'PYTHONPATH':str(archived),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    command=[sys.executable,'-m','hulhe_escher.train','--config',str(path),'--iterations','1',
        '--workers','1','--threads','1','--seed','123','--policy-every','0',
        '--policy-seconds','1000000000','--output',str(baseline/'training')]
    result=subprocess.run(command,cwd=archived,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr[-4000:]
    publish_policy(baseline/'training',baseline,labels[0])
    hashes={p:sha256_file(p) for p in baseline.rglob('*') if p.is_file()}
    output=tmp_path/'continued';uploaded=[]
    def upload(root,uri,label):
        assert json.loads((root/f'reload-{label}.json').read_text())['status']=='VERIFIED'
        uploaded.append(label)
    monkeypatch.setattr('hulhe_escher.continuation.upload_checkpoint',upload)
    report=run_continuation(baseline,archived,output,stage_seconds=3,wall_seconds=240,
        baseline_hours=baseline_hours,
        allow_smoke=True,require_objective_mass=True,verify_each_stage=True,
        require_protected_sequential=True,upload_uri='gs://test/offline')
    assert report['status']=='TRAINING_COMPLETE' and report['final_recovery_verified']
    assert uploaded==labels
    for i,label in enumerate(uploaded[1:],1):
        record=report['checkpoints'][label]
        assert record['target_active_seconds']==report['checkpoints'][labels[0]]['active_seconds']+3*i
        assert record['active_seconds']>=record['target_active_seconds']
        assert record['metrics']['policy_validation']['protected_training']['coefficient']==.3
        assert record['metrics']['policy_validation']['policy_sampling_seed']==[123,record['outer_iteration']]
    assert all(sha256_file(p)==value for p,value in hashes.items())
    assert sha256_file(output/f'{labels[-1]}.pkl')!=sha256_file(output/f'{labels[0]}.pkl')
    assert len(list((output/'training').glob('state_*')))==1


def test_evaluation_design_is_fresh_and_adjusted(tmp_path,monkeypatch):
    from hulhe_escher import continuation_evaluate as evaluation
    labels=['P8','P14','P20','P26','P32']
    root=tmp_path/'run';root.mkdir()
    (root/'continuation.json').write_text(json.dumps(dict(status='TRAINING_COMPLETE',
        final_recovery_verified=True,smoke=False,baseline_hours=8,
        checkpoints={label:dict(sha256=label) for label in labels})))
    monkeypatch.setattr(evaluation,'wait_policy',lambda r,l,d:(r/f'{l}.pkl',dict(sha256=l)))
    matches=[];opponents=[]
    def match(*a,**kw):
        matches.append(kw);return dict(status='COMPLETE',result=None)
    def fo2(*a,**kw):
        opponents.append(kw);return dict(status='COMPLETE')
    monkeypatch.setattr(evaluation,'run_evaluation',match)
    monkeypatch.setattr(evaluation,'run_fo2',fo2)
    result=evaluation.run_suite(root,tmp_path/'eval',extra_seed_offset=3000000)
    assert result['status']=='COMPLETE' and result['primary']=='P32-vs-P8'
    assert [r['seed'] for r in matches]==[3028301+10*i for i in range(5)]
    assert [r['alpha'] for r in matches]==[.05]+[.05/8]*4
    assert opponents[0]['seed']==3029501 and opponents[0]['labels']==tuple(labels)
    assert not result['primary_passed']
