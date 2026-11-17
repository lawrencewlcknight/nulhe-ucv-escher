"""Bounded real continuation and evaluation checks; no network or paid VMs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from hulhe_escher.continuation import LABELS, run_continuation, trainer_command
from hulhe_escher.continuation_cloud import extract_source
from hulhe_escher.continuation_evaluate import SECONDARY_ALPHA, fo2_plan, gate_secondary
from hulhe_escher.evaluation_stats import estimate
from hulhe_escher.experiment import publish_policy

REPO=Path(__file__).resolve().parents[1]


def launcher_module():
    sys.path.insert(0,str(REPO/'tools'))
    try:
        spec=importlib.util.spec_from_file_location('continuation_launcher',REPO/'tools/launch_continuation.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


@pytest.fixture(scope='module')
def continued(tmp_path_factory):
    directory=tmp_path_factory.mktemp('continuation')
    archived=directory/'archived'
    module=launcher_module()
    module.source_archive(REPO,directory/'source.tar.gz')
    extract_source(directory/'source.tar.gz',archived)
    baseline=directory/'baseline';baseline.mkdir()
    env={**os.environ,'PYTHONPATH':str(archived),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    result=subprocess.run([sys.executable,'-m','hulhe_escher.train','--smoke','--iterations','1',
        '--workers','1','--threads','1','--seed','123','--policy-every','0',
        '--policy-seconds','1000000000','--output',str(baseline/'training')],
        cwd=archived,env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr[-4000:]
    publish_policy(baseline/'training',baseline,'P8')
    output=directory/'continued'
    # Real subprocesses and exact original source checks on every stage.
    report=run_continuation(baseline,archived,output,stage_seconds=.001,
                            wall_seconds=180,allow_smoke=True)
    assert report['status']=='TRAINING_COMPLETE'
    return baseline,archived,output


def test_full_continuation_preserves_baseline_and_exports_four_policies(continued):
    baseline,archived,root=continued
    report=json.loads((root/'continuation.json').read_text())
    assert report['final_recovery_verified']
    assert list(report['checkpoints'])==list(LABELS)
    assert [report['checkpoints'][label]['outer_iteration'] for label in LABELS]==[1,2,3,4,5]
    assert (baseline/'training/state_000001').exists()
    assert [p.name for p in (root/'training').glob('state_*')]==['state_000005']
    assert (root/'P8.pkl').read_bytes()==(baseline/'P8.pkl').read_bytes()
    for label in LABELS:
        assert (root/f'{label}.pkl').exists()
    assert json.loads((root/'reload-P32.json').read_text())['status']=='VERIFIED'
    assert json.loads((root/'training/manifest.json').read_text())==json.loads((baseline/'training/manifest.json').read_text())


def test_changed_archived_source_is_rejected(continued,tmp_path):
    baseline,archived,_=continued
    import shutil
    source=tmp_path/'changed'
    shutil.copytree(archived,source)
    with (source/'hulhe_escher/config.py').open('a') as handle:handle.write('\n# incompatible source\n')
    with pytest.raises(RuntimeError,match='reload verification failed'):
        run_continuation(baseline,source,tmp_path/'out',stage_seconds=.001,wall_seconds=30,allow_smoke=True)
    assert not (tmp_path/'out/P14.pkl').exists()
    assert json.loads((tmp_path/'out/continuation.json').read_text())['status']=='FAILED'


def test_production_rejects_smoke_and_same_directory(continued,tmp_path):
    baseline,archived,_=continued
    with pytest.raises(ValueError,match='smoke'):
        run_continuation(baseline,archived,tmp_path/'out')
    with pytest.raises(ValueError,match='separate'):
        run_continuation(baseline,archived,baseline,allow_smoke=True)


def test_stage_command_keeps_resume_contract(continued):
    baseline,_,_=continued
    contract=json.loads((baseline/'training/manifest.json').read_text())
    command=trainer_command(baseline/'training',contract,21600)
    assert command[command.index('--policy-seconds')+1]==str(contract['policy_seconds'])
    assert command[command.index('--additional-training-seconds')+1]=='21600'
    assert '--resume' in command


def test_secondary_plan_and_primary_gate():
    pilot=[{label:{'mean_bb':(-1)**i*(j+1)+i*.01} for j,label in enumerate(LABELS)} for i in range(100)]
    plan=fo2_plan(pilot,minimum=100,maximum=1000000,affordable=1000000)
    assert plan['alpha']==.05/8 and plan['pilot_excluded']
    assert plan['selected']>=max(p['required'] for p in plan['differences'].values())
    assert not fo2_plan(pilot,minimum=100,maximum=1000000,affordable=99)['minimum_fits_budget']
    result=estimate([1.,2.]*100,alpha=SECONDARY_ALPHA)
    assert result['confidence']==.99375
    assert gate_secondary(result,False,False)['confirmatory'] is False


def test_real_offline_evaluation_suite(continued,tmp_path):
    from hulhe_escher.continuation_evaluate import run_suite
    _,_,root=continued
    report=run_suite(root,tmp_path/'evaluation',workers=1,max_seconds=240,smoke=True,opponent='check_call')
    assert report['status']=='COMPLETE'
    assert len(report['head_to_head'])==5
    assert report['fo2']['completed_groups']==8
    assert set(report['fo2']['estimates'])==set(LABELS)
    assert all(not x['result']['confirmatory'] for x in report['head_to_head'].values())
    assert all(not x['confirmatory'] for x in report['fo2']['improvements'].values())
    rows=[json.loads(x) for x in (tmp_path/'evaluation/fo2/confirmatory.jsonl').read_text().splitlines()]
    assert all(set(r)=={'pair_id',*LABELS} for r in rows)
    assert (tmp_path/'evaluation/continuation_report.md').exists()


def test_archive_rejects_traversal(tmp_path):
    import io
    with tarfile.open(tmp_path/'bad.tar.gz','w:gz') as handle:
        info=tarfile.TarInfo('../outside');info.size=1;handle.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='safe regular'):
        extract_source(tmp_path/'bad.tar.gz',tmp_path/'extract')
    assert not (tmp_path/'outside').exists()


@pytest.mark.parametrize('baseline_hours', [8,32])
def test_launch_bundle_has_one_bounded_vm_and_no_secrets(tmp_path, baseline_hours):
    module=launcher_module()
    plan=module.build_bundle(tmp_path/'bundle',project='test-project',zone='europe-west1-b',
        bucket='gs://test-bucket/hulhe',service_account='runner@test-project.iam.gserviceaccount.com',
        run_name='hulhe-continuation-test',source_uri='gs://test-bucket/hulhe/original',expected_p8_sha256='a'*64,
        baseline_hours=baseline_hours)
    creates=[x for x in plan['commands'] if x[1:4]==['compute','instances','create']]
    assert len(creates)==1 and '--max-run-duration' in creates[0] and '44h' in creates[0]
    assert plan['additional_active_hours']==24 and plan['stage_hours']==6
    assert plan['labels']==[f'P{baseline_hours+6*i}' for i in range(5)]
    assert plan['primary']==f'P{baseline_hours+24} vs P{baseline_hours}, alpha 0.05'
    script=(tmp_path/'bundle/startup.sh').read_text()
    assert 'continuation_cloud' in script and "trap 'shutdown -h now' EXIT" in script
    with tarfile.open(tmp_path/'bundle/source.tar.gz') as handle:
        names=handle.getnames()
    assert 'hulhe_escher/verify_recovery.py' in names
    assert all('.env' not in x and not x.startswith(('outputs/','.git/')) for x in names)


def test_six_hour_stage_boundaries_preserve_learning_state(continued,tmp_path):
    """Four restart/export boundaries must match uninterrupted learner updates."""
    import shutil
    import torch
    from hulhe_escher.training_state import read_training_state
    baseline,archived,root=continued
    training=tmp_path/'training'
    shutil.copytree(baseline/'training',training)
    contract=json.loads((training/'manifest.json').read_text())
    command=trainer_command(training,contract,1000000)
    command[command.index('--iterations')+1]='5'
    result=subprocess.run(command,cwd=archived,env={**os.environ,'PYTHONPATH':str(archived)},
                          capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr[-3000:]
    a=read_training_state(training/'state_000005')
    b=read_training_state(root/'training/state_000005')
    def equal(x,y):
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for key in x:equal(x[key],y[key])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for xx,yy in zip(x,y):equal(xx,yy)
        elif isinstance(x,np.ndarray):np.testing.assert_array_equal(x,y)
        elif isinstance(x,torch.Tensor):assert torch.equal(x,y)
        else:assert x==y
    for key in ('regret_trainers','q_ensemble','calibration','gate_controller','rng','actors','replay_rng','parallel_contract'):
        equal(a[key],b[key])
    equal(a['average_policy_trainer']['buffer'],b['average_policy_trainer']['buffer'])


def test_failed_p14_publication_resumes_without_repeating_training(continued, tmp_path, monkeypatch):
    from hulhe_escher import continuation
    baseline, archived, complete = continued
    output = tmp_path/'retry'
    uploads = []
    def fail_p14(root, uri, label):
        uploads.append(label)
        if label == 'P14':
            raise subprocess.CalledProcessError(1, ['gcloud','storage','rsync'], stderr='HTTPError 503')
    monkeypatch.setattr(continuation, 'upload_checkpoint', fail_p14)
    with pytest.raises(subprocess.CalledProcessError):
        run_continuation(baseline, archived, output, stage_seconds=.001, wall_seconds=180,
                         allow_smoke=True, upload_uri='gs://test/run')
    failed = json.loads((output/'continuation.json').read_text())
    assert failed['status'] == 'FAILED' and list(failed['checkpoints']) == ['P8','P14']
    p14 = failed['checkpoints']['P14']
    original_policy = (output/'P14.pkl').read_bytes()
    original_log = (output/'train-P14.log').read_bytes()
    assert p14['outer_iteration'] == 2
    # Incorrect checkpoint identity must fail before mutation or verification.
    before = (output/'continuation.json').read_bytes()
    with pytest.raises(ValueError, match='Latest full recovery'):
        run_continuation(baseline, archived, output, stage_seconds=.001, allow_smoke=True,
                         resume_checkpoint='P14', expected_resume_sha256='0'*64)
    assert (output/'continuation.json').read_bytes() == before
    uploads.clear()
    def publish(root, uri, label):
        # A full fresh-process restore must precede the repaired P14 publication.
        if label == 'P14':
            assert json.loads((root/'retry-1-reload-P14.json').read_text())['status'] == 'VERIFIED'
            assert not (root/'retry-1-train-P20.log').exists()
        uploads.append(label)
    monkeypatch.setattr(continuation, 'upload_checkpoint', publish)
    report = run_continuation(baseline, archived, output, stage_seconds=.001, wall_seconds=180,
        allow_smoke=True, upload_uri='gs://test/run', resume_checkpoint='P14',
        expected_resume_sha256=p14['sha256'])
    assert uploads == ['P14','P20','P26','P32']
    assert report['status'] == 'TRAINING_COMPLETE' and report['resume_recovery_verified']
    assert report['final_recovery_verified']
    assert [report['checkpoints'][name]['outer_iteration'] for name in LABELS] == [1,2,3,4,5]
    assert (output/'P14.pkl').read_bytes() == original_policy
    assert (output/'train-P14.log').read_bytes() == original_log
    assert not (output/'retry-1-train-P14.log').exists()
    assert json.loads((output/'retry-1-previous-report.json').read_text())['status'] == 'FAILED'
    assert json.loads((output/'retry-1-reload-P32.json').read_text())['status'] == 'VERIFIED'
    with pytest.raises(ValueError, match='failed run'):
        run_continuation(baseline, archived, output, stage_seconds=.001, allow_smoke=True,
                         resume_checkpoint='P32', expected_resume_sha256=report['checkpoints']['P32']['sha256'])


def test_chained_p32_continuation_retries_and_evaluates_shifted_grid(continued, tmp_path, monkeypatch):
    from hulhe_escher import continuation
    from hulhe_escher.continuation_evaluate import run_suite
    _, archived, baseline = continued
    root = tmp_path/'next24'
    original = (baseline/'P32.pkl').read_bytes()
    def fail(root, uri, label):
        if label == 'P38':
            raise subprocess.CalledProcessError(1, ['gcloud','storage','rsync'])
    monkeypatch.setattr(continuation, 'upload_checkpoint', fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_continuation(baseline, archived, root, baseline_hours=32,
            stage_seconds=.001, wall_seconds=180, allow_smoke=True, upload_uri='gs://test/next24')
    failed = json.loads((root/'continuation.json').read_text())
    assert list(failed['checkpoints']) == ['P32','P38']
    uploads=[]
    monkeypatch.setattr(continuation, 'upload_checkpoint', lambda root,uri,label: uploads.append(label))
    report = run_continuation(baseline, archived, root, baseline_hours=32,
        stage_seconds=.001, wall_seconds=180, allow_smoke=True, upload_uri='gs://test/next24',
        resume_checkpoint='P38', expected_resume_sha256=failed['checkpoints']['P38']['sha256'])
    labels = ['P32','P38','P44','P50','P56']
    assert report['labels'] == labels and report['final_recovery_verified']
    assert uploads == labels[1:]
    assert [report['checkpoints'][p]['outer_iteration'] for p in labels] == [5,6,7,8,9]
    assert (root/'P32.pkl').read_bytes() == original == (baseline/'P32.pkl').read_bytes()
    assert (baseline/'training/state_000005').exists()
    assert json.loads((root/'retry-1-reload-P56.json').read_text())['status'] == 'VERIFIED'
    result = run_suite(root, tmp_path/'evaluation', workers=1,max_seconds=240,smoke=True,opponent='check_call')
    assert result['status'] == 'COMPLETE'
    assert result['primary'] == 'P56-vs-P32'
    assert list(result['head_to_head']) == ['P56-vs-P32','P38-vs-P32','P44-vs-P38','P50-vs-P44','P56-vs-P50']
    assert set(result['fo2']['estimates']) == set(labels)
    assert set(result['fo2']['improvements']) == set(labels[1:])
    design=json.loads((tmp_path/'evaluation/design.json').read_text())
    assert design['head_to_head_pilot_seeds'] == [52301,52311,52321,52331,52341]
    assert design['fo2_pilot_seed'] == 53501
    text=(tmp_path/'evaluation/continuation_report.md').read_text()
    assert 'Primary: P56 vs P32' in text and 'P8' not in text


def test_cloud_chaining_uses_original_trainer_archive(continued,tmp_path,monkeypatch):
    import shutil
    from hulhe_escher import continuation_cloud as cloud
    _,archived,baseline=continued
    calls=[]
    def storage(*args):
        calls.append(args)
        if args[0]=='cp':
            name=args[1].removeprefix('gs://test/source/')
            src=baseline.parent/'source.tar.gz' if name=='training_source.tar.gz' else baseline/name
            shutil.copy2(src,args[2])
        else:
            assert args[:2]==('rsync','--recursive')
            shutil.copytree(baseline/'training/state_000005',args[-1],dirs_exist_ok=True)
    monkeypatch.setattr(cloud,'storage',storage)
    marker=json.loads((baseline/'P32.json').read_text())
    source,trainer=cloud.download_source('gs://test/source',tmp_path/'download',marker['sha256'],32)
    assert (source/'P32.pkl').read_bytes()==(baseline/'P32.pkl').read_bytes()
    assert (trainer/'hulhe_escher/train.py').read_bytes()==(archived/'hulhe_escher/train.py').read_bytes()
    assert any(args[1]=='gs://test/source/training_source.tar.gz' for args in calls)
    assert not any(args[1]=='gs://test/source/source.tar.gz' for args in calls)
    with pytest.raises(ValueError,match='requested baseline'):
        cloud.download_source('gs://test/source',tmp_path/'wrong','0'*64,32)


@pytest.mark.parametrize('hours', [0,14,31,True,32.0])
def test_invalid_baseline_rejected(hours):
    from hulhe_escher.continuation import checkpoint_labels
    with pytest.raises(ValueError,match='Baseline hours'):
        checkpoint_labels(hours)
