"""P32 continuation routing, immutable provenance and independent evaluation."""
import json
import pickle
import shutil
import subprocess
import sys

import pytest

from test_sequential_continuation import REPO, pinned
from hulhe_escher import continuation_cloud as cloud, sequential_continuation as sequential
from hulhe_escher.checkpointing import sha256_file


@pytest.fixture
def p32(pinned):
    source, trainer, spec = pinned
    (source/'source.tar.gz').rename(source/'training_source.tar.gz')
    (source/'training/state_000170').rename(source/'training/state_000659')
    payload = pickle.loads((source/'P8.pkl').read_bytes())
    payload.update(outer_iteration=659, training_elapsed_seconds=115582.36501615302)
    (source/'P32.pkl').write_bytes(pickle.dumps(payload))
    marker = json.loads((source/'P8.json').read_text())
    marker.update(label='P32', outer_iteration=659, recovery='state_000659',
        active_seconds=payload['training_elapsed_seconds'], sha256=sha256_file(source/'P32.pkl'))
    marker['metrics']['policy_validation']['policy_sampling_seed'] = [20261115, 659]
    (source/'P32.json').write_text(json.dumps(marker))
    (source/'training/latest.json').write_text(json.dumps(dict(iteration=659, state='state_000659')))
    spec.update(iteration=659, active_seconds=marker['active_seconds'],
        policy_sha256=marker['sha256'], evaluation_seed_offset=6000000)
    return source, trainer, spec


def test_p32_plan_and_source_protection(tmp_path):
    sys.path.insert(0, str(REPO/'tools'))
    import launch_sequential_p32_continuation_batch as launcher
    plan = launcher.build_bundle(tmp_path/'plan', run_name='hulhe-sequential-p32-plus24-test')
    assert plan['baseline'] == sequential.source_spec(32)
    assert plan['baseline']['iteration'] == 659
    assert plan['labels'] == ['P32','P38','P44','P50','P56']
    assert plan['evaluation_seed_offset'] == 6000000
    assert plan['primary'] == 'Sequential-P56 versus its own P32; alpha .05'
    assert plan['additional_active_hours'] == 24 and plan['stage_hours'] == 6
    assert plan['verify_each_stage'] and not plan['source_modified'] and not plan['config_changed']
    script = (tmp_path/'plan/batch.sh').read_text()
    assert '--baseline-hours 32' in script and '--sequential-source' in script
    assert '--require-corrected-sampler' in script
    task = json.loads((tmp_path/'plan/job.json').read_text())['taskGroups'][0]['taskSpec']
    assert task['maxRunDuration'] == '158400s' and task['maxRetryCount'] == 0
    subprocess.run(['bash', '-n', str(tmp_path/'plan/batch.sh')], check=True)
    for hours in (8, 32):
        with pytest.raises(ValueError, match='independent'):
            launcher.build_bundle(tmp_path/f'bad{hours}',
                run_name=sequential.source_spec(hours)['source_uri'].split('/')[-1])


def test_p32_verified_archive_and_checkpoint(p32):
    source, trainer, spec = p32
    receipt = sequential.verify_sequential_source(source, trainer, 32)
    assert receipt['status'] == 'VERIFIED' and receipt['baseline_iteration'] == 659
    (source/'training_source.tar.gz').write_bytes(b'coordinator archive is not the trainer')
    with pytest.raises(ValueError, match='archived trainer'):
        sequential.verify_sequential_source(source, trainer, 32)


def test_p32_cloud_routing(p32, tmp_path, monkeypatch):
    source, trainer, spec = p32
    def download(*args, **kwargs):
        assert args[-1] == 32 and kwargs['expected_iteration'] == 659
        assert kwargs['expected_state_manifest_sha256'] == spec['state_manifest_sha256']
        assert kwargs['expected_archive_sha256'] == spec['source_archive_sha256']
        return source, trainer
    monkeypatch.setattr(cloud, 'download_source', download)
    transfers = []
    monkeypatch.setattr(cloud, 'storage', lambda *args: transfers.append(args))
    monkeypatch.setattr(cloud, 'run_transfer', lambda *args, **kwargs: None)
    monkeypatch.setattr(cloud, 'prepare', lambda p: dict(fingerprint='fo2'))
    def train(src, archived, output, **kwargs):
        assert src == source and archived == trainer
        assert kwargs['baseline_hours'] == 32 and kwargs['stage_count'] == 4
        assert kwargs['verify_each_stage'] and kwargs['require_protected_sequential']
        assert kwargs['require_objective_mass']
        output.mkdir(parents=True)
        return dict(training_source_fingerprint=spec['source_fingerprint'])
    def evaluate(*args, **kwargs):
        assert kwargs['extra_seed_offset'] == 6000000
        return dict(status='COMPLETE')
    monkeypatch.setattr(cloud, 'run_continuation', train)
    monkeypatch.setattr(cloud, 'run_suite', evaluate)
    root = tmp_path/'cloud'; root.mkdir()
    cloud.run_cloud(root, spec['source_uri'], 'gs://test/continued', spec['policy_sha256'],
        baseline_hours=32, sequential_source=True, require_corrected_sampler=True)
    assert any(args[-2:] == (source/'training_source.tar.gz', 'gs://test/continued/training_source.tar.gz')
        for args in transfers)


@pytest.mark.parametrize('change', ['hash', 'stages', 'sampler', 'old_output'])
def test_p32_bad_request_rejected_before_download(tmp_path, monkeypatch, change):
    spec = sequential.source_spec(32)
    options = dict(baseline_hours=32, sequential_source=True, require_corrected_sampler=True)
    output, digest = 'gs://test/new', spec['policy_sha256']
    if change == 'hash': digest = 'bad'
    elif change == 'stages': options['stage_count'] = 8
    elif change == 'sampler': options['require_corrected_sampler'] = False
    else: output = sequential.source_spec(8)['source_uri']
    monkeypatch.setattr(cloud, 'download_source', lambda *a, **k: pytest.fail('Must reject before download'))
    with pytest.raises(ValueError):
        cloud.run_cloud(tmp_path, spec['source_uri'], output, digest, **options)


def test_p32_manifest_rejected_before_replay(p32, tmp_path, monkeypatch):
    source, trainer, spec = p32
    transfers = []
    def transfer(*args):
        transfers.append(args)
        assert args[0] == 'cp', 'Must not download full replay'
        shutil.copyfile(source/args[1].removeprefix(spec['source_uri']+'/'), args[-1])
    monkeypatch.setattr(cloud, 'storage', transfer)
    with pytest.raises(ValueError, match='manifest mismatch'):
        cloud.download_source(spec['source_uri'], tmp_path/'download', spec['policy_sha256'], 32,
            expected_archive_sha256=spec['source_archive_sha256'], expected_iteration=659,
            expected_state_manifest_sha256='bad')
    assert any(args[1].endswith('/training_source.tar.gz') for args in transfers)
    assert not any(args[1].endswith('/source.tar.gz') for args in transfers)


def test_p32_evaluation_is_fresh(tmp_path, monkeypatch):
    from hulhe_escher import continuation_evaluate as evaluation
    labels = ['P32','P38','P44','P50','P56']
    root = tmp_path/'run'; root.mkdir()
    (root/'continuation.json').write_text(json.dumps(dict(status='TRAINING_COMPLETE',
        final_recovery_verified=True, smoke=False, baseline_hours=32,
        checkpoints={label:dict(sha256=label) for label in labels})))
    monkeypatch.setattr(evaluation, 'wait_policy', lambda r,l,d:(r/f'{l}.pkl',dict(sha256=l)))
    matches, opponents = [], []
    def match(*args, **kwargs):
        matches.append(kwargs); return dict(status='COMPLETE', result=None)
    def fo2(*args, **kwargs):
        opponents.append(kwargs); return dict(status='COMPLETE')
    monkeypatch.setattr(evaluation, 'run_evaluation', match)
    monkeypatch.setattr(evaluation, 'run_fo2', fo2)
    report = evaluation.run_suite(root, tmp_path/'eval', extra_seed_offset=6000000)
    assert report['primary'] == 'P56-vs-P32' and report['status'] == 'COMPLETE'
    assert [r['seed'] for r in matches] == [6052301+10*i for i in range(5)]
    assert [r['alpha'] for r in matches] == [.05]+[.05/8]*4
    assert opponents[0]['seed'] == 6053501 and opponents[0]['labels'] == tuple(labels)
