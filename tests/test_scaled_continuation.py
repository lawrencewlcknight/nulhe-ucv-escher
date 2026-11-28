"""Pinned architecture continuations, complete-state resume and independent Batch plans."""
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import pytest

from hulhe_escher import continuation_cloud as cloud, scaled_continuation as scaled
from hulhe_escher.continuation import run_continuation
from hulhe_escher.continuation_evaluate import run_suite
from hulhe_escher.checkpointing import sha256_file
from hulhe_escher.experiment import publish_policy
from hulhe_escher.scaling_config import scaling_config, deep_scaling_config, hybrid_scaling_config

REPO = Path(__file__).resolve().parents[1]


def launcher():
    sys.path.insert(0, str(REPO/'tools'))
    import launch_scaled_continuation_batch
    return launch_scaled_continuation_batch


@pytest.mark.parametrize('variant,iteration,parameters', [('wide',953,777731), ('deep',868,749827)])
def test_batch_plans_pin_the_right_state_and_do_not_launch(tmp_path, variant, iteration, parameters):
    module = launcher()
    root = tmp_path/'plan'
    plan = module.build_bundle(root, variant=variant, run_name=f'hulhe-{variant}-p32-plus24-test')
    assert plan['baseline'] == scaled.source_spec(variant)
    assert plan['baseline']['iteration'] == iteration and plan['per_network_parameters'] == parameters
    assert plan['labels'] == ['P32','P38','P44','P50','P56']
    assert plan['additional_active_hours'] == 24 and plan['verify_each_stage']
    assert not plan['source_modified'] and not plan['config_changed']
    assert plan['policy_group_sampling'] == 'objective_mass'
    job = json.loads((root/'job.json').read_text())
    task = job['taskGroups'][0]['taskSpec']
    assert task['maxRunDuration'] == '158400s' and task['maxRetryCount'] == 0
    assert task['computeResource'] == dict(cpuMilli=16000, memoryMib=60000)
    assert job['allocationPolicy']['instances'][0]['policy']['bootDisk']['sizeGb'] == 200
    script = (root/'batch.sh').read_text()
    assert f'--scaled-variant {variant}' in script and '--require-corrected-sampler' in script
    assert '--corrected-sampler ' not in script and '--baseline-hours 32' in script
    assert plan['baseline']['source_uri'] in script and plan['baseline']['policy_sha256'] in script
    subprocess.run(['bash','-n',str(root/'batch.sh')], check=True)
    with pytest.raises(ValueError, match='independent'):
        module.build_bundle(tmp_path/'bad', variant=variant, run_name=f'hulhe-{variant}32-batch-20261122')


@pytest.fixture(params=['wide', 'deep'])
def source_fixture(request, tmp_path, monkeypatch):
    variant = request.param
    source, trainer = tmp_path/'source', tmp_path/'trainer'
    state = source/'training/state_000953'
    state.mkdir(parents=True); trainer.mkdir()
    config = deep_scaling_config() if variant == 'deep' else scaling_config()
    # Small metadata-only fixture. A real recovery is exercised separately below.
    contract = dict(config=config, source_fingerprint=scaled.fingerprint(trainer),
        smoke=False, workers=8, threads=8, seed=20261115, policy_every=0, checkpoint_every=0,
        recovery_seconds=21600., policy_seconds=1e9)
    payload = dict(training_config=config, seed=20261115, outer_iteration=953, training_elapsed_seconds=115210.)
    with (source/'P32.pkl').open('wb') as f: pickle.dump(payload, f)
    (source/'training_source.tar.gz').write_bytes(b'fixture archive')
    (state/'manifest.json').write_text('{}')
    marker = dict(label='P32', outer_iteration=953, recovery=state.name, active_seconds=115210.,
        sha256=sha256_file(source/'P32.pkl'), smoke=False,
        metrics=dict(policy_validation=dict(group_sampling='objective_mass', policy_sampling_seed=[20261115,953])))
    for path, value in [('training/manifest.json',contract), ('P32.json',marker),
                        ('training/latest.json',dict(iteration=953,state=state.name))]:
        (source/path).write_text(json.dumps(value))
    spec = dict(source_uri='gs://test/original', policy_sha256=marker['sha256'], iteration=953,
        source_archive_sha256=sha256_file(source/'training_source.tar.gz'),
        source_fingerprint=contract['source_fingerprint'],
        state_manifest_sha256=sha256_file(state/'manifest.json'), evaluation_seed_offset=1000000)
    monkeypatch.setattr(scaled, 'source_spec', lambda _: spec)
    monkeypatch.setattr(scaled, 'load_checkpoint_payload', lambda p: pickle.loads(p.read_bytes()))
    return variant, source, trainer, spec


def test_source_verification_and_sampling_guard(source_fixture):
    variant, source, trainer, spec = source_fixture
    receipt = scaled.verify_scaled_source(source, trainer, variant)
    assert receipt['status'] == 'VERIFIED' and not receipt['source_modified']
    marker = json.loads((source/'P32.json').read_text())
    marker['metrics']['policy_validation']['policy_sampling_seed'] = [0,953]
    (source/'P32.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='sampling provenance'):
        scaled.verify_scaled_source(source, trainer, variant)


@pytest.mark.parametrize('what', ['archive','config','recovery','source','smoke'])
def test_source_drift_rejected(source_fixture, what):
    variant, source, trainer, spec = source_fixture
    if what == 'archive': (source/'training_source.tar.gz').write_bytes(b'changed')
    elif what == 'source': spec['source_fingerprint'] = 'wrong'
    elif what == 'recovery': (source/'training/state_000953/manifest.json').write_text('{"changed":true}')
    else:
        path = source/'training/manifest.json'; contract = json.loads(path.read_text())
        if what == 'config': contract['config']['ave_policy_buffer_size'] = 1
        else: contract['smoke'] = True
        path.write_text(json.dumps(contract))
    with pytest.raises(ValueError): scaled.verify_scaled_source(source, trainer, variant)


def test_pin_failures_prevent_large_download_and_extract(source_fixture, tmp_path, monkeypatch):
    variant, source, trainer, spec = source_fixture
    calls = []
    def transfer(*a):
        calls.append(a)
        assert a[0] == 'cp', 'Must reject metadata before downloading replay'
        relative = a[1].split('gs://test/original/')[1]
        shutil.copyfile(source/relative, a[-1])
    monkeypatch.setattr(cloud, 'storage', transfer)
    monkeypatch.setattr(cloud, 'extract_source', lambda *a: pytest.fail('Do not extract wrong source'))
    for name, overrides in [('archive',dict(expected_archive_sha256='bad')),
                            ('iteration',dict(expected_iteration=999)),
                            ('manifest',dict(expected_state_manifest_sha256='bad'))]:
        with pytest.raises(ValueError):
            cloud.download_source(spec['source_uri'], tmp_path/name, spec['policy_sha256'], 32, **overrides)
    assert calls


def test_scaled_cloud_preserves_trainer_and_applies_verification(source_fixture, tmp_path, monkeypatch):
    variant, source, trainer, spec = source_fixture
    calls = []
    def download(*a, **kw):
        assert kw == dict(expected_archive_sha256=spec['source_archive_sha256'],
            expected_iteration=spec['iteration'], expected_state_manifest_sha256=spec['state_manifest_sha256'])
        return source, trainer
    monkeypatch.setattr(cloud, 'download_source', download)
    monkeypatch.setattr(cloud, 'storage', lambda *a: calls.append(a))
    monkeypatch.setattr(cloud, 'run_transfer', lambda *a, **k: None)
    monkeypatch.setattr(cloud, 'prepare', lambda p: dict(fingerprint='test-fo2'))
    def train(src, archived, output, **kw):
        assert src == source and archived == trainer
        assert kw['verify_each_stage'] and kw['require_objective_mass'] and kw['baseline_hours'] == 32
        output.mkdir(parents=True)
        return dict(training_source_fingerprint=spec['source_fingerprint'])
    def evaluate(*a, **kw):
        assert kw['extra_seed_offset'] == spec['evaluation_seed_offset']
        return dict(status='COMPLETE')
    monkeypatch.setattr(cloud, 'run_continuation', train)
    monkeypatch.setattr(cloud, 'run_suite', evaluate)
    root = tmp_path/'cloud'; root.mkdir()
    cloud.run_cloud(root, spec['source_uri'], 'gs://test/continued', spec['policy_sha256'],
        baseline_hours=32, require_corrected_sampler=True, scaled_variant=variant)
    assert any(a[-1] == 'gs://test/continued/training_source.tar.gz' for a in calls)
    provenance = json.loads((root/'run/provenance.json').read_text())
    assert provenance['scaled_variant'] == variant
    assert (source/'training/state_000953/manifest.json').read_text() == '{}'


def test_wrong_variant_source_rejected_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, 'download_source', lambda *a, **k: pytest.fail('Do not download'))
    wide, deep = scaled.source_spec('wide'), scaled.source_spec('deep')
    with pytest.raises(ValueError, match='pinned corrected P32'):
        cloud.run_cloud(tmp_path, wide['source_uri'], 'gs://test/new', deep['policy_sha256'],
            baseline_hours=32, require_corrected_sampler=True, scaled_variant='deep')
    with pytest.raises(ValueError, match='preserve both'):
        cloud.run_cloud(tmp_path, wide['source_uri'], deep['source_uri'], wide['policy_sha256'],
            baseline_hours=32, require_corrected_sampler=True, scaled_variant='wide')


@pytest.mark.parametrize('variant', ['wide', 'deep'])
def test_continuation_evaluation_has_fresh_variant_seeds(tmp_path, monkeypatch, variant):
    from hulhe_escher import continuation_evaluate as evaluation
    labels = ['P32','P38','P44','P50','P56']
    root = tmp_path/'run'; root.mkdir()
    (root/'continuation.json').write_text(json.dumps(dict(status='TRAINING_COMPLETE',
        final_recovery_verified=True, smoke=False, baseline_hours=32,
        checkpoints={label:dict(sha256=label) for label in labels})))
    monkeypatch.setattr(evaluation, 'wait_policy', lambda r,l,d: (r/f'{l}.pkl',dict(sha256=l)))
    matches, opponent = [], []
    def match(*a, **kw):
        matches.append(kw); return dict(status='COMPLETE',result=None)
    def fo2(*a, **kw):
        opponent.append(kw); return dict(status='COMPLETE')
    monkeypatch.setattr(evaluation, 'run_evaluation', match)
    monkeypatch.setattr(evaluation, 'run_fo2', fo2)
    extra = scaled.source_spec(variant)['evaluation_seed_offset']
    result = run_suite(root, tmp_path/'eval', extra_seed_offset=extra)
    assert result['status'] == 'COMPLETE' and result['primary'] == 'P56-vs-P32'
    assert [m['seed'] for m in matches] == [52301+extra+10*i for i in range(5)]
    assert [m['alpha'] for m in matches] == [.05]+[.05/8]*4
    assert opponent[0]['seed'] == 53501+extra and opponent[0]['labels'] == tuple(labels)
    design = json.loads((tmp_path/'eval/design.json').read_text())
    assert design['extra_seed_offset'] == extra and design['pilot_excluded']


@pytest.mark.parametrize('variant,baseline_hours,stage_count', [
    ('wide',32,4), ('deep',32,4), ('wide',56,8), ('deep',56,8), ('hybrid',32,8), ('wide',104,8)])
def test_real_full_network_continuation_preserves_source(tmp_path, variant, monkeypatch, baseline_hours, stage_count):
    from hulhe_escher.continuation import checkpoint_labels
    labels = checkpoint_labels(baseline_hours, stage_count)
    baseline_label, final_label = labels[0], labels[-1]
    module = launcher()
    archived = tmp_path/'archived'
    # An optional local archived trainer makes this exercise the actual production
    # source. The default remains offline, reproducible and self-contained.
    old = os.environ.get('HULHE_SCALED_ARCHIVES')
    archive = (Path(old)/f'hulhe-{variant}-p32-trainer-20261124.tar.gz' if old else tmp_path/'source.tar.gz')
    if not old: module.source_archive(REPO, archive)
    cloud.extract_source(archive, archived)
    baseline = tmp_path/'baseline'; baseline.mkdir()
    config = {'wide': scaling_config, 'deep': deep_scaling_config,
              'hybrid': hybrid_scaling_config}[variant](small=True)
    config_path = tmp_path/'config.json'; config_path.write_text(json.dumps(config))
    env = {**os.environ, 'PYTHONPATH':str(archived), 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'}
    command = [sys.executable,'-m','hulhe_escher.train','--config',str(config_path),'--iterations','1',
        '--workers','1','--threads','1','--seed','123','--policy-every','0',
        '--policy-seconds','1000000000','--output',str(baseline/'training')]
    result = subprocess.run(command, cwd=archived, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr[-4000:]
    publish_policy(baseline/'training', baseline, baseline_label)
    hashes = {p:sha256_file(p) for p in baseline.rglob('*') if p.is_file()}
    output = tmp_path/'continued'
    uploaded = []
    def upload(root, uri, label):
        assert json.loads((root/f'reload-{label}.json').read_text())['status'] == 'VERIFIED'
        uploaded.append(label)
    monkeypatch.setattr('hulhe_escher.continuation.upload_checkpoint', upload)
    report = run_continuation(baseline, archived, output, baseline_hours=baseline_hours, stage_count=stage_count,
        stage_seconds=3, wall_seconds=240, allow_smoke=True,
        require_objective_mass=True, verify_each_stage=True, upload_uri='gs://test/offline')
    assert report['status'] == 'TRAINING_COMPLETE' and report['final_recovery_verified']
    assert list(report['checkpoints']) == list(labels)
    assert report['additional_active_seconds'] == stage_count*3
    assert uploaded == list(report['checkpoints'])
    initial = report['checkpoints'][baseline_label]['active_seconds']
    for i, label in enumerate(labels[1:],1):
        record = report['checkpoints'][label]
        assert record['target_active_seconds'] == initial+3*i
        assert record['active_seconds'] >= initial+3*i
        assert record['metrics']['policy_validation']['policy_sampling_seed'] == [123,record['outer_iteration']]
        assert json.loads((output/f'reload-{label}.json').read_text())['status'] == 'VERIFIED'
    assert (output/f'{baseline_label}.pkl').read_bytes() == (baseline/f'{baseline_label}.pkl').read_bytes()
    assert json.loads((output/'training/manifest.json').read_text())['config'] == config
    assert all(sha256_file(p) == value for p,value in hashes.items())
    assert sha256_file(output/f'{final_label}.pkl') != sha256_file(output/f'{baseline_label}.pkl')
