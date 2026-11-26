"""Hybrid P32 plus48 immutable source, Batch contract and evaluation routing."""
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest
from test_scaled_continuation import source_fixture
from hulhe_escher import continuation_cloud as cloud, continuation_evaluate as evaluation
from hulhe_escher import scaled_continuation as scaled
from hulhe_escher.scaling_config import hybrid_scaling_config
from hulhe_escher.checkpointing import sha256_file

REPO = Path(__file__).resolve().parents[1]
LABELS = tuple(f'P{32+6*i}' for i in range(9))

@pytest.fixture
def hybrid_source(source_fixture, monkeypatch):
    _, source, trainer, spec = source_fixture
    config = hybrid_scaling_config()
    path = source/'training/manifest.json'
    contract = json.loads(path.read_text()); contract['config'] = config
    path.write_text(json.dumps(contract))
    path = source/'P32.pkl'
    payload = pickle.loads(path.read_bytes()); payload['training_config'] = config
    path.write_bytes(pickle.dumps(payload))
    path = source/'P32.json'
    marker = json.loads(path.read_text()); marker['sha256'] = sha256_file(source/'P32.pkl')
    path.write_text(json.dumps(marker)); spec['policy_sha256'] = marker['sha256']
    monkeypatch.setattr(scaled, 'source_spec', lambda variant, baseline_hours=32: spec)
    return 'hybrid', source, trainer, spec


def test_hybrid_source_validation_rejects_changed_architecture(hybrid_source):
    variant, source, trainer, spec = hybrid_source
    assert scaled.verify_scaled_source(source, trainer, variant)['status'] == 'VERIFIED'
    path = source/'training/manifest.json'
    contract = json.loads(path.read_text())
    contract['config']['regret_policy_architecture'] = 'deep_residual_v1'
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='production contract'):
        scaled.verify_scaled_source(source, trainer, variant)


def test_invalid_hybrid_source_and_stage_rejected_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, 'download_source', lambda *a, **kw: pytest.fail('No download'))
    pin = scaled.source_spec('hybrid')
    for changes in [dict(stage_count=4), dict(source_uri='gs://wrong/source'),
                    dict(expected_p8_sha256='wrong'), dict(require_corrected_sampler=False)]:
        args = dict(workspace=tmp_path, source_uri=pin['source_uri'], output_uri='gs://test/new',
                    expected_p8_sha256=pin['policy_sha256'], baseline_hours=32,stage_count=8,
                    scaled_variant='hybrid', require_corrected_sampler=True)
        args.update(changes)
        with pytest.raises(ValueError, match='pinned corrected'):
            cloud.run_cloud(**args)
    with pytest.raises(ValueError, match='No pinned'):
        scaled.source_spec('hybrid',56)


def test_hybrid_plus48_batch_plan(tmp_path):
    sys.path.insert(0, str(REPO/'tools'))
    import launch_hybrid_continuation_batch as launcher
    root = tmp_path/'plan'
    plan = launcher.build_bundle(root, run_name='hulhe-hybrid-p32-plus48-test')
    assert plan['baseline'] == scaled.source_spec('hybrid')
    assert plan['baseline']['iteration'] == 777
    assert plan['labels'] == list(LABELS) and plan['stage_count'] == 8
    assert plan['additional_active_hours'] == 48 and plan['stage_hours'] == 6
    assert plan['verify_each_stage'] and not plan['config_changed'] and not plan['source_modified']
    assert plan['policy_group_sampling'] == 'objective_mass'
    assert (plan['training_wall_hours'], plan['evaluation_wall_hours'], plan['max_job_hours']) == (60,12,74)
    job = json.loads((root/'job.json').read_text())
    task = job['taskGroups'][0]['taskSpec']
    assert task['maxRunDuration'] == '266400s' and task['maxRetryCount'] == 0
    assert job['allocationPolicy']['instances'][0]['policy']['machineType'] == 'c4-standard-16'
    script = (root/'batch.sh').read_text()
    for value in ('--baseline-hours 32', '--stage-count 8', '--training-wall-seconds 216000',
                  '--require-corrected-sampler', '--scaled-variant hybrid'):
        assert value in script
    subprocess.run(['bash','-n',str(root/'batch.sh')], check=True)
    for hours in (32,56):
        for other in ('wide','deep'):
            name = scaled.source_spec(other,hours)['source_uri'].rsplit('/',1)[1]
            with pytest.raises(ValueError, match='independent'):
                launcher.build_bundle(tmp_path/'bad', run_name=name)

    assert plan['per_network_parameters'] == 1245699
    name = scaled.source_spec('hybrid')['source_uri'].rsplit('/',1)[1]
    with pytest.raises(ValueError, match='independent'):
        launcher.build_bundle(tmp_path/'bad-hybrid', run_name=name)


def test_cloud_routes_eight_stages_to_original_trainer(hybrid_source, tmp_path, monkeypatch):
    variant, source, trainer, spec = hybrid_source
    def download(*a, **kw):
        assert a[3] == 32
        assert kw['expected_iteration'] == spec['iteration']
        assert kw['expected_state_manifest_sha256'] == spec['state_manifest_sha256']
        return source, trainer
    monkeypatch.setattr(cloud,'download_source',download)
    monkeypatch.setattr(cloud,'storage',lambda *a:None)
    monkeypatch.setattr(cloud,'run_transfer',lambda *a,**kw:None)
    monkeypatch.setattr(cloud,'prepare',lambda p:dict(fingerprint='fo2'))
    def train(src, archived, output, **kw):
        assert src==source and archived==trainer
        assert kw['stage_count']==8 and kw['baseline_hours']==32
        assert kw['wall_seconds']==216000 and kw['verify_each_stage'] and kw['require_objective_mass']
        output.mkdir(parents=True)
        return dict(training_source_fingerprint=spec['source_fingerprint'])
    def evaluate(*a,**kw):
        assert kw['extra_seed_offset']==spec['evaluation_seed_offset']
        return dict(status='COMPLETE')
    monkeypatch.setattr(cloud,'run_continuation',train)
    monkeypatch.setattr(cloud,'run_suite',evaluate)
    root=tmp_path/'cloud'; root.mkdir()
    cloud.run_cloud(root,spec['source_uri'],'gs://test/new',spec['policy_sha256'],
        baseline_hours=32,stage_count=8,training_wall_seconds=216000,
        require_corrected_sampler=True,scaled_variant=variant)
    provenance=json.loads((root/'run/provenance.json').read_text())
    assert provenance['additional_active_hours']==48 and provenance['stage_count']==8


def test_nine_h2h_matches_and_sixteen_test_family(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir()
    (root/'continuation.json').write_text(json.dumps(dict(status='TRAINING_COMPLETE',
        final_recovery_verified=True,smoke=False,baseline_hours=32,stage_count=8,labels=list(LABELS),
        checkpoints={label:dict(sha256=label) for label in LABELS})))
    monkeypatch.setattr(evaluation,'wait_policy',lambda r,l,d:(r/f'{l}.pkl',dict(sha256=l)))
    matches=[];opponents=[]
    def match(a,b,output,**kw):
        matches.append((Path(a).stem,Path(b).stem,kw))
        return dict(status='COMPLETE',result=dict(confirmatory=True,conclusion='improved',
            bb_per_100=3.,interval=[1.,5.],confidence=1-kw['alpha']))
    def fo2(*a,**kw):
        opponents.append(kw)
        return dict(status='COMPLETE')
    monkeypatch.setattr(evaluation,'run_evaluation',match)
    monkeypatch.setattr(evaluation,'run_fo2',fo2)
    extra=scaled.source_spec('hybrid')['evaluation_seed_offset']
    result=evaluation.run_suite(root,tmp_path/'eval',extra_seed_offset=extra)
    assert result['primary']=='P80-vs-P32' and result['status']=='COMPLETE'
    assert result['secondary_family_size']==16 and result['secondary_alpha']==.05/16
    assert [(a,b) for a,b,_ in matches]==[('P80','P32'),*zip(LABELS[1:],LABELS[:-1])]
    assert [kw['alpha'] for _,_,kw in matches]==[.05]+[.05/16]*8
    assert [kw['seed'] for _,_,kw in matches]==[52301+extra+10*i for i in range(9)]
    assert opponents[0]['labels']==LABELS and opponents[0]['primary_passed']
    assert opponents[0]['seed']==53501+extra
    text=(tmp_path/'eval/continuation_report.md').read_text()
    assert '48-hour' in text and '99.6875%' in text and 'across 16' in text

