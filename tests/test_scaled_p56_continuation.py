"""Eight-stage full-state continuations, pinned P56 identities and 16-test family."""
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from test_scaled_continuation import source_fixture
from test_policy_sampling_experiment import frozen_source
from hulhe_escher import continuation as continuation
from hulhe_escher import continuation_cloud as cloud
from hulhe_escher import continuation_evaluate as evaluation
from hulhe_escher import scaled_continuation as scaled
from hulhe_escher.checkpointing import sha256_file

REPO = Path(__file__).resolve().parents[1]
LABELS = tuple(f'P{56+6*i}' for i in range(9))


@pytest.mark.parametrize('variant,iteration', [('wide',1703),('deep',1516)])
def test_p56_plus48_batch_plans(tmp_path, variant, iteration):
    sys.path.insert(0, str(REPO/'tools'))
    import launch_scaled_p56_continuation_batch as launcher
    root = tmp_path/'plan'
    plan = launcher.build_bundle(root, variant=variant, run_name=f'hulhe-{variant}-p56-plus48-test')
    assert plan['baseline'] == scaled.source_spec(variant, 56)
    assert plan['baseline']['iteration'] == iteration
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
    for value in ('--baseline-hours 56', '--stage-count 8', '--training-wall-seconds 216000',
                  '--require-corrected-sampler', f'--scaled-variant {variant}'):
        assert value in script
    subprocess.run(['bash','-n',str(root/'batch.sh')], check=True)
    for hours in (32,56):
        for other in ('wide','deep'):
            name = scaled.source_spec(other,hours)['source_uri'].rsplit('/',1)[1]
            with pytest.raises(ValueError, match='independent'):
                launcher.build_bundle(tmp_path/'bad', variant=variant, run_name=name)


@pytest.fixture
def p56_source(source_fixture, monkeypatch):
    variant, source, trainer, spec = source_fixture
    (source/'P32.pkl').rename(source/'P56.pkl')
    payload = pickle.loads((source/'P56.pkl').read_bytes())
    payload['training_elapsed_seconds'] = 201700.
    (source/'P56.pkl').write_bytes(pickle.dumps(payload))
    marker = json.loads((source/'P32.json').read_text())
    marker.update(label='P56', active_seconds=201700., sha256=sha256_file(source/'P56.pkl'))
    (source/'P56.json').write_text(json.dumps(marker))
    spec.update(policy_sha256=marker['sha256'])
    monkeypatch.setattr(scaled, 'source_spec', lambda variant, baseline_hours=32: spec)
    return variant, source, trainer, spec


def test_p56_verification_and_wrong_baseline_rejected(p56_source):
    variant, source, trainer, _ = p56_source
    assert scaled.verify_scaled_source(source,trainer,variant,56)['status']=='VERIFIED'
    marker = json.loads((source/'P56.json').read_text())
    marker['label'] = 'P32'
    (source/'P56.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError,match='P56 recovery'):
        scaled.verify_scaled_source(source,trainer,variant,56)


def test_cloud_routes_eight_stages_to_original_trainer(p56_source, tmp_path, monkeypatch):
    variant, source, trainer, spec = p56_source
    def download(*a, **kw):
        assert a[3] == 56
        assert kw['expected_iteration'] == spec['iteration']
        assert kw['expected_state_manifest_sha256'] == spec['state_manifest_sha256']
        return source, trainer
    monkeypatch.setattr(cloud,'download_source',download)
    monkeypatch.setattr(cloud,'storage',lambda *a:None)
    monkeypatch.setattr(cloud,'run_transfer',lambda *a,**kw:None)
    monkeypatch.setattr(cloud,'prepare',lambda p:dict(fingerprint='fo2'))
    def train(src, archived, output, **kw):
        assert src==source and archived==trainer
        assert kw['stage_count']==8 and kw['baseline_hours']==56
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
        baseline_hours=56,stage_count=8,training_wall_seconds=216000,
        require_corrected_sampler=True,scaled_variant=variant)
    provenance=json.loads((root/'run/provenance.json').read_text())
    assert provenance['additional_active_hours']==48 and provenance['stage_count']==8


def test_wrong_stage_count_and_other_source_are_rejected_before_download(tmp_path,monkeypatch):
    monkeypatch.setattr(cloud,'download_source',lambda *a,**kw:pytest.fail('No download'))
    wide,deep=scaled.source_spec('wide',56),scaled.source_spec('deep',56)
    with pytest.raises(ValueError,match='stage count'):
        cloud.run_cloud(tmp_path,wide['source_uri'],'gs://test/new',wide['policy_sha256'],
            baseline_hours=56,stage_count=4,scaled_variant='wide',require_corrected_sampler=True)
    with pytest.raises(ValueError,match='preserve both'):
        cloud.run_cloud(tmp_path,wide['source_uri'],deep['source_uri'],wide['policy_sha256'],
            baseline_hours=56,stage_count=8,scaled_variant='wide',require_corrected_sampler=True)


@pytest.mark.parametrize('bad', [0,1,7,9,True,8.0])
def test_invalid_stage_counts_rejected(bad):
    with pytest.raises(ValueError,match='stages'):
        continuation.checkpoint_labels(56,bad)


@pytest.mark.parametrize('variant', ['wide','deep'])
def test_nine_h2h_matches_and_sixteen_test_family(tmp_path,monkeypatch,variant):
    root=tmp_path/'run';root.mkdir()
    (root/'continuation.json').write_text(json.dumps(dict(status='TRAINING_COMPLETE',
        final_recovery_verified=True,smoke=False,baseline_hours=56,stage_count=8,labels=list(LABELS),
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
    extra=scaled.source_spec(variant,56)['evaluation_seed_offset']
    result=evaluation.run_suite(root,tmp_path/'eval',extra_seed_offset=extra)
    assert result['primary']=='P104-vs-P56' and result['status']=='COMPLETE'
    assert result['secondary_family_size']==16 and result['secondary_alpha']==.05/16
    assert [(a,b) for a,b,_ in matches]==[('P104','P56'),*zip(LABELS[1:],LABELS[:-1])]
    assert [kw['alpha'] for _,_,kw in matches]==[.05]+[.05/16]*8
    assert [kw['seed'] for _,_,kw in matches]==[76301+extra+10*i for i in range(9)]
    assert opponents[0]['labels']==LABELS and opponents[0]['primary_passed']
    assert opponents[0]['seed']==77501+extra
    text=(tmp_path/'eval/continuation_report.md').read_text()
    assert '48-hour' in text and '99.6875%' in text and 'across 16' in text


def test_fo2_power_plan_uses_nine_policies_and_eight_changes():
    pilot=[{label:dict(mean_bb=(-1)**i*j+i*.1) for j,label in enumerate(LABELS)} for i in range(100)]
    plan=evaluation.fo2_plan(pilot,minimum=100,maximum=1000000,affordable=1000000,labels=LABELS)
    assert plan['alpha']==.05/16 and len(plan['differences'])==8
    assert len(plan['precision_required'])==9 and plan['unit']=='shared_deal_eighteen_hand_group'
    assert all(p['alpha']==.05/16 for p in plan['differences'].values())


def test_completed_stage_retry_retains_48_hour_contract(tmp_path, monkeypatch):
    root=tmp_path/'run'; (root/'training').mkdir(parents=True)
    contract=dict(source_fingerprint='original',smoke=False)
    (root/'training/manifest.json').write_text(json.dumps(contract))
    records={}
    for i,label in enumerate(LABELS[:5]):
        (root/f'{label}.pkl').write_bytes(label.encode())
        record=dict(sha256=sha256_file(root/f'{label}.pkl'),outer_iteration=1703+i,
                    recovery=f'state_{1703+i:06}',active_seconds=201700.+21600*i)
        (root/f'{label}.json').write_text(json.dumps(record)); records[label]=record
    last=records['P80']
    (root/'training/latest.json').write_text(json.dumps(dict(iteration=last['outer_iteration'],state=last['recovery'])))
    report=dict(status='FAILED',checkpoints=records,stage_seconds=21600.,additional_active_seconds=48*3600,
        training_source_fingerprint='original',smoke=False,stages={'P80':dict(returncode=0)})
    path=root/'continuation.json';path.write_text(json.dumps(report))
    monkeypatch.setattr(continuation,'load_checkpoint_payload',lambda p:dict(
        outer_iteration=last['outer_iteration'],training_elapsed_seconds=last['active_seconds']))
    restored,index=continuation.resume_record(root,contract,records['P56'],'P80',last['sha256'],21600.,LABELS)
    assert index==4 and restored['additional_active_seconds']==48*3600
    report['additional_active_seconds']=24*3600
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='contract'):
        continuation.resume_record(root,contract,records['P56'],'P80',last['sha256'],21600.,LABELS)


def test_real_nine_policy_offline_evaluation(frozen_source, tmp_path):
    import shutil
    _, policy = frozen_source
    root=tmp_path/'nine-policies';root.mkdir()
    records={}
    for label in LABELS:
        shutil.copyfile(policy,root/f'{label}.pkl')
        record=dict(label=label,sha256=sha256_file(root/f'{label}.pkl'),outer_iteration=10,
                    active_seconds=1.,smoke=True)
        (root/f'{label}.json').write_text(json.dumps(record));records[label]=record
    (root/'continuation.json').write_text(json.dumps(dict(status='TRAINING_COMPLETE',
        final_recovery_verified=True,smoke=True,baseline_hours=56,stage_count=8,
        labels=list(LABELS),checkpoints=records)))
    result=evaluation.run_suite(root,tmp_path/'evaluation',workers=1,max_seconds=480,
                                smoke=True,opponent='check_call')
    assert result['status']=='COMPLETE' and len(result['head_to_head'])==9
    assert result['fo2']['completed_groups']==8
    assert result['fo2']['completed_hands_per_policy']==16
    assert len(result['fo2']['estimates'])==9 and len(result['fo2']['improvements'])==8
    assert result['fo2']['alpha']==.05/16
    assert all(not r['confirmatory'] for r in result['fo2']['improvements'].values())
    assert all(r['confidence']==1-.05/16 for r in result['fo2']['improvements'].values())
