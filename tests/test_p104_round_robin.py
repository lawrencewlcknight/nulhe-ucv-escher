"""P104 identities, unchanged policies, fresh deals and independent Batch execution."""
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tarfile

import pytest

from test_p56_round_robin import pinned, frozen_source
from hulhe_escher import p56_round_robin as comparison, p56_round_robin_cloud as cloud
from hulhe_escher.checkpointing import load_checkpoint_payload, sha256_file

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def p104(pinned):
    files, policies = {}, {}
    for variant, iteration in [('Wide',3028), ('Deep',2802), ('Small',3489)]:
        label = variant+'-P104'
        payload = load_checkpoint_payload(pinned/(variant+'-P56.pkl'))
        payload.update(outer_iteration=iteration, training_elapsed_seconds=374800.)
        (pinned/(label+'.pkl')).write_bytes(pickle.dumps(payload))
        marker = dict(label='P104', outer_iteration=iteration, smoke=False, active_seconds=374800.,
            sha256=sha256_file(pinned/(label+'.pkl')),
            metrics=dict(policy_validation=dict(group_sampling='objective_mass')))
        (pinned/(label+'.json')).write_text(json.dumps(marker))
        policies[label] = dict(iteration=iteration, active_seconds=374800.)
        for suffix in ('json','pkl'):
            name = label+'.'+suffix
            files[name] = dict(uri='gs://test/source/'+name, sha256=sha256_file(pinned/name))
    comparison.source_file(104).write_text(json.dumps(dict(files=files, policies=policies)))
    return pinned


def test_production_pins_select_historic_small():
    sources = json.loads(comparison.source_file(104).read_text())
    assert [v['iteration'] for v in sources['policies'].values()] == [3028,2802,3489]
    assert sources['files']['Small-P104.pkl']['uri'].endswith(
        '/hulhe-p80-plus24-batch-20261120-171431/P104.pkl')
    assert len(sources['files']) == 6
    for name, pin in sources['files'].items():
        if name.endswith('.json'):
            run = pin['uri'].split('/')[-2]
            local = REPO/'outputs'/f'{run}-results'/'P104.json'
            if local.exists():
                assert sha256_file(local) == pin['sha256']
                assert json.loads(local.read_text())['sha256'] == sources['files'][name[:-5]+'.pkl']['sha256']


def test_p104_hashes_checked_before_pickle(p104, monkeypatch):
    specs, _ = comparison.verify_inputs(p104, checkpoint_hours=104)
    assert set(specs) == {'Wide-P104','Deep-P104','Small-P104'}
    (p104/'Small-P104.pkl').write_bytes(b'wrong')
    monkeypatch.setattr(comparison, 'load_checkpoint_payload', lambda p: pytest.fail('Must verify hashes first'))
    with pytest.raises(ValueError, match='checksum'):
        comparison.verify_inputs(p104, checkpoint_hours=104)


@pytest.mark.parametrize('change', ['label','sampler','smoke','hours'])
def test_p104_marker_validation(p104, change):
    path = p104/'Small-P104.json'; marker = json.loads(path.read_text())
    if change == 'label': marker['label'] = 'P56'
    elif change == 'sampler': marker['metrics']['policy_validation']['group_sampling'] = 'uniform'
    elif change == 'smoke': marker['experimental_smoke'] = True
    else: marker['active_seconds'] = 56*3600
    path.write_text(json.dumps(marker))
    sources = json.loads(comparison.source_file(104).read_text())
    sources['files'][path.name]['sha256'] = sha256_file(path)
    comparison.source_file(104).write_text(json.dumps(sources))
    with pytest.raises(ValueError, match='provenance'):
        comparison.verify_inputs(p104, checkpoint_hours=104)


def test_p104_actual_mixed_architecture_smoke(p104, tmp_path):
    before = {p:sha256_file(p) for p in p104.iterdir()}
    output = tmp_path/'evaluation'
    report = comparison.run_evaluation(p104, output, checkpoint_hours=104,
        workers=2, max_seconds=120, smoke=True)
    assert report['status'] == 'COMPLETE' and report['completed_groups'] == 8
    assert report['completed_hands'] == 48 and len(report['results']) == 3
    for result in report['results'].values():
        assert result['observations'] == [8] and not result['confirmatory']
        assert result['confidence'] == pytest.approx(1-.05/3)
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['pilot_seed'] == 928941 and manifest['confirmation_seed'] == 928942
    assert manifest['matchups'] == [list(row) for row in comparison.matchups_for(104)]
    assert 'P104 Wide / Deep' in (output/'comparison_report.md').read_text()
    assert all(sha256_file(p) == digest for p,digest in before.items())


@pytest.mark.parametrize('mode', ['complete','failure','timeout'])
def test_p104_cloud_routing_and_publication(p104, tmp_path, monkeypatch, mode):
    calls = []
    def storage(*args):
        calls.append(args)
        if args[0] == 'cp' and str(args[1]).startswith('gs://test/source/'):
            shutil.copyfile(p104/str(args[1]).rsplit('/',1)[1], args[2])
    def child(command, log, **kwargs):
        assert kwargs['timeout'] == 11100
        assert command[command.index('--checkpoint-hours')+1] == '104'
        assert command[command.index('--workers')+1] == '8'
        output = Path(command[command.index('--output')+1]); output.mkdir()
        (output/'report.json').write_text(json.dumps(dict(status='COMPLETE' if mode=='complete' else 'EVALUATION_ERROR')))
        return dict(returncode=int(mode!='complete'), timed_out=mode=='timeout')
    monkeypatch.setattr(cloud, 'storage', storage)
    monkeypatch.setattr(cloud, 'run_child', child)
    if mode == 'complete':
        assert cloud.run_cloud(tmp_path/'cloud', 'gs://test/result', 104)['status'] == 'COMPLETE'
    else:
        with pytest.raises((RuntimeError, TimeoutError)):
            cloud.run_cloud(tmp_path/'cloud', 'gs://test/result', 104)
    downloads = [args for args in calls if args[0]=='cp' and str(args[1]).startswith('gs://test/source/')]
    assert len(downloads) == 6 and all('P104' in str(args[1]) for args in downloads)
    assert calls[-2][0] == 'rsync' and calls[-1][-1] == 'gs://test/result/cloud_status.json'
    status = json.loads((tmp_path/'cloud/results/cloud_status.json').read_text())
    assert status['status'] == ('COMPLETE' if mode=='complete' else 'FAILED')
    for uri in ('gs://test/source','gs://test/source/child','gs://test'):
        with pytest.raises(ValueError, match='overlaps'):
            cloud.validate_output(uri, 104)


def test_p104_bundle_and_prepared_submission(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO/'tools'))
    import launch_p104_round_robin as launcher
    output = tmp_path/'bundle'
    plan = launcher.build_bundle(output, run_name='hulhe-p104-round-robin-test')
    assert plan['checkpoint_hours'] == 104 and plan['pilot_seed'] == 928041
    assert plan['confirmation_seed'] == 928042
    assert plan['primary'] == ['Wide-P104 vs Deep-P104','Wide-P104 vs Small-P104','Deep-P104 vs Small-P104']
    assert plan['alpha_per_test'] == .05/3 and plan['minimum_groups'] == 100000
    assert plan['maximum_groups'] == 1000000 and not plan['retraining'] and not plan['replay_download']
    task = json.loads((output/'job.json').read_text())['taskGroups'][0]['taskSpec']
    assert task['maxRunDuration'] == '14400s' and task['maxRetryCount'] == 0
    assert task['computeResource'] == dict(cpuMilli=16000, memoryMib=60000)
    assert '--checkpoint-hours 104' in (output/'batch.sh').read_text()
    subprocess.run(['bash','-n',str(output/'batch.sh')], check=True)
    with tarfile.open(output/'source.tar.gz') as archive:
        names = archive.getnames()
    assert 'hulhe_escher/data/p104_round_robin_sources.json' in names
    assert not any(n.endswith('.pkl') or '.env' in n or n.startswith('outputs/') for n in names)
    for pin in plan['sources']['files'].values():
        with pytest.raises(ValueError, match='overlaps'):
            launcher.build_bundle(tmp_path/'bad', run_name=pin['uri'].split('/')[-2])
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs:calls.append(command))
    launcher.submit(output)
    assert len(calls) == 4 and calls[0][-1] == plan['uri']+'/launch.json'
    assert all('--if-generation-match=0' in command for command in calls[:3])
    assert calls[-1][:5] == ['gcloud','batch','jobs','submit',plan['job_name']]
    (output/'source.tar.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        launcher.submit(output)
    assert len(calls) == 4
