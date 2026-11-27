"""Evaluation-only cloud boundaries, publication ordering and Batch packaging."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

import pytest

from hulhe_escher import sequential_p32_comparison_cloud as cloud
from hulhe_escher.checkpointing import sha256_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import launch_sequential_p32_comparison as launcher


@pytest.fixture
def pins(tmp_path, monkeypatch):
    inputs = tmp_path/'pinned'
    inputs.mkdir()
    files = {}
    for name, suffix in [('Sequential-P32.pkl', 'P32.pkl'), ('P32-fit.json', 'fit/experiment.json')]:
        path = inputs/name
        path.write_bytes(name.encode())
        files[name] = dict(uri='gs://test/source/'+suffix, sha256=sha256_file(path))
    source = tmp_path/'sources.json'
    source.write_text(json.dumps(dict(files=files)))
    monkeypatch.setattr(cloud, 'SOURCES', source)
    monkeypatch.setattr(launcher, 'SOURCES', source)
    return inputs, dict(files=files)


@pytest.mark.parametrize('mode', ['complete', 'incomplete', 'nonzero', 'timeout', 'missing'])
def test_cloud_only_pinned_inputs_and_status_last(pins, tmp_path, monkeypatch, mode):
    inputs, sources = pins
    calls, verified = [], []
    by_uri = {r['uri']: inputs/name for name, r in sources['files'].items()}
    workspace = tmp_path/'cloud'

    def storage(*args):
        if args[0] == 'cp' and str(args[1]).startswith('gs://'):
            assert args[1] in by_uri
            shutil.copyfile(by_uri[args[1]], args[2])
            calls.append(('download', args))
        elif args[0] == 'rsync':
            assert Path(args[2]) == workspace/'results'
            assert not any(p.suffix == '.pkl' for p in Path(args[2]).rglob('*'))
            status = json.loads((Path(args[2])/'cloud_status.json').read_text())
            calls.append(('results', status['status']))
        else:
            assert args[-1] == 'gs://test/new-run/cloud_status.json'
            calls.append(('status', json.loads(Path(args[1]).read_text())['status']))

    def verify(path):
        assert {p.name for p in path.iterdir()} == set(sources['files'])
        assert all(sha256_file(path/name) == record['sha256'] for name, record in sources['files'].items())
        verified.append(path)

    def child(command, log, **kw):
        assert command[command.index('--workers')+1] == '8'
        assert command[command.index('--max-seconds')+1] == '21600'
        assert command[command.index('--inputs')+1] == str(workspace/'inputs')
        assert kw['timeout'] == 21900
        output = Path(command[command.index('--output')+1])
        output.mkdir()
        if mode != 'missing':
            report = dict(status='INCOMPLETE_TIME_LIMIT' if mode == 'incomplete' else 'COMPLETE')
            (output/'report.json').write_text(json.dumps(report))
        return dict(returncode=int(mode == 'nonzero'), timed_out=mode == 'timeout')

    monkeypatch.setattr(cloud, 'storage', storage)
    monkeypatch.setattr(cloud, 'verify_inputs', verify)
    monkeypatch.setattr(cloud, 'run_child', child)
    if mode == 'complete':
        result = cloud.run_cloud(workspace, 'gs://test/new-run/')
        assert result['status'] == 'COMPLETE'
    else:
        error = TimeoutError if mode == 'timeout' else RuntimeError
        with pytest.raises(error):
            cloud.run_cloud(workspace, 'gs://test/new-run')
    assert len([c for c in calls if c[0] == 'download']) == len(sources['files'])
    assert len(verified) == (2 if mode == 'complete' else 1)
    assert calls[-2:] == [('results', 'UPLOADING'),
                         ('status', 'COMPLETE' if mode == 'complete' else 'FAILED')]


def test_bad_input_hash_fails_before_verification_or_child(pins, tmp_path, monkeypatch):
    calls = []

    def storage(*args):
        calls.append(args)
        if args[0] == 'cp' and str(args[1]).startswith('gs://'):
            Path(args[2]).write_bytes(b'corrupt')

    monkeypatch.setattr(cloud, 'storage', storage)
    monkeypatch.setattr(cloud, 'verify_inputs', lambda p: pytest.fail('checksum must precede deserialization'))
    monkeypatch.setattr(cloud, 'run_child', lambda *a, **k: pytest.fail('invalid inputs cannot run'))
    with pytest.raises(ValueError, match='checksum'):
        cloud.run_cloud(tmp_path/'cloud', 'gs://test/new-run')
    assert calls[-1][-1] == 'gs://test/new-run/cloud_status.json'
    status = json.loads((tmp_path/'cloud/results/cloud_status.json').read_text())
    assert status['status'] == 'FAILED'


def test_result_upload_failure_never_publishes_complete(pins, tmp_path, monkeypatch):
    inputs, sources = pins
    by_uri = {r['uri']: inputs/name for name, r in sources['files'].items()}
    calls = []

    def storage(*args):
        calls.append(args)
        if args[0] == 'rsync':
            assert json.loads((Path(args[2])/'cloud_status.json').read_text())['status'] == 'UPLOADING'
            raise RuntimeError('upload failed')
        if str(args[1]).startswith('gs://'):
            shutil.copyfile(by_uri[args[1]], args[2])

    def child(command, *args, **kw):
        output = Path(command[command.index('--output')+1])
        output.mkdir()
        (output/'report.json').write_text('{"status":"COMPLETE"}')
        return dict(returncode=0, timed_out=False)

    monkeypatch.setattr(cloud, 'storage', storage)
    monkeypatch.setattr(cloud, 'run_child', child)
    monkeypatch.setattr(cloud, 'verify_inputs', lambda p: None)
    with pytest.raises(RuntimeError, match='upload failed'):
        cloud.run_cloud(tmp_path/'cloud', 'gs://test/new-run')
    assert calls[-1][0] == 'rsync'


def test_source_overlap_and_invalid_output(pins):
    _, sources = pins
    for uri in ('gs://test/source', 'gs://test/source/', 'gs://test/source/child',
                'gs://test/source/fit/child', 'gs://test'):
        with pytest.raises(ValueError, match='overlaps'):
            cloud.validate_output(uri)
        with pytest.raises(ValueError, match='overlaps'):
            launcher.validate_output(uri, sources)
    for uri in ('/tmp/out', 'gs://test/unsafe;command', 'https://test/out'):
        with pytest.raises(ValueError, match='Invalid'):
            cloud.validate_output(uri)
        with pytest.raises(ValueError, match='Invalid'):
            launcher.validate_output(uri, sources)
    assert cloud.validate_output('gs://test/source-sibling/') == 'gs://test/source-sibling'


def test_offline_bundle_and_archive(tmp_path):
    output = tmp_path/'launch'
    plan = launcher.build_bundle(output, run_name='hulhe-sequential-p32-vs-all-test')
    assert len(plan['primary']) == 4 and plan['alpha_per_test'] == .0125
    assert plan['confidence'] == .9875 and plan['familywise_alpha'] == .05
    assert (plan['pilot_seed'], plan['confirmation_seed'], plan['smoke_seed']) == (927321, 927322, 927921)
    assert plan['hands_per_group'] == 10
    assert (plan['pilot_groups'], plan['minimum_groups'], plan['maximum_groups']) == (10000, 100000, 1000000)
    assert (plan['evaluation_seconds'], plan['process_guard_seconds'], plan['max_task_seconds']) == (21600, 21900, 25200)
    assert plan['workers'] == 8 and plan['max_task_hours'] == 7
    assert not any(plan[k] for k in ('collection', 'retraining', 'replay_download', 'fo2'))
    assert hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest() == plan['source_sha256']
    job = json.loads((output/'job.json').read_text())
    assert job['taskGroups'][0]['taskSpec']['maxRunDuration'] == '25200s'
    assert job['taskGroups'][0]['taskSpec']['maxRetryCount'] == 0
    vm = job['allocationPolicy']['instances'][0]['policy']
    assert vm['machineType'] == 'c4-standard-16' and vm['bootDisk']['sizeGb'] == 50
    subprocess.run(['bash', '-n', str(output/'batch.sh')], check=True)
    with tarfile.open(output/'source.tar.gz') as archive:
        names = archive.getnames()
    assert 'hulhe_escher/data/sequential_p32_comparison_sources.json' in names
    assert 'hulhe_escher/sequential_p32_comparison.py' in names
    assert 'hulhe_escher/sequential_p32_comparison_cloud.py' in names
    assert not any('.env' in n or n.startswith('outputs/') or n.endswith('.pkl') for n in names)
    with pytest.raises(ValueError, match='overlaps'):
        launcher.build_bundle(tmp_path/'bad', run_name='hulhe-sequential-p8-plus24-20261125-054522')


def test_prepared_submit_hash_and_create_only_uploads(tmp_path, monkeypatch):
    output = tmp_path/'launch'
    plan = launcher.build_bundle(output, run_name='hulhe-sequential-p32-vs-all-test')
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'run', lambda command, **kw: calls.append(command))
    launcher.submit(output)
    assert len(calls) == 4
    assert all(c[:4] == ['gcloud', 'storage', 'cp', '--if-generation-match=0'] for c in calls[:3])
    assert calls[0][-1] == plan['uri']+'/launch.json'
    assert calls[-1][:5] == ['gcloud', 'batch', 'jobs', 'submit', plan['job_name']]
    calls.clear()
    (output/'source.tar.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        launcher.submit(output)
    assert not calls


def test_documented_download_excludes_inputs_and_journals():
    document = (launcher.REPO/'docs/SEQUENTIAL_P32_COMPARISON.md').read_text()
    pattern = re.search(r"--exclude='([^']+)'", document).group(1)
    for name in ('launch.json', 'sources.json', 'cloud_status.json', 'evaluation/report.json',
                 'evaluation/plan.json', 'evaluation/manifest.json', 'evaluation/comparison_report.md'):
        assert not re.search(pattern, name)
    for name in ('source.tar.gz', 'inputs/P32.json', 'inputs/P32.pkl', 'evaluation/confirmation.jsonl',
                 'evaluation/pilot.jsonl', 'evaluation.log', 'state_123/manifest.json', 'training/report.json'):
        assert re.search(pattern, name)
