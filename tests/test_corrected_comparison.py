"""Fixed-policy provenance, correlated-deal statistics, and bounded Batch execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from hulhe_escher import corrected_comparison as comparison
from hulhe_escher import corrected_comparison_cloud as cloud
from hulhe_escher.evaluation_stats import estimate


def test_statistics_average_correlated_refits_before_uncertainty():
    rows = [{'h2h_1701': {'mean_bb': a}, 'h2h_1702': {'mean_bb': b}}
            for a, b in [(1, 3), (2, 4), (-3, -1), (0, 2)]]
    observations = comparison.observations(rows)
    np.testing.assert_array_equal(observations, [2, 3, -2, 1])
    result = estimate(observations)
    assert result['observations'] == [4]  # Not 8 matchups or 16 hands.
    assert result['bb_per_100'] == 100
    assert result['standard_error_bb100'] == pytest.approx(100*np.std([2,3,-2,1], ddof=1)/2)


@pytest.fixture
def pinned_inputs(tmp_path, monkeypatch):
    root = tmp_path/'inputs'
    root.mkdir()
    fit = dict(status='COMPLETE', smoke=False, source_iteration=1132, results=[])
    files = {}
    for label in ('P56', 'P32-corrected-1701', 'P32-corrected-1702'):
        path = root/(label+'.pkl')
        path.write_bytes(label.encode())
        files[path.name] = dict(uri='gs://test/source/'+path.name,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for seed in comparison.SEEDS:
        fit['results'].append(dict(arm='objective_mass', fit_seed=seed,
            fit=dict(group_sampling='objective_mass'), sha256=files[f'P32-corrected-{seed}.pkl']['sha256']))
    final = dict(outer_iteration=1943, smoke=False, sha256=files['P56.pkl']['sha256'],
                 metrics=dict(policy_validation=dict(group_sampling='objective_mass')))
    for name, value in [('P32-fit.json', fit), ('P56.json', final)]:
        (root/name).write_text(json.dumps(value))
        files[name] = dict(uri='gs://test/source/'+name, sha256=hashlib.sha256((root/name).read_bytes()).hexdigest())
    sources = tmp_path/'sources.json'
    sources.write_text(json.dumps(dict(files=files, fit_seeds=list(comparison.SEEDS))))
    monkeypatch.setattr(comparison, 'SOURCES', sources)
    monkeypatch.setattr(cloud, 'SOURCES', sources)
    payload = {field: 'same' for field in ('game', 'policy_model', 'feature_encoder',
                                          'policy_network_layers', 'input_size', 'num_actions')}
    monkeypatch.setattr(comparison, 'load_checkpoint_payload', lambda path: payload.copy())
    return root


def test_provenance_and_hash_failure_before_deserialisation(pinned_inputs, monkeypatch):
    specs, _ = comparison.verify_inputs(pinned_inputs)
    assert set(specs) == {'P56', 'P32-corrected-1701', 'P32-corrected-1702'}
    (pinned_inputs/'P56.pkl').write_bytes(b'wrong')
    monkeypatch.setattr(comparison, 'load_checkpoint_payload', lambda p: pytest.fail('must check hashes first'))
    with pytest.raises(ValueError, match='checksum'):
        comparison.verify_inputs(pinned_inputs)


def test_architecture_mismatch_rejected(pinned_inputs, monkeypatch):
    monkeypatch.setattr(comparison, 'load_checkpoint_payload', lambda p: dict(game=str(p)))
    with pytest.raises(ValueError, match='matched on game'):
        comparison.verify_inputs(pinned_inputs)


class FakePool:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


@pytest.mark.parametrize('mode', ['complete', 'partial', 'error', 'smoke'])
def test_pilot_exclusion_and_incomplete_claims(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(comparison, 'verify_inputs', lambda p: ({}, {}))
    monkeypatch.setattr(comparison.mp, 'get_context', lambda _: type('Context', (), {'Pool': FakePool}))
    monkeypatch.setattr(comparison, 'runtime_plan', lambda *a, **kw: dict(selected=4,
        minimum_fits_budget=True, capped=False, estimated_detectable_effect_bb100=2.))
    calls = []
    def collect(pool, *, count, seed, rows=None, **kwargs):
        calls.append(seed)
        if rows is None:
            return [{'h2h_1701': {'mean_bb': 999}, 'h2h_1702': {'mean_bb': 999}} for _ in range(count)]
        for v in [1., 2., 3., 4.][:2 if mode in ('partial', 'error') else 4]:
            rows.append({'h2h_1701': {'mean_bb': v}, 'h2h_1702': {'mean_bb': v+2}})
        if mode == 'error': raise RuntimeError('worker failed')
        return rows
    monkeypatch.setattr(comparison, 'collect', collect)
    report = comparison.run_evaluation(tmp_path, tmp_path/'result', smoke=mode=='smoke')
    seed = comparison.SMOKE_SEED if mode == 'smoke' else comparison.DEAL_SEED
    assert calls == [seed, seed+1]
    assert report['result']['bb_per_100'] == (250 if mode in ('partial','error') else 350)
    assert report['result']['confirmatory'] == (mode == 'complete')
    assert all(not r['confirmatory'] for r in report['by_fit_seed'].values())
    assert report['completed_hands'] == 4*report['completed_groups']
    assert (tmp_path/'result/comparison_report.md').exists()


@pytest.mark.parametrize('failure', [False, True])
def test_cloud_only_small_inputs_uploads_status_last(pinned_inputs, tmp_path, monkeypatch, failure):
    calls = []
    def storage(*args):
        calls.append(args)
        if args[0] == 'cp' and str(args[1]).startswith('gs://test/source/'):
            shutil.copy2(pinned_inputs/str(args[1]).rsplit('/',1)[1], args[2])
    def child(command, log, **kwargs):
        output = Path(command[command.index('--output')+1])
        output.mkdir()
        (output/'report.json').write_text(json.dumps(dict(status='COMPLETE' if not failure else 'INCOMPLETE_TIME_LIMIT')))
        return dict(timed_out=False, returncode=0 if not failure else 1)
    monkeypatch.setattr(cloud, 'storage', storage)
    monkeypatch.setattr(cloud, 'run_child', child)
    if failure:
        with pytest.raises(RuntimeError, match='did not complete'):
            cloud.run_cloud(tmp_path/'cloud', 'gs://test/result')
    else:
        assert cloud.run_cloud(tmp_path/'cloud', 'gs://test/result')['status'] == 'COMPLETE'
    assert len([c for c in calls if str(c[1]).startswith('gs://test/source/')]) == 5
    assert calls[-2][0] == 'rsync'
    assert calls[-2][-2] == tmp_path/'cloud/results'
    assert calls[-1][-1] == 'gs://test/result/cloud_status.json'
    status = json.loads((tmp_path/'cloud/results/cloud_status.json').read_text())
    assert status['status'] == ('FAILED' if failure else 'COMPLETE')


def test_launch_bundle_and_source_isolation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'tools'))
    try:
        spec = importlib.util.spec_from_file_location('comparison_launcher', root/'tools/launch_corrected_comparison.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plan = module.build_bundle(tmp_path/'launch', run_name='hulhe-p56-vs-p32c-test')
        with pytest.raises(ValueError, match='overlaps'):
            module.build_bundle(tmp_path/'bad', run_name='hulhe-p32-plus24-batch-20261118-1105')
    finally:
        sys.path.pop(0)
    job = json.loads((tmp_path/'launch/job.json').read_text())
    task = job['taskGroups'][0]['taskSpec']
    assert task['maxRetryCount'] == 0 and task['maxRunDuration'] == '10800s'
    assert plan['workers'] == 8 and plan['fit_seeds'] == [1701, 1702]
    assert plan['retraining'] is False and plan['replay_download'] is False
    subprocess.run(['bash', '-n', str(tmp_path/'launch/batch.sh')], check=True)
    with tarfile.open(tmp_path/'launch/source.tar.gz') as archive:
        names = archive.getnames()
    assert 'hulhe_escher/data/corrected_comparison_sources.json' in names
    assert not any('.env' in n or n.startswith('outputs/') or '.git/' in n for n in names)
