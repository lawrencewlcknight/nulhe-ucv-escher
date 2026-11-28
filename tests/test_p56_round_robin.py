"""Pinned P56 identities, paired inference, bounded evaluation and Batch packaging."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pytest
import torch

from test_policy_sampling_experiment import frozen_source
from hulhe_escher import p56_round_robin as comparison
from hulhe_escher import p56_round_robin_cloud as cloud
from hulhe_escher.checkpointing import load_checkpoint_payload, sha256_file
from hulhe_escher.features import make_feature_encoder, StructuredHULHEMLP
from hulhe_escher.deep_model import DeepResidualHULHEMLP
from hulhe_escher.scaling_config import scaling_config, deep_scaling_config


@pytest.fixture
def pinned(frozen_source, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    _, policy = frozen_source
    base = load_checkpoint_payload(policy)
    root = tmp_path/'inputs'
    root.mkdir()
    files, policies = {}, {}
    for label, iteration in zip(comparison.LABELS, (1703, 1516, 1943)):
        payload = deepcopy(base)
        if label != 'Small-P56':
            deep = label.startswith('Deep')
            config = deep_scaling_config(small=True) if deep else scaling_config(small=True)
            encoder = make_feature_encoder(config['feature_encoder_id'])
            cls = DeepResidualHULHEMLP if deep else StructuredHULHEMLP
            model = cls(encoder.policy_layout, config['average_policy_network_layers'], 3,
                        branch_width=config['structured_branch_width'])
            payload.update(training_config=config, policy_state_dict=model.state_dict(),
                policy_model=model.checkpoint_metadata(),
                policy_network_layers=config['average_policy_network_layers'])
        payload.update(outer_iteration=iteration, training_elapsed_seconds=201700.)
        with (root/(label+'.pkl')).open('wb') as f:
            pickle.dump(payload, f)
        marker = dict(label='P56', outer_iteration=iteration, smoke=False, active_seconds=201700.,
            sha256=sha256_file(root/(label+'.pkl')),
            metrics=dict(policy_validation=dict(group_sampling='objective_mass')))
        (root/(label+'.json')).write_text(json.dumps(marker))
        policies[label] = dict(iteration=iteration, active_seconds=201700.)
    for path in root.iterdir():
        files[path.name] = dict(uri='gs://test/source/'+path.name, sha256=sha256_file(path))
    sources = tmp_path/'sources.json'
    sources.write_text(json.dumps(dict(files=files, policies=policies)))
    monkeypatch.setattr(comparison, 'SOURCES', sources)
    return root


def test_hashes_checked_before_deserialisation(pinned, monkeypatch):
    specs, _ = comparison.verify_inputs(pinned)
    assert len(specs) == 3
    (pinned/'Deep-P56.pkl').write_bytes(b'wrong')
    monkeypatch.setattr(comparison, 'load_checkpoint_payload', lambda p: pytest.fail('hash check first'))
    with pytest.raises(ValueError, match='checksum'):
        comparison.verify_inputs(pinned)


@pytest.mark.parametrize('field,value', [('outer_iteration', 953), ('training_elapsed_seconds', 115200.),
                                         ('feature_encoder', {'wrong': True})])
def test_wrong_identity_or_contract_rejected(pinned, monkeypatch, field, value):
    original = comparison.load_checkpoint_payload
    def wrong(path):
        p = original(path)
        if path.name.startswith('Deep'):
            p[field] = value
        return p
    monkeypatch.setattr(comparison, 'load_checkpoint_payload', wrong)
    with pytest.raises(ValueError, match='mismatch|matched'):
        comparison.verify_inputs(pinned)


def test_uncorrected_export_rejected_even_with_matching_file_hash(pinned):
    path = pinned/'Small-P56.json'
    marker = json.loads(path.read_text())
    marker['metrics']['policy_validation']['group_sampling'] = 'uniform'
    path.write_text(json.dumps(marker))
    sources = json.loads(comparison.SOURCES.read_text())
    sources['files'][path.name]['sha256'] = sha256_file(path)
    comparison.SOURCES.write_text(json.dumps(sources))
    with pytest.raises(ValueError, match='provenance'):
        comparison.verify_inputs(pinned)


def rows(values):
    return [{name: dict(mean_bb=v*(i+1), legs_bb=[v*(i+1)-1, v*(i+1)+1])
             for i, (name, *_) in enumerate(comparison.MATCHUPS)} for v in values]


def test_three_matchups_and_correlated_seat_means():
    data = rows([1., 2., 3., 4.])
    np.testing.assert_array_equal(comparison.observations(data, 'Wide-vs-Deep'), [1, 2, 3, 4])
    plan = comparison.common_plan(data, 1000., 10000., minimum=100000, maximum=1000000)
    assert plan['selected'] == max(p['selected'] for p in plan['by_matchup'].values())
    assert plan['alpha_per_test'] == .05/3 and plan['familywise_alpha'] == .05
    assert all(p['alpha'] == .05/3 and p['pilot_excluded'] for p in plan['by_matchup'].values())
    short = comparison.common_plan(data, 1., 1., minimum=100000, maximum=1000000)
    assert not short['full_selected_sample_fits'] and short['selected'] == plan['selected']
    assert len({frozenset([a,b]) for _,a,b,_ in comparison.MATCHUPS}) == 3
    assert len({s for *_,s in comparison.MATCHUPS}) == 3
    capped = comparison.common_plan(rows([-100.,100.]), 1e9, 1e9, minimum=8, maximum=10)
    assert all(p['capped'] for p in capped['by_matchup'].values())


def test_actual_spawned_mixed_architecture_smoke(pinned, tmp_path):
    before = {p:sha256_file(p) for p in pinned.iterdir()}
    result = comparison.run_evaluation(pinned, tmp_path/'evaluation', workers=2, max_seconds=90, smoke=True)
    assert result['status'] == 'COMPLETE'
    assert result['completed_groups'] == 8 and result['completed_hands'] == 48
    assert len(result['results']) == 3
    for r in result['results'].values():
        assert r['observations'] == [8] and not r['confirmatory']
        assert r['confidence'] == pytest.approx(1-.05/3)
    assert all(sha256_file(p) == sha for p,sha in before.items())


class FakePool:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


@pytest.mark.parametrize('mode', ['complete', 'partial', 'error', 'pilot', 'budget'])
@pytest.mark.parametrize('checkpoint_hours', [56, 104])
def test_pilot_exclusion_and_no_partial_significance(tmp_path, monkeypatch, mode, checkpoint_hours):
    monkeypatch.setattr(comparison, 'verify_inputs', lambda p, **kw: ({}, {}))
    monkeypatch.setattr(comparison.mp, 'get_context', lambda _: type('Context', (), {'Pool':FakePool}))
    monkeypatch.setattr(comparison, 'common_plan', lambda *a, **kw: dict(selected=4,
        by_matchup={}, full_selected_sample_fits=mode!='budget'))
    seeds = []
    def collect(pool, *, seed, count, rows=None, **kw):
        seeds.append(seed)
        if rows is None:
            return globals()['rows']([999.]*(1 if mode=='pilot' else count))
        rows.extend(globals()['rows']([1.,2.,3.,4.][:2 if mode in ('error','partial') else 4]))
        if mode == 'error': raise RuntimeError('failure')
        return rows
    monkeypatch.setattr(comparison, 'collect', collect)
    report = comparison.run_evaluation(tmp_path, tmp_path/'out', checkpoint_hours=checkpoint_hours)
    if mode in ('pilot', 'budget'):
        assert not report['results'] and len(seeds)==1
    else:
        seed = comparison.DEAL_SEED if checkpoint_hours == 56 else 928041
        assert seeds == [seed, seed+1]
        r = report['results']['Wide-vs-Deep']
        assert r['bb_per_100'] == (150 if mode in ('partial','error') else 250)
        assert r['confirmatory'] == (mode=='complete')
        assert r['observations'] == ([2] if mode in ('partial','error') else [4])


@pytest.mark.parametrize('failure', [False, True])
def test_cloud_sources_and_status_publication(pinned, tmp_path, monkeypatch, failure):
    calls = []
    def storage(*args):
        calls.append(args)
        if args[0]=='cp' and str(args[1]).startswith('gs://test/source/'):
            shutil.copyfile(pinned/str(args[1]).rsplit('/',1)[1], args[2])
    def child(command, log, **kw):
        assert kw['timeout'] == 11100
        output = Path(command[command.index('--output')+1]); output.mkdir()
        (output/'report.json').write_text(json.dumps(dict(status='COMPLETE' if not failure else 'INCOMPLETE_TIME_LIMIT')))
        return dict(returncode=int(failure), timed_out=False)
    monkeypatch.setattr(cloud, 'storage', storage)
    monkeypatch.setattr(cloud, 'run_child', child)
    if failure:
        with pytest.raises(RuntimeError, match='did not complete'):
            cloud.run_cloud(tmp_path/'cloud', 'gs://test/results')
    else:
        assert cloud.run_cloud(tmp_path/'cloud', 'gs://test/results')['status']=='COMPLETE'
    assert len([c for c in calls if c[0]=='cp' and str(c[1]).startswith('gs://test/source/')])==6
    assert calls[-1][-1]=='gs://test/results/cloud_status.json'
    for uri in ['gs://test/source', 'gs://test/source/child', 'gs://test']:
        with pytest.raises(ValueError, match='overlaps'):
            cloud.validate_output(uri)


def test_launcher_and_source_pins(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
    import launch_p56_round_robin as launcher
    plan = launcher.build_bundle(tmp_path/'launch', run_name='hulhe-p56-round-robin-test')
    assert plan['alpha_per_test'] == .05/3 and len(plan['primary']) == 3
    assert plan['max_task_hours'] == 4 and plan['workers'] == 8 and not plan['retraining']
    assert len(plan['sources']['files']) == 6
    job = json.loads((tmp_path/'launch/job.json').read_text())
    assert job['taskGroups'][0]['taskSpec']['maxRunDuration'] == '14400s'
    assert job['taskGroups'][0]['taskSpec']['maxRetryCount'] == 0
    subprocess.run(['bash', '-n', str(tmp_path/'launch/batch.sh')], check=True)
    with tarfile.open(tmp_path/'launch/source.tar.gz') as archive:
        names = archive.getnames()
    assert 'hulhe_escher/data/p56_round_robin_sources.json' in names
    assert 'hulhe_escher/p56_round_robin.py' in names
    assert not any('.env' in n or n.startswith('outputs/') for n in names)
    with pytest.raises(ValueError, match='overlaps'):
        launcher.build_bundle(tmp_path/'bad', run_name='hulhe-deep-p32-plus24-20261124-112723')
