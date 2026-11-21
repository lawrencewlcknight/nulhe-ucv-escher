"""Production evaluation sizing, gating, and cloud failure propagation."""
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from hulhe_escher import continuation_evaluate as evaluation
from hulhe_escher.evaluation_stats import estimate


def test_p104_production_design(tmp_path, monkeypatch):
    labels = ['P104', 'P110', 'P116', 'P122', 'P128']
    root = tmp_path / 'run'
    root.mkdir()
    (root / 'continuation.json').write_text(json.dumps(dict(
        status='TRAINING_COMPLETE', final_recovery_verified=True, smoke=False,
        baseline_hours=104, checkpoints={p: dict(sha256=p) for p in labels})))
    monkeypatch.setattr(evaluation, 'wait_policy', lambda root, label, deadline:
                        (root / f'{label}.pkl', dict(sha256=label)))
    clock = [0.]
    monkeypatch.setattr(evaluation.time, 'monotonic', lambda: clock[0])
    matches, fo2_calls = [], []

    def match(a, b, output, **kwargs):
        matches.append((a.stem, kwargs))
        clock[0] += kwargs['max_seconds']
        return dict(status='COMPLETE', result=estimate([-1., 1.]))

    def fo2(root, output, **kwargs):
        fo2_calls.append(kwargs)
        return dict(status='COMPLETE')

    monkeypatch.setattr(evaluation, 'run_evaluation', match)
    monkeypatch.setattr(evaluation, 'run_fo2', fo2)
    result = evaluation.run_suite(root, tmp_path / 'evaluation',
        primary_minimum=400000, primary_effect_bb100=1.)
    assert result['status'] == 'COMPLETE' and not result['primary_passed']
    primary = matches[0][1]
    assert primary['minimum'] == 400000 and primary['effect_bb100'] == 1.
    assert primary['alpha'] == .05 and primary['max_seconds'] == 4320
    assert primary['seed'] == 124301
    for _, secondary in matches[1:]:
        assert secondary.get('minimum', 100000) == 100000
        assert secondary['effect_bb100'] == 2. and secondary['alpha'] == .05 / 8
        assert secondary['max_seconds'] == 1080
    assert fo2_calls[0]['max_seconds'] == 34560
    assert fo2_calls[0]['seed'] == 125501 and not fo2_calls[0]['primary_passed']
    assert all(item['result']['conclusion'] == 'descriptive_primary_gate_not_passed'
               for item in list(result['head_to_head'].values())[1:])
    design = json.loads((tmp_path / 'evaluation/design.json').read_text())
    assert design['primary_minimum_pairs'] == 400000
    assert design['maximum_pairs'] == 1000000 and design['pilot_pairs'] == 10000
    assert design['primary_time_weight'] == 4. and design['power'] == .8


@pytest.mark.parametrize('primary_passed,complete,degenerate,conclusion', [
    (False, True, False, 'descriptive_primary_gate_not_passed'),
    (True, True, False, 'inconclusive'),
    (False, False, False, 'incomplete_or_degenerate'),
    (False, True, True, 'incomplete_or_degenerate'),
])
def test_fo2_completion_distinct_from_primary_gate(tmp_path, monkeypatch,
        primary_passed, complete, degenerate, conclusion):
    labels = ['P104', 'P110', 'P116', 'P122', 'P128']
    monkeypatch.setattr(evaluation, 'wait_policy', lambda root, label, deadline:
        (root / f'{label}.pkl', dict(sha256=label, outer_iteration=3489, active_seconds=374815)))
    monkeypatch.setattr(evaluation, 'verify_build', lambda _: 'pinned-build')
    monkeypatch.setattr(evaluation.mp, 'get_context', lambda _:
        SimpleNamespace(Pool=lambda *args, **kwargs: nullcontext(None)))

    def collect(pool, *, count, seed, rows=None, **kwargs):
        n = count if rows is None or complete else count - 1
        data = [{label: dict(mean_bb=0. if i == 0 or degenerate else (-1.) ** j,
                            adaptations=0) for i, label in enumerate(labels)} for j in range(n)]
        if rows is not None:
            rows.extend(data)
        return data

    monkeypatch.setattr(evaluation, 'collect', collect)
    report = evaluation.run_fo2(tmp_path, tmp_path / 'fo2', workers=1,
        pilot_pairs=4, minimum=8, maximum=8, max_seconds=60,
        labels=labels, primary_passed=primary_passed)
    assert report['status'] == ('COMPLETE' if complete else 'INCOMPLETE_TIME_LIMIT')
    for item in report['improvements'].values():
        assert item['conclusion'] == conclusion
        assert item['confirmatory'] == (primary_passed and complete and not degenerate)


def test_cloud_preserves_incomplete_evaluation_and_fails_job(tmp_path, monkeypatch):
    from hulhe_escher import continuation_cloud as cloud
    source, trainer = tmp_path / 'source', tmp_path / 'trainer'
    source.mkdir()
    (source / 'training_source.tar.gz').write_bytes(b'archived trainer')
    monkeypatch.setattr(cloud, 'download_source', lambda *args: (source, trainer))
    monkeypatch.setattr(cloud, 'storage', lambda *args: None)
    transfers = []
    monkeypatch.setattr(cloud, 'run_transfer', lambda *args, **kwargs: transfers.append(args))
    monkeypatch.setattr(cloud, 'prepare', lambda _: dict(fingerprint='fo2-build'))

    def train(source, trainer, output, **kwargs):
        output.mkdir(parents=True)
        return dict(training_source_fingerprint='trainer')

    def evaluate(root, output, **kwargs):
        assert kwargs['primary_minimum'] == 400000 and kwargs['primary_effect_bb100'] == 1.
        output.mkdir()
        (output / 'report.json').write_text('{"status":"EVALUATION_INCOMPLETE"}')
        return dict(status='EVALUATION_INCOMPLETE')

    monkeypatch.setattr(cloud, 'run_continuation', train)
    monkeypatch.setattr(cloud, 'run_suite', evaluate)
    workspace = tmp_path / 'workspace'
    with pytest.raises(RuntimeError, match='evaluation incomplete'):
        cloud.run_cloud(workspace, 'gs://test/source', 'gs://test/run', 'hash',
            baseline_hours=104, primary_minimum=400000, primary_effect_bb100=1.)
    assert (workspace / 'run/evaluation/report.json').exists()
    assert 'evaluation incomplete' in (workspace / 'run/cloud_failure.json').read_text()
    assert len(transfers) == 2  # Initial and mandatory final publication.


def test_bad_primary_design_fails_before_source_download(tmp_path, monkeypatch):
    from hulhe_escher import continuation_cloud as cloud
    def download(*args):
        pytest.fail('Invalid design must be rejected before cloud work')
    monkeypatch.setattr(cloud, 'download_source', download)
    with pytest.raises(ValueError):
        cloud.run_cloud(tmp_path, 'gs://test/source', 'gs://test/run', 'hash',
            baseline_hours=104, primary_minimum=1000001)
