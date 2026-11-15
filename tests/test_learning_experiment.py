import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hulhe_escher.checkpointing import load_checkpoint_payload
from hulhe_escher.evaluation_stats import estimate, sample_plan


def test_statistical_units_power_and_incomplete_claims():
    pilot = [-20., 20.]*5000
    plan = sample_plan(pilot, minimum=100000, maximum=1000000)
    assert 313000 < plan["selected"] < 315000
    independent = sample_plan(pilot, other=pilot, minimum=2, maximum=1000000)
    assert abs(independent["required"]-2*plan["required"]) <= 1
    capped = sample_plan(pilot, minimum=2, maximum=1000)
    assert capped["capped"] and capped["estimated_detectable_effect_bb100"] > 10
    assert estimate([1., 2.]*100)["conclusion"] == "improved"
    assert estimate([1., 2.]*100, complete=False)["confirmatory"] is False
    assert estimate([1., 1.]*100)["confirmatory"] is False
    result = estimate([2., 3.]*100, other=[1., 2.]*100, alpha=.025)
    assert result["bb_per_100"] == 100 and result["confidence"] == .975


def test_report_does_not_claim_strength_from_smoke(tmp_path):
    from hulhe_escher.experiment_report import render
    payload = dict(status="TRAINING_COMPLETE", smoke=True,
                   evaluations={"primary": {"status": "COMPLETE", "result": estimate([1., 2.]*100)}})
    (tmp_path / "experiment.json").write_text(json.dumps(payload))
    report = render(tmp_path)
    assert report["strength_evidence"].startswith("SMOKE ONLY")
    payload["smoke"] = False
    payload["evaluations"]["primary"]["result"] = estimate([-1., -2.]*100)
    (tmp_path / "experiment.json").write_text(json.dumps(payload))
    assert "regression" in render(tmp_path)["strength_evidence"]


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    output = tmp_path_factory.mktemp("learning") / "run"
    command = [sys.executable, "-m", "hulhe_escher.experiment", "--output", str(output),
               "--smoke", "--stage-seconds", ".001", "--training-wall-seconds", "180",
               "--evaluation-seconds", "180"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout[-2000:]+result.stderr[-5000:]
    return output


def test_real_restart_and_playable_checkpoints(experiment):
    report = json.loads((experiment / "experiment.json").read_text())
    assert report["status"] == "TRAINING_COMPLETE"
    a, b = [load_checkpoint_payload(experiment / f"{label}.pkl") for label in ("P4", "P8")]
    assert a["outer_iteration"] == 1 and b["outer_iteration"] == 2
    assert b["training_elapsed_seconds"] > a["training_elapsed_seconds"]
    assert len(list((experiment / "training").glob("state_*"))) == 1
    primary = report["evaluations"]["primary"]
    assert primary["status"] == "COMPLETE"
    assert primary["completed_pairs"] == 8
    assert len((experiment / "primary" / "pilot.jsonl").read_text().splitlines()) == 4
    assert all(item["status"] == "COMPLETE" for item in report["evaluations"].values())
    assert report["training_diagnostics"]["finite_exported_policy_weights"]
    assert report["training_diagnostics"]["finite_recorded_metrics"]
    assert report["training_diagnostics"]["exported_policy_changed"]
    assert (experiment / "learning_report.md").exists()


def test_multiprocess_evaluation_is_deterministic(experiment, tmp_path):
    from hulhe_escher.learning_evaluate import run_evaluation
    a, b = experiment / "P8.pkl", str(experiment / "P4.pkl")
    for workers in (1, 2):
        report = run_evaluation(a, b, tmp_path / str(workers), workers=workers,
                                pilot_pairs=4, minimum=8, maximum=8, max_seconds=60)
        assert report["status"] == "COMPLETE"
    for filename in ("pilot.jsonl", "confirmatory.jsonl"):
        assert (tmp_path / "1" / filename).read_bytes() == (tmp_path / "2" / filename).read_bytes()


def test_cloud_plan_is_bounded_and_source_archive_excludes_secrets(tmp_path):
    import importlib.util
    import tarfile
    script = Path(__file__).resolve().parents[1] / "tools" / "launch_learning_check.py"
    spec = importlib.util.spec_from_file_location("launch", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.build_bundle(tmp_path / "plan", project="test-project", zone="us-central1-a",
                bucket="gs://test-bucket/hulhe", service_account="runner@test-project.iam.gserviceaccount.com",
                run_name="test-run")
    assert len(plan["commands"]) == 3 and plan["training_active_hours"] == 8
    assert "15h" in plan["commands"][1] and "26h" in plan["commands"][2]
    for command in plan["commands"][1:]:
        assert "--no-restart-on-failure" in command and "STOP" in command
    with tarfile.open(tmp_path / "plan" / "source.tar.gz") as archive:
        names = archive.getnames()
    assert "hulhe_escher/experiment.py" in names
    assert any(name.endswith(".java") for name in names)
    assert "c4-standard-16" in plan["commands"][1]
    assert "50GB" in plan["commands"][2]
    assert plan["estimated_compute_ceiling_usd"] == pytest.approx(16.99962)
    assert not any(".env" in name or ".git" in name or "outputs/" in name for name in names)
    for role in ("train", "external"):
        subprocess.run(["bash", "-n", str(tmp_path / "plan" / f"{role}-startup.sh")], check=True)



def test_cloud_recovery_publication_prunes_only_after_success(tmp_path, monkeypatch):
    from hulhe_escher.experiment import upload_recovery
    training = tmp_path / "training"
    training.mkdir()
    (training / "latest.json").write_text(json.dumps({"state": "state_000002"}))
    (tmp_path / "cloud_checkpoint.json").write_text(json.dumps({"state": "state_000001"}))
    commands = []
    def run(command, **kwargs):
        commands.append(command)
    monkeypatch.setattr(subprocess, "run", run)
    upload_recovery(tmp_path, "gs://bucket/run")
    assert commands[0][2] == "rsync"
    assert commands[2][-1] == "gs://bucket/run/training/latest.json"
    assert commands[-1] == ["gcloud", "storage", "rm", "--recursive", "gs://bucket/run/training/state_000001"]
    (tmp_path / "cloud_checkpoint.json").write_text(json.dumps({"state": "state_000001"}))
    commands.clear()
    def fail(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        upload_recovery(tmp_path, "gs://bucket/run")
    assert len(commands) == 1
    assert json.loads((tmp_path / "cloud_checkpoint.json").read_text())["state"] == "state_000001"


def test_invalid_active_budget_rejected_before_training(tmp_path):
    for value in ("0", "-1", "nan", "inf"):
        result = subprocess.run([sys.executable, "-m", "hulhe_escher.train", "--smoke",
                  "--output", str(tmp_path / "unused"), "--additional-training-seconds", value],
                  capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
    assert not (tmp_path / "unused").exists()


def test_duplicate_cards_seats_and_rng():
    from hulhe_escher.learning_evaluate import duplicate_pair
    from hulhe_escher.evaluate import AlwaysCallPolicy, UniformPolicy
    from hulhe_escher.game import load_hulhe_game
    game = load_hulhe_game()
    before = np.random.get_state()
    for i in range(20):
        result = duplicate_pair(game, [AlwaysCallPolicy(), AlwaysCallPolicy()], 123, i)
        assert result["mean_bb"] == 0
    policies = [UniformPolicy(), AlwaysCallPolicy()]
    assert duplicate_pair(game, policies, 4, 2) == duplicate_pair(game, policies, 4, 2)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_benchmark_plan_uses_paired_variance_and_absolute_precision():
    from hulhe_escher.benchmark import benchmark_plan
    early = np.tile([-5., 5.], 5000)
    late = early + np.tile([.01, .03], 5000)
    plan = benchmark_plan(early, late, minimum=100000, maximum=1000000, affordable=1000000)
    assert plan["difference_required"] < 100
    assert plan["required"] > 240000  # Absolute-score precision dominates here.
    assert plan["alpha"] == .05 and plan["pilot_excluded"]
    short = benchmark_plan(early, late, minimum=100000, maximum=1000000, affordable=500)
    assert not short["minimum_fits_budget"]


def test_offline_benchmark_protocol(experiment, tmp_path):
    from hulhe_escher.benchmark import run_benchmark
    result = run_benchmark(experiment, tmp_path / "benchmark", opponent="check_call",
                           workers=2, pilot_pairs=4, minimum=8, maximum=8, max_seconds=60)
    assert result["status"] == "COMPLETE"
    assert result["completed_groups"] == 8
    assert result["completed_hands_per_policy"] == 16
    assert result["smoke"] and not result["improvement"]["confirmatory"]
    assert result["improvement"]["confidence"] == .95
    rows = [json.loads(line) for line in (tmp_path / "benchmark/confirmatory.jsonl").read_text().splitlines()]
    assert all("P4" in row and "P8" in row for row in rows)
    differences = [r["P8"]["mean_bb"]-r["P4"]["mean_bb"] for r in rows]
    assert result["improvement"]["bb_per_100"] == pytest.approx(100*np.mean(differences))


def test_benchmark_hash_mismatch_fails_closed(experiment, tmp_path):
    from hulhe_escher.benchmark import run_benchmark
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "P4.pkl").write_bytes((experiment / "P4.pkl").read_bytes())
    (broken / "P4.json").write_text(json.dumps({"sha256": "wrong"}))
    result = run_benchmark(broken, tmp_path / "out", opponent="check_call", max_seconds=10)
    assert result["status"] == "EVALUATION_ERROR" and result["improvement"] is None


def test_benchmark_wait_timeout(tmp_path):
    from hulhe_escher.benchmark import run_benchmark
    result = run_benchmark(tmp_path, tmp_path / "out", opponent="check_call", max_seconds=.01)
    assert result["status"] == "INCOMPLETE_TIME_LIMIT" and result["completed_groups"] == 0


def test_small_memory_probe(tmp_path):
    from hulhe_escher.memory_preflight import probe
    result = probe(tmp_path / "memory.json", small=True)
    assert result["checkpoint_roundtrip"] and not result["full_capacity"]


def test_secondary_claim_requires_primary_and_complete_non_smoke(tmp_path):
    from hulhe_escher.experiment_report import render
    payload = dict(status="TRAINING_COMPLETE", smoke=False, evaluations={})
    (tmp_path / "experiment.json").write_text(json.dumps(payload))
    external = tmp_path / "benchmarks/fo2"
    external.mkdir(parents=True)
    good = estimate([1., 2.]*100)
    (external / "report.json").write_text(json.dumps(dict(status="COMPLETE", improvement=good)))
    assert not render(tmp_path)["external"]["fo2"]["supports_overall_improvement_claim"]
    payload["evaluations"]["primary"] = dict(status="COMPLETE", result=good)
    (tmp_path / "experiment.json").write_text(json.dumps(payload))
    assert render(tmp_path)["external"]["fo2"]["supports_overall_improvement_claim"]


@pytest.mark.fo2
@pytest.mark.skipif(__import__("os").environ.get("RUN_FO2_TESTS") != "1", reason="opt-in real Java bridge")
def test_real_fo2_benchmark(experiment, tmp_path):
    from hulhe_escher.benchmark import run_benchmark
    result = run_benchmark(experiment, tmp_path / "fo2", workers=2, pilot_pairs=4,
                           minimum=8, maximum=8, max_seconds=90)
    assert result["status"] == "COMPLETE", result
    assert result["completed_groups"] == 8
    assert not result["improvement"]["confirmatory"]  # Smoke training remains smoke.


def test_runtime_plan_does_not_lower_preregistered_minimum():
    from hulhe_escher.learning_evaluate import runtime_plan
    plan = runtime_plan([-1., 1.]*100, 1., .1, minimum=100000,
                        maximum=1000000, effect_bb100=2.)
    assert plan["selected"] == 100000
    assert not plan["minimum_fits_budget"]


def test_benchmark_error_keeps_partial_results_descriptive(experiment, tmp_path, monkeypatch):
    import hulhe_escher.benchmark as benchmark
    original = benchmark.collect
    def fail_main(*args, **kwargs):
        if Path(kwargs["path"]).name == "confirmatory.jsonl":
            kwargs["rows"].extend([dict(pair_id=i, P4=dict(mean_bb=0., adaptations=0),
                P8=dict(mean_bb=float(i+1), adaptations=0)) for i in range(2)])
            raise RuntimeError("Injected opponent failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(benchmark, "collect", fail_main)
    result = benchmark.run_benchmark(experiment, tmp_path / "partial", opponent="check_call",
                workers=1, pilot_pairs=2, minimum=8, maximum=8, max_seconds=60)
    assert result["status"] == "EVALUATION_ERROR"
    assert result["completed_groups"] == 2
    assert result["improvement"]["bb_per_100"] == 150
    assert result["improvement"]["confirmatory"] is False


def test_publication_marks_smoke_and_verifies_policies(experiment):
    from hulhe_escher.checkpointing import sha256_file
    for label in ("P4", "P8"):
        marker = json.loads((experiment / f"{label}.json").read_text())
        assert marker["smoke"] and marker["sha256"] == sha256_file(experiment / f"{label}.pkl")


def test_no_limit_statistical_implementation_stays_aligned():
    parent = Path(__file__).resolve().parents[2]
    other = parent / "hunl-ucv-escher/hunl_escher/evaluation_stats.py"
    if not other.exists():
        pytest.skip("No sibling no-limit checkout")
    ours = parent / "hulhe-ucv-escher/hulhe_escher/evaluation_stats.py"
    assert ours.read_bytes() == other.read_bytes()
