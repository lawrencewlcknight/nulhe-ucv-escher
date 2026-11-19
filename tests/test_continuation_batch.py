"""Real bounded migration/continuation plus Batch deployment contract checks."""
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from hulhe_escher.corrected_continuation import fingerprint, patch_trainer, rebind_metadata
from hulhe_escher.continuation import run_continuation
from hulhe_escher.continuation_cloud import extract_source
from hulhe_escher.experiment import publish_policy
from hulhe_escher.training_state import read_training_state

REPO = Path(__file__).resolve().parents[1]


def launcher():
    sys.path.insert(0, str(REPO / "tools"))
    try:
        spec = importlib.util.spec_from_file_location("batch_launcher", REPO / "tools/launch_continuation_batch.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b)
    else:
        assert a == b


def test_migration_preserves_all_learning_state_and_continues(tmp_path):
    module = launcher()
    module.source_archive(REPO, tmp_path / "source.tar.gz")
    archived = tmp_path / "legacy"
    extract_source(tmp_path / "source.tar.gz", archived)
    shutil.copyfile(REPO / "tests/fixtures/p32_legacy_holdem_fitting.py.txt",
                    archived / "unbiased_escher/holdem_fitting.py")
    (archived / "unbiased_escher/policy_sampling.py").unlink()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    env = {**os.environ, "PYTHONPATH": str(archived), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-m", "hulhe_escher.train", "--smoke", "--iterations", "1",
        "--workers", "1", "--threads", "1", "--seed", "123", "--policy-every", "0",
        "--policy-seconds", "1000000000", "--output", str(baseline / "training")],
        cwd=archived, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr[-4000:]
    publish_policy(baseline / "training", baseline, "P32")
    original = read_training_state(baseline / "training/state_000001")
    original_metadata = (baseline / "training/state_000001/metadata.pt").read_bytes()
    patched = tmp_path / "patched"
    shutil.copytree(archived, patched)
    before = {str(p.relative_to(patched)): p.read_bytes() for p in patched.rglob("*.py")}
    changes = patch_trainer(patched, 123)
    assert set(changes) == {"unbiased_escher/holdem_fitting.py", "unbiased_escher/policy_sampling.py"}
    for name, value in before.items():
        if name not in changes:
            assert (patched / name).read_bytes() == value
    migrated = tmp_path / "migrated"
    rebind_metadata(baseline, migrated, fingerprint(patched))
    state = read_training_state(migrated / "training/state_000001")  # verifies every shard hash
    assert (baseline / "training/state_000001/metadata.pt").read_bytes() == original_metadata
    assert original["repository_commit"] != state["repository_commit"]
    for key in original:
        if key != "repository_commit":
            equal(original[key], state[key])
    assert (migrated / "P32.pkl").read_bytes() == (baseline / "P32.pkl").read_bytes()
    report = run_continuation(migrated, patched, tmp_path / "continued", baseline_hours=32,
        stage_seconds=.001, wall_seconds=180, allow_smoke=True)
    assert report["status"] == "TRAINING_COMPLETE" and report["final_recovery_verified"]
    assert list(report["checkpoints"]) == ["P32", "P38", "P44", "P50", "P56"]
    for label in ("P38", "P44", "P50", "P56"):
        diag = report["checkpoints"][label]["metrics"]["policy_validation"]
        assert diag["group_sampling"] == "objective_mass"
        assert diag["policy_sampling_seed"] == [123, report["checkpoints"][label]["outer_iteration"]]
    # Chain the corrected full P56 state without another migration or reset.
    from hulhe_escher.continuation_evaluate import run_suite
    continued = tmp_path / "continued"
    original_p56 = (continued / "P56.pkl").read_bytes()
    next_run = tmp_path / "p56-plus24"
    uploads = []
    from unittest.mock import patch
    with patch("hulhe_escher.continuation.upload_checkpoint",
               side_effect=lambda root, uri, label: uploads.append(label)):
        next_report = run_continuation(continued, patched, next_run, baseline_hours=56,
            stage_seconds=.001, wall_seconds=180, allow_smoke=True,
            require_objective_mass=True, upload_uri="gs://test/p56")
    labels = ["P56", "P62", "P68", "P74", "P80"]
    assert next_report["labels"] == labels and next_report["final_recovery_verified"]
    assert uploads == labels
    assert [next_report["checkpoints"][p]["outer_iteration"] for p in labels] == [5, 6, 7, 8, 9]
    assert (next_run / "P56.pkl").read_bytes() == original_p56 == (continued / "P56.pkl").read_bytes()
    assert (continued / "training/state_000005").is_dir()
    assert json.loads((next_run / "training/manifest.json").read_text()) == json.loads((continued / "training/manifest.json").read_text())
    for label in labels[1:]:
        assert next_report["checkpoints"][label]["metrics"]["policy_validation"]["group_sampling"] == "objective_mass"
    evaluation = run_suite(next_run, tmp_path / "evaluation", workers=1,
        max_seconds=240, smoke=True, opponent="check_call")
    assert evaluation["status"] == "COMPLETE"
    assert list(evaluation["head_to_head"]) == ["P80-vs-P56", "P62-vs-P56", "P68-vs-P62", "P74-vs-P68", "P80-vs-P74"]
    assert set(evaluation["fo2"]["estimates"]) == set(labels)
    assert set(evaluation["fo2"]["improvements"]) == set(labels[1:])
    design = json.loads((tmp_path / "evaluation/design.json").read_text())
    assert design["head_to_head_pilot_seeds"] == [76301, 76311, 76321, 76331, 76341]
    assert design["fo2_pilot_seed"] == 77501
    with pytest.raises(ValueError, match="checksum"):
        patch_trainer(patched, 123)


@pytest.mark.parametrize("baseline_hours", [32, 56])
def test_batch_contract_and_shell(tmp_path, baseline_hours):
    module = launcher()
    bundle = tmp_path / "bundle"
    plan = module.build_bundle(bundle, project="test", region="europe-west1", bucket="gs://test/hulhe",
        service_account="runner@test.iam.gserviceaccount.com", run_name="hulhe-batch-test",
        baseline_hours=baseline_hours)
    job = json.loads((bundle / "job.json").read_text())
    spec = job["taskGroups"][0]["taskSpec"]
    assert spec["maxRetryCount"] == 0 and spec["maxRunDuration"] == "158400s"
    assert spec["computeResource"] == dict(cpuMilli=16000, memoryMib=60000)
    assert job["allocationPolicy"]["instances"][0]["policy"]["bootDisk"]["type"] == "hyperdisk-balanced"
    assert "batch-cos" in plan["vm_image"] and "@sha256:" in plan["python_container"]
    assert plan["policy_group_sampling"] == "objective_mass"
    assert plan["labels"] == [f"P{baseline_hours+6*i}" for i in range(5)]
    assert plan["primary"] == f"P{baseline_hours+24} vs P{baseline_hours}, alpha 0.05"
    assert job["labels"]["experiment"] == f"hulhe-p{baseline_hours}-plus24"
    script = (bundle / "batch.sh").read_text()
    assert "shutdown" not in script
    if baseline_hours == 32:
        assert "--corrected-sampler" in script
    else:
        assert "--require-corrected-sampler" in script and "--corrected-sampler" not in script
        assert module.P56_SHA256 in script and module.P56_SOURCE_URI in script
    assert "pip==25.3" in script and "--no-deps --only-binary=:all: torch==2.7.0+cpu" in script
    assert "APT::Update::Error-Mode=any" in script
    assert "free -m || true" in script and "python3-venv procps" in script
    subprocess.run(["bash", "-n", str(bundle / "batch.sh")], check=True)


def test_corrected_trainer_guard(tmp_path, monkeypatch):
    from hulhe_escher import corrected_continuation as corrected
    source, trainer = tmp_path / "source", tmp_path / "trainer"
    (source / "training").mkdir(parents=True)
    trainer.mkdir()
    contract = dict(source_fingerprint=corrected.CORRECTED_SOURCE, smoke=False,
                    workers=8, threads=8, seed=20261115)
    manifest = source / "training/manifest.json"
    manifest.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="audited objective_mass"):
        corrected.verify_corrected_source(source, trainer)
    monkeypatch.setattr(corrected, "fingerprint", lambda _: corrected.CORRECTED_SOURCE)
    assert corrected.verify_corrected_source(source, trainer)["source_modified"] is False
    contract["seed"] = 123
    manifest.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="production contract"):
        corrected.verify_corrected_source(source, trainer)


def test_p56_output_cannot_overlap_source(tmp_path):
    module = launcher()
    with pytest.raises(ValueError, match="independent output"):
        module.build_bundle(tmp_path / "bundle", project="test", region="europe-west1",
            bucket=module.P56_SOURCE_URI, service_account="test", run_name="nested-run",
            baseline_hours=56)
    assert not (tmp_path / "bundle").exists()
