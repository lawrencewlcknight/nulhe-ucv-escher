"""Offline checks for bounded launch and mandatory result publication."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from hulhe_escher import policy_sampling_cloud as cloud


def launcher():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    try:
        spec = importlib.util.spec_from_file_location("sampler_launcher", root / "tools/launch_policy_sampler.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_launch_is_bounded_and_archive_contains_only_source(tmp_path):
    plan = launcher().build_bundle(tmp_path / "launch", project="test-project", zone="europe-west1-b",
        bucket="gs://test-results/hulhe", service_account="runner@test-project.iam.gserviceaccount.com",
        run_name="hulhe-sampler-test", source_uri="gs://test-results/old-run",
        expected_policy_sha256="a" * 64)
    command = plan["commands"][-1]
    assert command[command.index("--max-run-duration") + 1] == "6h"
    assert command[command.index("--machine-type") + 1] == "c4-standard-16"
    assert command[command.index("--boot-disk-size") + 1] == "50GB"
    assert all("--if-generation-match=0" in c for c in plan["commands"][:-1])
    script = tmp_path / "launch/startup.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert "policy_sampling_cloud" in script.read_text()
    assert "shutdown -h now" in script.read_text()
    with tarfile.open(tmp_path / "launch/source.tar.gz") as archive:
        names = archive.getnames()
    assert "hulhe_escher/policy_sampling_cloud.py" in names
    assert not any("outputs/" in n or ".env" in n or ".git/" in n for n in names)


@pytest.mark.parametrize("evaluation_status", ["COMPLETE", "INCOMPLETE_TIME_LIMIT"])
def test_cloud_stages_and_uploads_only_results(tmp_path, monkeypatch, evaluation_status):
    transfers, fits = [], []

    def transfer(command, **kwargs):
        transfers.append(command)
        if command[2] == "cp" and command[3].endswith("training/policy_001132.pkl"):
            Path(command[4]).write_bytes(b"source-policy")

    def fit(state, policy, output, **kwargs):
        fits.append(kwargs)
        output.mkdir()
        (output / "experiment.json").write_text('{}')

    monkeypatch.setattr(cloud, "run_transfer", transfer)
    monkeypatch.setattr(cloud, "prepare", lambda *args: None)
    monkeypatch.setattr(cloud, "prepare_fo2", lambda *args: {"fingerprint": "test"})
    monkeypatch.setattr(cloud, "run_fit", fit)
    monkeypatch.setattr(cloud, "run_evaluation", lambda *args, **kwargs: {"status": evaluation_status})
    result = cloud.run_cloud(tmp_path / "work", "gs://test/source", "gs://test/result",
                            hashlib.sha256(b"source-policy").hexdigest())
    assert result["status"] == ("COMPLETE" if evaluation_status == "COMPLETE" else "EVALUATION_" + evaluation_status)
    assert fits == [{"seeds": (1701, 1702), "threads": 8}]
    assert all(c[-2] == str(tmp_path / "work/results") for c in transfers if c[2] == "rsync")
    assert transfers[-1][-1] == "gs://test/result/cloud_status.json"


def test_bad_source_hash_prevents_fit_and_publishes_failure(tmp_path, monkeypatch):
    transfers = []

    def transfer(command, **kwargs):
        transfers.append(command)
        if command[2] == "cp" and command[3].endswith("training/policy_001132.pkl"):
            Path(command[4]).write_bytes(b"bad-policy")

    monkeypatch.setattr(cloud, "run_transfer", transfer)
    with pytest.raises(ValueError, match="SHA-256"):
        cloud.run_cloud(tmp_path / "work", "gs://test/source", "gs://test/result", "a" * 64)
    result = json.loads((tmp_path / "work/results/cloud_status.json").read_text())
    assert result["status"] == "FAILED"
    assert transfers[-1][-1] == "gs://test/result/cloud_status.json"
