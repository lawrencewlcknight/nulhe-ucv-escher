#!/usr/bin/env python3
"""Build a P32 bottleneck Google Batch job; submit only with --execute."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from batch_runtime import PYTHON_IMAGE, VM_IMAGE, bootstrap, job_spec
from launch_learning_check import source_archive


def build_bundle(output, *, project, bucket, service_account, run_name, source_uri,
                 expected_policy_sha256, region="europe-west1", iteration=1132):
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,61}[a-z0-9]", run_name):
        raise ValueError("Invalid Batch job name")
    for uri in (bucket, source_uri):
        if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?", uri):
            raise ValueError("Invalid GCS bucket/prefix")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_policy_sha256):
        raise ValueError("Expected policy SHA-256 must contain 64 lowercase hex digits")
    if iteration <= 0:
        raise ValueError("Checkpoint iteration must be positive")
    source_uri = source_uri.rstrip("/")
    uri = bucket.rstrip("/") + "/" + run_name
    if uri == source_uri or uri.startswith(source_uri + "/") or source_uri.startswith(uri + "/"):
        raise ValueError("Investigation must have an independent output prefix")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], output / "source.tar.gz")
    digest = hashlib.sha256((output / "source.tar.gz").read_bytes()).hexdigest()
    command = ["python", "-m", "hulhe_escher.bottleneck_cloud", "--workspace", "/workspace/bottleneck",
        "--source-uri", source_uri, "--output-uri", uri,
        "--expected-policy-sha256", expected_policy_sha256, "--iteration", str(iteration)]
    script = bootstrap(command, uri, digest)
    job = job_spec(script, service_account, boot_disk_gb=150, max_run_seconds=14400,
        labels=dict(experiment="hulhe-bottleneck", sampler="objective-mass"))
    commands = [["gcloud", "storage", "cp", "--if-generation-match=0",
        str(output / name), uri + "/" + name] for name in ("source.tar.gz", "launch.json", "job.json")]
    commands.append(["gcloud", "batch", "jobs", "submit", run_name, "--project", project,
        "--location", region, "--config", str(output / "job.json")])
    plan = dict(experiment="hulhe_bottleneck_v1", deployment="google_batch",
        job_name=run_name, project=project, region=region, uri=uri, source_uri=source_uri,
        expected_policy_sha256=expected_policy_sha256, iteration=iteration, source_sha256=digest,
        machine="c4-standard-16", boot_disk_gb=150, collection_workers=8,
        evaluation_workers=8, fitting_threads=8, fit_seeds=[2701, 2702],
        arms=["corrected_current_lr", "corrected_low_lr", "residual_low_lr"],
        policy_group_sampling="objective_mass", python_container=PYTHON_IMAGE, vm_image=VM_IMAGE,
        evaluation_max_seconds=3600, max_task_hours=4,
        upstream_guard_seconds=900, refit_guard_seconds=3600, automatic_retries=0,
        data="Exactly one isolated full P32 recovery; source run read-only. Target samples capped at 4096 per learner.",
        workload="One production iteration, six 20000-update supervised refits, one selected-policy evaluation",
        cleanup="Batch releases its VM and boot disk; diagnostic results remain in GCS",
        selection="Mean validation KL across seeds; first seed policy from winning arm; untouched audit; fresh paired deals",
        commands=commands)
    (output / "batch.sh").write_text(script)
    for name, data in (("job.json", job), ("launch.json", plan)):
        (output / name).write_text(json.dumps(data, indent=2) + "\n")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "bucket", "service-account", "run-name", "source-uri", "expected-policy-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--region", default="europe-west1")
    parser.add_argument("--iteration", type=int, default=1132)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args(argv))
    execute = args.pop("execute")
    plan = build_bundle(**args)
    print(json.dumps(plan, indent=2))
    if execute:
        for command in plan["commands"]:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
