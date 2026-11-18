#!/usr/bin/env python3
"""Build/submit the corrected P32+24 Google Batch continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from launch_learning_check import source_archive

from batch_runtime import PYTHON_IMAGE, VM_IMAGE, bootstrap, job_spec

BASELINE_SHA256 = "9783cb191b46e22f759d7fe8215f342e4611ffbe05ad670525f057690d25a01b"
SOURCE_URI = "gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-plus24-20261116-063535"


def build_bundle(output, *, project, region, bucket, service_account, run_name):
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,61}[a-z0-9]", run_name):
        raise ValueError("Invalid Batch job name")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?", bucket):
        raise ValueError("Invalid output bucket prefix")
    uri = bucket.rstrip("/") + "/" + run_name
    if uri == SOURCE_URI or uri.startswith(SOURCE_URI + "/") or SOURCE_URI.startswith(uri + "/"):
        raise ValueError("Replacement must have an independent output prefix")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], output / "source.tar.gz")
    digest = hashlib.sha256((output / "source.tar.gz").read_bytes()).hexdigest()
    command = ["python", "-m", "hulhe_escher.continuation_cloud", "--workspace", "/workspace/continuation",
        "--source-uri", SOURCE_URI, "--output-uri", uri, "--expected-baseline-sha256", BASELINE_SHA256,
        "--baseline-hours", "32", "--corrected-sampler"]
    script = bootstrap(command, uri, digest)
    job = job_spec(script, service_account, boot_disk_gb=200, max_run_seconds=158400,
        labels=dict(experiment="hulhe-p32-plus24", sampler="objective-mass"))
    plan = dict(job_name=run_name, project=project, region=region, uri=uri, source_uri=SOURCE_URI,
        source_sha256=digest, baseline_sha256=BASELINE_SHA256,
        machine="c4-standard-16", collection_workers=8, training_threads=8,
        labels=["P32", "P38", "P44", "P50", "P56"], additional_active_hours=24,
        policy_group_sampling="objective_mass", python_container=PYTHON_IMAGE, vm_image=VM_IMAGE,
        max_job_hours=44, training_wall_hours=30, evaluation_wall_hours=12,
        automatic_retries=0, retry_reason="Prevent silent repeated 24-hour training; full six-hour recovery remains in GCS",
        primary="P56 vs P32, alpha 0.05", secondary="Adjacent policies and paired Fell Omen 2 improvements, alpha 0.05/8",
        cleanup="Batch releases its VM and boot disk; results and full recovery remain in GCS")
    for name, data in (("job.json", job), ("launch.json", plan)):
        (output / name).write_text(json.dumps(data, indent=2) + "\n")
    (output / "batch.sh").write_text(script)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="clever-overview-399515")
    parser.add_argument("--region", default="europe-west1")
    parser.add_argument("--bucket", default="gs://clever-overview-399515-fhp-escher-results/hulhe")
    parser.add_argument("--service-account", default="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args())
    execute = args.pop("execute")
    plan = build_bundle(**args)
    print(json.dumps(plan, indent=2))
    if execute:
        # Immutable creates also fail on existing prefixes; never overwrite a run.
        for name in ("source.tar.gz", "launch.json", "job.json"):
            subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0",
                str(args["output"] / name), plan["uri"] + "/" + name], check=True)
        subprocess.run(["gcloud", "batch", "jobs", "submit", plan["job_name"],
            "--project", plan["project"], "--location", plan["region"],
            "--config", str(args["output"] / "job.json")], check=True)


if __name__ == "__main__":
    main()
