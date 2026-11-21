#!/usr/bin/env python3
"""Build/submit a corrected P32, P56, P80 or P104 +24-hour Batch continuation."""
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
P56_SOURCE_URI = "gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-p32-plus24-batch-20261118-1105"
P56_SHA256 = "35b5348443678a2738d8ae52b1b313370739be90b1e0c298d3f73afd72adc30e"
P80_SOURCE_URI = "gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-p56-plus24-batch-20261119-144002"
P80_SHA256 = "822e1d499ce2688f362e46595220b341dbff6adf8546e9ca65a2e459f4020e8c"
P104_SOURCE_URI = "gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-p80-plus24-batch-20261120-171431"
P104_SHA256 = "7266950fc5fd4adfa37319e41bc69264e938e98c46754874c99e23483e442ad2"
AUDITED_BASELINES = {
    32: (SOURCE_URI, BASELINE_SHA256),
    56: (P56_SOURCE_URI, P56_SHA256),
    80: (P80_SOURCE_URI, P80_SHA256),
    104: (P104_SOURCE_URI, P104_SHA256),
}


def build_bundle(output, *, project, region, bucket, service_account, run_name, baseline_hours=32):
    if baseline_hours not in AUDITED_BASELINES:
        raise ValueError("Batch launch requires an audited P32, P56, P80 or P104 baseline")
    source_uri, baseline_sha256 = AUDITED_BASELINES[baseline_hours]
    labels = [f"P{baseline_hours + 6*i}" for i in range(5)]
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,61}[a-z0-9]", run_name):
        raise ValueError("Invalid Batch job name")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?", bucket):
        raise ValueError("Invalid output bucket prefix")
    uri = bucket.rstrip("/") + "/" + run_name
    if any(uri == source or uri.startswith(source + "/") or source.startswith(uri + "/")
           for source, _ in AUDITED_BASELINES.values()):
        raise ValueError("Replacement must have an independent output prefix")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], output / "source.tar.gz")
    digest = hashlib.sha256((output / "source.tar.gz").read_bytes()).hexdigest()
    command = ["python", "-m", "hulhe_escher.continuation_cloud", "--workspace", "/workspace/continuation",
        "--source-uri", source_uri, "--output-uri", uri, "--expected-baseline-sha256", baseline_sha256,
        "--baseline-hours", str(baseline_hours)]
    command += ["--corrected-sampler" if baseline_hours == 32 else "--require-corrected-sampler"]
    primary_minimum = 400000 if baseline_hours == 104 else 100000
    primary_effect = 1.0 if baseline_hours == 104 else 2.0
    command += ["--primary-minimum", str(primary_minimum), "--primary-effect-bb100", str(primary_effect)]
    script = bootstrap(command, uri, digest)
    job = job_spec(script, service_account, boot_disk_gb=200, max_run_seconds=158400,
        labels=dict(experiment=f"hulhe-p{baseline_hours}-plus24", sampler="objective-mass"))
    plan = dict(job_name=run_name, project=project, region=region, uri=uri, source_uri=source_uri,
        source_sha256=digest, baseline_sha256=baseline_sha256, baseline_hours=baseline_hours,
        machine="c4-standard-16", collection_workers=8, training_threads=8,
        labels=labels, additional_active_hours=24, stage_hours=6,
        policy_group_sampling="objective_mass", python_container=PYTHON_IMAGE, vm_image=VM_IMAGE,
        max_job_hours=44, training_wall_hours=30, evaluation_wall_hours=12,
        automatic_retries=0, retry_reason="Prevent silent repeated 24-hour training; full six-hour recovery remains in GCS",
        primary=f"{labels[-1]} vs {labels[0]}, alpha 0.05", secondary="Adjacent policies and paired Fell Omen 2 improvements, alpha 0.05/8",
        primary_minimum_pairs=primary_minimum, primary_maximum_pairs=1000000,
        primary_effect_bb100=primary_effect, pilot_pairs=10000, power=.8,
        primary_time_weight=primary_minimum/100000, secondary_effect_bb100=2.,
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
    parser.add_argument("--baseline-hours", type=int, choices=tuple(AUDITED_BASELINES), default=32)
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
