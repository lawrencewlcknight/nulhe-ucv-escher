#!/usr/bin/env python3
"""Build the pinned P80 position audit and paired P56 comparison; submit with --execute."""
import argparse
import json
from pathlib import Path
import subprocess

from launch_position_investigation import build_bundle as position_bundle

SOURCE_URI = "gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-p56-plus24-batch-20261119-144002"
P80_SHA256 = "822e1d499ce2688f362e46595220b341dbff6adf8546e9ca65a2e459f4020e8c"
P56_AUDIT_URI = "gs://clever-overview-399515-fhp-escher-results/hulhe/hulhe-positions-p56-20261119-143942"
P56_POLICY_URI = SOURCE_URI + "/P56.pkl"
P56_SHA256 = "35b5348443678a2738d8ae52b1b313370739be90b1e0c298d3f73afd72adc30e"
P56_LIBRARY_SHA256 = "ee1e877034f84335282d87e620f1f0295a046c2efde0427cdc93cbe4f42c065e"


def build_bundle(output, *, project, region, bucket, service_account, run_name):
    return position_bundle(output, project=project, region=region, bucket=bucket,
        service_account=service_account, run_name=run_name,
        checkpoint_label="P80", source_uri=SOURCE_URI, iteration=2711,
        expected_policy_sha256=P80_SHA256, comparison_label="P56",
        comparison_uri=P56_AUDIT_URI, comparison_policy_uri=P56_POLICY_URI,
        comparison_policy_sha256=P56_SHA256, comparison_library_sha256=P56_LIBRARY_SHA256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="clever-overview-399515")
    parser.add_argument("--region", default="europe-west1")
    parser.add_argument("--bucket", default="gs://clever-overview-399515-fhp-escher-results/hulhe")
    parser.add_argument("--service-account", default="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com")
    parser.add_argument("--run-name", required=True)
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
