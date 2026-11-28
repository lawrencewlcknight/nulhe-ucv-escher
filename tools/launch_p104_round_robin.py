#!/usr/bin/env python3
"""Prepare the P104 Wide/Deep/historical Small round robin; submission is explicit."""
import argparse
import json
from pathlib import Path

from launch_p56_round_robin import build_bundle as _build_bundle
from launch_scaled_continuation_batch import submit


def build_bundle(output, *, run_name):
    return _build_bundle(output, run_name=run_name, checkpoint_hours=104)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--submit-prepared', action='store_true')
    args = parser.parse_args()
    if args.submit_prepared:
        if args.run_name or args.execute:
            parser.error('Use --submit-prepared with --output only')
        submit(args.output)
    else:
        if not args.run_name:
            parser.error('Preparing requires --run-name')
        print(json.dumps(build_bundle(args.output, run_name=args.run_name), indent=2))
        if args.execute:
            submit(args.output)


if __name__ == '__main__':
    main()
