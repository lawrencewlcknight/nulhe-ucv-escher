#!/usr/bin/env python3
"""Prepare the pinned sequential P32 to P56 Batch continuation; submission is explicit."""
from launch_sequential_continuation_batch import build_bundle as _build_bundle, main


def build_bundle(output, *, run_name):
    return _build_bundle(output, run_name=run_name, baseline_hours=32)


if __name__ == '__main__':
    main(baseline_hours=32)
