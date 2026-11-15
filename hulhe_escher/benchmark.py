"""Paired P4/P8 evaluation against a pinned, explicitly adapted Fell Omen 2."""
import argparse
import json
import math
import multiprocessing as mp
from importlib.metadata import version
import subprocess
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import norm

from .checkpointing import sha256_file
from .evaluation_stats import estimate, sample_plan
from .fo2_setup import DEFAULT_ASSETS, asset_manifest, bridge_fingerprint, verify_build
from .game import serialisable_game_definition
from .learning_evaluate import initialise, collect, validate_design
from .train import write_json


def wait_policy(root, label, deadline):
    while time.monotonic() < deadline:
        marker, policy = root / f"{label}.json", root / f"{label}.pkl"
        if marker.exists() and policy.exists():
            record = json.loads(marker.read_text())
            if sha256_file(policy) != record["sha256"]:
                raise ValueError(f"{label} policy hash differs from publication marker")
            return policy, record
        if (root / "training_failed.json").exists():
            raise RuntimeError("Training failed before policy publication")
        time.sleep(min(5, max(0, deadline-time.monotonic())))
    raise TimeoutError(f"No verified {label} before deadline")


def benchmark_plan(early, late, *, minimum, maximum, affordable, effect_bb100=2., half_width_bb100=2.):
    early, late = np.asarray(early, dtype=float), np.asarray(late, dtype=float)
    if early.shape != late.shape:
        raise ValueError("Benchmark pilots must contain matched pairs")
    difference = sample_plan(late-early, minimum=minimum, maximum=maximum, effect_bb100=effect_bb100)
    variance = {"P4": float(early.var(ddof=1)), "P8": float(late.var(ddof=1))}
    precision_n = {label: max(2, math.ceil(norm.ppf(.975)**2*v/(half_width_bb100/100)**2))
                   for label, v in variance.items()}
    required = max(difference["required"], *precision_n.values())
    selected = max(minimum, min(maximum, affordable, max(minimum, required)))
    return dict(required=required, selected=selected, capped=selected < required,
        difference_required=difference["required"], precision_required=precision_n,
        target_effect_bb100=effect_bb100, target_half_width_bb100=half_width_bb100,
        estimated_detectable_effect_bb100=100*(norm.ppf(.975)+norm.ppf(.8))*
            math.sqrt(difference["pilot_variance_bb2"]/selected),
        estimated_half_width_bb100={label: 100*norm.ppf(.975)*math.sqrt(v/selected)
                                    for label, v in variance.items()},
        minimum_fits_budget=affordable >= minimum, affordable_pairs=affordable,
        alpha=.05, power=.8, pilot_excluded=True, unit="shared_deal_four_hand_group")


def run_benchmark(experiment, output, *, assets=DEFAULT_ASSETS, seed=9501, workers=4,
                  pilot_pairs=10000, minimum=100000, maximum=1000000, max_seconds=86400,
                  effect_bb100=2., opponent="fo2"):
    validate_design(seed, workers, pilot_pairs, minimum, maximum, max_seconds, effect_bb100)
    if opponent not in ("fo2", "check_call", "random"):
        raise ValueError("Unknown opponent")
    root, output = Path(experiment), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic()+max_seconds
    report = dict(provider="fo2", status="WAITING", improvement=None, estimates={},
        primary_metric="raw_bb", opponent_variant="Fell Omen 2, free-fold-to-check adaptation"
            if opponent == "fo2" else "OFFLINE TEST: "+opponent,
        claim_scope="Secondary improvement claim requires primary P8-vs-P4 test to pass",
        alpha=.05, pilot_pairs=pilot_pairs, checkpoints={}, smoke=False)
    manifest = dict(seed=seed, workers=workers, pilot_seed=seed, confirmation_seed=seed+1,
        minimum=minimum, maximum=maximum, max_seconds=max_seconds, effect_bb100=effect_bb100,
        game=serialisable_game_definition(), python=platform.python_version(),
        runtime={name: version(name) for name in ("numpy", "torch", "open_spiel")},
        action_streams={"P4": 100, "P8": 200}, opponent_variant=report["opponent_variant"])
    main, pilot, specs, pilot_seconds = [], {}, {}, 0.
    def save():
        write_json(output / "report.json", report)
    save()
    try:
        if opponent == "fo2":
            manifest["fo2"] = dict(upstream_commit=asset_manifest()["commit"],
                bridge_fingerprint=bridge_fingerprint(), build=verify_build(Path(assets)),
                free_fold_rule="check", java=subprocess.check_output(
                    ["java", "-version"], stderr=subprocess.STDOUT, text=True, timeout=10))
        write_json(output / "manifest.json", manifest)
        specs["opponent"] = dict(kind=opponent, assets=str(Path(assets).resolve()))
        for label, stream in (("P4", 100), ("P8", 200)):
            policy, record = wait_policy(root, label, deadline)
            specs[label] = dict(kind="checkpoint", checkpoint=str(policy))
            report["checkpoints"][label] = record
            report["smoke"] |= record.get("smoke", False)
            report["status"] = "PILOT_"+label
            save()
            started = time.monotonic()
            with mp.get_context("spawn").Pool(workers, initializer=initialise, initargs=(specs,)) as pool:
                pilot[label] = collect(pool, count=pilot_pairs, seed=seed,
                    matchups=[(label, label, "opponent", stream)], path=output / f"pilot-{label}.jsonl",
                    deadline=deadline, workers=workers)
            pilot_seconds += time.monotonic()-started
            if len(pilot[label]) != pilot_pairs:
                raise TimeoutError("Incomplete pilot")
        # Both pilots used identical deal IDs, independently sampled policy actions.
        affordable = int(.8*pilot_pairs/pilot_seconds*max(0, deadline-time.monotonic()))
        plan = benchmark_plan(*[[r[label]["mean_bb"] for r in pilot[label]] for label in ("P4", "P8")],
            minimum=minimum, maximum=maximum, affordable=affordable, effect_bb100=effect_bb100)
        plan["pilot_groups_per_second"] = pilot_pairs/pilot_seconds
        report["plan"] = plan
        write_json(output / "plan.json", plan)  # Frozen before any confirmation deals.
        if not plan["minimum_fits_budget"]:
            report["status"] = "INSUFFICIENT_TIME_FOR_MINIMUM"
        else:
            report["status"] = "MAIN"
            save()
            with mp.get_context("spawn").Pool(workers, initializer=initialise, initargs=(specs,)) as pool:
                collect(pool, count=plan["selected"], seed=seed+1,
                    matchups=[("P4", "P4", "opponent", 100), ("P8", "P8", "opponent", 200)],
                    path=output / "confirmatory.jsonl", deadline=deadline, workers=workers, rows=main)
            report["status"] = "COMPLETE" if len(main) == plan["selected"] else "INCOMPLETE_TIME_LIMIT"
            if opponent == "fo2" and verify_build(Path(assets)) != manifest["fo2"]["build"]:
                report["status"] = "INVALID_OPPONENT_VERSION_CHANGE"
    except TimeoutError as exc:
        report.update(status="INCOMPLETE_TIME_LIMIT", reason=str(exc))
    except Exception as exc:
        report.update(status="EVALUATION_ERROR", error_type=type(exc).__name__, reason=str(exc))
    finally:
        usable = report["status"] == "COMPLETE" and opponent == "fo2" and not report["smoke"]
        scores = {label: [r[label]["mean_bb"] for r in main] for label in ("P4", "P8")}
        report["completed_groups"] = len(main)
        report["completed_hands_per_policy"] = 2*len(main)
        report["estimates"] = {label: estimate(x, complete=False) if len(x) >= 2 else None
                               for label, x in scores.items()}
        # Absolute scores are descriptive, not additional claims that we beat FO2.
        report["improvement"] = estimate(np.asarray(scores["P8"])-scores["P4"], complete=usable) if len(main) >= 2 else None
        report["adaptations"] = {label: sum(r[label]["adaptations"] for r in main) for label in scores}
        save()
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--seed", type=int, default=9501)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pilot-pairs", type=int, default=10000)
    p.add_argument("--minimum", type=int, default=100000)
    p.add_argument("--maximum", type=int, default=1000000)
    p.add_argument("--max-seconds", type=float, default=86400)
    result = run_benchmark(**vars(p.parse_args(argv)))
    print(json.dumps(result, indent=2))
    if result["status"] == "EVALUATION_ERROR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
