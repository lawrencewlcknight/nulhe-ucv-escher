"""Two predeclared tests of frozen policy refits, conditional on one replay source."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .checkpointing import sha256_file
from .evaluation_stats import estimate, sample_plan
from .fo2_setup import DEFAULT_ASSETS, verify_build
from .learning_evaluate import collect, initialise, validate_design
from .train import write_json

ALPHA = .05 / 2  # Head-to-head and improvement against FO2, one family.


def paired_observations(rows, seeds):
    """One independent unit is a deal group, not each fit seed or each hand."""
    return {
        "head_to_head": np.asarray([np.mean([r[f"h2h_{s}"]["mean_bb"] for s in seeds]) for r in rows]),
        "opponent_improvement": np.asarray([np.mean([
            r[f"objective_mass_{s}"]["mean_bb"] - r[f"uniform_{s}"]["mean_bb"]
            for s in seeds]) for r in rows]),
    }


def run_evaluation(experiment, output, *, assets=DEFAULT_ASSETS, opponent="fo2", workers=8,
                   pilot_pairs=10000, minimum=100000, maximum=1000000,
                   max_seconds=7200, effect_bb100=2., seed=471901):
    validate_design(seed, workers, pilot_pairs, minimum, maximum, max_seconds, effect_bb100)
    if opponent not in ("fo2", "check_call", "random"):
        raise ValueError("Invalid opponent")
    root, output = Path(experiment).resolve(), Path(output)
    design = json.loads((root / "experiment.json").read_text())
    if design["status"] != "COMPLETE":
        raise ValueError("Both refit arms must finish before evaluation")
    seeds = design["fit_seeds"]
    specs = {"opponent": dict(kind=opponent, assets=str(Path(assets).resolve()))}
    hashes = {}
    for s in seeds:
        initial_hashes = []
        for arm in ("uniform", "objective_mass"):
            matches = [r for r in design["results"] if r["arm"] == arm and r["fit_seed"] == s]
            if len(matches) != 1:
                raise ValueError("Incomplete or duplicate matched refit")
            row = matches[0]
            path = root / row["checkpoint"]
            if sha256_file(path) != row["sha256"]:
                raise ValueError("Refit checkpoint checksum mismatch")
            name = f"{arm}_{s}"
            specs[name] = dict(kind="checkpoint", checkpoint=str(path))
            hashes[name] = row["sha256"]
            initial_hashes.append(row["initial_model_sha256"])
        if len(set(initial_hashes)) != 1:
            raise ValueError("A/B networks did not start from identical weights")
    matchups = []
    for i, s in enumerate(seeds):
        control, candidate = f"uniform_{s}", f"objective_mass_{s}"
        matchups.extend([(f"h2h_{s}", candidate, control, i * 100 + 10),
                         (control, control, "opponent", i * 100 + 20),
                         (candidate, candidate, "opponent", i * 100 + 20)])
    output.mkdir(parents=True, exist_ok=False)
    smoke = bool(design["smoke"] or opponent != "fo2")
    report = dict(status="STARTING", smoke=smoke, results={}, completed_groups=0,
        alpha_per_test=ALPHA, fit_seeds=seeds, opponent=opponent,
        inference_scope="Conditional on these fixed refits from one training replay; not across independent self-play runs")
    write_json(output / "manifest.json", dict(experiment_sha256=sha256_file(root / "experiment.json"),
        policies=hashes, fit_seeds=seeds, pilot_seed=seed, confirmation_seed=seed + 1,
        pilot_pairs=pilot_pairs, minimum=minimum, maximum=maximum, max_seconds=max_seconds,
        workers=workers, effect_bb100=effect_bb100, alpha_per_test=ALPHA,
        tests=["head_to_head", "opponent_improvement"], matchups=matchups,
        unit="shared deal group, seat averaged then averaged across all fit seeds",
        opponent_variant="FO2 free-fold-to-check adaptation" if opponent == "fo2" else opponent,
        smoke=smoke, pilot_excluded=True))
    deadline, main = time.monotonic() + max_seconds, []
    try:
        build = verify_build(assets) if opponent == "fo2" else None
        report["opponent_build"] = build
        with mp.get_context("spawn").Pool(workers, initializer=initialise, initargs=(specs,)) as pool:
            tick = time.monotonic()
            pilot = collect(pool, count=pilot_pairs, seed=seed, matchups=matchups,
                path=output / "pilot.jsonl", deadline=deadline, workers=workers)
            if len(pilot) != pilot_pairs:
                report["status"] = "INCOMPLETE_PILOT"
            else:
                plans = {k: sample_plan(v, minimum=minimum, maximum=maximum,
                    effect_bb100=effect_bb100, alpha=ALPHA) for k, v in paired_observations(pilot, seeds).items()}
                rate = pilot_pairs / (time.monotonic() - tick)
                affordable = int(.8 * rate * max(0, deadline - time.monotonic()))
                required = max(p["required"] for p in plans.values())
                selected = min(maximum, affordable, max(minimum, required))
                report["plan"] = dict(tests=plans, selected=selected, required=required,
                    capped=selected < required, affordable=affordable, pilot_groups_per_second=rate,
                    pilot_excluded=True)
                write_json(output / "plan.json", report["plan"])
                if selected < minimum:
                    report["status"] = "INSUFFICIENT_TIME_FOR_MINIMUM"
                else:
                    collect(pool, count=selected, seed=seed + 1, matchups=matchups,
                        path=output / "confirmatory.jsonl", deadline=deadline, workers=workers, rows=main)
                    report["status"] = "COMPLETE" if len(main) == selected else "INCOMPLETE_TIME_LIMIT"
        if opponent == "fo2" and verify_build(assets) != build:
            report["status"] = "INVALID_OPPONENT_VERSION_CHANGE"
    except Exception as exc:
        report.update(status="EVALUATION_ERROR", error=f"{type(exc).__name__}: {exc}")
    finally:
        if len(main) >= 2:
            complete = report["status"] == "COMPLETE" and not smoke
            report["results"] = {k: estimate(v, alpha=ALPHA, complete=complete)
                                 for k, v in paired_observations(main, seeds).items()}
            report["descriptive_opponent_scores"] = {name: estimate(
                [r[name]["mean_bb"] for r in main], complete=False) for name in hashes}
        report["completed_groups"] = len(main)
        report["completed_hands"] = len(main) * 2 * len(matchups)
        write_json(output / "report.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--opponent", choices=("fo2", "check_call", "random"), default="fo2")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pilot-pairs", type=int, default=10000)
    p.add_argument("--minimum", type=int, default=100000)
    p.add_argument("--maximum", type=int, default=1000000)
    p.add_argument("--max-seconds", type=float, default=7200)
    p.add_argument("--effect-bb100", type=float, default=2.)
    p.add_argument("--seed", type=int, default=471901)
    report = run_evaluation(**vars(p.parse_args(argv)))
    print(json.dumps(report, indent=2))
    if report["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
