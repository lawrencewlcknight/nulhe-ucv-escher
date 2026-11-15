"""Fixed-sample duplicate-deal comparisons, separate from the existing FO2 smoke CLI."""
import argparse
import atexit
import json
import math
import signal
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .checkpointing import LoadedHULHEPolicy, sha256_file
from .evaluate import AlwaysCallPolicy, UniformPolicy, play_hand
from .evaluation_stats import estimate, sample_plan
from .fell_omen2 import FellOmen2Policy
from .fo2_setup import DEFAULT_ASSETS
from .game import load_hulhe_game, serialisable_game_definition, SMALL_BET_CHIPS
from .train import write_json

_game = None
_policies = None
_initialisation_error = None


def load_policy(spec, game):
    kind = spec.get("kind", "checkpoint")
    if kind == "check_call":
        return AlwaysCallPolicy()
    if kind == "random":
        return UniformPolicy()
    if kind == "fo2":
        policy = FellOmen2Policy(game, spec["assets"], free_fold="check")
        atexit.register(policy.close)
        return policy
    if kind != "checkpoint":
        raise ValueError("Unknown policy kind")
    return LoadedHULHEPolicy(game, spec["checkpoint"])


def initialise(specs):
    global _game, _policies, _initialisation_error
    import torch
    from multiprocessing.util import Finalize
    torch.set_num_threads(1)
    _policies = {}
    def close():
        for policy in _policies.values():
            if hasattr(policy, "close"):
                policy.close()
    def terminate(signum, frame):
        close()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    Finalize(None, close, exitpriority=10)
    try:
        _game = load_hulhe_game()
        for label, spec in specs.items():
            _policies[label] = load_policy(spec, _game)
    except Exception as exc:
        # Do not let Pool repeatedly respawn an initializer that cannot load assets.
        _initialisation_error = f"{type(exc).__name__}: {exc}"
        close()


def duplicate_pair(game, policies, seed, pair_id, *, action_seed=None):
    """Cards stay in their seats; policies swap. The board cannot depend on betting."""
    deal = np.random.default_rng(np.random.SeedSequence([seed, pair_id, 314159])).choice(
        52, size=9, replace=False).tolist()
    legs, adaptations = [], 0
    for leg in (0, 1):
        seated = policies if leg == 0 else policies[::-1]
        record = play_hand(game, seated, deal, seed=seed if action_seed is None else action_seed,
                           pair=pair_id, leg=leg)
        legs.append(record["returns_chips"][leg] / SMALL_BET_CHIPS)
        adaptations += sum((a.get("opponent_lookup") or {}).get("legalization") ==
                           "explicit_free_fold_to_check" for a in record["actions"])
    return dict(legs_bb=legs, mean_bb=sum(legs)/2, adaptations=adaptations)


def batch(task):
    if _initialisation_error:
        raise RuntimeError(_initialisation_error)
    seed, start, count, matchups = task
    rows = []
    for pair_id in range(start, start+count):
        row = {"pair_id": pair_id}
        for label, a, b, stream in matchups:
            row[label] = duplicate_pair(_game, [_policies[a], _policies[b]], seed, pair_id,
                                       action_seed=seed+stream)
        rows.append(row)
    return rows


def collect(pool, *, count, seed, matchups, path, deadline, workers, rows=None):
    """Bounded dispatch, durable complete-pair journal, hard timeout on slow workers."""
    rows = [] if rows is None else rows
    with Path(path).open("x") as handle:
        for start in range(0, count, workers*32):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                break
            stop = min(count, start+workers*32)
            tasks = [(seed, i, min(8, stop-i), matchups) for i in range(start, stop, 8)]
            pending = pool.imap(batch, tasks)
            for _ in tasks:
                try:
                    group = pending.next(timeout=max(.001, deadline-time.monotonic()))
                except mp.TimeoutError:
                    return rows
                for row in group:
                    handle.write(json.dumps(row)+"\n")
                    rows.append(row)
                handle.flush()
    return rows


def validate_design(seed, workers, pilot_pairs, minimum, maximum, max_seconds, effect):
    if workers < 1 or pilot_pairs < 2 or seed < 0 or not 2 <= minimum <= maximum:
        raise ValueError("Invalid evaluation sample/worker/seed limits")
    if not all(math.isfinite(x) and x > 0 for x in (max_seconds, effect)):
        raise ValueError("Time and effect must be finite and positive")


def runtime_plan(pilot, rate, remaining, *, minimum, maximum, effect_bb100):
    # Do not silently lower the preregistered minimum when it will not fit.
    affordable = max(0, int(.8*rate*max(0, remaining)))
    plan = sample_plan(pilot, minimum=minimum, maximum=max(minimum, min(maximum, affordable)),
                       effect_bb100=effect_bb100)
    plan.update(pilot_pairs_per_second=rate, requested_maximum=maximum,
                affordable_pairs=affordable, minimum_fits_budget=affordable >= minimum)
    return plan


def run_evaluation(a, b, output, *, seed=8301, workers=8, pilot_pairs=10000,
                   minimum=100000, maximum=1000000, max_seconds=7200, effect_bb100=2.):
    validate_design(seed, workers, pilot_pairs, minimum, maximum, max_seconds, effect_bb100)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    specs = {"a": dict(kind="checkpoint", checkpoint=str(a)),
             "b": dict(kind=str(b)) if str(b) in ("random", "check_call") else
                  dict(kind="checkpoint", checkpoint=str(b))}
    identities = {k: dict(**v, sha256=sha256_file(v["checkpoint"]) if "checkpoint" in v else None)
                  for k, v in specs.items()}
    write_json(root / "manifest.json", dict(policies=identities, game=serialisable_game_definition(),
        seed=seed, workers=workers, pilot_pairs=pilot_pairs, minimum=minimum, maximum=maximum,
        max_seconds=max_seconds, effect_bb100=effect_bb100, unit="duplicate_pair_average_bb",
        pilot_seed=seed, confirmation_seed=seed+1))
    deadline = time.monotonic()+max_seconds
    report = dict(status="STARTING", result=None)
    try:
        with mp.get_context("spawn").Pool(workers, initializer=initialise, initargs=(specs,)) as pool:
            started = time.monotonic()
            pilot = collect(pool, count=pilot_pairs, seed=seed, matchups=[("score", "a", "b", 0)],
                            path=root / "pilot.jsonl", deadline=deadline, workers=workers)
            if len(pilot) != pilot_pairs:
                report.update(status="INCOMPLETE_PILOT", completed_pairs=len(pilot))
            else:
                plan = runtime_plan([r["score"]["mean_bb"] for r in pilot],
                    pilot_pairs/(time.monotonic()-started), deadline-time.monotonic(),
                    minimum=minimum, maximum=maximum, effect_bb100=effect_bb100)
                write_json(root / "plan.json", plan)
                report["plan"] = plan
                if not plan["minimum_fits_budget"]:
                    report.update(status="INSUFFICIENT_TIME_FOR_MINIMUM", completed_pairs=0)
                else:
                    main = collect(pool, count=plan["selected"], seed=seed+1,
                        matchups=[("score", "a", "b", 0)], path=root / "confirmatory.jsonl",
                        deadline=deadline, workers=workers)
                    complete = len(main) == plan["selected"]
                    report.update(status="COMPLETE" if complete else "INCOMPLETE_TIME_LIMIT",
                        completed_pairs=len(main), result=estimate([r["score"]["mean_bb"] for r in main],
                        complete=complete) if len(main) >= 2 else None)
    except Exception as exc:
        report.update(status="EVALUATION_ERROR", error_type=type(exc).__name__, reason=str(exc))
    finally:
        write_json(root / "report.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=8301)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pilot-pairs", type=int, default=10000)
    p.add_argument("--minimum", type=int, default=100000)
    p.add_argument("--maximum", type=int, default=1000000)
    p.add_argument("--max-seconds", type=float, default=7200)
    p.add_argument("--effect-bb100", type=float, default=2.)
    args = vars(p.parse_args(argv))
    result = run_evaluation(**args)
    print(json.dumps(result, indent=2))
    if result["status"] == "EVALUATION_ERROR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
