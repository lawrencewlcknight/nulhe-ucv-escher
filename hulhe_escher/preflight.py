"""Bounded hardware profiling; discarded data, full network widths, no strength tuning."""
import argparse
import json
from pathlib import Path
import sys
import time

from .experiment_runtime import run_child
from .train import write_json


def profile_case(workers, threads, output):
    import torch
    from .config import baseline_config, make_solver
    from .train import learner_metrics
    config = baseline_config()
    for key in ("advantage_buffer_size", "ave_policy_buffer_size", "baseline_buffer_size", "calibration_buffer_size"):
        config[key] = 65536
    config["num_traversals"] = 1024
    config["average_policy_train_steps"] = 500
    config["policy_validation_steps"] = [250, 500]
    torch.set_num_threads(threads)
    solver = make_solver(config, seed=771, workers=workers)
    rows = []
    try:
        for _ in range(3):
            started = time.monotonic()
            solver.iteration()
            elapsed = time.monotonic()-started
            rows.append(dict(seconds=elapsed, **learner_metrics(solver, elapsed)))
        started = time.monotonic()
        solver.train_average_policy()
        result = dict(workers=workers, threads=threads, iterations=rows,
                      policy_500_updates_seconds=time.monotonic()-started,
                      config=config, full_capacity_certified=False)
        write_json(output, result)
    finally:
        solver.close() if hasattr(solver, "close") else None


def preflight(output, *, max_seconds=1800):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic()+max_seconds
    memory = run_child([sys.executable, "-m", "hulhe_escher.memory_preflight",
                        "--output", str(output / "memory.json")],
                       output / "memory.log", timeout=min(600, max_seconds))
    import psutil
    if memory["returncode"] != 0 or not (output / "memory.json").exists():
        raise RuntimeError("Memory preflight failed; inspect logs and use a larger VM if needed")
    # A conservative headroom gate; host memory avoids RSS shared-page double counting.
    if (not memory["monitoring_available"] or
            memory["peak_host_unavailable_bytes"] > .85*psutil.virtual_memory().total):
        raise RuntimeError("Memory preflight lacks verified 15% host headroom; review or use highmem")
    results = []
    for workers, threads in ((8, 4), (12, 4), (8, 2), (8, 8)):
        remaining = deadline-time.monotonic()
        if remaining < 10:
            break
        name = f"w{workers}-t{threads}"
        path = output / f"{name}.json"
        resource = run_child([sys.executable, "-m", "hulhe_escher.preflight",
                    "--case-workers", str(workers), "--case-threads", str(threads),
                    "--output", str(path)], output / f"{name}.log", timeout=min(270, remaining))
        if resource["returncode"] == 0 and path.exists():
            case = json.loads(path.read_text())
            # Warm iteration includes collection, sequential network fitting and merge.
            results.append(dict(workers=workers, threads=threads,
                                seconds=sum(r["seconds"] for r in case["iterations"][-2:])/2,
                                warm_seconds=[r["seconds"] for r in case["iterations"][-2:]], resources=resource))
    if not results:
        raise RuntimeError("No hardware preflight completed; inspect preflight logs before training")
    default = next((r for r in results if (r["workers"], r["threads"]) == (8, 4)), None)
    best = min(results, key=lambda r: r["seconds"])
    consistent_gain = default and all(b < .9*d for b, d in zip(best["warm_seconds"], default["warm_seconds"]))
    chosen = best if default is None or consistent_gain else default
    result = dict(memory_probe=memory, workers=chosen["workers"], threads=chosen["threads"], candidates=results,
                  selection="two warm iterations; require 10% gain in both to replace 8 workers/4 threads",
                  limitations="Short early-replay proxy, not a full-capacity or global VM optimum certification")
    write_json(output / "selected.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-workers", type=int)
    p.add_argument("--case-threads", type=int, default=4)
    args = p.parse_args(argv)
    if args.case_workers:
        profile_case(args.case_workers, args.case_threads, args.output)
    else:
        preflight(args.output)


if __name__ == "__main__":
    main()
