"""Train HULHE UCV-ESCHER in the Fell Omen 2 fixed-limit game.

Bounded local training and resumable smoke runs; never submits cloud jobs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from importlib.metadata import version

from .config import baseline_config, smoke_config, make_solver
from .game import serialisable_game_definition
from .checkpointing import save_policy_checkpoint
from .checkpoint_schedule import CheckpointSchedule
from .training_state import build_training_state, restore_training_state, save_training_state, read_training_state


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for package in ("hulhe_escher", "unbiased_escher", "adaptive_escher", "vr_deep_cfr"):
        for path in sorted((root / package).rglob("*")):
            if path.suffix in (".py", ".json"):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")
    temp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--iterations", type=int, default=2, help="Total completed iterations, including restored ones")
    parser.add_argument("--workers", type=int, default=1, help="Use 8 for the inherited production collection layout")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Optional legacy iteration interval triggering both saves and policy fits")
    parser.add_argument("--recovery-seconds", type=float, default=3600,
                        help="Active learner seconds between full recovery saves (default: 3600)")
    parser.add_argument("--policy-seconds", type=float, default=21600,
                        help="Active learner seconds between playable-policy fits (default: 21600)")
    args = parser.parse_args(argv)
    if min(args.iterations, args.workers, args.threads) < 1:
        parser.error("iterations, workers and threads must be positive")
    if args.config and args.smoke:
        parser.error("Choose either a custom config or smoke")
    checkpoint_every = args.checkpoint_every if args.checkpoint_every is not None else (1 if args.smoke else None)
    if checkpoint_every is not None and checkpoint_every < 1:
        parser.error("checkpoint-every must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in (args.recovery_seconds, args.policy_seconds)):
        parser.error("recovery-seconds and policy-seconds must be finite and positive")
    config = (json.loads(args.config.read_text()) if args.config else
              smoke_config() if args.smoke else baseline_config())
    contract = {"config": config, "seed": args.seed, "workers": args.workers,
                "threads": args.threads, "checkpoint_every": checkpoint_every,
                "recovery_seconds": args.recovery_seconds, "policy_seconds": args.policy_seconds,
                "source_fingerprint": source_fingerprint(),
                "game": serialisable_game_definition(), "smoke": args.smoke,
                "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "open_spiel": version("open_spiel"),
                            "ray": version("ray") if args.workers > 1 else None}}
    root = args.output
    manifest = root / "manifest.json"
    if args.resume:
        if not manifest.exists() or json.loads(manifest.read_text()) != contract:
            raise ValueError("Resume requires the same source, configuration and runtime")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Use an empty output directory or --resume")
        root.mkdir(parents=True, exist_ok=True)
        write_json(manifest, contract)
    torch.set_num_threads(args.threads)
    solver = make_solver(config, seed=args.seed, workers=args.workers)
    try:
        kwargs = dict(seed=args.seed, repository_commit=contract["source_fingerprint"], config=config)
        latest = root / "latest.json"
        schedule = CheckpointSchedule()
        if args.resume:
            pointer = json.loads(latest.read_text())
            payload = read_training_state(root / pointer["state"])
            restore_training_state(solver, payload, **kwargs)
            schedule = CheckpointSchedule(**payload["checkpoint_schedule"])
        while solver.num_iteration < args.iterations:
            started = time.perf_counter()
            solver.iteration()
            learner_seconds = time.perf_counter() - started
            solver._resume_training_elapsed_seconds += learner_seconds
            iteration = solver.num_iteration
            active_seconds = solver._training_elapsed_seconds()
            recovery, policy = schedule.due(iteration=iteration, final=iteration == args.iterations,
                active_seconds=active_seconds, recovery_seconds=args.recovery_seconds,
                policy_seconds=args.policy_seconds, legacy_every=checkpoint_every)
            policy_seconds = 0.0
            if policy:
                rng = solver._capture_rng_state()
                started_policy = time.perf_counter()
                try:
                    solver.train_average_policy()
                finally:
                    solver._restore_rng_state(rng)
                policy_seconds = time.perf_counter() - started_policy
            row = {"iteration": iteration, "nodes": solver.nodes_touched, "trajectories": solver.episode,
                   "learner_seconds": learner_seconds, "policy_fit_seconds": policy_seconds,
                   "learning_seconds": learner_seconds + policy_seconds,
                   "training_elapsed_seconds": solver._training_elapsed_seconds(),
                   "policy_fitted": policy, "recovery_saved": recovery,
                   "policy_loss": solver._last_average_policy_loss if policy else None,
                   "policy_validation": getattr(solver.ave_policy_trainer, "last_fit_diagnostics", {}) if policy else None,
                   "regret_updates_per_player": config["advantage_network_train_steps"],
                   "regret_cache": [getattr(t, "last_target_cache_stats", {}) for t in solver.regret_trainers],
                   "critic_target_versions": [m.target_version for m in solver.q_value_trainer.members],
                   "critic_updates_per_member": config["baseline_network_train_steps"],
                   "critic_cache": [getattr(m, "last_target_cache_stats", {}) for m in solver.q_value_trainer.members]}
            persistence_started = time.perf_counter()
            schedule.completed(active_seconds, recovery=recovery, policy=policy)
            if policy:
                save_policy_checkpoint(solver, root / f"policy_{iteration:06d}.pkl",
                                       seed=args.seed, config=config, checkpoint_row=row)
            if recovery:
                state_name = f"state_{iteration:06d}"
                payload = build_training_state(solver, **kwargs, checkpoint_id=str(iteration), captured_checkpoints=[])
                payload["checkpoint_schedule"] = schedule.state()
                save_training_state(root / state_name, payload)
                write_json(latest, {"iteration": iteration, "state": state_name})
            row["persistence_seconds"] = time.perf_counter() - persistence_started
            row["iteration_wall_seconds"] = time.perf_counter() - started
            write_json(root / f"metrics_{iteration:06d}.json", row)
            print(json.dumps(row), flush=True)
    finally:
        if hasattr(solver, "close"):
            solver.close()


if __name__ == "__main__":
    main()
