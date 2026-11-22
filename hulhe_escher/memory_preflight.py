"""Synthetic full-replay grouping and recovery probe; never supplies training data."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch

from .config import baseline_config, smoke_config, make_solver, validate_config
from .training_state import build_training_state, save_training_state, read_training_state, restore_training_state
from .train import write_json
from unbiased_escher.replay_grouping import learner_replay_allocation


def probe(output, *, small=False, config=None, workers=1, threads=4):
    import psutil
    if not small and psutil.virtual_memory().total < 48*1024**3:
        raise RuntimeError("Full memory preflight requires a VM with at least 48 GiB RAM")
    config = dict(baseline_config() if config is None else config)
    validate_config(config)
    if small:
        tiny = smoke_config()
        for key in ("num_traversals", "advantage_network_train_steps", "baseline_network_train_steps",
                    "calibration_train_steps", "critic_min_train_steps", "calibration_min_train_steps",
                    "advantage_batch_size", "baseline_batch_size", "calibration_batch_size"):
            config[key] = tiny[key]
        for key in ("advantage_buffer_size", "ave_policy_buffer_size", "baseline_buffer_size", "calibration_buffer_size"):
            config[key] = 64
        config["ave_policy_batch_size"] = 8
    config["average_policy_train_steps"] = 2 if small else 500
    config["policy_validation_steps"] = [1, 2] if small else [250, 500]
    torch.set_num_threads(threads)
    solver = make_solver(config, seed=771, workers=workers)
    try:
        # Exercise real collection and critic/regret fits before filling replay.
        # Synthetic rows are then used only for the isolated capacity check.
        solver.iteration()
        buffers = [solver.ave_policy_trainer.buffer, *[t.buffer for t in solver.regret_trainers],
                   *[m.buffer for m in solver.q_value_trainer.members], solver.calibration_trainer.buffer]
        # Touch every allocated page, not just virtual allocations or occupied early replay.
        for buffer in buffers:
            for value in vars(buffer).values():
                if isinstance(value, np.ndarray):
                    value.fill(0)
        for trainer in [solver.ave_policy_trainer, *solver.regret_trainers]:
            buffer = trainer.buffer
            n = buffer.buffer_size
            for start in range(0, n, 4096):
                end = min(n, start+4096)
                ids = np.arange(start, end)
                buffer.infostate_buf[start:end, 0] = ids / n
                p, s = solver.feature_encoder.replay_context_columns()
                buffer.infostate_buf[start:end, p] = np.eye(2)[ids % 2]
                buffer.infostate_buf[start:end, s] = np.eye(4)[ids % 4]
            buffer.q_value_buf[:] = 1/3
            buffer.q_value_mask_buf[:] = 1
            buffer.iteration_buf[:] = 1
            buffer.cur_id = n
        for member in solver.q_value_trainer.members:
            member.buffer.size = member.buffer.buffer_size
        solver.calibration_trainer.buffer.size = solver.calibration_trainer.buffer.capacity
        solver.num_iteration = 1
        solver.train_average_policy()
        kwargs = dict(seed=771, repository_commit="synthetic-memory-probe", config=config)
        output = Path(output)
        with tempfile.TemporaryDirectory(prefix="memory-probe-", dir=output.parent) as directory:
            path = Path(directory) / "state"
            payload = build_training_state(solver, **kwargs, checkpoint_id="probe", captured_checkpoints=[])
            save_training_state(path, payload)
            del payload
            restored = read_training_state(path)
            restore_training_state(solver, restored, **kwargs)
            del restored
        result = dict(dense_replay_bytes=learner_replay_allocation(solver),
                      full_capacity=not small, grouping_and_policy_fit=True, checkpoint_roundtrip=True,
                      workers=workers, threads=threads,
                      policy_group_sampling=solver.ave_policy_trainer.last_fit_diagnostics['group_sampling'],
                      limitations="Real collection followed by synthetic full replay; not a proof of every future collection/cache peak")
        write_json(output, result)
        return result
    finally:
        if hasattr(solver, "close"):
            solver.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--small", action="store_true", help="Offline plumbing test only")
    p.add_argument("--config", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args(argv)
    import json
    probe(args.output, small=args.small, workers=args.workers, threads=args.threads,
          config=json.loads(args.config.read_text()) if args.config else None)


if __name__ == "__main__":
    main()
