"""Synthetic full-replay grouping and recovery probe; never supplies training data."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
import torch

from .config import baseline_config, make_solver
from .training_state import build_training_state, save_training_state, read_training_state, restore_training_state
from .train import write_json
from unbiased_escher.replay_grouping import learner_replay_allocation


def probe(output, *, small=False):
    import psutil
    if not small and psutil.virtual_memory().total < 48*1024**3:
        raise RuntimeError("Full memory preflight requires a VM with at least 48 GiB RAM")
    config = baseline_config()
    if small:
        for key in ("advantage_buffer_size", "ave_policy_buffer_size", "baseline_buffer_size", "calibration_buffer_size"):
            config[key] = 64
        config["ave_policy_batch_size"] = 8
    config["average_policy_train_steps"] = 2 if small else 500
    config["policy_validation_steps"] = [1, 2] if small else [250, 500]
    torch.set_num_threads(4)
    solver = make_solver(config, seed=771, workers=1)
    try:
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
                      limitations="Synthetic single-learner probe; excludes actor peaks and full critic target caches")
        write_json(output, result)
        return result
    finally:
        if hasattr(solver, "close"):
            solver.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--small", action="store_true", help="Offline plumbing test only")
    args = p.parse_args(argv)
    probe(args.output, small=args.small)


if __name__ == "__main__":
    main()
