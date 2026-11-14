"""HULHE learning settings and reduced-budget, full-architecture smoke runs."""
from copy import deepcopy
import json
from pathlib import Path

from .features import ENCODER_ID


def baseline_config():
    return json.loads((Path(__file__).parent / "data/baseline.json").read_text())


def smoke_config():
    config = baseline_config()
    for key in ("advantage_buffer_size", "ave_policy_buffer_size",
                "baseline_buffer_size", "calibration_buffer_size"):
        config[key] = 256
    for key in ("advantage_batch_size", "ave_policy_batch_size",
                "baseline_batch_size", "calibration_batch_size"):
        config[key] = 8
    for key in ("advantage_network_train_steps", "ave_policy_network_train_steps",
                "baseline_network_train_steps", "calibration_train_steps",
                "average_policy_train_steps"):
        config[key] = 2
    config["num_traversals"] = 16
    config["policy_validation_steps"] = [1, 2]
    return config


def validate_config(config):
    expected = baseline_config()
    if set(config) != set(expected):
        raise ValueError("Configuration keys differ from the versioned baseline")
    for key in ("advantage_buffer_size", "ave_policy_buffer_size", "baseline_buffer_size",
                "calibration_buffer_size", "num_traversals", "advantage_batch_size",
                "ave_policy_batch_size", "baseline_batch_size", "calibration_batch_size",
                "advantage_network_train_steps", "baseline_network_train_steps",
                "calibration_train_steps", "average_policy_train_steps",
                "num_layers", "num_hiddens", "structured_branch_width",
                "calibration_branch_width"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("critic_network_layers", "calibration_network_layers",
                "average_policy_network_layers", "paired_legacy_network_layers"):
        widths = config[key]
        if (not isinstance(widths, list) or not widths
                or any(type(width) is not int or width <= 0 for width in widths)):
            raise ValueError(f"{key} must be a nonempty list of positive integers")
    if config["game_name"] != "HULHE" or config["feature_encoder_id"] != ENCODER_ID:
        raise ValueError("Use the versioned HULHE game and encoder")
    if config["q_ensemble_size"] != 2 or config["use_instantaneous_predictor"]:
        raise ValueError("Baseline requires two cross-fit critics and no predictor")
    if config["evaluation_frequency"] != 0 or config["evaluate_initial_policy"]:
        raise ValueError("The HULHE runner owns evaluation and checkpoint boundaries")
    for key in ("cache_frozen_regret_targets", "uniform_regret_ties"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    fraction = config["policy_validation_fraction"]
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError("policy_validation_fraction must be between zero and one")
    steps = config["policy_validation_steps"]
    if (not isinstance(steps, list) or not steps
            or any(type(step) is not int or step <= 0 for step in steps)
            or steps != sorted(set(steps))):
        raise ValueError("policy_validation_steps must be increasing positive integers")


def make_solver(config=None, *, seed=0, workers=1):
    from unbiased_escher.hulhe_structured_solver import StructuredHULHEGroupedWideUCVEscher
    from unbiased_escher.grouped_parallel_solver import ParallelStructuredGroupedUCVEscher
    from vr_deep_cfr.logger import Logger

    config = deepcopy(baseline_config() if config is None else config)
    validate_config(config)
    if workers < 1:
        raise ValueError("workers must be positive")
    kwargs = dict(config)
    cache = kwargs.pop("cache_frozen_critic_targets")
    runtime = {key: kwargs.pop(key) for key in (
        "max_num_iterations", "preserve_evaluation_rng", "evaluate_initial_policy",
        "early_evaluation_node_thresholds")}
    kwargs.update(seed=seed, logger=Logger(), num_episodes=2 * config["num_traversals"])
    if workers == 1:
        solver = StructuredHULHEGroupedWideUCVEscher(**kwargs)
    else:
        solver = ParallelStructuredGroupedUCVEscher(
            **kwargs, parallel_num_workers=workers, parallel_run_seed=seed,
            parallel_collection_chunk_size=min(1200, config["num_traversals"]),
            parallel_ray_object_store_memory=128 * 1024**2,
            parallelize_independent_learners=False, parallel_cache_actor_snapshots=True)
    for key, value in runtime.items():
        setattr(solver, key, value)
    for member in solver.q_value_trainer.members:
        member.cache_frozen_targets = cache
    return solver
