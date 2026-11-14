"""Dense float32 replay persistence; no FHP-specific feature codebook."""
import numpy as np


def feature_state(features, size):
    return np.asarray(features[:size])


def restore_features(destination, state, size):
    if not isinstance(state, np.ndarray) or state.shape != (size, destination.shape[1]):
        raise ValueError("Replay shape differs from checkpoint contract")
    destination[:size] = state
