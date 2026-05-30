import numpy as np

from adsl.detection import compute_window_features


class FeatureConfig:
    reward_stats = True
    action_stats = True
    state_shift = True
    temporal_delta = True


def test_compute_window_features_shape():
    window = {
        "obs": np.zeros((8, 4), dtype=np.float32),
        "act": np.zeros((8, 1), dtype=np.float32),
        "rew": np.zeros((8, 1), dtype=np.float32),
        "obs2": np.ones((8, 4), dtype=np.float32),
        "done": np.zeros((8, 1), dtype=np.float32),
    }
    feats = compute_window_features(window, FeatureConfig())
    assert feats.ndim == 1
    assert feats.shape[0] > 0

