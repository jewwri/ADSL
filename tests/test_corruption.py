import numpy as np

from adsl.corruption import CorruptionEngine
from adsl.config import CorruptionConfig


def test_random_sparse_clean_when_disabled():
    config = CorruptionConfig(enabled=False)
    engine = CorruptionEngine(config, seed=0)
    outcome = engine.apply(np.array([1.0, 2.0], dtype=np.float32), np.array([0.0], dtype=np.float32), 1.0, 0)
    assert outcome.corrupted is False
    assert outcome.corruption_type == "clean"

