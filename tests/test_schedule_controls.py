import numpy as np

from adsl.config import CorruptionConfig
from adsl.corruption import CorruptionEngine


def test_corruption_respects_start_step_for_bursty_schedule():
    config = CorruptionConfig(
        enabled=True,
        type="reward_poisoning",
        schedule="bursty",
        start_step=10,
        burst_length=5,
        burst_period=20,
        reward_flip_p=1.0,
    )
    engine = CorruptionEngine(config, seed=0)
    obs = np.array([1.0, 2.0], dtype=np.float32)
    act = np.array([0.0], dtype=np.float32)

    early = engine.apply(obs, act, 1.0, 5)
    active = engine.apply(obs, act, 1.0, 10)

    assert early.corrupted is False
    assert active.corrupted is True
