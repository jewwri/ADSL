import numpy as np
import torch

from adsl.config import ControllerConfig
from adsl.control import BaselineReference, LookaheadContext, LookaheadController, ShadowLearnerState
from adsl.rl import Actor, Critic


def _make_batch(batch_size: int, obs_dim: int, act_dim: int) -> dict[str, np.ndarray]:
    return {
        "obs": np.random.randn(batch_size, obs_dim).astype(np.float32),
        "act": np.random.randn(batch_size, act_dim).astype(np.float32),
        "rew": np.random.randn(batch_size, 1).astype(np.float32),
        "obs2": np.random.randn(batch_size, obs_dim).astype(np.float32),
        "done": np.zeros((batch_size, 1), dtype=np.float32),
        "corrupted": np.ones((batch_size, 1), dtype=np.float32),
        "corruption_type": np.asarray(["reward_poisoning"] * batch_size, dtype=object),
    }


def _make_state(obs_dim: int = 3, act_dim: int = 2) -> tuple[ShadowLearnerState, Actor]:
    device = torch.device("cpu")
    actor = Actor(obs_dim, act_dim, 1.0).to(device)
    critic1 = Critic(obs_dim, act_dim).to(device)
    critic2 = Critic(obs_dim, act_dim).to(device)
    target1 = Critic(obs_dim, act_dim).to(device)
    target2 = Critic(obs_dim, act_dim).to(device)
    target1.load_state_dict(critic1.state_dict())
    target2.load_state_dict(critic2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    state = ShadowLearnerState(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        target1=target1,
        target2=target2,
        log_alpha=log_alpha,
        device=device,
        gamma=0.99,
        tau=0.005,
        target_entropy=-act_dim,
        lr=3e-4,
    )
    return state, actor


def test_controller_accepts_when_baseline_not_ready():
    cfg = ControllerConfig(enabled=True, mode="mcts", mcts_simulations=8)
    controller = LookaheadController(cfg)
    state, _ = _make_state()
    baseline = BaselineReference(capacity=8)
    batch = _make_batch(batch_size=8, obs_dim=3, act_dim=2)

    result = controller.decide(
        learner_state=state,
        baseline=baseline,
        ctx=LookaheadContext(
            detector_risk=0.8,
            flagged_batch=batch,
            clean_batch=batch,
            reference_obs=batch["obs"],
            reward_shift=1.0,
            action_shift=1.0,
            obs_shift=1.0,
            replay_size=len(batch["obs"]),
            clean_replay_size=len(batch["obs"]),
            replay_clean_fraction=1.0,
            global_step=100,
        ),
    )

    assert result.action == "accept"
    assert result.trust == 1.0


def test_controller_runs_mcts_when_baseline_ready():
    cfg = ControllerConfig(enabled=True, mode="mcts", mcts_simulations=8, baseline_reference_size=8)
    controller = LookaheadController(cfg)
    state, actor = _make_state()
    baseline = BaselineReference(capacity=8)
    for _ in range(8):
        baseline.update_memory(np.random.randn(3).astype(np.float32))
    baseline.capture_actor(actor)
    flagged = _make_batch(batch_size=8, obs_dim=3, act_dim=2)
    clean = _make_batch(batch_size=8, obs_dim=3, act_dim=2)

    result = controller.decide(
        learner_state=state,
        baseline=baseline,
        ctx=LookaheadContext(
            detector_risk=0.7,
            flagged_batch=flagged,
            clean_batch=clean,
            reference_obs=flagged["obs"],
            reward_shift=0.5,
            action_shift=0.5,
            obs_shift=0.5,
            replay_size=len(flagged["obs"]),
            clean_replay_size=len(clean["obs"]),
            replay_clean_fraction=1.0,
            global_step=100,
        ),
    )

    assert result.action in {"accept", "sanitize"}
    assert sum(result.visit_counts.values()) == cfg.mcts_simulations
    assert 0.0 <= result.trust <= 1.0


def test_sanitize_uses_clean_replacement_batch():
    cfg = ControllerConfig(enabled=True, mode="mcts")
    controller = LookaheadController(cfg)
    flagged = _make_batch(batch_size=4, obs_dim=3, act_dim=2)
    clean = _make_batch(batch_size=4, obs_dim=3, act_dim=2)
    sanitized = controller._batch_for_action("sanitize", flagged, clean)

    assert sanitized is clean
