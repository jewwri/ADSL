from __future__ import annotations

import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim

from .rl import Actor, Critic, sac_update


INTERVENTION_ACTIONS = ("accept", "attenuate", "block", "sanitize")


@dataclass
class ControlDecision:
    risk: float
    action: str
    trust: float


@dataclass
class ShadowLearnerState:
    actor: Actor
    critic1: Critic
    critic2: Critic
    target1: Critic
    target2: Critic
    log_alpha: torch.Tensor
    device: torch.device
    gamma: float
    tau: float
    target_entropy: float
    lr: float


@dataclass
class LookaheadContext:
    # MCTS state = learner snapshot + suspicious window summary + replay context.
    detector_risk: float
    flagged_batch: dict[str, np.ndarray]
    clean_batch: dict[str, np.ndarray]
    reference_obs: np.ndarray
    reward_shift: float
    action_shift: float
    obs_shift: float
    replay_size: int
    clean_replay_size: int
    replay_clean_fraction: float
    global_step: int


@dataclass
class MCTSResult:
    action: str
    score: float
    trust: float
    visit_counts: dict[str, int]
    action_values: dict[str, float]
    predicted_deviation: float
    predicted_return_drop: float


class BaselineReference:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.obs_memory: deque[np.ndarray] = deque(maxlen=capacity)
        self.actor: Actor | None = None

    def update_memory(self, obs: np.ndarray) -> None:
        self.obs_memory.append(np.asarray(obs, dtype=np.float32).copy())

    def ready(self) -> bool:
        return len(self.obs_memory) >= 8 and self.actor is not None

    def capture_actor(self, actor: Actor) -> None:
        # This is a clean-policy reference snapshot, not an optimal or frozen expert.
        device = next(actor.parameters()).device
        self.actor = deepcopy(actor).to(device)
        self.actor.eval()

    def reference_obs(self, fallback: np.ndarray) -> np.ndarray:
        if not self.obs_memory:
            return np.asarray(fallback, dtype=np.float32)
        return np.stack(list(self.obs_memory), axis=0).astype(np.float32)


class MCTSNode:
    def __init__(self, depth: int, prior_actions: tuple[str, ...] = INTERVENTION_ACTIONS):
        self.depth = depth
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[str, MCTSNode] = {}
        self.prior_actions = prior_actions

    @property
    def value(self) -> float:
        return self.value_sum / max(1, self.visit_count)

    def expanded(self) -> bool:
        return bool(self.children)


class LookaheadController:
    def __init__(self, config):
        self.config = config

    def decide(self, learner_state: ShadowLearnerState, baseline: BaselineReference, ctx: LookaheadContext) -> MCTSResult:
        if self.config.mode == "none" or not baseline.ready():
            return MCTSResult(
                action="accept",
                score=0.0,
                trust=1.0,
                visit_counts={name: 0 for name in INTERVENTION_ACTIONS},
                action_values={name: 0.0 for name in INTERVENTION_ACTIONS},
                predicted_deviation=0.0,
                predicted_return_drop=0.0,
            )

        root = MCTSNode(depth=0)
        action_stats = {
            name: {"visits": 0, "value_sum": 0.0, "deviation_sum": 0.0, "return_drop_sum": 0.0}
            for name in INTERVENTION_ACTIONS
        }

        for _ in range(self.config.mcts_simulations):
            node = root
            path: list[tuple[MCTSNode, str]] = []

            while node.expanded():
                action = self._select_ucb(node)
                path.append((node, action))
                node = node.children[action]

            if node.depth < self.config.mcts_horizon:
                for action in INTERVENTION_ACTIONS:
                    node.children[action] = MCTSNode(depth=node.depth + 1)

            root_action = path[0][1] if path else self._select_rollout_action()
            value, deviation, return_drop = self._simulate_action_value(
                learner_state=learner_state,
                baseline=baseline,
                ctx=ctx,
                root_action=root_action,
            )

            for parent, action in path:
                child = parent.children[action]
                child.visit_count += 1
                child.value_sum += value
            node.visit_count += 1
            node.value_sum += value

            stats = action_stats[root_action]
            stats["visits"] += 1
            stats["value_sum"] += value
            stats["deviation_sum"] += deviation
            stats["return_drop_sum"] += return_drop

        best_action = max(
            INTERVENTION_ACTIONS,
            key=lambda name: (
                action_stats[name]["value_sum"] / max(1, action_stats[name]["visits"]),
                action_stats[name]["visits"],
            ),
        )
        best_stats = action_stats[best_action]
        best_score = best_stats["value_sum"] / max(1, best_stats["visits"])
        predicted_deviation = best_stats["deviation_sum"] / max(1, best_stats["visits"])
        predicted_return_drop = best_stats["return_drop_sum"] / max(1, best_stats["visits"])
        trust = max(0.0, 1.0 - max(predicted_deviation, predicted_return_drop))

        return MCTSResult(
            action=best_action,
            score=1.0 - best_score,
            trust=trust,
            visit_counts={name: action_stats[name]["visits"] for name in INTERVENTION_ACTIONS},
            action_values={
                name: action_stats[name]["value_sum"] / max(1, action_stats[name]["visits"])
                for name in INTERVENTION_ACTIONS
            },
            predicted_deviation=predicted_deviation,
            predicted_return_drop=predicted_return_drop,
        )

    def _select_ucb(self, node: MCTSNode) -> str:
        total_visits = max(1, node.visit_count)
        best_action = INTERVENTION_ACTIONS[0]
        best_score = -float("inf")
        for action, child in node.children.items():
            q = child.value
            u = self.config.mcts_exploration_c * math.sqrt(math.log(total_visits + 1) / max(1, child.visit_count))
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _select_rollout_action(self) -> str:
        return np.random.choice(INTERVENTION_ACTIONS).item()

    def _simulate_action_value(
        self,
        learner_state: ShadowLearnerState,
        baseline: BaselineReference,
        ctx: LookaheadContext,
        root_action: str,
    ) -> tuple[float, float, float]:
        # MCTS transition model: copy the current SAC learner, apply the candidate
        # intervention to the suspicious window plus replay context, then perform a
        # shadow SAC update and score the resulting actor against the clean-policy
        # reference actor.
        shadow = self._copy_shadow_state(learner_state)
        simulated_batch = self._batch_for_action(root_action, ctx.flagged_batch, ctx.clean_batch)
        if simulated_batch is not None:
            self._shadow_sac_update(shadow, simulated_batch)

        reference_obs = baseline.reference_obs(ctx.reference_obs)
        baseline_actions = self._actor_actions(baseline.actor, reference_obs, learner_state.device)
        candidate_actions = self._actor_actions(shadow.actor, reference_obs, learner_state.device)
        deviation = float(np.mean(np.linalg.norm(candidate_actions - baseline_actions, axis=1)))
        return_drop = self._predicted_return_drop(shadow, baseline.actor, reference_obs)

        # MCTS reward/score: weighted combination of deviation from clean-policy
        # behavior, predicted return drop, and detector risk.
        weights = {
            "baseline_deviation": 0.55,
            "predicted_return_drop": 0.25,
            "detector_risk": 0.20,
            **self.config.harm_weights,
        }
        base_penalty = (
            float(weights["baseline_deviation"]) * min(deviation / max(self.config.deviation_threshold, 1e-6), 1.0)
            + float(weights["predicted_return_drop"]) * min(return_drop, 1.0)
            + float(weights["detector_risk"]) * min(ctx.detector_risk, 1.0)
        )
        value = max(0.0, 1.0 - base_penalty)
        return float(value), deviation, return_drop

    def _batch_for_action(
        self,
        action: str,
        flagged_batch: dict[str, np.ndarray],
        clean_batch: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray] | None:
        if action == "block":
            return None
        if action == "accept":
            return flagged_batch
        if action == "sanitize":
            return clean_batch
        if action == "attenuate":
            ratio = float(self.config.attenuate_clean_ratio)
            batch = {}
            for key in flagged_batch:
                if key == "corruption_type":
                    batch[key] = flagged_batch[key]
                else:
                    batch[key] = (1.0 - ratio) * flagged_batch[key] + ratio * clean_batch[key]
            return batch
        return flagged_batch

    def _copy_shadow_state(self, learner_state: ShadowLearnerState) -> ShadowLearnerState:
        actor = deepcopy(learner_state.actor)
        critic1 = deepcopy(learner_state.critic1)
        critic2 = deepcopy(learner_state.critic2)
        target1 = deepcopy(learner_state.target1)
        target2 = deepcopy(learner_state.target2)
        log_alpha = learner_state.log_alpha.detach().clone().requires_grad_(True)
        return ShadowLearnerState(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            target1=target1,
            target2=target2,
            log_alpha=log_alpha,
            device=learner_state.device,
            gamma=learner_state.gamma,
            tau=learner_state.tau,
            target_entropy=learner_state.target_entropy,
            lr=learner_state.lr,
        )

    def _shadow_sac_update(self, shadow: ShadowLearnerState, batch: dict[str, np.ndarray]) -> None:
        opt_actor = optim.Adam(shadow.actor.parameters(), lr=shadow.lr)
        opt_critic = optim.Adam(
            list(shadow.critic1.parameters()) + list(shadow.critic2.parameters()),
            lr=shadow.lr,
        )
        opt_alpha = optim.Adam([shadow.log_alpha], lr=shadow.lr)
        sac_update(
            actor=shadow.actor,
            critic1=shadow.critic1,
            critic2=shadow.critic2,
            target1=shadow.target1,
            target2=shadow.target2,
            log_alpha=shadow.log_alpha,
            opt_actor=opt_actor,
            opt_critic=opt_critic,
            opt_alpha=opt_alpha,
            batch=batch,
            gamma=shadow.gamma,
            tau=shadow.tau,
            target_entropy=shadow.target_entropy,
            device=shadow.device,
        )

    def _actor_actions(self, actor: Actor, obs: np.ndarray, device: torch.device) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            mu, _ = actor(obs_t)
            actions = torch.tanh(mu) * actor.act_limit
        return actions.cpu().numpy()

    def _predicted_return_drop(self, shadow: ShadowLearnerState, baseline_actor: Actor, obs: np.ndarray) -> float:
        baseline_actions = self._actor_actions(baseline_actor, obs, shadow.device)
        candidate_actions = self._actor_actions(shadow.actor, obs, shadow.device)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=shadow.device)
        act_baseline_t = torch.as_tensor(baseline_actions, dtype=torch.float32, device=shadow.device)
        act_candidate_t = torch.as_tensor(candidate_actions, dtype=torch.float32, device=shadow.device)
        with torch.no_grad():
            q_base = torch.min(shadow.critic1(obs_t, act_baseline_t), shadow.critic2(obs_t, act_baseline_t)).mean()
            q_cand = torch.min(shadow.critic1(obs_t, act_candidate_t), shadow.critic2(obs_t, act_candidate_t)).mean()
        drop = torch.relu(q_base - q_cand).item()
        return float(min(drop / 5.0, 1.0))
