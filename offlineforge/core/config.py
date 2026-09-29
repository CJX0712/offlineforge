"""OfflineForge 全局配置 — 与 forge 系列一致的 from_env 覆盖约定."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    """OfflineForge 运行配置.

    所有随机性来源统一由 random_state / benchmark_seeds 控制, 保证可复现.
    """

    random_state: int = 42
    benchmark_seeds: tuple[int, int, int] = (42, 123, 7)

    # ---- GridWorld MDP ----
    grid_size: int = 5
    gamma: float = 0.95
    slip_prob: float = 0.1
    step_cost: float = -0.04
    goal_reward: float = 1.0
    max_steps: int = 50

    # ---- 离线数据集生成 ----
    n_episodes: int = 400
    behavior_epsilon: float = 0.3
    behavior_policy: str = "epsilon_greedy"

    # ---- 离线策略学习 ----
    cql_alpha: float = 1.0
    fqe_iterations: int = 300
    fqe_tol: float = 1e-6
    bc_learning_rate: float = 0.5
    bc_epochs: int = 60
    ood_floor: float = -50.0  # 未见 (s,a) 的保守 Q 下限

    # ---- OPE ----
    ope_methods: tuple[str, ...] = (
        "dm",
        "sis",
        "wis",
        "tis",
        "dr",
        "fqe",
    )

    # ---- 后端 ----
    use_d3rlpy: bool = True

    @classmethod
    def from_env(cls) -> Config:
        """允许通过 ENV_OFF_*/ENV_RF_* 覆盖关键超参 (与 forge 系列一致)."""
        kw: dict[str, Any] = {}
        mapping = {
            "OFF_GRID_SIZE": ("grid_size", int),
            "OFF_N_EPISODES": ("n_episodes", int),
            "OFF_GAMMA": ("gamma", float),
            "OFF_BEHAVIOR_EPS": ("behavior_epsilon", float),
            "OFF_SEED": ("random_state", int),
            "OFF_CQL_ALPHA": ("cql_alpha", float),
        }
        for env_key, (attr, caster) in mapping.items():
            val = os.environ.get(env_key)
            if val is not None:
                try:
                    kw[attr] = caster(val)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"无效环境变量 {env_key}={val!r}") from exc
        return cls(**kw)

    def as_dict(self) -> dict[str, Any]:
        return {
            "random_state": self.random_state,
            "benchmark_seeds": list(self.benchmark_seeds),
            "grid_size": self.grid_size,
            "gamma": self.gamma,
            "slip_prob": self.slip_prob,
            "step_cost": self.step_cost,
            "goal_reward": self.goal_reward,
            "max_steps": self.max_steps,
            "n_episodes": self.n_episodes,
            "behavior_epsilon": self.behavior_epsilon,
            "behavior_policy": self.behavior_policy,
            "cql_alpha": self.cql_alpha,
            "fqe_iterations": self.fqe_iterations,
            "fqe_tol": self.fqe_tol,
            "bc_learning_rate": self.bc_learning_rate,
            "bc_epochs": self.bc_epochs,
            "ood_floor": self.ood_floor,
            "ope_methods": list(self.ope_methods),
            "use_d3rlpy": self.use_d3rlpy,
        }


__all__ = ["Config"]
