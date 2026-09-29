"""OfflineForge 核心数据类型 — 与 forge 系列统一的 dataclass 约定."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Transition:
    """单条离线转移样本 (s, a, r, s', done), 并记录行为策略概率 pi_b(a|s)."""

    s: int
    a: int
    r: float
    s_next: int
    done: bool
    pi_b: float
    t: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "s": self.s,
            "a": self.a,
            "r": self.r,
            "s_next": self.s_next,
            "done": self.done,
            "pi_b": self.pi_b,
            "t": self.t,
        }


@dataclass
class Dataset:
    """离线数据集: 多条 episode, 每条为 Transition 列表.

    pi_b 已随样本记录, 故 OPE 的重要性权重 rho = pi_e(a|s)/pi_b(a|s) 可直接计算.
    """

    name: str
    n_states: int
    n_actions: int
    gamma: float
    episodes: list[list[Transition]] = field(default_factory=list)
    behavior_policy: str = "epsilon_greedy"
    start_state: int = 0
    goal_state: int = 0
    grid_size: int = 0

    @property
    def n_transitions(self) -> int:
        return sum(len(ep) for ep in self.episodes)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "n_episodes": self.n_episodes,
            "n_transitions": self.n_transitions,
            "behavior_policy": self.behavior_policy,
            "start_state": self.start_state,
            "goal_state": self.goal_state,
            "grid_size": self.grid_size,
        }


@dataclass
class OpeResult:
    """单个 OPE 估计器的评估结果."""

    method: str
    estimate: float
    true_value: float
    abs_error: float
    bias: float
    std: float
    n_used: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "estimate": self.estimate,
            "true_value": self.true_value,
            "abs_error": self.abs_error,
            "bias": self.bias,
            "std": self.std,
            "n_used": self.n_used,
        }


@dataclass
class EvalRow:
    """流水线单行评估结果 (与 forge 系列一致)."""

    algorithm: str
    backend: str
    target: str
    seed: int
    best_ope: str
    best_ope_error: float
    true_value: float
    estimates: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "backend": self.backend,
            "target": self.target,
            "seed": self.seed,
            "best_ope": self.best_ope,
            "best_ope_error": self.best_ope_error,
            "true_value": self.true_value,
            "estimates": self.estimates,
        }


@dataclass
class StreamSummary:
    """流水线汇总 (按算法排序的均值误差)."""

    rows: list[EvalRow] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"rows": [r.as_dict() for r in self.rows]}


__all__ = ["Transition", "Dataset", "OpeResult", "EvalRow", "StreamSummary"]
