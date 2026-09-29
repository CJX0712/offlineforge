"""OfflineForge 抽象接口 — 与 forge 系列一致的 Base* 约定."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    """目标/行为策略接口: 给定状态返回动作概率分布与贪心动作."""

    @abstractmethod
    def action_probs(self, s: int) -> list[float]: ...

    @abstractmethod
    def act(self, s: int) -> int: ...

    @abstractmethod
    def available(self) -> bool: ...

    def clone(self) -> BasePolicy:
        raise NotImplementedError("clone 未实现")


class BaseOfflineLearner(ABC):
    """离线策略学习器: 从 Dataset 学出一个 BasePolicy."""

    @abstractmethod
    def fit(self, dataset: Any) -> BaseOfflineLearner: ...

    @abstractmethod
    def policy(self) -> BasePolicy: ...

    @abstractmethod
    def available(self) -> bool: ...


class BaseOpeEstimator(ABC):
    """OPE 估计器: 用离线数据集估计目标策略价值 V^pi_e(s0)."""

    @abstractmethod
    def estimate(self, dataset: Any, target: BasePolicy, **kw: Any) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class BaseMetric(ABC):
    """评估指标: 对比 OPE 估计与 DP 真值."""

    @abstractmethod
    def score(self, estimate: float, true_value: float) -> float: ...


__all__ = [
    "BasePolicy",
    "BaseOfflineLearner",
    "BaseOpeEstimator",
    "BaseMetric",
]
