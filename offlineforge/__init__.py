"""OfflineForge — 离线强化学习 (Offline RL) + 离线策略评估 (OPE) 框架.

作者: 晨星
领域: 离线强化学习 / 离线策略评估 (Offline RL + Offline Policy Evaluation)
旗舰: OfflineForgePipeline —— 从固定离线数据集学习策略 (BC / CQL-lite / FQE),
      并用多种 OPE 估计器 (DM / SIS / WIS / TIS / DR / FQE) 评估目标策略价值,
      对照动态规划 (DP) 真值给出误差与排序.

零外部依赖内核: 纯 numpy 实现 GridWorld MDP、行为克隆、CQL-lite 保守 Q 学习、
表格 FQE 与全部 OPE 估计器; 可选复用 d3rlpy (CQL/BCQ) 作为顶级开源背书.
"""

from .core.config import Config
from .core.errors import OfflineForgeError
from .core.types import (
    Dataset,
    EvalRow,
    OpeResult,
    StreamSummary,
    Transition,
)

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = [
    "Config",
    "Dataset",
    "Transition",
    "OpeResult",
    "EvalRow",
    "StreamSummary",
    "OfflineForgeError",
    "__version__",
    "__author__",
]
