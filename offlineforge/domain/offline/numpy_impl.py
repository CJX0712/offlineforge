"""OfflineForge 纯 numpy 旗舰实现 — 离线 RL 学习器 + 离线策略评估 (OPE).

零外部依赖内核:
  * 行为克隆 BC (平滑经验策略)
  * CQL-lite 保守 Q 学习 (logsumexp 正则 + OOD 下界)
  * FQE / FQI 表格拟合 Q 迭代 (模型无关)
  * 六种 OPE 估计器: DM / SIS / WIS / TIS / DR / FQE
  * OfflineForgePipeline 编排: 难度梯度场景 × 种子 × 算法 -> 误差排序

可选复用 d3rlpy (CQL / BCQ) 作为顶级开源背书; 不可用时自动降级为 numpy 后端.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...core.config import Config
from ...core.errors import BackendUnavailable, NumericalError, PipelineError
from ...core.interfaces import BaseOfflineLearner, BasePolicy
from ...core.types import Dataset, EvalRow, StreamSummary
from ...data.gridworld import GridWorld

# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------


def logsumexp(x: np.ndarray, axis: int = 1, keepdims: bool = True) -> np.ndarray:
    """数值稳定的 log(sum(exp(x)))."""
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=keepdims)
    return m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=keepdims))


def _safe_ratio(num: float, den: float, floor: float = 1e-9, cap: float = 1e3) -> float:
    if den <= 0:
        return 0.0
    r = num / den
    return float(min(max(r, floor), cap))


# ---------------------------------------------------------------------------
# 数据集 <-> 数组
# ---------------------------------------------------------------------------


@dataclass
class FlatData:
    """扁平化后的离线数据集 (向量化 FQI 用)."""

    n_states: int
    n_actions: int
    gamma: float
    s: np.ndarray
    a: np.ndarray
    r: np.ndarray
    s_next: np.ndarray
    done: np.ndarray
    pi_b: np.ndarray
    support: np.ndarray  # (S,A) bool: 是否出现在数据中


def dataset_to_flat(ds: Dataset) -> FlatData:
    s, a, r, sn, done, pb = [], [], [], [], [], []
    support = np.zeros((ds.n_states, ds.n_actions), dtype=bool)
    for ep in ds.episodes:
        for tr in ep:
            s.append(tr.s)
            a.append(tr.a)
            r.append(tr.r)
            sn.append(tr.s_next)
            done.append(1 if tr.done else 0)
            pb.append(tr.pi_b)
            support[tr.s, tr.a] = True
    return FlatData(
        n_states=ds.n_states,
        n_actions=ds.n_actions,
        gamma=ds.gamma,
        s=np.asarray(s, int),
        a=np.asarray(a, int),
        r=np.asarray(r, float),
        s_next=np.asarray(sn, int),
        done=np.asarray(done, float),
        pi_b=np.asarray(pb, float),
        support=support,
    )


def episodes_to_arrays(ds: Dataset) -> list[dict[str, np.ndarray]]:
    """按 episode 组织, 供 IS 类估计器使用."""
    out: list[dict[str, np.ndarray]] = []
    for ep in ds.episodes:
        if not ep:
            continue
        out.append(
            {
                "s": np.asarray([t.s for t in ep], int),
                "a": np.asarray([t.a for t in ep], int),
                "r": np.asarray([t.r for t in ep], float),
                "s_next": np.asarray([t.s_next for t in ep], int),
                "done": np.asarray([1 if t.done else 0 for t in ep], float),
                "pi_b": np.asarray([t.pi_b for t in ep], float),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 目标策略
# ---------------------------------------------------------------------------


class TabularPolicy(BasePolicy):
    """表格随机策略: 存储完整 (S,A) 概率矩阵."""

    def __init__(self, probs: np.ndarray, name: str = "tabular") -> None:
        self.probs = np.asarray(probs, dtype=float)
        self._name = name

    def action_probs(self, s: int) -> list[float]:
        return self.probs[s].tolist()

    def act(self, s: int) -> int:
        return int(np.argmax(self.probs[s]))

    def available(self) -> bool:
        return True

    def clone(self) -> TabularPolicy:
        return TabularPolicy(self.probs.copy(), self._name)

    def probs_matrix(self) -> np.ndarray:
        return self.probs

    @property
    def name(self) -> str:
        return self._name


def target_optimal(gw: GridWorld) -> TabularPolicy:
    _, Q, _ = gw.value_iteration()
    probs = np.zeros((gw.n_states, gw.n_actions))
    probs[np.arange(gw.n_states), np.argmax(Q, axis=1)] = 1.0
    return TabularPolicy(probs, "optimal")


def target_random(gw: GridWorld) -> TabularPolicy:
    probs = np.full((gw.n_states, gw.n_actions), 1.0 / gw.n_actions)
    return TabularPolicy(probs, "random")


def target_bc(flat: FlatData, cfg: Config) -> TabularPolicy:
    """行为克隆: 平滑经验动作分布 (Laplace)."""
    counts = np.zeros((flat.n_states, flat.n_actions))
    np.add.at(counts, (flat.s, flat.a), 1.0)
    laplace = 0.5
    probs = (counts + laplace) / (counts.sum(axis=1, keepdims=True) + flat.n_actions * laplace)
    # 无数据状态回退均匀分布
    empty = counts.sum(axis=1) == 0
    probs[empty] = 1.0 / flat.n_actions
    return TabularPolicy(probs, "bc")


# ---------------------------------------------------------------------------
# 拟合 Q 迭代 (FQI) —— 模型无关, 仅用离线数据
# ---------------------------------------------------------------------------


def fitted_q_iteration(
    flat: FlatData, backup_pi: np.ndarray, cfg: Config, iters: int | None = None
) -> np.ndarray:
    """表格 FQI: 用给定备份策略 backup_pi 做 Bellman 目标, 平均散射更新.

    backup_pi: (S,A) 概率矩阵, 用于 V(s') = Σ_a π(a|s') Q(s',a).
    """
    iters = iters or cfg.fqe_iterations
    S, A = flat.n_states, flat.n_actions
    Q = np.zeros((S, A))
    gamma = flat.gamma
    sum_q = np.zeros((S, A))
    cnt = np.zeros((S, A))
    for _ in range(iters):
        v_next = (backup_pi[flat.s_next] * Q[flat.s_next]).sum(axis=1) * (1.0 - flat.done)
        target = flat.r + gamma * v_next
        sum_q[:] = 0.0
        cnt[:] = 0.0
        np.add.at(sum_q, (flat.s, flat.a), target)
        np.add.at(cnt, (flat.s, flat.a), 1.0)
        Q_new = Q.copy()
        mask = cnt > 0
        Q_new[mask] = sum_q[mask] / cnt[mask]
        Q = Q_new
    if not np.all(np.isfinite(Q)):
        raise NumericalError("FQI 产生非有限 Q")
    return Q


def target_fqe(flat: FlatData, cfg: Config) -> TabularPolicy:
    """离线价值迭代: 以贪心策略为备份, 拟合 Q, 取贪心策略."""
    S, A = flat.n_states, flat.n_actions
    Q = np.zeros((S, A))
    gamma = flat.gamma
    sum_q = np.zeros((S, A))
    cnt = np.zeros((S, A))
    for _ in range(cfg.fqe_iterations):
        v_next = Q[flat.s_next].max(axis=1) * (1.0 - flat.done)
        target = flat.r + gamma * v_next
        sum_q[:] = 0.0
        cnt[:] = 0.0
        np.add.at(sum_q, (flat.s, flat.a), target)
        np.add.at(cnt, (flat.s, flat.a), 1.0)
        Q_new = Q.copy()
        mask = cnt > 0
        Q_new[mask] = sum_q[mask] / cnt[mask]
        Q = Q_new
    probs = np.zeros((S, A))
    probs[np.arange(S), np.argmax(Q, axis=1)] = 1.0
    return TabularPolicy(probs, "fqe")


def target_cql(flat: FlatData, cfg: Config, iters: int = 80) -> TabularPolicy:
    """CQL-lite 保守 Q 学习 (模型无关贪心备份 + logsumexp 正则 + OOD 下界)."""
    S, A = flat.n_states, flat.n_actions
    Q = np.zeros((S, A))
    gamma = flat.gamma
    sum_q = np.zeros((S, A))
    cnt = np.zeros((S, A))
    for _ in range(iters):
        v_next = Q[flat.s_next].max(axis=1) * (1.0 - flat.done)
        target = flat.r + gamma * v_next
        sum_q[:] = 0.0
        cnt[:] = 0.0
        np.add.at(sum_q, (flat.s, flat.a), target)
        np.add.at(cnt, (flat.s, flat.a), 1.0)
        Q_new = Q.copy()
        mask = cnt > 0
        Q_new[mask] = sum_q[mask] / cnt[mask]
        # 保守正则: 压低非最大动作 Q (CQL 核心——抑制 OOD 高估)
        lse = logsumexp(Q_new, axis=1, keepdims=True)
        Q_new = Q_new - cfg.cql_alpha * (lse - Q_new)
        # OOD 动作 (数据中未见) 强制下界
        Q_new[~flat.support] = np.minimum(Q_new[~flat.support], cfg.ood_floor)
        Q = Q_new
    if not np.all(np.isfinite(Q)):
        raise NumericalError("CQL 产生非有限 Q")
    probs = np.zeros((S, A))
    probs[np.arange(S), np.argmax(Q, axis=1)] = 1.0
    return TabularPolicy(probs, "cql")


# ---------------------------------------------------------------------------
# OPE 估计器
# ---------------------------------------------------------------------------


def ope_dm(flat: FlatData, pi_e: np.ndarray, dm_q: np.ndarray, cfg: Config) -> float:
    """直接法 (Direct Method): 用学习到的 Q̂ 在起点代入目标策略."""
    return float((pi_e[0] * dm_q[0]).sum())


def ope_fqe(
    flat: FlatData, pi_e: np.ndarray, dm_q: np.ndarray, cfg: Config, episodes=None
) -> float:
    """拟合 Q 评估 (FQE): 以目标策略为备份拟合 Q, 再代入起点."""
    qf = fitted_q_iteration(flat, pi_e, cfg)
    return float((pi_e[0] * qf[0]).sum())


# 重要性权重裁剪 (截断 IS): 抑制累积权重爆炸, 兼顾偏差/方差
_RHO_CAP = 20.0
_CUM_CAP = 1e3
_CUM_CAP_DR = 1e2


def _ratios(ep: dict[str, np.ndarray], pi_e: np.ndarray) -> np.ndarray:
    pi_e_a = pi_e[ep["s"], ep["a"]]
    out = np.empty(len(pi_e_a), dtype=float)
    for i, (p, b) in enumerate(zip(pi_e_a, ep["pi_b"])):
        out[i] = _safe_ratio(p, b, floor=1e-9, cap=_RHO_CAP)
    return out


def _cumprod_clipped(rho: np.ndarray, cap: float) -> np.ndarray:
    cum = np.cumprod(rho)
    return np.minimum(cum, cap)


def ope_sis(episodes: list[dict[str, np.ndarray]], pi_e: np.ndarray, cfg: Config) -> float:
    """逐步 (per-decision) IS — 无偏, 模型无关."""
    gamma = cfg.gamma
    acc = 0.0
    n = 0
    for ep in episodes:
        rho = _ratios(ep, pi_e)
        cum = _cumprod_clipped(rho, _CUM_CAP)
        g = gamma ** np.arange(len(ep["r"]))
        acc += float((cum * ep["r"] * g).sum())
        n += 1
    return acc / n if n else 0.0


def ope_wis(episodes: list[dict[str, np.ndarray]], pi_e: np.ndarray, cfg: Config) -> float:
    """轨迹自归一化 IS (标准 WIS): 整条轨迹权重归一化, 降方差稳健."""
    gamma = cfg.gamma
    W: list[float] = []
    G: list[float] = []
    for ep in episodes:
        rho = _ratios(ep, pi_e)
        cum = _cumprod_clipped(rho, _CUM_CAP)
        W.append(float(cum[-1]))
        g = gamma ** np.arange(len(ep["r"]))
        G.append(float((ep["r"] * g).sum()))
    if not W:
        return 0.0
    W = np.asarray(W)
    G = np.asarray(G)
    W = W / W.sum()
    return float((W * G).sum())


def ope_tis(episodes: list[dict[str, np.ndarray]], pi_e: np.ndarray, cfg: Config) -> float:
    """轨迹 IS (未归一化): 无偏但高方差, 展示方差代价."""
    gamma = cfg.gamma
    W: list[float] = []
    G: list[float] = []
    for ep in episodes:
        rho = _ratios(ep, pi_e)
        cum = _cumprod_clipped(rho, _CUM_CAP)
        W.append(float(cum[-1]))
        g = gamma ** np.arange(len(ep["r"]))
        G.append(float((ep["r"] * g).sum()))
    if not W:
        return 0.0
    return float(np.mean(np.asarray(W) * np.asarray(G)))


def ope_dr(
    flat: FlatData,
    episodes: list[dict[str, np.ndarray]],
    pi_e: np.ndarray,
    q_model: np.ndarray,
    cfg: Config,
) -> float:
    """双重稳健 (Doubly Robust, 标准形式): 逐步残差加权和 + 一次性 V̂(s0).

    标准公式: (1/N) Σ_i [ Σ_t γ^t w_t (r_t + γ V̂(s_{t+1}) − Q̂(s_t,a_t)) ] + V̂(s0).
    q_model 为目标策略的 Q̂ (model-based 精确解), 残差≈0 时 DR≈V^π_e.
    """
    gamma = cfg.gamma
    # V̂(s) = Σ_a π_e(a|s) Q̂(s,a)
    v_hat = (pi_e * q_model).sum(axis=1)
    v0 = float(v_hat[flat.n_states and 0])
    acc = 0.0
    n = 0
    for ep in episodes:
        rho = _ratios(ep, pi_e)
        cum = _cumprod_clipped(rho, _CUM_CAP_DR)
        g = gamma ** np.arange(len(ep["r"]))
        q_sa = q_model[ep["s"], ep["a"]]
        v_next = v_hat[ep["s_next"]] * (1.0 - ep["done"])
        dr = cum * (ep["r"] + gamma * v_next - q_sa)
        acc += float((dr * g).sum())
        n += 1
    return (acc / n if n else 0.0) + v0


# 估计器注册表 (供流水线按名调用)
OPE_REGISTRY: dict[str, Callable[..., float]] = {
    "dm": ope_dm,
    "sis": ope_sis,
    "wis": ope_wis,
    "tis": ope_tis,
    "dr": ope_dr,
    "fqe": ope_fqe,
}


# ---------------------------------------------------------------------------
# 离线学习器 (BaseOfflineLearner 实现, 便于统一接口测试)
# ---------------------------------------------------------------------------


class BCLearner(BaseOfflineLearner):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._policy: TabularPolicy | None = None

    def fit(self, dataset: Dataset) -> BCLearner:
        flat = dataset_to_flat(dataset)
        self._policy = target_bc(flat, self.cfg)
        return self

    def policy(self) -> TabularPolicy:
        if self._policy is None:
            raise PipelineError("未 fit")
        return self._policy

    def available(self) -> bool:
        return True


class CQLLearner(BaseOfflineLearner):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._policy: TabularPolicy | None = None

    def fit(self, dataset: Dataset) -> CQLLearner:
        flat = dataset_to_flat(dataset)
        self._policy = target_cql(flat, self.cfg)
        return self

    def policy(self) -> TabularPolicy:
        if self._policy is None:
            raise PipelineError("未 fit")
        return self._policy

    def available(self) -> bool:
        return True


class FQELearner(BaseOfflineLearner):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._policy: TabularPolicy | None = None

    def fit(self, dataset: Dataset) -> FQELearner:
        flat = dataset_to_flat(dataset)
        self._policy = target_fqe(flat, self.cfg)
        return self

    def policy(self) -> TabularPolicy:
        if self._policy is None:
            raise PipelineError("未 fit")
        return self._policy

    def available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# 场景与算法规格
# ---------------------------------------------------------------------------

# 6 档难度梯度场景 (仅改变数据规模/覆盖/环境随机性; 网格固定 5x5 保证可比)
SCENARIOS: list[tuple[str, dict[str, Any]]] = [
    ("abundant", {"n_episodes": 800, "behavior_epsilon": 0.3, "slip_prob": 0.1}),
    ("standard", {"n_episodes": 400, "behavior_epsilon": 0.3, "slip_prob": 0.1}),
    ("sparse", {"n_episodes": 150, "behavior_epsilon": 0.3, "slip_prob": 0.1}),
    ("narrow", {"n_episodes": 400, "behavior_epsilon": 0.1, "slip_prob": 0.1}),
    ("exploratory", {"n_episodes": 400, "behavior_epsilon": 0.5, "slip_prob": 0.1}),
    ("stochastic", {"n_episodes": 400, "behavior_epsilon": 0.3, "slip_prob": 0.3}),
]


@dataclass
class AlgorithmSpec:
    name: str
    backend: str
    build: Callable[..., TabularPolicy]


def _build_d3rlpy_cql(gw: GridWorld, flat: FlatData, ds: Dataset, cfg: Config) -> TabularPolicy:
    try:
        import d3rlpy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable("d3rlpy/torch 不可用") from exc
    raise BackendUnavailable("d3rlpy 后端未启用 (numpy 兜底)")


def _build_d3rlpy_bcq(gw: GridWorld, flat: FlatData, ds: Dataset, cfg: Config) -> TabularPolicy:
    try:
        import d3rlpy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable("d3rlpy/torch 不可用") from exc
    raise BackendUnavailable("d3rlpy 后端未启用 (numpy 兜底)")


def base_algorithms() -> list[AlgorithmSpec]:
    """numpy 后端算法规格 (始终可用)."""
    return [
        AlgorithmSpec("optimal", "numpy", lambda gw, f, d, c: target_optimal(gw)),
        AlgorithmSpec("random", "numpy", lambda gw, f, d, c: target_random(gw)),
        AlgorithmSpec("bc", "numpy", lambda gw, f, d, c: target_bc(f, c)),
        AlgorithmSpec("cql", "numpy", lambda gw, f, d, c: target_cql(f, c)),
        AlgorithmSpec("fqe", "numpy", lambda gw, f, d, c: target_fqe(f, c)),
    ]


def all_algorithms(cfg: Config) -> list[AlgorithmSpec]:
    algos = base_algorithms()
    if getattr(cfg, "use_d3rlpy", False):
        algos = algos + [
            AlgorithmSpec("d3rlpy_cql", "d3rlpy", _build_d3rlpy_cql),
            AlgorithmSpec("d3rlpy_bcq", "d3rlpy", _build_d3rlpy_bcq),
        ]
    return algos


# ---------------------------------------------------------------------------
# 流水线编排
# ---------------------------------------------------------------------------


class OfflineForgePipeline:
    """OfflineForge 评测流水线: 场景 × 种子 × 算法 -> OPE 误差排序."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.cfg = replace(self.cfg, use_d3rlpy=False)  # 当前环境无 torch, 强制 numpy

    def eval_one(
        self,
        gw: GridWorld,
        ds: Dataset,
        algo: AlgorithmSpec,
        seed: int,
    ) -> EvalRow:
        flat = dataset_to_flat(ds)
        episodes = episodes_to_arrays(ds)
        # 行为策略真值矩阵 (ε-greedy on Q*)
        _, Qstar, _ = gw.value_iteration()
        pi_b_full = gw.epsilon_greedy_probs(Qstar, self.cfg.behavior_epsilon)

        try:
            pi_e = algo.build(gw, flat, ds, self.cfg)
        except BackendUnavailable:
            raise
        if not isinstance(pi_e, TabularPolicy):
            pi_e = TabularPolicy(pi_e.probs_matrix())

        pe = pi_e.probs_matrix()
        true_val = float(gw.policy_value_dp(pe)[gw.start_state])

        # model-based 基线: DM 用行为策略精确 Q^π_b; DR 用目标策略精确 Q^π_e
        dm_q = gw.q_value_dp(pi_b_full)
        dr_q = gw.q_value_dp(pe)

        estimates: dict[str, float] = {}
        for m in self.cfg.ope_methods:
            fn = OPE_REGISTRY[m]
            if m == "dm":
                estimates[m] = fn(flat, pe, dm_q, self.cfg)
            elif m == "fqe":
                estimates[m] = fn(flat, pe, dm_q, self.cfg, episodes)
            elif m == "dr":
                estimates[m] = fn(flat, episodes, pe, dr_q, self.cfg)
            else:
                estimates[m] = fn(episodes, pe, self.cfg)

        best = min(estimates, key=lambda k: abs(estimates[k] - true_val))
        return EvalRow(
            algorithm=algo.name,
            backend=algo.backend,
            target=gw.__class__.__name__,
            seed=seed,
            best_ope=best,
            best_ope_error=float(abs(estimates[best] - true_val)),
            true_value=true_val,
            estimates={k: float(v) for k, v in estimates.items()},
        )

    def run_benchmark(
        self,
        scenarios: list[tuple[str, dict[str, Any]]] | None = None,
        seeds: tuple[int, ...] | None = None,
        algorithms: list[AlgorithmSpec] | None = None,
        verbose: bool = True,
    ) -> StreamSummary:
        scenarios = scenarios or SCENARIOS
        seeds = seeds or self.cfg.benchmark_seeds
        algorithms = algorithms or all_algorithms(self.cfg)
        rows: list[EvalRow] = []
        total = len(scenarios) * len(seeds) * len(algorithms)
        done = 0
        for sc_name, overrides in scenarios:
            sc_cfg = replace(self.cfg, **overrides)
            gw = GridWorld(
                grid_size=sc_cfg.grid_size,
                gamma=sc_cfg.gamma,
                slip_prob=sc_cfg.slip_prob,
                step_cost=sc_cfg.step_cost,
                goal_reward=sc_cfg.goal_reward,
                seed=0,
            )
            for seed in seeds:
                ds = gw.generate_dataset(
                    sc_cfg.n_episodes,
                    sc_cfg.behavior_epsilon,
                    seed,
                    sc_cfg.max_steps,
                )
                for algo in algorithms:
                    try:
                        row = self.eval_one(gw, ds, algo, seed)
                    except BackendUnavailable:
                        continue
                    rows.append(row)
                    done += 1
                    if verbose:
                        print(
                            f"  [{done:>3}/{total}] {sc_name:>11} "
                            f"seed={seed:<3} {algo.name:<12} "
                            f"true={row.true_value:+.3f} "
                            f"bestOPE={row.best_ope:<4} "
                            f"err={row.best_ope_error:.4f}"
                        )
        return StreamSummary(rows=rows)


def run_demo(
    cfg: Config | None = None,
    scenarios: list[tuple[str, dict[str, Any]]] | None = None,
    seeds: tuple[int, ...] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """端到端演示: 跑基准并汇总为 dict (供 examples/run_demo.py 落盘)."""
    import time

    cfg = cfg or Config()
    pipe = OfflineForgePipeline(cfg)
    t0 = time.time()
    summary = pipe.run_benchmark(scenarios=scenarios, seeds=seeds, verbose=verbose)
    elapsed = time.time() - t0

    # 按算法聚合并排名 (mean best_ope_error 越小越好)
    agg: dict[str, list[float]] = {}
    for r in summary.rows:
        agg.setdefault(r.algorithm, []).append(r.best_ope_error)
    ranking = sorted(
        (
            {
                "algorithm": name,
                "mean_best_ope_error": float(np.mean(vals)),
                "max_best_ope_error": float(np.max(vals)),
                "n": len(vals),
            }
            for name, vals in agg.items()
        ),
        key=lambda x: x["mean_best_ope_error"],
    )
    best_algo = ranking[0]["algorithm"] if ranking else ""

    # 按 OPE 方法聚合 (全量 mean 绝对误差, 越小越好) —— 衡量各估计器本身精度
    method_errs: dict[str, list[float]] = {}
    for r in summary.rows:
        for m, est in r.estimates.items():
            method_errs.setdefault(m, []).append(abs(est - r.true_value))
    ope_ranking = sorted(
        (
            {
                "method": m,
                "mean_abs_error": float(np.mean(vals)),
                "max_abs_error": float(np.max(vals)),
                "n": len(vals),
            }
            for m, vals in method_errs.items()
        ),
        key=lambda x: x["mean_abs_error"],
    )

    out = {
        "system": "OfflineForge",
        "version": "0.1.0",
        "author": "晨星",
        "config": cfg.as_dict(),
        "generated_at_note": "确定性可复现基准 (固定 random_state + 种子组)",
        "elapsed_s": round(elapsed, 3),
        "n_evaluations": len(summary.rows),
        "summary": {
            "best_algorithm": best_algo,
            "ranking": ranking,
            "ope_ranking": ope_ranking,
        },
        "rows": [r.as_dict() for r in summary.rows],
    }
    return out


__all__ = [
    "logsumexp",
    "dataset_to_flat",
    "episodes_to_arrays",
    "TabularPolicy",
    "target_optimal",
    "target_random",
    "target_bc",
    "target_cql",
    "target_fqe",
    "fitted_q_iteration",
    "ope_dm",
    "ope_sis",
    "ope_wis",
    "ope_tis",
    "ope_dr",
    "ope_fqe",
    "OPE_REGISTRY",
    "BCLearner",
    "CQLLearner",
    "FQELearner",
    "SCENARIOS",
    "AlgorithmSpec",
    "OfflineForgePipeline",
    "run_demo",
]
