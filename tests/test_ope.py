"""OPE 估计器不变量测试 (有限性 / 交叉验证 / 重要性权重裁剪)."""

from __future__ import annotations

import math

from offlineforge.core.config import Config
from offlineforge.data.gridworld import GridWorld
from offlineforge.domain.offline.numpy_impl import (
    OPE_REGISTRY,
    dataset_to_flat,
    episodes_to_arrays,
    ope_dm,
    ope_dr,
    ope_fqe,
    ope_sis,
    ope_tis,
    ope_wis,
    target_bc,
    target_cql,
    target_optimal,
)


def _make(seed: int = 42, n: int = 600, eps: float = 0.3):
    cfg = Config()
    gw = GridWorld(grid_size=5, gamma=0.95, slip_prob=0.1, seed=0)
    ds = gw.generate_dataset(n, eps, seed, 50)
    flat = dataset_to_flat(ds)
    eps_arr = episodes_to_arrays(ds)
    _, Q, _ = gw.value_iteration()
    pi_b = gw.epsilon_greedy_probs(Q, eps)
    return cfg, gw, ds, flat, eps_arr, pi_b


def test_ope_registry_complete():
    assert set(OPE_REGISTRY) == {"dm", "sis", "wis", "tis", "dr", "fqe"}


def test_all_estimators_finite_and_reasonable():
    cfg, gw, ds, flat, eps_arr, pi_b = _make()
    dm_q = gw.q_value_dp(pi_b)
    pol = target_bc(flat, cfg)
    pm = pol.probs_matrix()
    dr_q = gw.q_value_dp(pm)
    est = {
        "dm": ope_dm(flat, pm, dm_q, cfg),
        "fqe": ope_fqe(flat, pm, dm_q, cfg, eps_arr),
        "sis": ope_sis(eps_arr, pm, cfg),
        "wis": ope_wis(eps_arr, pm, cfg),
        "tis": ope_tis(eps_arr, pm, cfg),
        "dr": ope_dr(flat, eps_arr, pm, dr_q, cfg),
    }
    for name, val in est.items():
        assert math.isfinite(val), f"{name} 非有限: {val}"
    # 经验范围: 价值应在合理区间
    for val in est.values():
        assert -5.0 < val < 5.0


def test_dr_near_exact_for_optimal():
    """DR (model-based 精确 Q + IS) 评估最优策略应近精确 (误差 < 1e-2)."""
    cfg, gw, ds, flat, eps_arr, pi_b = _make(n=800, eps=0.3)
    pol = target_optimal(gw)
    pm = pol.probs_matrix()
    tv = gw.policy_value_dp(pm)[gw.start_state]
    dr_q = gw.q_value_dp(pm)
    est = ope_dr(flat, eps_arr, pm, dr_q, cfg)
    assert abs(est - tv) < 1e-2


def test_dm_is_biased_for_optimal():
    """DM 用行为策略 Q 评估最优策略 -> 应系统性低估 (偏差)."""
    cfg, gw, ds, flat, eps_arr, pi_b = _make(n=800, eps=0.3)
    pol = target_optimal(gw)
    pm = pol.probs_matrix()
    tv = gw.policy_value_dp(pm)[gw.start_state]
    dm_q = gw.q_value_dp(pi_b)
    est = ope_dm(flat, pm, dm_q, cfg)
    # 最优策略价值高于行为策略, DM 用行为 Q 低估
    assert est < tv - 0.05


def test_fqe_close_for_optimal():
    cfg, gw, ds, flat, eps_arr, pi_b = _make(n=800, eps=0.3)
    pol = target_optimal(gw)
    pm = pol.probs_matrix()
    tv = gw.policy_value_dp(pm)[gw.start_state]
    dm_q = gw.q_value_dp(pi_b)
    est = ope_fqe(flat, pm, dm_q, cfg, eps_arr)
    assert abs(est - tv) < 0.05


def test_estimators_stable_across_seeds():
    """同场景不同种子: 估计量应有限且不爆炸 (权重裁剪生效)."""
    for seed in (42, 123, 7):
        cfg, gw, ds, flat, eps_arr, pi_b = _make(seed=seed)
        pol = target_cql(flat, cfg)
        pm = pol.probs_matrix()
        dm_q = gw.q_value_dp(pi_b)
        dr_q = gw.q_value_dp(pm)
        for m in ("dm", "sis", "wis", "tis", "dr"):
            if m == "dm":
                v = ope_dm(flat, pm, dm_q, cfg)
            elif m == "dr":
                v = ope_dr(flat, eps_arr, pm, dr_q, cfg)
            else:
                v = OPE_REGISTRY[m](eps_arr, pm, cfg)
            assert math.isfinite(v)
