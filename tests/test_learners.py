"""离线学习器 (BC / CQL / FQE) 与 FQI 不变量测试."""

from __future__ import annotations

import numpy as np

from offlineforge.core.config import Config
from offlineforge.data.gridworld import GridWorld
from offlineforge.domain.offline.numpy_impl import (
    BCLearner,
    CQLLearner,
    FQELearner,
    dataset_to_flat,
    fitted_q_iteration,
    target_bc,
    target_cql,
    target_fqe,
    target_optimal,
    target_random,
)


def _make(seed: int = 42):
    cfg = Config()
    gw = GridWorld(grid_size=5, gamma=0.95, slip_prob=0.1, seed=0)
    ds = gw.generate_dataset(300, 0.3, seed, 50)
    flat = dataset_to_flat(ds)
    return cfg, gw, ds, flat


def test_policy_prob_mass_one():
    cfg, gw, ds, flat = _make()
    for pol in [
        target_optimal(gw),
        target_random(gw),
        target_bc(flat, cfg),
        target_cql(flat, cfg),
        target_fqe(flat, cfg),
    ]:
        pm = pol.probs_matrix()
        assert np.allclose(pm.sum(axis=1), 1.0, atol=1e-9)
        assert (pm >= 0).all()
        assert pol.available()


def test_bc_resembles_behavior():
    """BC 在高频 (s,a) 上应与行为策略接近 (平滑经验分布)."""
    cfg, gw, ds, flat = _make()
    bc = target_bc(flat, cfg)
    _, Q, _ = gw.value_iteration()
    pi_b = gw.epsilon_greedy_probs(Q, 0.3)
    # 在支撑状态上, BC 的最大动作应与行为贪心动作一致
    agree = 0
    total = 0
    counts = np.zeros((gw.n_states, gw.n_actions))
    np.add.at(counts, (flat.s, flat.a), 1.0)
    for s in range(gw.n_states):
        if counts[s].sum() > 5:
            total += 1
            if np.argmax(bc.probs_matrix()[s]) == np.argmax(pi_b[s]):
                agree += 1
    assert total > 0
    assert agree / total >= 0.7


def test_cql_fqe_distinct_from_random():
    cfg, gw, ds, flat = _make()
    cql = target_cql(flat, cfg).probs_matrix()
    fqe = target_fqe(flat, cfg).probs_matrix()
    rnd = target_random(gw).probs_matrix()
    # 学习到的策略不应等于均匀随机 (信息量更高)
    assert not np.allclose(cql, rnd, atol=1e-3)
    assert not np.allclose(fqe, rnd, atol=1e-3)


def test_fqi_finite():
    cfg, gw, ds, flat = _make()
    _, Q, _ = gw.value_iteration()
    pi_b = gw.epsilon_greedy_probs(Q, 0.3)
    q = fitted_q_iteration(flat, pi_b, cfg)
    assert np.all(np.isfinite(q))
    assert q.shape == (gw.n_states, gw.n_actions)


def test_learner_interface():
    cfg, gw, ds, flat = _make()
    for L in (BCLearner, CQLLearner, FQELearner):
        learner = L(cfg).fit(ds)
        assert learner.available()
        pol = learner.policy()
        assert pol.available()
        assert np.allclose(pol.probs_matrix().sum(axis=1), 1.0, atol=1e-9)
