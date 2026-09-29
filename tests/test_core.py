"""OfflineForge 核心不变量测试 (DP 交叉验证 / 数据集合法性 / 类型)."""

from __future__ import annotations

import numpy as np
import pytest

from offlineforge.core.config import Config
from offlineforge.core.errors import DataError
from offlineforge.core.types import Transition
from offlineforge.data.gridworld import GridWorld


def test_value_iteration_matches_dp_optimal():
    """VI 最优值迭代 V* 必须等于 policy_value_dp(π_opt) (解析交叉验证)."""
    gw = GridWorld(grid_size=5, gamma=0.95, slip_prob=0.1, seed=0)
    V, Q, pi_opt = gw.value_iteration()
    pi_probs = np.zeros((gw.n_states, gw.n_actions))
    pi_probs[np.arange(gw.n_states), pi_opt] = 1.0
    v_dp = gw.policy_value_dp(pi_probs)
    assert np.allclose(V, v_dp, atol=1e-9)


def test_optimal_dominates_random():
    """最优策略 DP 价值 >= 随机策略 DP 价值 (最优性)."""
    gw = GridWorld(grid_size=5, gamma=0.95, slip_prob=0.1, seed=1)
    _, Q, pi_opt = gw.value_iteration()
    pi_opt_probs = np.zeros((gw.n_states, gw.n_actions))
    pi_opt_probs[np.arange(gw.n_states), pi_opt] = 1.0
    v_opt = gw.policy_value_dp(pi_opt_probs)[gw.start_state]
    pi_rand = np.full((gw.n_states, gw.n_actions), 1.0 / gw.n_actions)
    v_rand = gw.policy_value_dp(pi_rand)[gw.start_state]
    assert v_opt >= v_rand


def test_q_value_dp_consistency():
    """q_value_dp 与 policy_value_dp 在贪心下一致: V(s) = max_a Q(s,a)."""
    gw = GridWorld(grid_size=4, gamma=0.95, slip_prob=0.1, seed=2)
    _, Q, pi_opt = gw.value_iteration()
    pi_probs = np.zeros((gw.n_states, gw.n_actions))
    pi_probs[np.arange(gw.n_states), pi_opt] = 1.0
    Qdp = gw.q_value_dp(pi_probs)
    Vdp = gw.policy_value_dp(pi_probs)
    assert np.allclose(Qdp.max(axis=1), Vdp, atol=1e-9)


def test_generate_dataset_records_pi_b():
    """数据集每条样本的 pi_b 合法 ( >0 ), 且转移一致."""
    gw = GridWorld(grid_size=5, gamma=0.95, slip_prob=0.1, seed=3)
    ds = gw.generate_dataset(120, 0.3, 7, 50)
    assert ds.n_episodes == 120
    assert ds.n_transitions > 0
    for ep in ds.episodes:
        for tr in ep:
            assert tr.pi_b > 0.0
            assert 0 <= tr.s < gw.n_states
            assert 0 <= tr.a < gw.n_actions


def test_dataset_name_uses_grid_size():
    """回归: generate_dataset 的 name 应引用 self.grid_size (旧 bug: 未定义变量)."""
    gw = GridWorld(grid_size=6, gamma=0.95, slip_prob=0.1, seed=4)
    ds = gw.generate_dataset(50, 0.3, 7, 50)
    assert "grid6" in ds.name


def test_config_from_env(monkeypatch):
    """ENV 覆盖生效."""
    monkeypatch.setenv("OFF_GRID_SIZE", "7")
    monkeypatch.setenv("OFF_N_EPISODES", "222")
    cfg = Config.from_env()
    assert cfg.grid_size == 7
    assert cfg.n_episodes == 222


def test_config_invalid_env(monkeypatch):
    monkeypatch.setenv("OFF_GRID_SIZE", "notanint")
    with pytest.raises(ValueError):
        Config.from_env()


def test_dataset_transition_type():
    t = Transition(s=0, a=1, r=0.5, s_next=2, done=False, pi_b=0.25)
    d = t.as_dict()
    assert d["s"] == 0 and d["a"] == 1 and d["pi_b"] == 0.25


def test_gridworld_small_rejected():
    with pytest.raises(DataError):
        GridWorld(grid_size=1)
