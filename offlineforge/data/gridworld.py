"""GridWorld MDP — 纯 numpy 合成环境.

提供:
  * 随机/确定性转移 (含 slip), 价值迭代 (DP) 求 V*/Q*/pi_opt;
  * 任意目标策略的真值价值 policy_value_dp (线性系统求解, 作为 OPE 真值);
  * epsilon-greedy 行为策略与离线数据集生成 (记录 pi_b(a|s), 使 OPE 重要性权重精确).

全部随机性来自 seeded numpy Generator, 保证可复现.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError, NumericalError
from ..core.types import Dataset, Transition

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
_ACTIONS = (UP, DOWN, LEFT, RIGHT)
_PERP = {UP: (LEFT, RIGHT), DOWN: (LEFT, RIGHT), LEFT: (UP, DOWN), RIGHT: (UP, DOWN)}


class GridWorld:
    """方格世界 MDP."""

    def __init__(
        self,
        grid_size: int = 5,
        gamma: float = 0.95,
        slip_prob: float = 0.1,
        step_cost: float = -0.04,
        goal_reward: float = 1.0,
        seed: int = 0,
    ) -> None:
        if grid_size < 2:
            raise DataError("grid_size 必须 >= 2")
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4
        self.gamma = float(gamma)
        self.slip_prob = float(slip_prob)
        self.step_cost = float(step_cost)
        self.goal_reward = float(goal_reward)
        self.start_state = 0
        self.goal_state = self.n_states - 1
        self.rng = np.random.default_rng(seed)
        self._trans: list[list[list[tuple[int, float]]]] = self._build_transitions()

    # ---------- 几何 ----------
    def _rc(self, s: int) -> tuple[int, int]:
        return divmod(s, self.grid_size)

    def _idx(self, r: int, c: int) -> int:
        return r * self.grid_size + c

    def _intended_next(self, s: int, a: int) -> int:
        r, c = self._rc(s)
        if a == UP:
            r = max(0, r - 1)
        elif a == DOWN:
            r = min(self.grid_size - 1, r + 1)
        elif a == LEFT:
            c = max(0, c - 1)
        elif a == RIGHT:
            c = min(self.grid_size - 1, c + 1)
        return self._idx(r, c)

    def _build_transitions(self) -> list[list[list[tuple[int, float]]]]:
        trans: list[list[list[tuple[int, float]]]] = []
        for s in range(self.n_states):
            row_a: list[list[tuple[int, float]]] = []
            for a in _ACTIONS:
                ns = self._intended_next(s, a)
                perp = _PERP[a]
                p_perp = self.slip_prob / 2.0
                lst: list[tuple[int, float]] = [(ns, 1.0 - self.slip_prob)]
                for pa in perp:
                    lst.append((self._intended_next(s, pa), p_perp))
                # 合并同一目标
                merged: dict[int, float] = {}
                for nxt, p in lst:
                    merged[nxt] = merged.get(nxt, 0.0) + p
                row_a.append([(k, v) for k, v in merged.items()])
            trans.append(row_a)
        return trans

    def reward(self, s: int, ns: int) -> float:
        return self.goal_reward if ns == self.goal_state else self.step_cost

    def is_terminal(self, s: int) -> bool:
        return s == self.goal_state

    # ---------- 动态规划 ----------
    def value_iteration(
        self, max_iter: int = 5000, tol: float = 1e-10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """价值迭代: 返回 V* (S), Q* (S,A), pi_opt (S)."""
        V = np.zeros(self.n_states)
        Q = np.zeros((self.n_states, self.n_actions))
        for _ in range(max_iter):
            max_delta = 0.0
            for s in range(self.n_states):
                if self.is_terminal(s):
                    Q[s, :] = 0.0
                    V[s] = 0.0
                    continue
                for a in _ACTIONS:
                    q = 0.0
                    for ns, p in self._trans[s][a]:
                        q += p * (self.reward(s, ns) + self.gamma * V[ns])
                    Q[s, a] = q
                v_new = float(np.max(Q[s]))
                max_delta = max(max_delta, abs(v_new - V[s]))
                V[s] = v_new
            if max_delta < tol:
                break
        else:
            raise NumericalError("value_iteration 未收敛")
        pi_opt = np.argmax(Q, axis=1).astype(int)
        if not np.all(np.isfinite(V)):
            raise NumericalError("V* 含非有限值")
        return V, Q, pi_opt

    def q_value_dp(self, pi_probs: np.ndarray) -> np.ndarray:
        """任意目标策略的真值动作价值 Q^π (基于已知 MDP 模型), 供 model-based DM/DR 使用."""
        if pi_probs.shape != (self.n_states, self.n_actions):
            raise DataError("pi_probs 形状必须为 (S, A)")
        V = self.policy_value_dp(pi_probs)
        Q = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                q = 0.0
                for ns, p in self._trans[s][a]:
                    q += p * (
                        self.reward(s, ns)
                        + self.gamma * V[ns] * (0.0 if self.is_terminal(ns) else 1.0)
                    )
                Q[s, a] = q
        # 终止态不采取动作, Q=0 (与 policy_value_dp 中 V=0 一致)
        for s in range(self.n_states):
            if self.is_terminal(s):
                Q[s, :] = 0.0
        if not np.all(np.isfinite(Q)):
            raise NumericalError("q_value_dp 含非有限值")
        return Q

    def policy_value_dp(self, pi_probs: np.ndarray) -> np.ndarray:
        """任意目标策略的真值价值 V^pi (线性系统求解), 作为 OPE 真值.

        pi_probs: (S,A) 行概率分布.
        """
        if pi_probs.shape != (self.n_states, self.n_actions):
            raise DataError("pi_probs 形状必须为 (S, A)")
        if not np.allclose(pi_probs.sum(axis=1), 1.0, atol=1e-6):
            raise DataError("pi_probs 每行须为概率分布")
        P = np.zeros((self.n_states, self.n_states))
        R = np.zeros(self.n_states)
        for s in range(self.n_states):
            if self.is_terminal(s):
                continue
            for a in _ACTIONS:
                pa = pi_probs[s, a]
                for ns, p in self._trans[s][a]:
                    P[s, ns] += pa * p
                    R[s] += pa * p * self.reward(s, ns)
        A_mat = np.eye(self.n_states) - self.gamma * P
        try:
            V = np.linalg.solve(A_mat, R)
        except np.linalg.LinAlgError as exc:
            raise NumericalError("策略价值线性系统奇异") from exc
        if not np.all(np.isfinite(V)):
            raise NumericalError("policy_value_dp 含非有限值")
        return V

    # ---------- 行为策略与数据集 ----------
    def epsilon_greedy_probs(self, Q: np.ndarray, eps: float) -> np.ndarray:
        """基于 Q* 的 epsilon-greedy 行为策略 (保证动作覆盖, IS 比值有限)."""
        pi_b = np.full((self.n_states, self.n_actions), eps / self.n_actions)
        greedy = np.argmax(Q, axis=1)
        for s in range(self.n_states):
            if self.is_terminal(s):
                continue
            pi_b[s, greedy[s]] += 1.0 - eps
        # 归一化 (浮点修正)
        pi_b /= pi_b.sum(axis=1, keepdims=True)
        return pi_b

    def generate_dataset(
        self,
        n_episodes: int,
        eps: float,
        seed: int,
        max_steps: int = 50,
        pi_b: np.ndarray | None = None,
    ) -> Dataset:
        """生成离线数据集, 记录每条样本的 pi_b(a|s) (OPE 真值权重来源)."""
        rng = np.random.default_rng(seed)
        if pi_b is None:
            _, Q, _ = self.value_iteration()
            pi_b = self.epsilon_greedy_probs(Q, eps)
        episodes: list[list[Transition]] = []
        for ep in range(n_episodes):
            s = self.start_state
            t = 0
            traj: list[Transition] = []
            while not self.is_terminal(s) and t < max_steps:
                a = int(rng.choice(self.n_actions, p=pi_b[s]))
                pb = float(pi_b[s, a])
                ns = self._intended_next(s, a)
                # 随机 slip
                if rng.random() < self.slip_prob:
                    pa = int(rng.choice(_PERP[a]))
                    ns = self._intended_next(s, pa)
                r = self.reward(s, ns)
                done = self.is_terminal(ns)
                traj.append(Transition(s=s, a=a, r=r, s_next=ns, done=done, pi_b=pb, t=t))
                s = ns
                t += 1
                if done:
                    break
            episodes.append(traj)
        name = f"grid{self.grid_size}_eps{eps}_n{n_episodes}"
        return Dataset(
            name=name,
            n_states=self.n_states,
            n_actions=self.n_actions,
            gamma=self.gamma,
            episodes=episodes,
            behavior_policy="epsilon_greedy",
            start_state=self.start_state,
            goal_state=self.goal_state,
            grid_size=self.grid_size,
        )


__all__ = ["GridWorld", "UP", "DOWN", "LEFT", "RIGHT"]
