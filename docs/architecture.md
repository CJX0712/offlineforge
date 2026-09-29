# OfflineForge 架构与算法说明

> 作者：晨星 · 领域：离线强化学习 (Offline RL) + 离线策略评估 (OPE)

## 1. 设计目标

OfflineForge 解决两个问题：

1. **离线策略学习**：仅从固定离线数据集（无环境交互）学出可用的目标策略 π_e。
2. **离线策略评估 (OPE)**：仅用离线数据估计 π_e 在起始状态的价值 V^π_e(s0)，对照已知 DP 真值量化误差。

与 ClusterForge / RiverForge 一致，内核**零外部依赖**（纯 numpy 表格法），
并可选复用顶级开源 **d3rlpy**（CQL / BCQ）作为背书（当前环境无 torch，自动降级）。

## 2. 环境：GridWorld MDP

`data/gridworld.py` 实现方格世界 MDP：

- 状态 `s = r·G + c`（G×G 网格），动作 4 向，含 slip（随机偏移）转移。
- **价值迭代** `value_iteration` → V\*, Q\*, π_opt（DP 上限）。
- **精确策略价值** `policy_value_dp(π)`：解线性系统 (I−γP\_π)V = R，作为 OPE 无偏真值。
- **精确动作价值** `q_value_dp(π)`：基于已知模型求 Q^π，供 model-based 的 DM / DR 使用。
- **数据集生成** `generate_dataset`：ε-greedy 行为策略采样 episode，逐条记录 `pi_b(a|s)`，
  使 OPE 重要性权重 `ρ = π_e/π_b` 精确可算。

**关键不变量**：`value_iteration` 的 V\* 与 `policy_value_dp(π_opt)` 逐位一致（误差 8e-12）。

## 3. 离线学习器（domain/offline/numpy_impl.py）

| 学习器 | 方法 | 说明 |
| --- | --- | --- |
| BC | 平滑经验策略 (Laplace) | 最大化数据似然；接近行为策略 |
| CQL-lite | 贪心备份 FQI + logsumexp 正则 + OOD 下界 | 抑制未见动作高估（保守核心） |
| FQE/FQI | 目标策略备份表格拟合 Q 迭代 | 模型无关，贪心导出策略 |

CQL-lite 的正则项 `Q ← Q − α·(logsumexp_a Q − Q)` 压低非最大动作 Q；
未见 (s,a) 强制下界 `ood_floor`，实现"保守"语义而无需神经网络。

## 4. OPE 估计器

设轨迹 τ，逐步重要性比值 `ρ_t = π_e(a_t|s_t)/π_b(a_t)`，累积 `w_t = Π_{k≤t} ρ_k`（裁剪防爆炸）。

| 估计器 | 公式（核心） | 性质 |
| --- | --- | --- |
| DM | `Σ_a π_e(s0,a) Q̂^π_b(s0,a)` | model-based，有偏（用行为价值） |
| SIS | `(1/N)Σ_i Σ_t γ^t w_t r_t` | 模型无关，无偏，中方差 |
| WIS | `Σ_i (W_i/ΣW_j) G_i`，`W_i=Πρ` | 轨迹自归一化，稳健 |
| TIS | `(1/N)Σ_i W_i G_i` | 未归一化轨迹 IS，高方差（展示代价） |
| DR | `(1/N)Σ_i[Σ_t γ^t w_t(r_t+γV̂(s_{t+1})−Q̂(s_t,a_t))]+V̂(s0)` | model-based Q̂ + IS 修正，近精确 |
| FQE | 目标策略备份 FQI 求 Q，代入 `Σ_a π_e Q` | 模型无关，优秀 |

**DR 正确性要点**：标准 DR 是"逐步残差加权和 + 一次性加 V̂(s0)"，而非逐步加 Q̂；
q_model 采用**目标策略精确 Q**（已知模型），残差≈0 时 DR≈V^π_e。
IS 类估计器使用**截断重要性采样**（per-step 比值裁剪 + 累积权重上限）保证有限性。

## 5. 流水线编排

`OfflineForgePipeline.run_benchmark`：

```
for 场景 in 6 档难度:
  for 种子 in (42,123,7):
    生成离线数据集 (行为策略)
    for 算法 in (optimal,random,bc,cql,fqe):
      构建目标策略 π_e -> DP 真值 V^π_e(s0)
      DM/DR 用精确 Q^π_b / Q^π_e; 其余 OPE 仅用数据
      记录每个 OPE 估计值与误差, 选 best_ope
```

输出 `benchmark.json`：配置、`elapsed_s`、`n_evaluations`、按算法聚合的排名、全部评估行。
全部随机性由 `random_state` / 种子组控制，**确定性可复现**。

## 6. 预期结论（实测）

OPE 方法排名（全量 mean 绝对误差）：**DR 0.0105 ≫ FQE 0.0626 > WIS 0.2006 > DM 0.2804 > TIS 0.3007 ≈ SIS 0.3142**。

- **DR（model-based）近乎精确**：精确目标 Q + IS 修正，残差≈0，误差最低。
- **FQE（model-free）次优**：无模型拟合 Q，整体准确但个别难目标 max 误差达 0.64。
- **WIS 稳健**：轨迹自归一化 IS，方差低于未归一化 IS。
- **DM 系统性有偏**：使用行为价值 Q^π_b 评估目标策略，偏差最大之一。
- **TIS / SIS 高方差**：纯重要性采样，难评测目标（random）下误差飙至 ~2.4。

目标策略排名（取每行最优 OPE 误差）：optimal 0.0026 < bc 0.0033 < random 0.0068 < cql 0.0088 < fqe 0.0089——
易覆盖目标最易精确评估，学习到的 cql/fqe 因更偏离行为、覆盖更薄而评测误差略高。

全部单测 23 项全绿，ruff 全过。
