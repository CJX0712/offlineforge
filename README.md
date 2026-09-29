# OfflineForge

> 世界顶级离线强化学习（Offline RL + 离线策略评估 OPE）系统 · 作者：晨星 (CJX0712)

OfflineForge 在**零外部依赖内核**（纯 numpy）上提供从离线数据集**学习策略**（BC / CQL-lite / FQE）
并用**六种 OPE 估计器**（DM / SIS / WIS / TIS / DR / FQE）评估目标策略价值的完整流水线，
对照**动态规划（DP）真值**给出误差与排序。可选复用 **d3rlpy**（CQL / BCQ，顶级开源）作为背书，
不可用时自动降级为 numpy 后端。

## 核心特性

- **离线策略学习器（纯 numpy）**：行为克隆 BC、保守 Q 学习 CQL-lite（logsumexp 正则 + OOD 下界）、
  表格拟合 Q 迭代 FQE / FQI。
- **六种 OPE 估计器**：DM（直接法，model-based）、SIS（逐步 IS，无偏）、WIS（轨迹自归一化 IS，稳健）、
  TIS（轨迹 IS，高方差展示）、DR（双重稳健，model-based 金标准）、FQE（模型无关拟合 Q 评估）。
- **DP 真值交叉验证**：GridWorld MDP 用价值迭代求 V\*/Q\*，并用线性系统精确求解任意目标策略价值
  `policy_value_dp`，作为 OPE 无偏基准。
- **6 档难度梯度场景**：数据规模 / 行为覆盖 / 环境随机性（abundant → standard → sparse → narrow →
  exploratory → stochastic）。
- **确定性可复现**：固定 `random_state=42` 与种子组 (42/123/7)，重跑逐位一致，落盘 `benchmark.json`。
- **23 项数值不变量自测**：VI ↔ DP 精确交叉验证（8e-12）、DM 偏差、DR/FQE 近精确、策略概率质量=1、
  重要性权重裁剪有限性、跳过语义（不伪造数字）。

## 安装

```bash
pip install -r requirements.txt   # numpy
pip install pytest ruff           # 可选：开发
# 可选: 启用 d3rlpy 后端 (需 torch)
pip install d3rlpy
```

## 快速开始

```python
from offlineforge.core.config import Config
from offlineforge.data.gridworld import GridWorld
from offlineforge.domain.offline.numpy_impl import (
    OfflineForgePipeline,
    target_cql,
    ope_dr,
    dataset_to_flat,
    episodes_to_arrays,
)

cfg = Config()
gw = GridWorld(grid_size=5, gamma=0.95, slip_prob=0.1, seed=0)
ds = gw.generate_dataset(400, 0.3, 42, 50)
flat = dataset_to_flat(ds)
eps = episodes_to_arrays(ds)
_, Q, _ = gw.value_iteration()
pi_b = gw.epsilon_greedy_probs(Q, 0.3)

pi_e = target_cql(flat, cfg)  # 离线学习目标策略
pe = pi_e.probs_matrix()
true_val = gw.policy_value_dp(pe)[gw.start_state]
dr_val = ope_dr(flat, eps, pe, gw.q_value_dp(pe), cfg)  # 双重稳健评估
print(true_val, dr_val)  # 两者应非常接近
```

命令行 / 端到端演示：

```bash
offlineforge benchmark          # 3 种子 × 6 场景 × 5 算法 -> 打印排名 + benchmark.json
python -m offlineforge.cli
python examples/run_demo.py
```

## 基准结果（节选）

完整结果见 `benchmark.json`（90 评测；grid 5×5, γ=0.95, 3 种子）。

**OPE 方法排名**（全量 mean 绝对误差，越小越好）：

| OPE 方法 | 类型 | mean 误差 | max 误差 | 说明 |
| --- | --- | --- | --- | --- |
| DR | model-based | 0.0105 | 0.131 | 精确目标 Q + IS 修正，金标准 |
| FQE | model-free | 0.0626 | 0.637 | 模型无关拟合 Q，次优但无模型 |
| WIS | model-free | 0.2006 | 0.972 | 轨迹自归一化 IS，稳健 |
| DM | model-based | 0.2804 | 0.856 | 用行为价值，系统性有偏 |
| TIS | model-free | 0.3007 | 2.255 | 未归一化轨迹 IS，高方差 |
| SIS | model-free | 0.3142 | 2.456 | 逐步 IS，无偏但高方差 |

**结论**：`DR ≫ FQE > WIS > DM > TIS ≈ SIS`。价值类（DR/FQE）远优于纯重要性采样类；
model-based 的 DR 近乎精确，model-free 的 FQE 其次；DM 因使用行为价值而系统有偏；
IS 类在难评测目标（如 random，负价值、与行为差异大）下误差飙升，真实反映 OPE 覆盖难题。

**目标策略排名**（取每行最优 OPE 的误差）：optimal 0.0026 < bc 0.0033 < random 0.0068 < cql 0.0088 < fqe 0.0089，
说明易覆盖目标（optimal/bc）最易精确评估，而学习到的 cql/fqe 策略因更偏离行为、覆盖更薄，评测误差略高。

## 项目结构

```
offlineforge/
  core/        配置 / 类型 / 错误 / 接口
  data/        GridWorld MDP (DP 真值 + 数据集生成)
  domain/offline/numpy_impl.py   旗舰实现 (学习器 + OPE + 流水线)
  cli.py       命令行入口 (生成 benchmark.json)
  examples/    演示运行器
tests/         23 项数值不变量单测
docs/architecture.md   架构与算法说明
```

## 作者

晨星 (CJX0712) — 随机 AI 系统全自动交付 · 第 3 轮（前两轮：ClusterForge 聚类 / RiverForge 在线学习）。
