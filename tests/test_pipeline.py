"""流水线编排测试 (benchmark 规模 / 排序 / run_demo 结构)."""

from __future__ import annotations

import numpy as np

from offlineforge.core.config import Config
from offlineforge.domain.offline.numpy_impl import (
    SCENARIOS,
    OfflineForgePipeline,
    base_algorithms,
    run_demo,
)


def _fast_cfg() -> Config:
    return Config(fqe_iterations=40, cql_alpha=1.0)


def test_run_benchmark_small():
    cfg = _fast_cfg()
    pipe = OfflineForgePipeline(cfg)
    summary = pipe.run_benchmark(
        scenarios=SCENARIOS[:1],
        seeds=(42,),
        algorithms=base_algorithms(),
        verbose=False,
    )
    rows = summary.rows
    assert len(rows) == len(base_algorithms())  # 5 算法 × 1 场景 × 1 种子
    for r in rows:
        assert r.algorithm in {a.name for a in base_algorithms()}
        assert r.best_ope in {"dm", "sis", "wis", "tis", "dr", "fqe"}
        assert np.isfinite(r.best_ope_error)
        # best_ope_error 必须不劣于所有估计器误差
        assert r.best_ope_error <= min(abs(v - r.true_value) for v in r.estimates.values()) + 1e-9


def test_run_demo_structure():
    out = run_demo(
        cfg=_fast_cfg(),
        scenarios=SCENARIOS[:1],
        seeds=(42,),
        verbose=False,
    )
    assert out["system"] == "OfflineForge"
    assert out["version"] == "0.1.0"
    assert out["author"] == "晨星"
    assert "summary" in out and "ranking" in out["summary"]
    ranking = out["summary"]["ranking"]
    assert len(ranking) == len(base_algorithms())
    # 排名按 mean_best_ope_error 升序
    errs = [r["mean_best_ope_error"] for r in ranking]
    assert errs == sorted(errs)
    # 行数一致
    assert out["n_evaluations"] == len(out["rows"])
    # 最优算法应为已知算法之一 (其目标策略在基准中最易评估)
    assert out["summary"]["best_algorithm"] in {a.name for a in base_algorithms()}


def test_scenarios_count():
    assert len(SCENARIOS) == 6
