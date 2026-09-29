"""OfflineForge 命令行入口: 跑完整基准并落盘 benchmark.json."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from offlineforge.domain.offline.numpy_impl import run_demo


def main(argv: list[str] | None = None) -> int:
    out: dict[str, Any] = run_demo(verbose=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n=== OfflineForge 排名 (mean best OPE error, 越小越好) ===")
    for r in out["summary"]["ranking"]:
        print(f"  {r['algorithm']:<8}  {r['mean_best_ope_error']:.4f}")
    print(f"\nDONE -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
