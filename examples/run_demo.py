"""OfflineForge 端到端演示入口.

运行完整确定性基准 (6 难度场景 × 3 种子 × 5 算法 = 90 评测),
将结果落盘到仓库根目录 benchmark.json, 并打印排名.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offlineforge.cli import main

if __name__ == "__main__":
    sys.exit(main())
