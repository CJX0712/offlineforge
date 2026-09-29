#!/usr/bin/env python3
"""OfflineForge 一键交付：建仓库 + Git Data API 推送（绕过 git 智能协议的代理 502）。

幂等可重跑，覆盖全部已知情形：
- 仓库不存在 -> POST /user/repos 创建（REST，绕开 GraphQL 502）
- 空仓库 blob 409 -> 先 Contents API PUT README 建 bootstrap commit
- main 已存在 -> PATCH refs/heads/main (force) 追加提交；不存在 -> POST refs
- 偶发 502/Bad Gateway -> 按关键词退避重试
作者署名：晨星 <CJX0712@users.noreply.github.com>
"""

import base64
import json
import os
import subprocess
import sys
import time

REPO = "CJX0712/offlineforge"
ROOT = os.path.dirname(os.path.abspath(__file__))
AUTHOR = {"name": "晨星", "email": "CJX0712@users.noreply.github.com"}
MSG = (
    "OfflineForge v0.1.0 — 世界顶级离线强化学习（Offline RL + OPE）系统\n\n"
    "- 零依赖内核: 纯 numpy GridWorld MDP + BC/CQL-lite/FQE 离线学习器\n"
    "- 六种 OPE 估计器: DM/SIS/WIS/TIS/DR/FQE, 对照 DP 真值给误差排序\n"
    "- 6 档难度梯度场景 × 3 种子 × 5 算法 = 90 评测, benchmark.json 落盘\n"
    "- 23 项数值不变量单测全绿, ruff 全过; DR/FQE 近精确, DM 有偏, IS 呈方差代价\n"
    "- 可选复用 d3rlpy(CQL/BCQ) 作为顶级开源背书 (无 torch 时自动降级 numpy)\n"
    "- 作者: 晨星"
)
RETRY_KEYS = ("Bad Gateway", "502", "503", "Connection reset", "unexpected end")


def gh_api(method, endpoint, body=None, retries=8):
    cmd = ["gh", "api", endpoint, "-X", method]
    tmp = None
    if body is not None:
        tmp = os.path.join(os.environ.get("TEMP", "/tmp"), "_gh_body.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False)
        cmd += ["--input", tmp]
    last = ""
    for attempt in range(1, retries + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if r.returncode == 0:
                return json.loads(r.stdout) if r.stdout.strip() else {}
            lines = (r.stderr or "").strip().splitlines()
            last = lines[-1] if lines else "empty"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        if any(k in last for k in RETRY_KEYS) and attempt < retries:
            time.sleep(2.5)
            continue
        raise RuntimeError(f"{method} {endpoint}: {last}")
    raise RuntimeError(f"{method} {endpoint}: {last}")


def tracked_files():
    out = subprocess.check_output(["git", "-C", ROOT, "ls-files"], text=True)
    return [f for f in out.splitlines() if f.strip()]


def repo_exists():
    try:
        gh_api("GET", f"repos/{REPO}")
        return True
    except RuntimeError as e:
        if '"status":"404"' in str(e) or "Not Found" in str(e):
            return False
        raise


def ensure_repo():
    if repo_exists():
        print("[info] repo exists")
        return
    gh_api(
        "POST",
        "user/repos",
        {
            "name": REPO.split("/")[1],
            "description": (
                "OfflineForge — 世界顶级离线强化学习（Offline RL + OPE）系统（作者: 晨星）。"
                "零依赖 numpy 内核, BC/CQL-lite/FQE 学习器, 六种 OPE 估计器对照 DP 真值。"
            ),
            "private": False,
            "auto_init": False,
            "has_issues": True,
            "has_wiki": False,
        },
    )
    print("[info] repo created")
    time.sleep(2)


def main_head():
    try:
        r = gh_api("GET", f"repos/{REPO}/git/refs/heads/main")
        return r["object"]["sha"]
    except RuntimeError as e:
        if "404" in str(e) or "409" in str(e) or "Not Found" in str(e):
            return None
        raise


def bootstrap_if_empty():
    try:
        gh_api("POST", f"repos/{REPO}/git/blobs", {"content": "cHJvYmU=", "encoding": "base64"})
        return
    except RuntimeError as e:
        if "409" not in str(e) and "empty" not in str(e).lower():
            raise
    print("[info] empty repo -> bootstrap README via Contents API")
    with open(os.path.join(ROOT, "README.md"), "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    gh_api(
        "PUT",
        f"repos/{REPO}/contents/README.md",
        {
            "message": "OfflineForge v0.1.0 初始化（README）",
            "content": b64,
            "branch": "main",
            "author": {**AUTHOR, "date": now},
            "committer": {**AUTHOR, "date": now},
        },
    )
    time.sleep(1)


def main():
    ensure_repo()
    bootstrap_if_empty()

    files = tracked_files()
    print(f"[info] tracked files: {len(files)}")

    entries = []
    for i, rel in enumerate(files, 1):
        with open(os.path.join(ROOT, rel), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        resp = gh_api("POST", f"repos/{REPO}/git/blobs", {"content": b64, "encoding": "base64"})
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": resp["sha"]})
        print(f"  blob {i:>2}/{len(files)} {rel} -> {resp['sha'][:10]}")

    tree = gh_api("POST", f"repos/{REPO}/git/trees", {"tree": entries})
    print(f"[info] tree -> {tree['sha'][:10]} ({len(entries)} entries)")

    head = main_head()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commit = gh_api(
        "POST",
        f"repos/{REPO}/git/commits",
        {
            "message": MSG,
            "tree": tree["sha"],
            "parents": [head] if head else [],
            "author": {**AUTHOR, "date": now},
            "committer": {**AUTHOR, "date": now},
        },
    )
    print(f"[info] commit -> {commit['sha'][:10]} (parent={head[:10] if head else 'none'})")

    if head:
        gh_api(
            "PATCH",
            f"repos/{REPO}/git/refs/heads/main",
            {"sha": commit["sha"], "force": True},
        )
        print("[info] main ref updated (PATCH force)")
    else:
        gh_api("POST", f"repos/{REPO}/git/refs", {"ref": "refs/heads/main", "sha": commit["sha"]})
        print("[info] main ref created (POST)")
    gh_api("PATCH", f"repos/{REPO}", {"default_branch": "main"})

    try:
        tag = gh_api(
            "POST",
            f"repos/{REPO}/git/tags",
            {
                "tag": "v0.1.0",
                "message": "OfflineForge v0.1.0",
                "object": commit["sha"],
                "type": "commit",
                "tagger": {**AUTHOR, "date": now},
            },
        )
        gh_api("POST", f"repos/{REPO}/git/refs", {"ref": "refs/tags/v0.1.0", "sha": tag["sha"]})
        print("[info] tag v0.1.0 created")
    except RuntimeError as e:
        print(f"[warn] tag 跳过: {str(e)[:80]}")

    print(f"\nDONE: https://github.com/{REPO}/commit/{commit['sha']}")


if __name__ == "__main__":
    sys.exit(main())
