"""Day 7 B3：发布前仓库自检（脚本级检查，输出 PASS/FAIL 清单）。

检查项：
1. git 跟踪文件中是否含疑似 API key（正则 sk-[A-Za-z0-9]{20,} 等）
2. .gitignore 是否覆盖 .env / .venv/ / data/chroma/ / eval/.cache/ / __pycache__ / 模型缓存
3. requirements.txt 是否全部 == 精确 pin
4. docs/pitfalls.md 条目数 ≥12
5. Quickstart 可复现性：MOCK_LLM=1 下跑 graph --selftest 是否不报错
6. 无超大文件（>50MB）被 git 跟踪

运行：python scripts/repo_check.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEY_PATTERNS = [
    (r"sk-[A-Za-z0-9]{20,}", "疑似 API key（sk- 开头）"),
    (r"lsv2_[A-Za-z0-9]{20,}", "疑似 LangSmith key"),
    (r"AIza[0-9A-Za-z\-_]{30,}", "疑似 Google API key"),
]

GITIGNORE_REQUIRED = [".env", ".venv/", "data/chroma/", "eval/.cache/", "__pycache__/", "models/"]


def check(cond: bool) -> str:
    return "PASS" if cond else "FAIL"


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [f for f in out.stdout.splitlines() if f]


def check_secrets(files: list[str]) -> tuple[str, list[str]]:
    hits: list[str] = []
    for rel in files:
        path = PROJECT_ROOT / rel
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern, desc in KEY_PATTERNS:
            for i, line in enumerate(text.splitlines(), start=1):
                if re.search(pattern, line):
                    hits.append(f"{rel}:{i} [{desc}]")
    return check(not hits), hits


def check_gitignore() -> tuple[str, list[str]]:
    gi = PROJECT_ROOT / ".gitignore"
    content = gi.read_text(encoding="utf-8") if gi.exists() else ""
    missing = [item for item in GITIGNORE_REQUIRED if item not in content]
    return check(not missing), missing


def check_requirements_pin() -> tuple[str, list[str]]:
    req = PROJECT_ROOT / "requirements.txt"
    if not req.exists():
        return "FAIL", ["requirements.txt 不存在"]
    bad = []
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            bad.append(line)
    return check(not bad), bad


def check_pitfalls(min_items: int = 12) -> tuple[str, str]:
    path = PROJECT_ROOT / "docs" / "pitfalls.md"
    if not path.exists():
        return "FAIL", "docs/pitfalls.md 不存在"
    count = len(re.findall(r"^### 坑 \d+", path.read_text(encoding="utf-8"), re.M))
    return check(count >= min_items), f"条目数 {count}/{min_items}"


def check_quickstart_mock() -> tuple[str, str]:
    """MOCK_LLM=1 下跑 selftest，验证不依赖真实 API 也能跑通链路。"""
    env = {"MOCK_LLM": "1"}
    import os
    env_full = dict(os.environ, **env)
    proc = subprocess.run([sys.executable, "-m", "src.agent.graph", "--selftest"],
                          cwd=PROJECT_ROOT, capture_output=True, text=True,
                          env=env_full, timeout=600)
    ok = proc.returncode == 0
    return check(ok), ("" if ok else (proc.stderr or proc.stdout)[-500:])


def check_large_files(files: list[str]) -> tuple[str, list[str]]:
    big = []
    for rel in files:
        p = PROJECT_ROOT / rel
        if p.is_file() and p.stat().st_size > 50 * 1024 * 1024:
            big.append(f"{rel}（{p.stat().st_size / 1024 / 1024:.1f} MB）")
    return check(not big), big


def main() -> None:
    files = tracked_files()
    print("=== 仓库发布前自检 ===\n")
    results: list[tuple[str, str, object]] = []

    if not files:
        print("[提示] 当前不是 git 仓库或没有跟踪文件，跳过 git 相关检查")
    else:
        status, hits = check_secrets(files)
        results.append(("1. 无 API key 泄漏", status, hits))
        status, big = check_large_files(files)
        results.append(("6. 无超大文件(>50MB)入库", status, big))

    status, missing = check_gitignore()
    results.append(("2. .gitignore 覆盖必要项", status, missing))
    status, bad = check_requirements_pin()
    results.append(("3. requirements 全部精确 pin", status, bad))
    status, info = check_pitfalls()
    results.append(("4. 踩坑日志 ≥12 条", status, info))
    status, info = check_quickstart_mock()
    results.append(("5. MOCK_LLM 下 quickstart 可跑通", status, info))

    all_pass = True
    for name, status, detail in results:
        print(f"[{status}] {name}")
        if status == "FAIL" and detail:
            print(f"       → {detail}")
        all_pass = all_pass and status == "PASS"

    print(f"\n总体: {'全部 PASS ✔' if all_pass else '存在 FAIL，请修复后再发布 ✘'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
