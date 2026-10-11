"""配置加载：统一从 configs/*.yaml 读取参数，禁止在业务代码里硬编码。

用法：
    from src.config import load_config
    cfg = load_config()                      # 读 configs/default.yaml
    cfg = load_config("configs/exp_e1.yaml") # 读指定配置
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# 项目根目录（src/ 的上一级）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载 yaml 配置并返回 dict。文件不存在或格式错误时给出带路径的报错。"""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}（请检查路径或先创建该文件）")
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"配置文件解析失败: {config_path}: {exc}") from exc
    return cfg


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置：override 覆盖 base，用于对比实验的 config_override。"""
    merged: dict[str, Any] = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def dump_config(cfg: dict[str, Any]) -> str:
    """把配置转回 yaml 文本，用于报告里打印「本次运行的配置快照」。"""
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
