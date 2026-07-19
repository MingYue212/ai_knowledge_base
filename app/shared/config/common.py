"""
环境变量读取工具：提供类型安全的 env_str / env_bool / env_float 函数。
模块加载时自动执行 load_dotenv()，后续 import 本模块的文件可直接使用 os.getenv。
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value is not None else default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default

    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False

    return default


def env_float(name: str, default: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default

    try:
        return float(value)
    except ValueError:
        return default
