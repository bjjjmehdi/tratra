"""
utils/config.py
- 读取 config.yaml
- 自动加载同目录的 credentials.env
- 递归替换 ${ENV_VAR} 占位符
"""
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

def load_config() -> Dict[str, Any]:
    # 1) YAML 路径
    yaml_path = Path(__file__).parents[2] / "config" / "config.yaml"

    # 2) 加载 YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # 3) 加载 credentials.env（同目录）
    env_path = yaml_path.parent / "credentials.env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    # 4) 递归替换 ${VAR} 占位符
    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(item) for item in obj]
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            var = obj[2:-1]
            return os.getenv(var, obj)  # 找不到就保持原样
        return obj

    cfg = _resolve(cfg)
    return cfg