"""
配置管理 - 从config.json读取设置
支持：API配置、引擎参数、显示设置
"""

import os
import json

_CONFIG = None


def _get_config_path():
    """获取配置文件路径"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config():
    """加载配置文件"""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    config_path = _get_config_path()
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            _CONFIG = json.load(f)
    except FileNotFoundError:
        # 配置文件不存在时使用默认配置
        _CONFIG = {
            "api": {
                "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
                "base_url": "https://api.deepseek.com",
                "default_model": "deepseek-v4-flash",
                "default_max_tokens": 4096,
                "reasoning_effort": "medium",
                "thinking_mode": False,
                "agent_models": {
                    "synthesizer": {"model": "deepseek-v4-pro", "max_tokens": 8192}
                }
            },
            "engine": {
                "convergence_rounds": 2,
                "max_task_rounds": 5,
                "parallel_modules": True,
                "use_llm_expressor": True
            },
            "display": {
                "fragment_max_length": 600,
                "show_api_stats": True,
                "language": "zh"
            }
        }
        save_config(_CONFIG)
    return _CONFIG


def save_config(config=None):
    """保存配置文件"""
    if config is None:
        config = _CONFIG
    if config is None:
        return
    config_path = _get_config_path()
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[配置] 保存失败: {e}")


def get_api_config():
    """获取API配置"""
    cfg = load_config()
    return cfg.get("api", {})


def get_engine_config():
    """获取引擎配置"""
    cfg = load_config()
    return cfg.get("engine", {})


def get_agent_config(agent_name: str) -> dict:
    """获取指定Agent的模型配置。先查agent_models覆盖，未找到则返回default值。"""
    api_cfg = get_api_config()
    defaults = {
        "model": api_cfg.get("default_model", "deepseek-v4-flash"),
        "max_tokens": api_cfg.get("default_max_tokens", 4096),
        "reasoning_effort": api_cfg.get("reasoning_effort", "medium"),
        "thinking_mode": api_cfg.get("thinking_mode", False),
        "base_url": api_cfg.get("base_url", "https://api.deepseek.com"),
        "api_key": api_cfg.get("api_key", ""),
    }
    agent_models = api_cfg.get("agent_models", {})
    override = agent_models.get(agent_name, {})
    defaults.update(override)
    return defaults


def get_display_config():
    """获取显示配置"""
    cfg = load_config()
    return cfg.get("display", {})


def reload_config():
    """重新加载配置"""
    global _CONFIG
    _CONFIG = None
    return load_config()
