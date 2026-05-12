"""
提示词加载器 - 从 prompts/*.md 文件读取Agent提示词
用户可直接修改 markdown 文件自定义提示词，无需修改代码
"""

import os

_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(name: str, default: str) -> str:
    """
    加载指定名称的提示词
    优先从 prompts/{name}.md 读取，文件不存在时返回 default
    """
    global _PROMPT_CACHE

    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", f"{name}.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            _PROMPT_CACHE[name] = content
            return content
    except FileNotFoundError:
        _PROMPT_CACHE[name] = default
        return default


def reload_prompts():
    """清除缓存，重新加载所有提示词"""
    _PROMPT_CACHE.clear()
