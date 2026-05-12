"""API 连通性检测"""
import sys
from consciousness_sim.config import load_config


def check():
    cfg = load_config()
    api = cfg.get("api", {})
    api_key = api.get("api_key", "")
    base_url = api.get("base_url", "https://api.deepseek.com")
    model = api.get("default_model", "deepseek-v4-flash")

    if not api_key:
        print("[FAIL] api_key 未配置，请编辑 consciousness_sim\\config.json")
        return 1

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        # 仅列出模型，验证 key 和地址可达
        client.models.list()
        print(f"[OK] API 连接成功 (base_url={base_url}, model={model})")
        return 0
    except Exception as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            print("[FAIL] API Key 无效，请检查 config.json 中的 api_key")
        elif "Connection" in msg or "refused" in msg or "resolve" in msg.lower():
            print(f"[FAIL] 无法连接到 {base_url}，请检查网络和 base_url")
        else:
            print(f"[FAIL] API 检测失败: {e}")
            print("请检查 consciousness_sim\\config.json 中的 api_key 和 base_url")
        return 1


if __name__ == "__main__":
    sys.exit(check())
