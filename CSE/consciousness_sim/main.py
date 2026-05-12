"""
意识工程化构造实验 v5.0 - CLI主界面
Consciousness Engineering Construction Experiment v5.0 - CLI

v5.0 认知张力驱动版：用户输入作为种子注入，系统自主思考。
"""

import sys
import os
from .engine import ConsciousnessEngine


def print_banner():
    banner = r"""
╔══════════════════════════════════════════════════════════════╗
║      意识工程化构造实验 v5.0                                 ║
║  Consciousness Engineering Construction Experiment          ║
║                                                             ║
║  发散碎片 → 选择性合成 → 意识流写入 → 语言表达              ║
║  认知张力驱动 · 多Agent思维系统模拟器                        ║
╚══════════════════════════════════════════════════════════════╝

系统已启动。请输入您的初始种子以启动思考循环。

可用指令：
  · 输入任何问题 → 作为种子注入意识流
  · 「继续」→ 推进下一轮思考
  · 「debug」→ 显示系统状态信息
  · 「exit」→ 退出系统

"""
    print(banner)


def main():
    print_banner()
    engine = ConsciousnessEngine()

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n系统退出。")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\n系统退出。")
            break

        if user_input.lower() == "debug":
            status = engine.get_status()
            print("\n--- 系统状态 ---")
            for k, v in status.items():
                print(f"  {k}: {v}")
            print("-----------------")
            continue

        if user_input.lower() == "继续":
            if not engine.stream.goal and not engine._running:
                print("\n请先输入种子。")
                continue
            if engine.stream.waiting_for_user or engine.stream.paused or engine.stream.idle_mode:
                output = engine.continue_round()
                print("\n" + output)
            else:
                print("\n系统未暂停，无需继续。输入新种子以启动思考。")
            continue

        # v5.0: 用户输入作为种子注入意识流
        result = engine.handle_new_goal(user_input)
        print(f"\n{result}")


if __name__ == "__main__":
    main()
