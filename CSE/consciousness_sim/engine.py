"""
意识工程化构造实验 v5.0 - 循环引擎
Consciousness Engineering Construction Experiment v5.0 - Loop Engine

v5.0 认知张力驱动版：移除任务管理器，用户问题作为种子注入，
系统由悬置冲突、好奇漫游、认知自然漂移三种内在张力驱动。
"""

import os
from typing import Optional
from .models import (
    ConsciousnessStream, Fragment, Severity,
    SourceType, ConvergenceStatus, PauseDecision
)
from .agent_modules import ParallelExecutor, LLMClient
from .config import get_api_config, get_display_config
from .session_manager import SessionManager


DEFAULT_SESSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "conversations")


class ConsciousnessEngine:
    """意识循环引擎 v5.0 — 认知张力驱动"""

    def __init__(self, api_key: Optional[str] = None,
                 session_manager: Optional[SessionManager] = None):
        if api_key is None:
            api_cfg = get_api_config()
            api_key = api_cfg.get("api_key", "")
        self.stream = ConsciousnessStream()
        self.client = LLMClient(api_key=api_key)
        self.executor = ParallelExecutor(self.client)
        self._first_round = True
        self._running = False
        self.session_manager = session_manager

    # ============================================================
    # 公共接口
    # ============================================================

    def start(self, goal: str) -> str:
        """v5.0: 用户问题作为种子注入意识流，直接启动思考循环"""
        self._running = True
        self.stream.goal = goal
        self.stream.cognitive_thread = f"种子：{goal}"
        self.stream.add_round_record(
            f"初始种子「{goal}」已注入意识流。系统开始自主思考。")
        return self._run_round(first_round=True)

    def continue_round(self) -> str:
        """继续下一轮思考"""
        if self.stream.waiting_for_user:
            return "系统正在等待您的输入。请输入新目标或指令。"
        return self._run_round()

    def handle_new_goal(self, goal: str) -> str:
        """v5.0: 新种子插队，旧种子入队待恢复"""
        if self.stream.goal:
            self.stream.seed_queue.append(self.stream.goal)
        self._running = True
        self.stream.goal = goal
        self.stream.cognitive_thread = f"种子：{goal}"
        self.stream.add_round_record(
            f"新种子「{goal}」注入。旧种子「{self.stream.seed_queue[-1] if self.stream.seed_queue else '无'}」暂存。")
        return self._run_round()

    def get_status(self) -> dict:
        """获取系统当前状态"""
        return {
            "round": self.stream.round,
            "goal": self.stream.goal,
            "paused": self.stream.paused,
            "idle": self.stream.idle_mode,
            "emotional_color": self.stream.emotional_color,
            "active_conflicts": len([c for c in self.stream.unresolved_conflicts
                                     if c.status == "active" and not c.resolved]),
            "api_calls": self.client.stats.get("calls", 0),
            "api_tokens": self.client.stats.get("total_tokens", 0),
        }

    def create_session(self, base_dir: str = DEFAULT_SESSION_DIR, goal: str = "") -> str:
        sm = SessionManager.create_session(base_dir, goal or self.stream.goal)
        self.session_manager = sm
        return sm.session_dir

    @staticmethod
    def load_from_session(session_path: str, api_key: Optional[str] = None) -> "ConsciousnessEngine":
        sm = SessionManager(session_path)
        stream = sm.load_stream_state()
        if stream is None:
            raise ValueError(f"会话无有效状态: {session_path}")
        if api_key is None:
            api_cfg = get_api_config()
            api_key = api_cfg.get("api_key", "")
        engine = ConsciousnessEngine(api_key=api_key, session_manager=sm)
        engine.stream = stream
        eng_state = sm.load_engine_state()
        engine._first_round = eng_state.get("first_round", False)
        engine._running = True
        if not engine.stream.expression_history:
            expr = sm.load_expression_history(stream)
            if expr:
                engine.stream.expression_history = expr
        return engine

    def get_session_info(self) -> dict:
        if not self.session_manager:
            return {"active": False}
        return {"active": True, "path": str(self.session_manager.session_dir)}

    # ============================================================
    # 内部方法
    # ============================================================

    def _enter_idle_mode(self):
        seed = self.stream.pick_idle_seed()
        if seed is None:
            return False
        self.stream.idle_mode = True
        self.stream.idle_round_count = 0
        self.stream.cognitive_thread = f"[空转反思] {seed.description}"
        self.stream.add_round_record(
            f"[关键事件：进入空转] 认知收敛后系统自主反思。种子：{seed.description}")
        return True

    def _finish_with_pause(self, reason: str):
        self.stream.paused = True
        self.stream.waiting_for_user = True
        self.stream.pause_reason = reason
        self.stream.idle_mode = False

    def _run_round(self, first_round: bool = False) -> str:
        """执行一轮完整循环"""
        self.stream.advance_round()
        current_round = self.stream.round

        # === 空转模式：选取种子 ===
        if self.stream.idle_mode:
            seed = self.stream.pick_idle_seed()
            if seed:
                self.stream.cognitive_thread = f"[空转反思] {seed.description}"
                self.stream.idle_round_count += 1
            else:
                self._finish_with_pause("空转素材已穷尽")
                return "空转素材已穷尽，系统暂停。输入新目标启动新的思考。"

        # === 并行执行发散模块 ===
        all_fragments, synth_output, writer_output, expression = \
            self.executor.execute_all(self.stream)

        if expression and not expression.startswith("[API错误"):
            self.stream.expression_history.append(f"[第{self.stream.round}轮表达]: {expression}")
            self.stream.add_chat_message("assistant", expression)

        # === 检查元认知信号 ===
        meta_frag = all_fragments.get("元认知监控者")
        meta_extra = meta_frag.extra if meta_frag else {}
        converged = meta_extra.get("converged", ConvergenceStatus.NOT_CONVERGED)
        pause_decision = meta_extra.get("pause", PauseDecision.NO)

        # === 构建展示 ===
        display = self._build_display(current_round, all_fragments, synth_output, writer_output, first_round)

        # === 暂停检查 ===
        is_paused = pause_decision == PauseDecision.YES
        if is_paused:
            self._finish_with_pause(meta_extra.get("pause_reason", ""))

        # === 收敛跟踪 ===
        if converged == ConvergenceStatus.CONVERGED:
            self.stream.convergence_rounds += 1
        else:
            self.stream.convergence_rounds = 0

        # === 收敛后恢复旧种子 ===
        if self.stream.convergence_rounds >= 2 and self.stream.seed_queue and not is_paused:
            old = self.stream.seed_queue.pop()
            self.stream.goal = old
            self.stream.cognitive_thread = f"种子：{old}"
            self.stream.convergence_rounds = 0
            self.stream.add_round_record(
                f"当前种子已收敛，恢复旧种子「{old}」。")

        # === 收敛超过3轮 → 尝试空转（需允许空转） ===
        if self.stream.convergence_rounds >= 3 and not self.stream.idle_mode and not is_paused and self.stream.allow_idle:
            self.stream.rebuild_seed_pool()
            if self.stream.idle_seed_count() > 0:
                entered = self._enter_idle_mode()
                if entered:
                    is_paused = False

        # === 空转上限 ===
        if self.stream.idle_mode and not is_paused:
            if self.stream.idle_round_count >= 3:
                self._finish_with_pause("空转已运行3轮，自动暂停")
            elif self.stream.idle_seed_count() == 0:
                self._finish_with_pause("空转素材已穷尽")

        # === 组合输出 ===
        output_parts = [
            display,
            "\n**语言表达：**",
            expression,
        ]
        api_stats = self.client.stats
        output_parts.append(f"\n\n*API调用统计：{api_stats['calls']}次调用，{api_stats['total_tokens']} tokens*")

        if is_paused:
            output_parts.append(f"\n\n*思考暂停。原因：{self.stream.pause_reason}*\n*输入「继续」推进，或发送新目标。*")
        elif self.stream.idle_mode:
            output_parts.append(
                f"\n\n*[空转模式] 第{self.stream.idle_round_count}轮自主反思。"
                f"输入「继续」推进，或发送新目标。*"
            )
        elif first_round:
            output_parts.append("\n\n*系统正在自动推进。输入「继续」手动推进下一轮。*")

        self._first_round = False

        # === 自动保存 ===
        if self.session_manager:
            try:
                self.session_manager.save_round(
                    stream=self.stream, fragments=all_fragments,
                    synth_output=synth_output, writer_output=writer_output,
                    narrative="", expression=expression,
                    agent_contexts=self.executor.last_contexts,
                    prev_fragments=None, first_round=self._first_round,
                )
            except Exception:
                pass

        return "\n".join(output_parts)

    def _build_display(self, round_num: int, fragments: dict,
                       synth: Fragment, writer_out: str, first_round: bool) -> str:
        """构建格式化展示"""
        display_cfg = get_display_config()
        max_len = display_cfg.get("fragment_max_length", 800)

        lines = []
        idle_tag = " [空转]" if self.stream.idle_mode else ""
        lines.append("=" * 60)
        lines.append(f"**第{round_num}轮{idle_tag}**")
        lines.append("=" * 60)

        lines.append("\n**当前意识流（本轮处理前）：**")
        lines.append(self.stream.get_context_display())

        lines.append("\n**发散碎片：**")
        for name in ["逻辑推理者", "批判者", "联想与创意联结者",
                       "情感评估者", "动机检测者", "元认知监控者"]:
            frag = fragments.get(name)
            if frag:
                lines.append(f"\n- **{name}**：")
                lines.append(frag.content[:max_len])

        lines.append("\n**合成器输出：**")
        lines.append(synth.content[:max_len])

        lines.append("\n**意识流写入器输出：**")
        lines.append(writer_out[:max_len])

        return "\n".join(lines)
