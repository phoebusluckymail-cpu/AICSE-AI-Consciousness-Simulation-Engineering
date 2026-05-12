"""
LLM驱动的Agent模块 - 使用DeepSeek API实现v4.6规范中的各Agent角色
替代modules.py中的规则式实现，保留数据模型和任务启动器。
"""

import re
import json
import concurrent.futures
from typing import Optional, Callable
from .models import (
    Fragment, ConsciousnessStream,
    Severity, SourceType, ConvergenceStatus, SwitchSuggestion, PauseDecision,
)
from .llm_client import (
    LLMClient,
    LOGICAL_REASONER_PROMPT, CRITIC_PROMPT, ASSOCIATION_PROMPT,
    EMOTIONAL_PROMPT, MOTIVATION_PROMPT, METACOGNITIVE_PROMPT,
    SYNTHESIZER_PROMPT, STREAM_WRITER_PROMPT,
    build_context, build_synthesizer_context
)
from .config import get_engine_config
from .prompt_loader import load_prompt


# ============================================================
# v5.0: 任务管理器已移除，用户问题作为种子注入意识流


# 中文模块名 → config.json 中 agent_models 的 key
_AGENT_CONFIG_MAP = {
    "逻辑推理者": "logical_reasoner",
    "批判者": "critic",
    "联想与创意联结者": "association",
    "情感评估者": "emotional",
    "动机检测者": "motivation",
    "元认知监控者": "metacognitive",
}


class LLMModule:
    """LLM驱动的模块基类"""

    def __init__(self, name: str, system_prompt: str, client: LLMClient):
        self.name = name
        self.system_prompt = system_prompt
        self.client = client
        self.config_key = _AGENT_CONFIG_MAP.get(name, "")

    def generate_fragment(self, stream: ConsciousnessStream) -> Fragment:
        """生成LLM驱动的碎片（v5.0：独立思考，不读邻接模块）"""
        context = build_context(stream, self.name)
        content = self.client.call_module(self.system_prompt, context,
                                          agent_name=self.config_key)
        tags = self._extract_tags(content)
        extra = self._extract_extra(content)
        return Fragment(source=self.name, content=content, tags=tags, extra=extra)

    def _extract_tags(self, content: str) -> list[str]:
        """从LLM输出中提取标记"""
        tags = []
        tag_patterns = [
            r'\[致命\]', r'\[重要\]', r'\[次要\]',
            r'\[确定\]', r'\[推测\]', r'\[模糊直觉\]',
            r'\[逻辑瓶颈\]',
            r'\[强关联\]', r'\[弱启发\]',
            r'\[边界[^\]]*\]',
            r'\[悬置突破\]',
            r'\[紧急偏向警告\]',
            r'\[回避偏向\]', r'\[冒进偏向\]',
            r'\[情感偏差：已自审\]',
            r'\[动机偏差：已自审\]',
            r'\[元盲区：已自审\]',
            r'暂停建议[：:]',
            r'二阶批判',
            r'模糊批判[：:]',
            r'记忆修正申请',
            r'\[建设\]',
        ]
        for pattern in tag_patterns:
            if re.search(pattern, content):
                # 提取标签名（去掉方括号）
                match = re.search(pattern, content)
                tag = match.group(0).strip('[]').split('：')[0]
                if tag not in tags:
                    tags.append(tag)
        return tags

    def _extract_extra(self, content: str) -> dict:
        """子类覆写以提取额外元数据"""
        return {}


class LogicalReasoner(LLMModule):
    def __init__(self, client: LLMClient):
        super().__init__("逻辑推理者", LOGICAL_REASONER_PROMPT, client)


class Critic(LLMModule):
    def __init__(self, client: LLMClient):
        super().__init__("批判者", CRITIC_PROMPT, client)


class AssociationCreator(LLMModule):
    def __init__(self, client: LLMClient):
        super().__init__("联想与创意联结者", ASSOCIATION_PROMPT, client)


class EmotionalEvaluator(LLMModule):
    def __init__(self, client: LLMClient):
        super().__init__("情感评估者", EMOTIONAL_PROMPT, client)

    def _extract_extra(self, content: str) -> dict:
        """提取情感底色词"""
        extra = {}
        match = re.search(r'情感底色[：:]\s*(.+?)(?:\n|$)', content)
        if match:
            extra["emotional_color"] = match.group(1).strip()
        return extra


class MotivationDetector(LLMModule):
    def __init__(self, client: LLMClient):
        super().__init__("动机检测者", MOTIVATION_PROMPT, client)


class MetaCognitiveMonitor(LLMModule):
    def __init__(self, client: LLMClient):
        super().__init__("元认知监控者", METACOGNITIVE_PROMPT, client)

    def _extract_extra(self, content: str) -> dict:
        """提取必填字段: 收敛、切换、暂停"""
        extra = {}

        # 收敛判定
        conv_match = re.search(r'\[当前认知收敛[：:]\s*(是|否)\]', content)
        if conv_match:
            extra["converged"] = ConvergenceStatus.CONVERGED if conv_match.group(1) == "是" else ConvergenceStatus.NOT_CONVERGED

        # 优先级切换
        switch_match = re.search(r'\[涌现优先级切换建议[：:]\s*(是|否)\]', content)
        if switch_match:
            extra["switch"] = SwitchSuggestion.YES if switch_match.group(1) == "是" else SwitchSuggestion.NO
            if switch_match.group(1) == "是":
                dir_match = re.search(r'建议探索的涌现方向[：:]\s*([^\n]+)', content)
                if dir_match:
                    extra["switch_direction"] = dir_match.group(1).strip()

        # 暂停决策
        pause_match = re.search(r'\[暂停[：:]\s*(是|否)\]', content)
        if pause_match:
            extra["pause"] = PauseDecision.YES if pause_match.group(1) == "是" else PauseDecision.NO
            if pause_match.group(1) == "是":
                reason_match = re.search(r'暂停原因[：:]\s*([^\n]+)', content)
                if reason_match:
                    extra["pause_reason"] = reason_match.group(1).strip()

        return extra


class SelectiveSynthesizer:
    """LLM驱动的选择性合成器"""

    def __init__(self, client: LLMClient):
        self.client = client

    def synthesize(self, stream: ConsciousnessStream, all_fragments: dict) -> Fragment:
        """调用LLM合成碎片"""
        context = build_synthesizer_context(stream, all_fragments)
        content = self.client.call_synthesizer(SYNTHESIZER_PROMPT, context)

        tags = []
        if "[悬置" in content:
            tags.append("悬置")

        return Fragment(source="选择性合成器", content=content, tags=tags)


class StreamWriter:
    """意识流写入器 - LLM驱动的智能提取 + 规则式状态管理

    LLM负责：分级保真度提炼、来源标记、情感底色判断、跨字段关联检测、关键事件标注
    代码负责：冲突管理、探索分支、记忆衰减、记录写入、认知线程更新
    """

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client

    def write(self, stream: ConsciousnessStream, synthesizer_output: Fragment,
              all_fragments: dict) -> str:
        """调用LLM处理合成器输出，并维护系统状态"""


        # Step 1: 调用LLM进行智能提取
        llm_result = self._call_llm(stream, synthesizer_output)

        # Step 2: 从LLM输出中提取状态更新信号
        # === 检查暂停信号（机械逻辑，来自元认知监控者） ===
        meta = all_fragments.get("元认知监控者")
        meta_extra = meta.extra if meta else {}
        if meta_extra.get("pause") == PauseDecision.YES:
            stream.paused = True
            stream.pause_reason = meta_extra.get("pause_reason", "")
        else:
            stream.paused = False
            stream.pause_reason = ""

        # === 处理悬置冲突（来自合成器输出中的标记） ===
        if "[悬置：" in synthesizer_output.content:
            for line in synthesizer_output.content.split("\n"):
                if "[悬置：" in line:
                    desc = line.strip()[:100]
                    stream.add_conflict(desc, Severity.FATAL)

        # === 处理延后关注（来自合成器输出） ===
        in_deferred = False
        for line in synthesizer_output.content.split("\n"):
            if "[延后关注]" in line:
                in_deferred = True
                continue
            if in_deferred and line.strip().startswith("·"):
                item = line.strip().strip("· ")
                if item:
                    stream.add_deferred(item)
            elif in_deferred and not line.strip():
                continue
            elif in_deferred and not line.strip().startswith("·"):
                in_deferred = False



        # === 动态记忆衰减 + 冷冻冲突扫描 ===
        stream.decay_all()
        stream.thaw_frozen_conflicts()

        # === 处理记忆修正申请 ===
        if "记忆修正申请" in synthesizer_output.content:
            for line in synthesizer_output.content.split("\n"):
                if "记忆修正申请" in line:
                    correction_desc = line.strip()
                    stream.add_round_record(
                        f"[记忆修正] {correction_desc}（基于第{stream.round}轮合成器审查）")



        # === 记录到意识流历史 ===
        source_type = SourceType.SYNTHESIS
        if "悬置" in synthesizer_output.tags or "[悬置" in llm_result:
            source_type = SourceType.SUSPENDED
        stream.last_synthesizer_summary = synthesizer_output.content[:200]
        stream.add_round_record(synthesizer_output.content)

        # === 更新认知线程（v5.0：认知张力驱动） ===
        stream.cognitive_thread = stream.goal or "空转"

        return llm_result

    def _call_llm(self, stream: ConsciousnessStream, synth_output: Fragment) -> str:
        """调用LLM写入器"""
        if self.client is None:
            return self._fallback_output(stream, synth_output)

        try:
            context = self._build_context(stream, synth_output)
            result = self.client.call_module(STREAM_WRITER_PROMPT, context, temperature=0.3,
                                              agent_name="stream_writer")
            if result.startswith("[API错误"):
                return self._fallback_output(stream, synth_output)
            return result
        except Exception:
            return self._fallback_output(stream, synth_output)

    def _build_context(self, stream: ConsciousnessStream, synth_output: Fragment) -> str:
        """构建写入器上下文"""
        parts = []
        parts.append(f"=== 第{stream.round}轮 意识流 ===")
        parts.append(f"当前认知线程：{stream.cognitive_thread}")
        if stream.goal:
            parts.append(f"种子：{stream.goal}")
        parts.append("")

        active = [c for c in stream.unresolved_conflicts
                  if c.status == "active" and not c.resolved]
        if active:
            parts.append("未解决冲突：")
            for c in active:
                parts.append(f"- [{c.severity.value}] {c.description}（悬置于第{c.round_created}轮）")
            parts.append("")

        deferred = [d for d in stream.deferred_attention if not d.obsolete]
        if deferred:
            parts.append("延后关注：")
            for d in deferred:
                parts.append(f"- {d.description}")
            parts.append("")
        parts.append("=== 本轮合成器输出 ===")
        parts.append(synth_output.content)

        return "\n".join(parts)

    def _fallback_output(self, stream: ConsciousnessStream, synth: Fragment) -> str:
        """无LLM时的规则式回退输出"""
        lines = [f"--- 第{stream.round}轮写入记录 ---",
                 f"[合成] {synth.content[:500]}"]
        return "\n".join(lines)


class LanguageExpressor:
    """语言表达器 - 将意识流转化为用户语言（使用LLM获得高质量表达）"""

    EXPRESSOR_PROMPT = load_prompt("expressor", """你是语言表达器。你不参与思维过程。你的唯一任务，是将当前完整的意识流，转化为语法完整、流畅自然的语言表达。

要求：
1. 理解意识流中已经确定的核心结论和论证结构。
2. 将其转化为一个完整的、逻辑清晰的、面向用户的语言表达。
3. 不添加意识流中没有的新观点。
4. 认知张力保留：当意识流中存在悬置冲突或不确定性标记时，表达应保留这种不确定性。
5. 情感底色感知：可读取意识流头部的"当前情感底色"字段，在表达语气上做轻微匹配。
6. 关键事件凸出：当意识流中包含[关键事件]标注时，将其作为表达的重点。
7. 场景适配：
   - 首轮：用"关于你提出的问题……"自然引入
   - 后续轮次：参考"此前轮次的语言表达"字段，用"继续之前的思考……"自然承接
   - 暂停时：明确说明暂停原因
   - 最后一轮且任务完成：明确告知并呈现最终结论
   - 空转静默：空转轮次不输出
8. 不解释转化过程。""")

    def express(self, stream: ConsciousnessStream, writer_output: str,
                is_first_round: bool = False, is_paused: bool = False,
                is_idle: bool = False, client: Optional[LLMClient] = None) -> str:
        """将意识流转化为语言表达（v5.0：直接读取完整意识流，不需要合成器原文）"""
        if is_idle:
            return ""

        engine_cfg = get_engine_config()
        use_llm = engine_cfg.get("use_llm_expressor", True)

        if use_llm and client:
            try:
                context = self._build_express_context(stream, writer_output,
                                                       is_first_round, is_paused)
                result = client.call_module(self.EXPRESSOR_PROMPT, context, temperature=0.5,
                                              agent_name="expressor")
                if not result.startswith("[API错误"):
                    return result
            except Exception:
                pass

        return self._fallback_express(stream, is_first_round, is_paused)

    def _build_express_context(self, stream, writer_output,
                                is_first_round, is_paused) -> str:
        """构建表达器上下文（v5.0：本轮意识流 + 对话记录）"""
        parts = []
        parts.append(f"种子: {stream.goal} · 第{stream.round}轮 · 情感: {stream.emotional_color}")

        if is_paused:
            parts.append(f"暂停原因: {stream.pause_reason}")

        # 本轮新产生的意识流
        parts.append("\n本轮意识流：")
        parts.append(writer_output)

        # 与用户的对话记录
        if stream.chat_history:
            parts.append("\n=== 对话记录 ===")
            for msg in stream.chat_history[-10:]:
                role = "用户" if msg["role"] == "user" else "系统"
                parts.append(f"[{role}] {msg['content']}")

        return "\n".join(parts)

    def _fallback_express(self, stream, is_first_round, is_paused) -> str:
        """规则式回退表达（v5.0：基于意识流全貌，不依赖合成器原文）"""
        parts = []
        if is_first_round:
            parts.append(f"关于你提出的问题「{stream.goal}」……")
        else:
            parts.append("继续推进思考。")

        active = [c for c in stream.unresolved_conflicts if c.status == "active" and not c.resolved]
        if active:
            parts.append("\n[注意：当前存在未解决的悬置冲突。]")

        if is_paused:
            parts.append(f"\n思考暂停。原因：{stream.pause_reason}")

        return "\n".join(parts)


# ============================================================
# 并行执行器
# ============================================================

class ParallelExecutor:
    """并行执行发散模块"""

    def __init__(self, client: LLMClient):
        self.modules = {
            "逻辑推理者": LogicalReasoner(client),
            "批判者": Critic(client),
            "联想与创意联结者": AssociationCreator(client),
            "情感评估者": EmotionalEvaluator(client),
            "动机检测者": MotivationDetector(client),
            "元认知监控者": MetaCognitiveMonitor(client),
        }
        self.synthesizer = SelectiveSynthesizer(client)
        self.writer = StreamWriter(client)
        self.expressor = LanguageExpressor()
        self.client = client
        # 存储每轮各Agent看到的记忆上下文
        self.last_contexts: dict[str, str] = {}

    def execute_all(self, stream: ConsciousnessStream) -> tuple[dict, Fragment, str, str]:
        """
        并行执行所有发散模块，然后合成、写入、表达
        返回: (all_fragments, synth_output, writer_output, expression)
        """
        # 并行执行6个发散模块
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_map = {}
            for name, module in self.modules.items():
                future = executor.submit(module.generate_fragment, stream)
                future_map[future] = name

            all_fragments = {}
            for future in concurrent.futures.as_completed(future_map):
                name = future_map[future]
                try:
                    all_fragments[name] = future.result()
                except Exception as e:
                    all_fragments[name] = Fragment(
                        source=name,
                        content=f"[执行错误] {str(e)}",
                        tags=[]
                    )

        # 按标准顺序排列（便于展示）
        ordered = {}
        for name in ["逻辑推理者", "批判者", "联想与创意联结者",
                       "情感评估者", "动机检测者", "元认知监控者"]:
            if name in all_fragments:
                ordered[name] = all_fragments[name]
        all_fragments = ordered

        # 情感底色由情感评估者决定
        emotional = all_fragments.get("情感评估者")
        if emotional and "emotional_color" in emotional.extra:
            stream.emotional_color = emotional.extra["emotional_color"]

        # 合成
        synth_output = self.synthesizer.synthesize(stream, all_fragments)

        # 写入
        writer_output = self.writer.write(stream, synth_output, all_fragments)

        # 表达
        meta = all_fragments.get("元认知监控者")
        meta_extra = meta.extra if meta else {}
        is_paused = meta_extra.get("pause") == PauseDecision.YES
        is_first_round = (stream.round == 1)
        expression = self.expressor.express(
            stream, writer_output,
            is_first_round=is_first_round,
            is_paused=is_paused,
            client=self.client
        )

        # === 保存所有Agent的上下文（供UI展示） ===
        self.last_contexts = {}

        # 6个发散模块
        for name, mod in self.modules.items():
            ctx = build_context(stream, name)
            self.last_contexts[name] = ctx

        # 合成器
        synth_ctx = build_synthesizer_context(stream, all_fragments)
        self.last_contexts["选择性合成器"] = synth_ctx

        # 写入器
        writer_ctx = self.writer._build_context(stream, synth_output)
        self.last_contexts["意识流写入器"] = writer_ctx

        # 表达器
        express_ctx = self.expressor._build_express_context(
            stream, writer_output,
            is_first_round=(stream.round == 1), is_paused=is_paused
        )
        self.last_contexts["语言表达器"] = express_ctx

        return all_fragments, synth_output, writer_output, expression
