"""
意识工程化构造实验 v5.0 - 数据模型
Consciousness Engineering Construction Experiment v5.0 - Data Models
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import copy
import math


# ============================================================
# 枚举定义
# ============================================================

class Severity(Enum):
    FATAL = "致命"
    IMPORTANT = "重要"
    MINOR = "次要"


class SourceType(Enum):
    SYNTHESIS = "合成"
    SUSPENDED = "悬置"
    IDLE = "空转"
    RESIDUAL = "残注"
    MEMORY_CORRECTION = "记忆修正"


class FragmentTag(Enum):
    # 逻辑推理者
    CERTAIN = "确定"
    SPECULATIVE = "推测"
    VAGUE_INTUITION = "模糊直觉"
    LOGIC_BOTTLENECK = "逻辑瓶颈"
    # 批判者
    CRITICAL_FATAL = "致命"
    CRITICAL_IMPORTANT = "重要"
    CRITICAL_MINOR = "次要"
    VAGUE_CRITIQUE = "模糊批判"
    SECOND_ORDER = "二阶批判"
    PAUSE_SUGGESTION = "暂停建议"
    CONSTRUCTIVE = "建设"
    # 联想
    STRONG_RELEVANCE = "强关联"
    WEAK_HEURISTIC = "弱启发"
    BOUNDARY = "边界"
    SUSPENSION_BREAK = "悬置突破"
    # 情感
    EMOTIONAL_FIRST = "情感先行"
    EMOTIONAL_BIAS = "情感偏差"
    # 动机
    URGENT_BIAS_WARNING = "紧急偏向警告"
    AVOIDANCE_BIAS = "回避偏向"
    AGGRESSIVE_BIAS = "冒进偏向"
    MOTIVATION_BIAS = "动机偏差"
    # 元认知
    STRATEGY_SUGGESTION = "策略建议"
    META_BLIND_SPOT = "元盲区"


class ConvergenceStatus(Enum):
    CONVERGED = "是"
    NOT_CONVERGED = "否"


class SwitchSuggestion(Enum):
    YES = "是"
    NO = "否"


class PauseDecision(Enum):
    YES = "是"
    NO = "否"


# ============================================================
# 基础数据类
# ============================================================

@dataclass
class Fragment:
    """思维碎片 - 任何模块的输出"""
    source: str
    content: str
    tags: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # 额外信息（如情感底色词）

    def has_tag(self, tag: str) -> bool:
        return any(tag in t for t in self.tags)


@dataclass
class Conflict:
    """未解决冲突"""
    description: str
    round_created: int
    severity: Severity
    active_value: float = 1.0
    status: str = "active"  # active, sleeping, frozen
    last_referenced_round: int = 0
    resolved: bool = False
    resolve_round: Optional[int] = None
    resolve_description: str = ""


@dataclass
class DeferredItem:
    """延后关注项"""
    description: str
    round_created: int
    active_value: float = 1.0
    scanned_count: int = 0
    obsolete: bool = False


@dataclass
class ExplorationBranch:
    """探索分支"""
    description: str
    start_round: int
    original_thread: str
    return_anchor: str
    completed: bool = False
    min_rounds: int = 2  # 最少运行2轮
    rounds_run: int = 0

    def can_check_convergence(self) -> bool:
        return self.rounds_run >= self.min_rounds


@dataclass
class IdleSeed:
    """空转种子 — 任务完成后系统自主反思的素材"""
    description: str
    source_type: str = "conflict"  # conflict, memory, deferred
    arousal: float = 1.0
    pick_count: int = 0
    round_created: int = 0


@dataclass
class Record:
    """历史记录"""
    content: str
    source_type: SourceType
    round: int
    source_module: str = ""


# ============================================================
# 意识流主数据结构
# ============================================================

class ConsciousnessStream:
    """意识流 - 系统核心状态容器"""

    def __init__(self):
        self.round: int = 0
        self.emotional_color: str = "无"
        self.unresolved_conflicts: list[Conflict] = []
        self.deferred_attention: list[DeferredItem] = []
        self.cognitive_thread: str = ""
        self.round_records: list[str] = []  # 每轮意识流输出
        self.frozen_conflicts: list[Conflict] = []
        self.allow_idle: bool = True  # 是否允许空转
        self.idle_mode: bool = False
        self.paused: bool = False
        self.waiting_for_user: bool = False
        self.pause_reason: str = ""
        self.goal: str = ""
        self.seed_queue: list[str] = []  # 旧种子队列，当前种子收敛后恢复
        self.no_progress_count: int = 0  # 连续无推进计数
        self.last_synthesizer_summary: str = ""
        self.expression_history: list[str] = []  # 每轮系统表达（子Agent用）
        self.chat_history: list[dict] = []  # 完整对话 [{role, content}]（表达器用）
        self.convergence_rounds: int = 0  # 连续收敛计数（用于空转恢复判定）
        self.idle_seeds: list[IdleSeed] = []  # 空转种子池
        self.idle_round_count: int = 0  # 空转已运行轮数

    def add_chat_message(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})

    def advance_round(self):
        """推进轮次计数"""
        self.round += 1

    def add_round_record(self, content: str):
        """添加本轮意识流记录"""
        self.round_records.append(f"[第{self.round}轮] {content}")

    def add_conflict(self, description: str, severity: Severity):
        """添加未解决冲突（去重）"""
        # 检查是否已存在
        for c in self.unresolved_conflicts:
            if c.description == description and not c.resolved:
                c.last_referenced_round = self.round
                c.active_value = 1.0
                return c

        # 检查上限（5条）
        if len(self.unresolved_conflicts) >= 5:
            # 按"创建轮次最早"和"被引用次数最少"排序，移除最低优先级的
            sorted_conflicts = sorted(
                [c for c in self.unresolved_conflicts if c.status == "active" and not c.resolved],
                key=lambda c: (c.round_created, c.last_referenced_round)
            )
            if sorted_conflicts:
                to_freeze = sorted_conflicts[0]
                to_freeze.status = "frozen"
                self.frozen_conflicts.append(to_freeze)
                self.unresolved_conflicts.remove(to_freeze)

        conflict = Conflict(
            description=description,
            round_created=self.round,
            severity=severity,
            last_referenced_round=self.round
        )
        self.unresolved_conflicts.append(conflict)
        return conflict

    def resolve_conflict(self, description: str, resolve_text: str = ""):
        """消解冲突"""
        for c in self.unresolved_conflicts:
            if c.description == description and not c.resolved:
                c.resolved = True
                c.resolve_round = self.round
                c.resolve_description = resolve_text
                self.unresolved_conflicts.remove(c)
                return True
        return False

    def add_deferred(self, description: str):
        """添加延后关注项"""
        for d in self.deferred_attention:
            if d.description == description:
                d.active_value = 1.0
                return
        self.deferred_attention.append(DeferredItem(
            description=description,
            round_created=self.round
        ))

    def decay_all(self):
        """执行所有活跃值的动态衰减"""
        # 冲突衰减
        for c in self.unresolved_conflicts:
            if c.status != "active":
                continue
            # 检查本轮是否被引用
            referenced = (c.last_referenced_round == self.round)
            if not referenced:
                if c.severity == Severity.FATAL:
                    c.active_value *= 0.95
                elif c.severity == Severity.IMPORTANT:
                    c.active_value *= 0.9
                else:
                    c.active_value *= 0.8
                # 休眠判定：连续3轮未被引用
                if self.round - c.last_referenced_round >= 3:
                    c.status = "sleeping"
            else:
                c.active_value = 1.0

        # 延后关注衰减
        for d in self.deferred_attention:
            if d.obsolete:
                continue
            d.active_value *= 0.8
            if d.active_value < 0.3:
                d.obsolete = True

    def thaw_frozen_conflicts(self):
        """每3轮扫描一次冷冻区，移回未解决（休眠状态）"""
        if self.round % 3 != 0:
            return
        for fc in self.frozen_conflicts:
            fc.status = "sleeping"
            self.unresolved_conflicts.append(fc)
        self.frozen_conflicts = []

    def get_context_display(self) -> str:
        """构建简洁的意识流展示（不含历史记录，避免与写入器输出重复）"""
        return self.get_context(include_history=False)

    def get_context(self, include_history: bool = True) -> str:
        """构建当前意识流的文本上下文，供模块读取"""
        lines = []
        lines.append(f"### 当前轮次：第{self.round}轮")
        lines.append(f"当前情感底色：{self.emotional_color}")
        lines.append("")

        # 未解决冲突
        active_conflicts = [c for c in self.unresolved_conflicts if c.status == "active" and not c.resolved]
        if active_conflicts:
            lines.append("未解决冲突：")
            for c in active_conflicts:
                lines.append(f"· [{c.severity.value}]{c.description}（悬置于第{c.round_created}轮，活跃度：{c.active_value:.2f}）")
            lines.append("")

        # 延后关注
        active_deferred = [d for d in self.deferred_attention if not d.obsolete]
        if active_deferred:
            lines.append("延后关注：")
            for d in active_deferred:
                lines.append(f"· {d.description}（延后于第{d.round_created}轮，活跃度：{d.active_value:.2f}）")
            lines.append("")

        # 当前认知线程
        lines.append(f"当前认知线程：{self.cognitive_thread}")
        lines.append("")

        # 目标（种子）
        if self.goal:
            lines.append(f"目标：{self.goal}")
            lines.append("")

        # 本轮意识流记录
        if include_history and self.round_records:
            lines.append("近期思考：")
            for rec in self.round_records[-6:]:
                lines.append(f"{rec[:200]}...")
            lines.append("")

        return "\n".join(lines)

    # ============================================================
    # 空转种子池（规范三.空转种子池）
    # ============================================================

    def rebuild_seed_pool(self):
        """重建空转种子池。
        来源：未解决冲突 + 高唤醒历史记录（含[致命]/[关键事件]/[紧急偏向警告]标记）
        已存在的不重复添加，新种子追加。
        """
        import random as _random
        new_seeds: list[IdleSeed] = []

        # 来源1：未解决冲突（致命>重要>次要）
        for c in self.unresolved_conflicts:
            if c.status == "active" and not c.resolved:
                # 跳过已在池中的
                if any(s.description == c.description for s in self.idle_seeds):
                    continue
                arousal = {Severity.FATAL: 1.0, Severity.IMPORTANT: 0.7, Severity.MINOR: 0.4}.get(
                    c.severity, 0.5)
                new_seeds.append(IdleSeed(
                    description=f"[悬置冲突] {c.description}",
                    source_type="conflict",
                    arousal=arousal,
                    round_created=self.round
                ))

        # 来源2：近5轮中高唤醒标记的历史记录
        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        for rec in recent:
            if any(tag in rec.content for tag in ["[致命]", "[关键事件", "[紧急偏向警告]",
                                                    "[悬置消解]", "[记忆修正]"]):
                short = rec.content[:120]
                if any(s.description == short for s in self.idle_seeds):
                    continue
                new_seeds.append(IdleSeed(
                    description=short,
                    source_type="memory",
                    arousal=0.8,
                    round_created=self.round
                ))

        # 来源3：延后关注中未过时的
        for d in self.deferred_attention:
            if not d.obsolete:
                desc = f"[延后关注] {d.description}"
                if any(s.description == desc for s in self.idle_seeds):
                    continue
                new_seeds.append(IdleSeed(
                    description=desc,
                    source_type="deferred",
                    arousal=d.active_value,
                    round_created=self.round
                ))

        self.idle_seeds.extend(new_seeds)

    def pick_idle_seed(self) -> Optional[IdleSeed]:
        """从种子池中加权随机选取一个种子。
        权重 = arousal / (1 + pick_count)，被选次数越多权重越低。
        返回 None 表示候选池为空。
        """
        import random as _random
        active_seeds = [s for s in self.idle_seeds if s.round_created != self.round
                        or s.source_type == "conflict"]
        if not active_seeds:
            return None

        weights = [s.arousal / (1 + s.pick_count) for s in active_seeds]
        total = sum(weights)
        if total <= 0:
            return None

        # 加权随机选择
        r = _random.random() * total
        cumulative = 0.0
        for s, w in zip(active_seeds, weights):
            cumulative += w
            if r <= cumulative:
                s.pick_count += 1
                return s

        # fallback: 返回最后一个
        active_seeds[-1].pick_count += 1
        return active_seeds[-1]

    def idle_seed_count(self) -> int:
        """返回活跃种子数量"""
        return len(self.idle_seeds)

    # 注：模块视角的上下文实际由 llm_client.build_context() 构建（含互读映射），
    # 此处仅保留基础展示功能。

    def __str__(self) -> str:
        return self.get_context()


# ============================================================
# 交互读取数据
# ============================================================

@dataclass
class NeighborFragments:
    """邻接模块碎片（上一轮）"""
    logical_reasoner: Optional[Fragment] = None
    critic: Optional[Fragment] = None
    association: Optional[Fragment] = None
    emotional: Optional[Fragment] = None
    motivation: Optional[Fragment] = None
    metacognitive: Optional[Fragment] = None


# ============================================================
# 核心配置
# ============================================================

# 互读映射表
CROSS_READING_MAP = {
    "逻辑推理者": ["元认知监控者"],
    "批判者": ["逻辑推理者", "动机检测者"],
    "联想与创意联结者": ["情感评估者", "元认知监控者"],
    "情感评估者": ["联想与创意联结者"],
    "动机检测者": ["批判者", "情感评估者"],
    "元认知监控者": ["逻辑推理者", "批判者", "联想与创意联结者", "情感评估者", "动机检测者"],
}

