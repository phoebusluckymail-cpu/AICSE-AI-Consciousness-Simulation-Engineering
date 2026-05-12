"""
会话管理器 - 本地数据持久化
职责：保存/加载完整对话状态，支持断点续对话
"""

import json
import os
import shutil
import copy
from enum import Enum
from datetime import datetime
from typing import Optional
from pathlib import Path

from .models import (
    ConsciousnessStream, NeighborFragments, Fragment,
    Conflict, DeferredItem,
    Record, IdleSeed,
    Severity, SourceType, ConvergenceStatus, SwitchSuggestion, PauseDecision,
    CROSS_READING_MAP
)


# ============================================================
# JSON 序列化/反序列化
# ============================================================

_ENUM_MAP = {
    "Severity": Severity,
    "SourceType": SourceType,
    "ConvergenceStatus": ConvergenceStatus,
    "SwitchSuggestion": SwitchSuggestion,
    "PauseDecision": PauseDecision,
}


def _obj_to_dict(obj):
    """递归将对象转为JSON可序列化字典"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return {"__enum__": type(obj).__name__, "value": obj.value}
    if isinstance(obj, list):
        return [_obj_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _obj_to_dict(v) for k, v in obj.items()}

    # dataclass
    if hasattr(obj, "__dataclass_fields__"):
        result = {"__dataclass__": type(obj).__name__}
        for field_name in obj.__dataclass_fields__:
            result[field_name] = _obj_to_dict(getattr(obj, field_name))
        return result

    return str(obj)


def _dict_to_obj(data):
    """递归将字典还原为Python对象"""
    if data is None:
        return None
    if isinstance(data, list):
        return [_dict_to_obj(item) for item in data]
    if isinstance(data, dict):
        if "__enum__" in data:
            enum_cls = _ENUM_MAP.get(data["__enum__"])
            if enum_cls:
                for member in enum_cls:
                    if member.value == data["value"]:
                        return member
                return enum_cls(data["value"])
            return data["value"]

        if "__dataclass__" in data:
            dc_name = data["__dataclass__"]
            # 根据 __dataclass__ 名称重建对应的 dataclass
            kwargs = {k: _dict_to_obj(v) for k, v in data.items() if k != "__dataclass__"}
            _DC_MAP = {
                "Conflict": Conflict,
                "DeferredItem": DeferredItem,
                "Record": Record,
                "Fragment": Fragment,
                "NeighborFragments": NeighborFragments,
                "IdleSeed": IdleSeed,
            }
            if dc_name == "NeighborFragments":
                return NeighborFragments(**kwargs)
            cls = _DC_MAP.get(dc_name)
            if cls:
                return cls(**kwargs)
            return kwargs

        return {k: _dict_to_obj(v) for k, v in data.items()}

    return data


# ============================================================
# 意识流序列化
# ============================================================

def stream_to_dict(stream: ConsciousnessStream) -> dict:
    """将ConsciousnessStream序列化为可JSON的字典"""
    return {
        "round": stream.round,
        "emotional_color": stream.emotional_color,
        "unresolved_conflicts": [_obj_to_dict(c) for c in stream.unresolved_conflicts],
        "deferred_attention": [_obj_to_dict(d) for d in stream.deferred_attention],
        "cognitive_thread": stream.cognitive_thread,
        "round_records": stream.round_records,
        "frozen_conflicts": [_obj_to_dict(c) for c in stream.frozen_conflicts],
        "idle_mode": stream.idle_mode,
        "paused": stream.paused,
        "waiting_for_user": stream.waiting_for_user,
        "pause_reason": stream.pause_reason,
        "goal": stream.goal,
        "no_progress_count": stream.no_progress_count,
        "last_synthesizer_summary": stream.last_synthesizer_summary,
        "chat_history": stream.chat_history,
        "convergence_rounds": stream.convergence_rounds,
        "idle_seeds": [_obj_to_dict(s) for s in stream.idle_seeds],
        "idle_round_count": stream.idle_round_count,
    }


def dict_to_stream(data: dict) -> ConsciousnessStream:
    """从字典恢复ConsciousnessStream"""
    stream = ConsciousnessStream()
    stream.round = data.get("round", 0)
    stream.emotional_color = data.get("emotional_color", "无")
    stream.unresolved_conflicts = _dict_to_obj(data.get("unresolved_conflicts", []))
    stream.deferred_attention = _dict_to_obj(data.get("deferred_attention", []))
    stream.cognitive_thread = data.get("cognitive_thread", "")
    stream.round_records = data.get("round_records", [])
    stream.frozen_conflicts = _dict_to_obj(data.get("frozen_conflicts", []))
    stream.idle_mode = data.get("idle_mode", False)
    stream.paused = data.get("paused", False)
    stream.waiting_for_user = data.get("waiting_for_user", False)
    stream.pause_reason = data.get("pause_reason", "")
    stream.goal = data.get("goal", "")
    stream.no_progress_count = data.get("no_progress_count", 0)
    stream.last_synthesizer_summary = data.get("last_synthesizer_summary", "")
    stream.chat_history = data.get("chat_history", [])
    stream.convergence_rounds = data.get("convergence_rounds", 0)
    stream.idle_seeds = _dict_to_obj(data.get("idle_seeds", []))
    stream.idle_round_count = data.get("idle_round_count", 0)
    return stream


# ============================================================
# 会话管理器
# ============================================================

class SessionManager:
    """管理对话会话的本地持久化"""

    def __init__(self, session_dir: str):
        self.session_dir = Path(session_dir)
        self._ensure_dirs()

    def _ensure_dirs(self):
        """创建会话目录结构"""
        dirs = [
            self.session_dir,
            self.session_dir / "history",
            self.session_dir / "agent_contexts",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def records_md(self) -> Path:
        return self.session_dir / "records.md"

    @property
    def stream_state_json(self) -> Path:
        return self.session_dir / "stream_state.json"

    @property
    def narrative_md(self) -> Path:
        return self.session_dir / "narrative.md"

    @property
    def chat_history_json(self) -> Path:
        return self.session_dir / "chat_history.json"

    @property
    def metadata_json(self) -> Path:
        return self.session_dir / "metadata.json"

    # ============================================================
    # 创建会话
    # ============================================================

    @staticmethod
    def create_session(base_dir: str, goal: str = "") -> "SessionManager":
        """创建新的会话文件夹，返回SessionManager"""
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_goal = "".join(c if c.isalnum() or c in " _-" else "_" for c in goal)[:20]
        folder_name = f"{timestamp}_{safe_goal}" if safe_goal else timestamp
        session_path = base / folder_name

        sm = SessionManager(str(session_path))
        sm._save_metadata(goal)
        return sm

    def _save_metadata(self, goal: str = ""):
        """保存会话元数据"""
        meta = {
            "created_at": datetime.now().isoformat(),
            "goal": goal,
            "rounds": 0,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.metadata_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _update_metadata(self, round_num: int):
        """更新元数据（轮次、时间）"""
        if self.metadata_json.exists():
            with open(self.metadata_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        meta["rounds"] = round_num
        meta["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ============================================================
    # 保存（每轮结束后调用）
    # ============================================================

    def save_round(self, stream: ConsciousnessStream,
                   fragments: dict[str, Fragment],
                   synth_output: Fragment,
                   writer_output: str,
                   narrative: str,
                   expression: str,
                   agent_contexts: dict[str, str],
                   prev_fragments: Optional[NeighborFragments] = None,
                   first_round: bool = False):
        """保存一轮完整数据"""
        round_num = stream.round

        # 1. 保存意识流状态
        self.save_stream_state(stream)

        # 2. 追加到markdown记录
        self._append_markdown_record(round_num, fragments, synth_output,
                                      writer_output, narrative, expression)

        # 3. 保存综合意识流
        if narrative:
            with open(self.narrative_md, "w", encoding="utf-8") as f:
                f.write(narrative)

        # 4. 保存各Agent上下文
        self.save_agent_contexts(round_num, agent_contexts)

        # 5. 保存引擎状态（prev_fragments等，用于断点恢复）
        self._save_engine_state(prev_fragments, first_round)

        # 6. 更新元数据
        self._update_metadata(round_num)

    @property
    def engine_state_json(self) -> Path:
        return self.session_dir / "engine_state.json"

    def _save_engine_state(self, prev_fragments: Optional[NeighborFragments] = None,
                           first_round: bool = False):
        """保存引擎状态，用于断点恢复"""
        state = {
            "first_round": first_round,
        }
        if prev_fragments:
            state["prev_fragments"] = _obj_to_dict(prev_fragments)
        with open(self.engine_state_json, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_engine_state(self) -> dict:
        """加载引擎状态"""
        if not self.engine_state_json.exists():
            return {"first_round": True, "prev_fragments": None}
        with open(self.engine_state_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        prev = data.get("prev_fragments")
        if prev:
            data["prev_fragments"] = _dict_to_obj(prev)
        return data

    def save_stream_state(self, stream: ConsciousnessStream):
        """保存意识流状态到JSON"""
        data = stream_to_dict(stream)
        with open(self.stream_state_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_agent_contexts(self, round_num: int, contexts: dict[str, str]):
        """保存本轮各Agent的上下文"""
        round_dir = self.session_dir / "agent_contexts" / f"round_{round_num:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        for agent_name, context in contexts.items():
            safe_name = agent_name.replace(" ", "_").replace("/", "_")
            file_path = round_dir / f"{safe_name}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(context)

    def _append_markdown_record(self, round_num: int,
                                 fragments: dict[str, Fragment],
                                 synth_output: Fragment,
                                 writer_output: str,
                                 narrative: str,
                                 expression: str):
        """追加一轮的Markdown格式对话记录"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        lines.append(f"# 第{round_num}轮 ({timestamp})\n")

        # 发散碎片
        lines.append("## 发散碎片\n")
        for name in ["逻辑推理者", "批判者", "联想与创意联结者",
                       "情感评估者", "动机检测者", "元认知监控者"]:
            frag = fragments.get(name)
            if frag and frag.content:
                lines.append(f"### {name}\n")
                tags_str = f" `[{'; '.join(frag.tags)}]`" if frag.tags else ""
                lines.append(f"{frag.content}{tags_str}\n")

        # 合成器输出
        lines.append("## 合成器输出\n")
        lines.append(f"{synth_output.content}\n")

        # 写入器记录
        lines.append("## 意识流写入器记录\n")
        lines.append(f"{writer_output}\n")

        # 综合意识流
        if narrative:
            lines.append("## 综合意识流\n")
            lines.append(f"{narrative}\n")

        # 语言表达
        if expression:
            lines.append("## 语言表达\n")
            lines.append(f"{expression}\n")

        lines.append("---\n")
        lines.append("")

        # 追加到文件
        with open(self.records_md, "a", encoding="utf-8") as f:
            f.writelines(line + "\n" if not line.endswith("\n") else line for line in lines)

    # ============================================================
    # 加载
    # ============================================================

    @staticmethod
    def list_sessions(base_dir: str) -> list[dict]:
        """列出所有会话"""
        base = Path(base_dir)
        if not base.exists():
            return []
        sessions = []
        for folder in sorted(base.iterdir(), reverse=True):
            if folder.is_dir():
                meta_file = folder / "metadata.json"
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                else:
                    meta = {"goal": "", "rounds": 0}
                sessions.append({
                    "path": str(folder),
                    "name": folder.name,
                    "goal": meta.get("goal", ""),
                    "rounds": meta.get("rounds", 0),
                    "created_at": meta.get("created_at", ""),
                    "last_updated": meta.get("last_updated", ""),
                })
        return sessions

    def load_stream_state(self) -> Optional[ConsciousnessStream]:
        """从JSON文件加载意识流状态"""
        if not self.stream_state_json.exists():
            return None
        with open(self.stream_state_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return dict_to_stream(data)

    def load_narrative(self) -> str:
        """加载综合意识流"""
        if self.narrative_md.exists():
            with open(self.narrative_md, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def load_agent_contexts_for_round(self, round_num: int) -> dict[str, str]:
        """加载指定轮次的所有Agent上下文"""
        round_dir = self.session_dir / "agent_contexts" / f"round_{round_num:04d}"
        if not round_dir.exists():
            return {}
        contexts = {}
        for file_path in sorted(round_dir.iterdir()):
            if file_path.suffix == ".txt":
                agent_name = file_path.stem.replace("_", " ")
                with open(file_path, "r", encoding="utf-8") as f:
                    contexts[agent_name] = f.read()
        return contexts

    def load_all_contexts(self) -> list[dict]:
        """加载所有轮次的Agent上下文（按轮次排序）"""
        rounds = []
        contexts_dir = self.session_dir / "agent_contexts"
        if not contexts_dir.exists():
            return rounds
        for round_dir in sorted(contexts_dir.iterdir()):
            if round_dir.is_dir() and round_dir.name.startswith("round_"):
                round_num = int(round_dir.name.split("_")[1])
                contexts = self.load_agent_contexts_for_round(round_num)
                if contexts:
                    rounds.append({"round": round_num, "contexts": contexts})
        return rounds

    def load_records_markdown(self) -> str:
        """加载Mark格式对话记录"""
        if self.records_md.exists():
            with open(self.records_md, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def load_expression_history(self, stream: ConsciousnessStream) -> list[str]:
        """从历史记录中恢复表达历史"""
        return []  # v5.0: expression_history 已内联保存

    def save_chat_history(self, messages: list[dict]):
        """保存对话历史到JSON"""
        with open(self.chat_history_json, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2, default=str)

    def load_chat_history(self) -> list[dict]:
        """从JSON加载对话历史"""
        if not self.chat_history_json.exists():
            return []
        with open(self.chat_history_json, "r", encoding="utf-8") as f:
            return json.load(f)
