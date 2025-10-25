#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将 MAVEN 系列原始 JSON 转换为 SKIRL-RL 所需的五类下游文件。

脚本逻辑概览：
1. 读取 ``data/maven_raw`` 下的 JSON 文档或 ``train/valid/test.jsonl`` 拆分，并结合映射表构造 ``event.jsonl``。
2. 基于 Agent 论元汇聚事件，生成按月划分的 person 轨迹 ``traj.jsonl``。
3. 对每条轨迹构造降质版本，写出偏好对 ``pairs.jsonl``。
4. 将事件转写成指令微调样本 ``maven_sft.jsonl`` 及按拆分输出的 ``maven_sft_{split}.jsonl``。
5. 将轨迹截断为前缀提示生成 ``rl_prompts.jsonl``。

所有输出均通过 Pydantic schema 校验，同时生成统计信息 ``*.stats.json`` 与汇总 ``summary.stats.json``。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "data" / "maven_raw"
DEFAULT_DST = ROOT / "data" / "processed"
MAPPING_DIR = ROOT / "mapping"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("convert_maven")


# =============================
# Pydantic Schema Definitions
# =============================


class EventArgument(BaseModel):
    role: str
    entity_id: str
    span: List[int]

    @field_validator("span")
    @classmethod
    def _validate_span(cls, value: List[int]) -> List[int]:
        assert len(value) == 2 and value[0] <= value[1], "span must be [start, end]"
        return value


class EventTrigger(BaseModel):
    span: List[int]
    text: str
    type: str

    @field_validator("span")
    @classmethod
    def _validate_trigger_span(cls, value: List[int]) -> List[int]:
        assert len(value) == 2 and value[0] <= value[1], "trigger span must be [start, end]"
        return value


class EventRelation(BaseModel):
    type: str
    head: str
    tail: str


class EventRelations(BaseModel):
    temporal: List[EventRelation]
    causal: List[EventRelation]
    subevent: List[EventRelation]


class EventTime(BaseModel):
    value: str
    span: List[int]

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: str) -> str:
        datetime.strptime(value, "%Y-%m-%d")
        return value

    @field_validator("span")
    @classmethod
    def _validate_span(cls, value: List[int]) -> List[int]:
        assert len(value) == 2 and value[0] <= value[1], "time span must be [start, end]"
        return value


class EventConfidence(BaseModel):
    trigger_prob: float = Field(..., ge=0.0, le=1.0)
    arg_role_avg: float = Field(..., ge=0.0, le=1.0)


class EventEntry(BaseModel):
    doc_id: str
    event_id: str
    trigger: EventTrigger
    arguments: List[EventArgument]
    time: EventTime
    relations: EventRelations
    confidence: EventConfidence
    source: str
    mapping: Dict[str, str]
    split: Optional[str] = None


class TrajectoryStep(BaseModel):
    event_id: str
    time: str
    type: str
    roles: Dict[str, str]
    delta_days_from_prev: int
    text_refs: List[Dict[str, object]]
    skeleton_hits: List[str]

    @field_validator("time")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        datetime.strptime(value, "%Y-%m-%d")
        return value


class TrajectoryMeta(BaseModel):
    graph_nodes: List[str]
    graph_edges: List[List[str]]


class TrajectoryEntry(BaseModel):
    person_id: str
    trajectory_id: str
    label: str
    steps: List[TrajectoryStep]
    meta: TrajectoryMeta

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        assert value in {"expert", "candidate", "negative"}
        return value

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, value: List[TrajectoryStep]) -> List[TrajectoryStep]:
        assert value, "trajectory must have at least one step"
        return value


class PreferencePair(BaseModel):
    better: str
    worse: str
    reason: str


class SFTSample(BaseModel):
    prompt: str
    response: str


class RLPrompt(BaseModel):
    prompt: str
    response: str
    trajectory_id: str
    person_id: str


# =============================
# 工具函数与统计
# =============================


@dataclass
class DatasetStats:
    name: str
    num_records: int
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "num_records": self.num_records, **self.extra}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> DatasetStats:
    records = list(records)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            if isinstance(item, BaseModel):
                payload = item.model_dump()
            else:
                payload = item
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    extra: Dict[str, object] = {}
    if path.name == "event.jsonl":
        doc_ids = {r.doc_id for r in records if isinstance(r, EventEntry)}
        extra["num_docs"] = len(doc_ids)
    if path.name == "traj.jsonl":
        labels = defaultdict(int)
        lengths: List[int] = []
        for r in records:
            if isinstance(r, TrajectoryEntry):
                labels[r.label] += 1
                lengths.append(len(r.steps))
        if lengths:
            extra["avg_traj_len"] = sum(lengths) / len(lengths)
        extra["label_distribution"] = dict(labels)
    if path.name == "pairs.jsonl":
        unique_ids = set()
        for r in records:
            if isinstance(r, PreferencePair):
                unique_ids.add(r.better)
                unique_ids.add(r.worse)
        extra["num_unique_ids"] = len(unique_ids)
    stats = DatasetStats(name=path.name, num_records=len(records), extra=extra)
    stats_path = path.with_suffix(path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


# =============================
# 解析原始 JSON
# =============================


def load_mapping(path: Path, default: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    LOGGER.warning("未找到映射文件 %s，使用默认值。", path)
    return default or {}


def normalise_span(value: Dict[str, int] | List[int]) -> List[int]:
    if isinstance(value, list):
        assert len(value) == 2
        return [int(value[0]), int(value[1])]
    keys = ["start", "end"]
    if all(k in value for k in keys):
        return [int(value["start"]), int(value["end"])]
    if all(k in value for k in ("start_token", "end_token")):
        return [int(value["start_token"]), int(value["end_token"])]
    if "offset" in value and isinstance(value["offset"], list) and len(value["offset"]) == 2:
        return [int(value["offset"][0]), int(value["offset"][1])]
    raise ValueError(f"无法解析 span: {value}")


def ensure_time(event: Dict[str, object], fallback_date: str) -> Tuple[str, List[int]]:
    raw_time = event.get("time")
    if isinstance(raw_time, dict) and "value" in raw_time:
        try:
            datetime.strptime(raw_time["value"], "%Y-%m-%d")
            return raw_time["value"], normalise_span(raw_time.get("span", [0, 0]))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("时间字段解析失败 %s：%s，使用回退日期 %s", event.get("id"), exc, fallback_date)
    return fallback_date, [0, 0]


def extract_trigger_from_mentions(
    event: Dict[str, object],
    doc_content: Optional[List[Dict[str, object]]],
) -> Tuple[str, List[int]]:
    mentions = event.get("mention") or event.get("mentions") or []
    if isinstance(mentions, dict):
        mentions = [mentions]
    if not mentions:
        return "", [0, 0]
    mention = mentions[0]
    trigger_text = mention.get("trigger_word") or mention.get("text") or ""
    offset = mention.get("offset") or mention.get("span") or [0, 0]
    if not trigger_text and doc_content:
        sent_id = mention.get("sent_id")
        if isinstance(sent_id, int) and 0 <= sent_id < len(doc_content):
            tokens = doc_content[sent_id].get("tokens", [])
            if (
                isinstance(tokens, list)
                and isinstance(offset, list)
                and len(offset) == 2
            ):
                start, end = offset
                start = max(int(start), 0)
                end = max(int(end), start)
                slice_tokens = tokens[start:end]
                if slice_tokens and all(isinstance(tok, str) for tok in slice_tokens):
                    trigger_text = " ".join(slice_tokens).strip()
    try:
        span = normalise_span(offset)
    except Exception:  # noqa: BLE001
        span = [0, 0]
    return trigger_text, span


def parse_event(
    doc_id: str,
    event: Dict[str, object],
    skeleton_map: Dict[str, List[str] | str],
    cameo_map: Dict[str, str],
    fallback_date: str,
    doc_content: Optional[List[Dict[str, object]]] = None,
    split: Optional[str] = None,
) -> Optional[EventEntry]:
    trigger = event.get("trigger") or {}
    trigger_type = event.get("type", "Unknown")
    using_new_schema = "trigger" not in event or not trigger

    if using_new_schema:
        trigger_text, trigger_span = extract_trigger_from_mentions(event, doc_content)
        if not trigger_text:
            LOGGER.warning("事件 %s 缺少触发词，已跳过", event.get("id"))
            return None
        time_value, time_span = fallback_date, [0, 0]
        arguments: List[EventArgument] = []
        relations_dict: Dict[str, List[Dict[str, str]]] = {}
        source = "MAVEN-JSONL"
    else:
        trigger_text = trigger.get("text", "")
        if not trigger_text:
            LOGGER.warning("事件 %s 缺少 trigger.text，已跳过", event.get("id"))
            return None
        try:
            trigger_span = normalise_span(trigger)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("事件 %s 的 trigger span 解析失败：%s", event.get("id"), exc)
            trigger_span = [0, 0]
        time_value, time_span = ensure_time(event, fallback_date)

        arguments = []
        for arg in event.get("arguments", []):
            role = arg.get("role", "")
            if not role:
                continue
            entity_id = arg.get("entity_id") or arg.get("text") or f"{trigger_type}_{role}"
            try:
                span = normalise_span(arg)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("事件 %s 的 argument span 解析失败：%s", event.get("id"), exc)
                span = [0, 0]
            arguments.append(EventArgument(role=role, entity_id=str(entity_id), span=span))

        if not arguments:
            LOGGER.warning("事件 %s 不含 argument，已跳过", event.get("id"))
            return None

        relations_dict = event.get("relations", {})
        source = "MAVEN|MAVEN-Arg|MAVEN-ERE|RAMS"

    relations = EventRelations(
        temporal=[EventRelation(**rel) for rel in relations_dict.get("temporal", [])],
        causal=[EventRelation(**rel) for rel in relations_dict.get("causal", [])],
        subevent=[EventRelation(**rel) for rel in relations_dict.get("subevent", [])],
    )

    skeleton = skeleton_map.get(trigger_type) or []
    if isinstance(skeleton, str):
        skeleton_hits = [skeleton]
    else:
        skeleton_hits = list(skeleton)
    skeleton_type = skeleton_hits[0] if skeleton_hits else "UNKNOWN"
    mapping = {
        "cameo": cameo_map.get(trigger_type, "000"),
        "skeleton_type": skeleton_type,
    }

    confidence = EventConfidence(trigger_prob=0.9, arg_role_avg=0.85)

    try:
        entry = EventEntry(
            doc_id=doc_id,
            event_id=event.get("id", f"{doc_id}_event"),
            trigger=EventTrigger(span=trigger_span, text=trigger_text, type=trigger_type),
            arguments=arguments,
            time=EventTime(value=time_value, span=time_span),
            relations=relations,
            confidence=confidence,
            source=source,
            mapping=mapping,
            split=split,
        )
    except ValidationError as exc:
        LOGGER.error("事件 %s 校验失败：%s", event.get("id"), exc)
        return None
    return entry


def load_events(src_dir: Path, skeleton_map: Dict[str, object], cameo_map: Dict[str, str]) -> List[EventEntry]:
    events: List[EventEntry] = []
    if not src_dir.exists():
        LOGGER.error("原始目录 %s 不存在。", src_dir)
        return events

    split_files = {name: src_dir / f"{name}.jsonl" for name in ("train", "valid", "test")}
    has_jsonl = any(path.exists() for path in split_files.values())

    if has_jsonl:
        for split_name, jsonl_path in split_files.items():
            if not jsonl_path.exists():
                continue
            with jsonl_path.open("r", encoding="utf-8") as f:
                iterator = enumerate(tqdm(f, desc=f"加载{split_name}", unit="doc"), start=1)
                for line_idx, line in iterator:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:  # noqa: PERF203
                        LOGGER.error("解析 %s 第 %d 行失败：%s", jsonl_path.name, line_idx, exc)
                        continue
                    doc_id = data.get("id") or f"{split_name}_{line_idx:06d}"
                    fallback_date = (
                        data.get("publish_time")
                        or data.get("time")
                        or data.get("date")
                        or "2014-01-01"
                    )
                    doc_content = data.get("content")
                    raw_events = data.get("events") or []
                    if not raw_events and data.get("candidates"):
                        raw_events = [
                            {"id": cand.get("id"), "type": "Unknown", "mention": [cand]}
                            for cand in data.get("candidates", [])
                        ]
                    for event in raw_events:
                        entry = parse_event(
                            doc_id,
                            event,
                            skeleton_map,
                            cameo_map,
                            fallback_date,
                            doc_content=doc_content,
                            split=split_name,
                        )
                        if entry is not None:
                            events.append(entry)
        LOGGER.info(
            "共解析事件 %d 条，来源拆分：%s。",
            len(events),
            ", ".join(name for name, path in split_files.items() if path.exists()),
        )
        return events

    json_files = sorted(path for path in src_dir.glob("*.json"))
    if not json_files:
        LOGGER.error("原始目录 %s 下未找到 JSON/JSONL 文件。", src_dir)
        return events

    for json_path in tqdm(json_files, desc="加载文档"):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        doc_id = data.get("id") or json_path.stem
        fallback_date = data.get("publish_time") or "2014-01-01"
        raw_events = data.get("events") or []
        if not raw_events and data.get("candidates"):
            raw_events = [
                {"id": cand.get("id"), "type": "Unknown", "mention": [cand]}
                for cand in data.get("candidates", [])
            ]
        for event in raw_events:
            entry = parse_event(
                doc_id,
                event,
                skeleton_map,
                cameo_map,
                fallback_date,
            )
            if entry is not None:
                events.append(entry)
    LOGGER.info("共解析事件 %d 条。", len(events))
    return events


# =============================
# 构造轨迹与偏好对
# =============================


def month_key(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y%m")


def compute_delta_days(prev: Optional[str], current: str) -> int:
    if prev is None:
        return 0
    dt_prev = datetime.strptime(prev, "%Y-%m-%d")
    dt_cur = datetime.strptime(current, "%Y-%m-%d")
    return (dt_cur - dt_prev).days


def classify_label(skeleton_seq: List[str]) -> str:
    hits = set(skeleton_seq)
    if {"PREP", "PROBE", "EXECUTE", "CASHOUT"}.issubset(hits):
        return "expert"
    if len(hits) >= 2 and "EXECUTE" in hits:
        return "candidate"
    return "negative"


def build_trajectories(events: List[EventEntry], skeleton_map: Dict[str, object]) -> Tuple[List[TrajectoryEntry], Dict[str, str]]:
    grouped: Dict[Tuple[str, str], List[Tuple[EventEntry, str]]] = defaultdict(list)
    for event in events:
        agent_ids = [arg.entity_id for arg in event.arguments if arg.role.lower() == "agent".lower()]
        if not agent_ids:
            continue
        for agent in agent_ids:
            grouped[(agent, month_key(event.time.value))].append((event, agent))

    trajectories: List[TrajectoryEntry] = []
    base_to_person: Dict[str, str] = {}

    for (person_id, month), ev_list in grouped.items():
        ev_list.sort(key=lambda item: datetime.strptime(item[0].time.value, "%Y-%m-%d"))
        steps: List[TrajectoryStep] = []
        meta_nodes = set([person_id])
        meta_edges: List[List[str]] = []
        prev_time: Optional[str] = None
        skeleton_seq: List[str] = []
        for event, _agent in ev_list:
            roles = {arg.role: arg.entity_id for arg in event.arguments}
            meta_nodes.update(roles.values())
            step_skeleton = skeleton_map.get(event.trigger.type, [])
            if isinstance(step_skeleton, str):
                skeleton_hits = [step_skeleton]
            else:
                skeleton_hits = list(step_skeleton)
            skeleton_seq.extend(skeleton_hits)
            delta = compute_delta_days(prev_time, event.time.value)
            prev_time = event.time.value
            text_ref = {"doc_id": event.doc_id, "span": event.trigger.span}
            steps.append(
                TrajectoryStep(
                    event_id=event.event_id,
                    time=event.time.value,
                    type=event.trigger.type,
                    roles=roles,
                    delta_days_from_prev=delta,
                    text_refs=[text_ref],
                    skeleton_hits=skeleton_hits,
                )
            )
            for role, ent in roles.items():
                if role.lower() in {"target", "place"}:
                    meta_edges.append([person_id, event.trigger.type, ent, event.time.value])
        if not steps:
            continue
        label = classify_label(skeleton_seq)
        traj_id = f"traj_{person_id}_{month}"
        trajectory = TrajectoryEntry(
            person_id=person_id,
            trajectory_id=traj_id,
            label=label,
            steps=steps,
            meta=TrajectoryMeta(graph_nodes=sorted(meta_nodes), graph_edges=meta_edges),
        )
        trajectories.append(trajectory)
        base_to_person[traj_id] = person_id

    LOGGER.info("生成轨迹 %d 条。", len(trajectories))
    return trajectories, base_to_person


def degrade_trajectory(traj: TrajectoryEntry, suffix: str, rng: random.Random) -> Optional[TrajectoryEntry]:
    steps = list(traj.steps)
    if len(steps) < 2:
        return None
    mutated_steps = [step.model_copy(deep=True) for step in steps]
    choice = rng.choice(["drop", "swap"])
    if choice == "drop":
        drop_count = 1 if len(mutated_steps) <= 3 else 2
        drop_indices = sorted(rng.sample(range(len(mutated_steps)), drop_count), reverse=True)
        for idx in drop_indices:
            mutated_steps.pop(idx)
    else:  # swap
        idx = rng.randrange(len(mutated_steps) - 1)
        mutated_steps[idx], mutated_steps[idx + 1] = mutated_steps[idx + 1], mutated_steps[idx]
    if not mutated_steps:
        return None
    mutated_label = "candidate" if choice == "swap" else "negative"
    mutated_id = f"{traj.trajectory_id}_{suffix}"
    mutated = TrajectoryEntry(
        person_id=traj.person_id,
        trajectory_id=mutated_id,
        label=mutated_label,
        steps=mutated_steps,
        meta=traj.meta,
    )
    return mutated


def build_preference_pairs(
    trajectories: List[TrajectoryEntry],
    rng: random.Random,
) -> Tuple[List[TrajectoryEntry], List[PreferencePair]]:
    extra_trajs: List[TrajectoryEntry] = []
    pairs: List[PreferencePair] = []
    for traj in trajectories:
        mutated = degrade_trajectory(traj, "mut", rng)
        if mutated is None:
            continue
        extra_trajs.append(mutated)
        pairs.append(
            PreferencePair(
                better=traj.trajectory_id,
                worse=mutated.trajectory_id,
                reason="more complete EXECUTE→CASHOUT chain",
            )
        )
    LOGGER.info("生成偏好对 %d 组。", len(pairs))
    return extra_trajs, pairs


# =============================
# 构造 SFT 与 RL 数据
# =============================


def format_arguments(arguments: List[EventArgument]) -> str:
    lines = []
    for arg in arguments:
        lines.append(f"- {arg.role}: {arg.entity_id} span={arg.span}")
    return "\n".join(lines) if lines else "- None"


def build_sft_samples(events: List[EventEntry]) -> Tuple[List[SFTSample], Dict[str, List[SFTSample]]]:
    samples: List[SFTSample] = []
    grouped: Dict[str, List[SFTSample]] = defaultdict(list)
    for event in events:
        prompt = (
            f"[DOC] {event.doc_id}\n"
            f"[EVENT] {event.event_id}\n"
            f"[TRIGGER] {event.trigger.text} ({event.trigger.type})\n"
            "[ARGUMENTS]\n"
            f"{format_arguments(event.arguments)}\n"
            f"[TIME] {event.time.value if event.time else 'N/A'}\n"
        )
        agent_roles = [arg.entity_id for arg in event.arguments if arg.role.lower() == "agent"]
        agent_text = "、".join(agent_roles) if agent_roles else "未知主体"
        response = f"事件{event.event_id}描述了{event.trigger.text}，类型为{event.trigger.type}，主体为{agent_text}。"
        sample = SFTSample(prompt=prompt, response=response)
        samples.append(sample)
        if event.split:
            grouped[event.split].append(sample)
    LOGGER.info("构造 SFT 样本 %d 条。", len(samples))
    if grouped:
        for split_name, split_samples in grouped.items():
            LOGGER.info("  - %s: %d 条", split_name, len(split_samples))
    return samples, grouped


def build_rl_prompts(trajectories: List[TrajectoryEntry]) -> List[RLPrompt]:
    prompts: List[RLPrompt] = []
    for traj in trajectories:
        if len(traj.steps) < 2:
            continue
        for idx in range(1, len(traj.steps)):
            prefix = traj.steps[:idx]
            next_step = traj.steps[idx]
            prefix_lines = []
            for step in prefix:
                roles_str = ", ".join(f"{k}:{v}" for k, v in step.roles.items())
                skeleton_str = ",".join(step.skeleton_hits)
                prefix_lines.append(
                    f"* {step.time} {step.type} roles={{{roles_str}}} skeleton={skeleton_str}"
                )
            prompt = (
                f"[PERSON] {traj.person_id}\n"
                f"[TRAJ] {traj.trajectory_id}\n\n"
                + "\n".join(prefix_lines)
                + "\n请预测下一步骨架动作与核心论元。"
            )
            skeleton_str = ",".join(next_step.skeleton_hits)
            response = f"{next_step.type} | skeleton={skeleton_str}"
            prompts.append(
                RLPrompt(
                    prompt=prompt,
                    response=response,
                    trajectory_id=traj.trajectory_id,
                    person_id=traj.person_id,
                )
            )
    LOGGER.info("构造 RL 提示 %d 条。", len(prompts))
    return prompts


# =============================
# 主流程
# =============================


def build_summary(stats: List[DatasetStats], output_path: Path) -> None:
    summary = {item.name: item.to_dict() for item in stats}
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MAVEN raw JSON to processed SKIRL datasets")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="原始 JSON 目录")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="输出目录")
    args = parser.parse_args()

    ensure_dir(args.dst)

    skeleton_map = load_mapping(MAPPING_DIR / "event2skeleton.json", default={})
    cameo_map = load_mapping(MAPPING_DIR / "event2cameo.json", default={})

    events = load_events(args.src, skeleton_map, cameo_map)
    if not events:
        LOGGER.error("未生成任何事件，流程终止。")
        return

    event_stats = write_jsonl(args.dst / "event.jsonl", events)

    base_trajs, _ = build_trajectories(events, skeleton_map)
    extra_trajs, pairs = build_preference_pairs(base_trajs, random.Random(42))
    all_trajs = base_trajs + extra_trajs
    traj_stats = write_jsonl(args.dst / "traj.jsonl", all_trajs)
    pair_stats = write_jsonl(args.dst / "pairs.jsonl", pairs)

    sft_samples, sft_by_split = build_sft_samples(events)
    sft_stats = write_jsonl(args.dst / "maven_sft.jsonl", sft_samples)
    sft_split_stats: List[DatasetStats] = []
    for split_name, split_samples in sorted(sft_by_split.items()):
        split_path = args.dst / f"maven_sft_{split_name}.jsonl"
        sft_split_stats.append(write_jsonl(split_path, split_samples))

    rl_prompts = build_rl_prompts(all_trajs)
    rl_stats = write_jsonl(args.dst / "rl_prompts.jsonl", rl_prompts)

    summary_path = args.dst / "summary.stats.json"
    summary_items = [event_stats, traj_stats, pair_stats, sft_stats]
    summary_items.extend(sft_split_stats)
    summary_items.append(rl_stats)
    build_summary(summary_items, summary_path)
    LOGGER.info("处理完成，结果写入 %s", args.dst)


if __name__ == "__main__":
    main()
