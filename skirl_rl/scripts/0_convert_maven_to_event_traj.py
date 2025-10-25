#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate synthetic MAVEN-style samples for SKIRL-RL demo.

This script constructs minimal but schema-valid demo data under ``data/processed``.
It creates the following files together with ``*.stats.json`` companions:

* ``event.jsonl`` – event-level annotations
* ``traj.jsonl`` – person-centric trajectories labelled as expert/candidate/negative
* ``pairs.jsonl`` – preference pairs for IRL regularisation
* ``maven_sft.jsonl`` – conversational SFT samples converted from events
* ``rl_prompts.jsonl`` – PPO prompts based on partial trajectories

All structures are validated with ``pydantic`` models.  The script is idempotent and
can run without the original MAVEN corpus – it fabricates a consistent toy dataset
that exercises the full training pipeline offline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"


class EventArgument(BaseModel):
    role: str
    entity_id: str
    span: List[int]

    @field_validator("span")
    @classmethod
    def _validate_span(cls, v: List[int]) -> List[int]:
        assert len(v) == 2 and v[0] <= v[1], "span must be [start, end]"
        return v


class EventTrigger(BaseModel):
    span: List[int]
    text: str
    type: str

    @field_validator("span")
    @classmethod
    def _validate_trigger_span(cls, v: List[int]) -> List[int]:
        assert len(v) == 2 and v[0] <= v[1], "trigger span must be [start, end]"
        return v


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

    @field_validator("span")
    @classmethod
    def _validate_time_span(cls, v: List[int]) -> List[int]:
        assert len(v) == 2 and v[0] <= v[1], "time span must be [start, end]"
        return v

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v


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
    def _validate_step_time(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v


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
    def _validate_label(cls, v: str) -> str:
        assert v in {"expert", "candidate", "negative"}
        return v

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, v: List[TrajectoryStep]) -> List[TrajectoryStep]:
        assert v, "trajectory must contain at least one step"
        return v


class PreferencePair(BaseModel):
    better: str
    worse: str
    reason: str


@dataclass
class DatasetStats:
    name: str
    num_records: int = 0
    extra: Dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "num_records": self.num_records, **self.extra}, ensure_ascii=False, indent=2)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> DatasetStats:
    items = list(items)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
    stats = DatasetStats(name=path.name, num_records=len(items))
    if path.name == "event.jsonl":
        doc_counter = Counter(e.doc_id for e in items)
        stats.extra.update({"num_docs": len(doc_counter)})
    elif path.name == "traj.jsonl":
        label_counter = Counter(t.label for t in items)
        stats.extra.update({"label_distribution": label_counter})
    elif path.name == "pairs.jsonl":
        stats.extra.update({"num_unique_ids": len({p.better for p in items} | {p.worse for p in items})})
    stats_path = path.with_suffix(path.suffix + ".stats.json")
    stats_path.write_text(stats.to_json(), encoding="utf-8")
    return stats


def build_demo_events() -> List[EventEntry]:
    base = {
        "doc_id": "doc_demo_0001",
        "source": "MAVEN|MAVEN-Arg|MAVEN-ERE|RAMS",
        "mapping": {"cameo": "190", "skeleton_type": "EXECUTE"},
    }
    events: List[EventEntry] = []
    triggers = [
        ("doc_demo_0001_e1", "侦察", "Conflict.Patrol"),
        ("doc_demo_0001_e2", "突袭", "Conflict.Attack"),
        ("doc_demo_0001_e3", "爆炸", "Conflict.Explosion"),
    ]
    offsets = [(20, 22), (60, 62), (120, 122)]
    for idx, (event_id, text, ev_type) in enumerate(triggers):
        trigger_span = [offsets[idx][0], offsets[idx][1]]
        arguments = [
            EventArgument(role="Agent", entity_id="P_001" if idx < 2 else "P_044", span=[80 + idx * 10, 85 + idx * 10]),
            EventArgument(role="Target", entity_id="O_017", span=[140, 146]),
            EventArgument(role="Place", entity_id="L_Basra", span=[170, 175]),
        ]
        relations = EventRelations(
            temporal=[EventRelation(type="BEFORE", head=triggers[max(0, idx - 1)][0], tail=event_id)] if idx > 0 else [],
            causal=[EventRelation(type="CAUSE", head=event_id, tail="doc_demo_0001_e4")] if idx == 2 else [],
            subevent=[],
        )
        confidence = EventConfidence(trigger_prob=0.85 + idx * 0.04, arg_role_avg=0.8 - idx * 0.02)
        event = EventEntry(
            **base,
            event_id=event_id,
            trigger=EventTrigger(span=trigger_span, text=text, type=ev_type),
            arguments=arguments,
            time=EventTime(value=f"2014-03-{19 + idx:02d}", span=[40 + idx * 5, 50 + idx * 5]),
            relations=relations,
            confidence=confidence,
        )
        events.append(event)
    # Additional document with different mapping
    events.append(
        EventEntry(
            doc_id="doc_demo_0002",
            event_id="doc_demo_0002_e1",
            trigger=EventTrigger(span=[30, 33], text="谈判", type="Diplomacy.Talks"),
            arguments=[
                EventArgument(role="Agent", entity_id="P_099", span=[10, 12]),
                EventArgument(role="Target", entity_id="O_050", span=[40, 42]),
            ],
            time=EventTime(value="2014-04-02", span=[70, 80]),
            relations=EventRelations(temporal=[], causal=[], subevent=[]),
            confidence=EventConfidence(trigger_prob=0.9, arg_role_avg=0.88),
            source="MAVEN|MAVEN-ERE",
            mapping={"cameo": "050", "skeleton_type": "NEGOTIATE"},
        )
    )
    return events


def build_demo_trajectories(events: List[EventEntry]) -> List[TrajectoryEntry]:
    step_lookup = {event.event_id: event for event in events}
    trajectories: List[TrajectoryEntry] = []

    traj_specs = [
        (
            "P_001",
            "traj_P001_201403",
            "expert",
            [
                ("doc_demo_0001_e1", ["PREP", "PROBE"], 0),
                ("doc_demo_0001_e2", ["EXECUTE"], 2),
                ("doc_demo_0001_e3", ["EXECUTE", "CASHOUT"], 1),
            ],
        ),
        (
            "P_044",
            "traj_P044_201403",
            "candidate",
            [
                ("doc_demo_0001_e3", ["EXECUTE"], 0),
                ("doc_demo_0002_e1", ["NEGOTIATE"], 5),
            ],
        ),
        (
            "P_099",
            "traj_P099_201404",
            "negative",
            [
                ("doc_demo_0002_e1", ["NEGOTIATE"], 0),
            ],
        ),
    ]

    for person_id, traj_id, label, steps in traj_specs:
        traj_steps: List[TrajectoryStep] = []
        prev_time: Optional[datetime] = None
        for ev_id, skeleton_hits, delay in steps:
            event = step_lookup[ev_id]
            event_time = datetime.strptime(event.time.value, "%Y-%m-%d")
            if prev_time is None:
                delta = 0
            else:
                delta = (event_time - prev_time).days
            delta += delay
            prev_time = event_time
            traj_steps.append(
                TrajectoryStep(
                    event_id=ev_id,
                    time=event.time.value,
                    type=event.trigger.type,
                    roles={arg.role: arg.entity_id for arg in event.arguments},
                    delta_days_from_prev=delta,
                    text_refs=[{"doc_id": event.doc_id, "span": event.trigger.span}],
                    skeleton_hits=skeleton_hits,
                )
            )
        meta = TrajectoryMeta(
            graph_nodes=list({arg.entity_id for step in traj_steps for arg in step_lookup[step.event_id].arguments} | {person_id}),
            graph_edges=[[person_id, step_lookup[step.event_id].trigger.type, step.roles.get("Target", "UNK"), step.time] for step in traj_steps],
        )
        trajectories.append(
            TrajectoryEntry(
                person_id=person_id,
                trajectory_id=traj_id,
                label=label,
                steps=traj_steps,
                meta=meta,
            )
        )
    return trajectories


def build_pairs(trajectories: List[TrajectoryEntry]) -> List[PreferencePair]:
    pair_specs = [
        ("traj_P001_201403", "traj_P044_201403", "more complete EXECUTE→CASHOUT chain"),
        ("traj_P044_201403", "traj_P099_201404", "contains EXECUTE stage"),
    ]
    existing_ids = {traj.trajectory_id for traj in trajectories}
    pairs: List[PreferencePair] = []
    for better, worse, reason in pair_specs:
        assert better in existing_ids and worse in existing_ids, "pair references unknown trajectory"
        pairs.append(PreferencePair(better=better, worse=worse, reason=reason))
    return pairs


def build_sft_samples(events: List[EventEntry]) -> List[Dict[str, str]]:
    samples = []
    for event in events:
        prompt_lines = [
            f"[DOC] {event.doc_id}",
            f"[EVENT] {event.event_id}",
            f"[TRIGGER] {event.trigger.text} ({event.trigger.type})",
            "[ARGUMENTS]",
        ]
        for arg in event.arguments:
            prompt_lines.append(f"- {arg.role}: {arg.entity_id} span={arg.span}")
        prompt_lines.append(f"[TIME] {event.time.value}")
        response = (
            f"事件{event.event_id}描述了{event.trigger.text}，主要类型是{event.trigger.type}，"
            f"由{event.arguments[0].entity_id if event.arguments else '未知主体'}发起。"
        )
        samples.append({"prompt": "\n".join(prompt_lines), "response": response})
    return samples


def build_rl_prompts(trajectories: List[TrajectoryEntry]) -> List[Dict[str, object]]:
    prompts = []
    for traj in trajectories:
        partial = traj.steps[:-1] if len(traj.steps) > 1 else traj.steps
        context_lines = [
            f"[PERSON] {traj.person_id}",
            f"[TRAJ] {traj.trajectory_id}",
        ]
        for step in partial:
            context_lines.append(
                f"- {step.time} {step.type} roles={step.roles} skeleton={','.join(step.skeleton_hits)}"
            )
        target_step = traj.steps[-1]
        prompts.append(
            {
                "prompt": "\n".join(context_lines) + "\n请预测下一步骨架动作与核心论元。",
                "response": target_step.type,
                "trajectory_id": traj.trajectory_id,
            }
        )
    return prompts


def write_json(path: Path, payload: List[Dict[str, object]]) -> DatasetStats:
    with path.open("w", encoding="utf-8") as f:
        for item in payload:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    stats = DatasetStats(name=path.name, num_records=len(payload))
    stats_path = path.with_suffix(path.suffix + ".stats.json")
    stats.extra.update({"fields": list(payload[0].keys()) if payload else []})
    stats_path.write_text(stats.to_json(), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo MAVEN SKIRL-RL dataset")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    events = build_demo_events()
    trajectories = build_demo_trajectories(events)
    pairs = build_pairs(trajectories)

    event_stats = write_jsonl(out_dir / "event.jsonl", events)
    traj_stats = write_jsonl(out_dir / "traj.jsonl", trajectories)
    pair_stats = write_jsonl(out_dir / "pairs.jsonl", pairs)

    sft_samples = build_sft_samples(events)
    rl_prompts = build_rl_prompts(trajectories)

    sft_stats = write_json(out_dir / "maven_sft.jsonl", sft_samples)
    rl_stats = write_json(out_dir / "rl_prompts.jsonl", rl_prompts)

    summary = {
        "event": json.loads((out_dir / "event.jsonl.stats.json").read_text("utf-8")),
        "traj": json.loads((out_dir / "traj.jsonl.stats.json").read_text("utf-8")),
        "pairs": json.loads((out_dir / "pairs.jsonl.stats.json").read_text("utf-8")),
        "sft": json.loads((out_dir / "maven_sft.jsonl.stats.json").read_text("utf-8")),
        "rl": json.loads((out_dir / "rl_prompts.jsonl.stats.json").read_text("utf-8")),
    }
    (out_dir / "summary.stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Generated datasets:")
    for stats in [event_stats, traj_stats, pair_stats, sft_stats, rl_stats]:
        print(f"- {stats.name}: {stats.num_records} records")


if __name__ == "__main__":
    main()
