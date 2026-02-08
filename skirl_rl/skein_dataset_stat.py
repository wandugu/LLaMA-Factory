# -*- coding: utf-8 -*-
"""Compute detailed statistics for converted SKEIN datasets."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from skirl_rl.config_utils import resolve_processed_files

LOGGER = logging.getLogger(__name__)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("config yaml must be a mapping")
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                LOGGER.debug("Skip non-dict record at %s:%d", path, idx)
                continue
            records.append(payload)
    LOGGER.debug("Loaded %d records from %s", len(records), path)
    return records


def _length_stats(values: Iterable[int]) -> Dict[str, float]:
    values_list = [v for v in values if v is not None]
    if not values_list:
        return {}
    return {
        "min": float(min(values_list)),
        "max": float(max(values_list)),
        "avg": float(mean(values_list)),
        "median": float(median(values_list)),
    }


def _safe_parse_date(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None


def _trajectory_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    label_counter: Counter[str] = Counter()
    step_lengths: List[int] = []
    role_types: Counter[str] = Counter()
    skeleton_hits: Counter[str] = Counter()
    person_ids: set[str] = set()
    traj_ids: set[str] = set()
    graph_nodes: List[int] = []
    graph_edges: List[int] = []
    dates: List[datetime] = []

    for traj in records:
        person_id = traj.get("person_id")
        trajectory_id = traj.get("trajectory_id")
        if person_id:
            person_ids.add(str(person_id))
        if trajectory_id:
            traj_ids.add(str(trajectory_id))
        label = traj.get("label")
        if label:
            label_counter[str(label)] += 1
        steps = traj.get("steps") or []
        if isinstance(steps, list):
            step_lengths.append(len(steps))
            for step in steps:
                if not isinstance(step, dict):
                    continue
                roles = step.get("roles") or {}
                if isinstance(roles, dict):
                    for role_name in roles.keys():
                        role_types[str(role_name)] += 1
                skeletons = step.get("skeleton_hits") or []
                if isinstance(skeletons, list):
                    for item in skeletons:
                        skeleton_hits[str(item)] += 1
                time_value = step.get("time")
                if isinstance(time_value, str):
                    parsed = _safe_parse_date(time_value)
                    if parsed:
                        dates.append(parsed)
        meta = traj.get("meta") or {}
        if isinstance(meta, dict):
            nodes = meta.get("graph_nodes") or []
            edges = meta.get("graph_edges") or []
            if isinstance(nodes, list):
                graph_nodes.append(len(nodes))
            if isinstance(edges, list):
                graph_edges.append(len(edges))

    time_range: Dict[str, str] = {}
    if dates:
        time_range = {
            "start": min(dates).strftime("%Y-%m-%d"),
            "end": max(dates).strftime("%Y-%m-%d"),
        }

    return {
        "num_records": len(records),
        "unique_persons": len(person_ids),
        "unique_trajectories": len(traj_ids),
        "label_distribution": dict(label_counter),
        "step_length": _length_stats(step_lengths),
        "graph_nodes": _length_stats(graph_nodes),
        "graph_edges": _length_stats(graph_edges),
        "role_type_counts": dict(role_types.most_common()),
        "skeleton_hit_counts": dict(skeleton_hits.most_common(20)),
        "time_range": time_range,
    }


def _pairs_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    traj_ids: set[str] = set()
    reason_lengths: List[int] = []
    for item in records:
        better = item.get("better")
        worse = item.get("worse")
        if better:
            traj_ids.add(str(better))
        if worse:
            traj_ids.add(str(worse))
        reason = item.get("reason")
        if isinstance(reason, str):
            reason_lengths.append(len(reason))
    return {
        "num_records": len(records),
        "unique_trajectory_ids": len(traj_ids),
        "reason_length": _length_stats(reason_lengths),
    }


def _event_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    doc_ids: set[str] = set()
    event_types: Counter[str] = Counter()
    trigger_texts: Counter[str] = Counter()
    for item in records:
        doc_id = item.get("doc_id")
        if doc_id:
            doc_ids.add(str(doc_id))
        trigger = item.get("trigger") or {}
        if isinstance(trigger, dict):
            event_type = trigger.get("type")
            if event_type:
                event_types[str(event_type)] += 1
            text = trigger.get("text")
            if text:
                trigger_texts[str(text)] += 1
    return {
        "num_records": len(records),
        "unique_docs": len(doc_ids),
        "event_type_distribution": dict(event_types.most_common(20)),
        "trigger_text_distribution": dict(trigger_texts.most_common(20)),
    }


def _sft_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    instruction_lengths: List[int] = []
    input_lengths: List[int] = []
    output_lengths: List[int] = []
    system_lengths: List[int] = []
    history_lengths: List[int] = []

    for item in records:
        instruction = item.get("instruction")
        if isinstance(instruction, str):
            instruction_lengths.append(len(instruction))
        input_text = item.get("input")
        if isinstance(input_text, str):
            input_lengths.append(len(input_text))
        output_text = item.get("output")
        if isinstance(output_text, str):
            output_lengths.append(len(output_text))
        system_text = item.get("system")
        if isinstance(system_text, str):
            system_lengths.append(len(system_text))
        history = item.get("history")
        if isinstance(history, list):
            history_lengths.append(len(history))

    return {
        "num_records": len(records),
        "instruction_length": _length_stats(instruction_lengths),
        "input_length": _length_stats(input_lengths),
        "output_length": _length_stats(output_lengths),
        "system_length": _length_stats(system_lengths),
        "history_turns": _length_stats(history_lengths),
    }


def _rl_prompt_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_lengths: List[int] = []
    response_lengths: List[int] = []
    traj_ids: set[str] = set()
    person_ids: set[str] = set()

    for item in records:
        prompt = item.get("prompt")
        if isinstance(prompt, str):
            prompt_lengths.append(len(prompt))
        response = item.get("response")
        if isinstance(response, str):
            response_lengths.append(len(response))
        trajectory_id = item.get("trajectory_id") or item.get("_meta", {}).get("trajectory_id")
        if trajectory_id:
            traj_ids.add(str(trajectory_id))
        person_id = item.get("person_id") or item.get("_meta", {}).get("person_id")
        if person_id:
            person_ids.add(str(person_id))

    return {
        "num_records": len(records),
        "unique_trajectories": len(traj_ids),
        "unique_persons": len(person_ids),
        "prompt_length": _length_stats(prompt_lengths),
        "response_length": _length_stats(response_lengths),
    }


def _build_sample_trajectory() -> Dict[str, Any]:
    return {
        "person_id": "P_SAMPLE",
        "trajectory_id": "traj_sample_1",
        "label": "expert",
        "steps": [
            {
                "event_id": "event_sample_1",
                "type": "EXECUTE",
                "skeleton_hits": ["PREP", "EXECUTE"],
                "delta_days_from_prev": 0,
                "roles": {"Agent": "A_SAMPLE", "Target": "T_SAMPLE"},
                "time": "2014-03-21",
                "text_refs": [{"doc_id": "doc1", "span": "0-12"}],
            }
        ],
        "meta": {
            "graph_nodes": ["A_SAMPLE", "T_SAMPLE"],
            "graph_edges": [["A_SAMPLE", "EXECUTE", "T_SAMPLE", "2014-03-21"]],
        },
    }


def _build_sample_event() -> Dict[str, Any]:
    return {
        "doc_id": "doc1",
        "event_id": "event_sample_1",
        "trigger": {"text": "sample", "span": [0, 6], "type": "EXECUTE"},
        "arguments": [
            {"role": "Agent", "text": "A_SAMPLE", "span": [0, 3], "entity_id": "ent1"},
            {"role": "Target", "text": "T_SAMPLE", "span": [4, 7], "entity_id": "ent2"},
        ],
        "time": {"text": "2014-03-21", "span": [0, 10]},
        "relations": {"previous": [], "next": []},
        "confidence": {"trigger_prob": 0.9, "arg_role_avg": 0.8},
        "source": "SAMPLE",
        "mapping": {"event": "EXECUTE"},
        "split": "train",
    }


def _build_sample_rl_prompt() -> Dict[str, Any]:
    return {
        "prompt": "样例 prompt: 请描述该事件轨迹。",
        "response": "EXECUTE | skeleton=PREP->EXECUTE",
        "trajectory_id": "traj_sample_1",
        "person_id": "P_SAMPLE",
    }


def _build_sample_sft() -> Dict[str, Any]:
    return {
        "instruction": "请总结事件轨迹。",
        "input": "样例输入。",
        "output": "样例输出。",
        "system": "你是事件分析助手。",
        "history": [],
    }


def _build_sample_pair() -> Dict[str, Any]:
    return {
        "better": "traj_sample_1",
        "worse": "traj_sample_1",
        "reason": "样例偏好对用于占位统计。",
    }


def _ensure_sample_dataset(root: Path, cfg: Dict[str, Any], processed_files: Dict[str, str]) -> Dict[str, Path]:
    sample_dir = _resolve_path(root, cfg["sample_dir"])
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = {
        key: sample_dir / filename for key, filename in processed_files.items()
    }

    if sample_paths.get("traj") and not sample_paths["traj"].exists():
        sample_paths["traj"].write_text(
            json.dumps(_build_sample_trajectory(), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        LOGGER.debug("Sample trajectory written: %s", sample_paths["traj"])

    if sample_paths.get("event") and not sample_paths["event"].exists():
        sample_paths["event"].write_text(
            json.dumps(_build_sample_event(), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        LOGGER.debug("Sample event written: %s", sample_paths["event"])

    if sample_paths.get("pairs") and not sample_paths["pairs"].exists():
        sample_paths["pairs"].write_text(
            json.dumps(_build_sample_pair(), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        LOGGER.debug("Sample pairs written: %s", sample_paths["pairs"])

    if sample_paths.get("sft") and not sample_paths["sft"].exists():
        sample_paths["sft"].write_text(
            json.dumps(_build_sample_sft(), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        LOGGER.debug("Sample sft written: %s", sample_paths["sft"])

    if sample_paths.get("rl_prompts") and not sample_paths["rl_prompts"].exists():
        sample_paths["rl_prompts"].write_text(
            json.dumps(_build_sample_rl_prompt(), ensure_ascii=False) + "\n", encoding="utf-8"
        )
        LOGGER.debug("Sample rl prompts written: %s", sample_paths["rl_prompts"])

    return sample_paths


def _collect_examples(
    available: Dict[str, List[Dict[str, Any]]],
    ordered_files: List[str],
    max_examples: int,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for name in ordered_files:
        items = available.get(name) or []
        if not items:
            continue
        for item in items[: max_examples - len(examples)]:
            examples.append({"source": name, "data": item})
            if len(examples) >= max_examples:
                break
        if len(examples) >= max_examples:
            break
    return examples


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统计转换后的 SKEIN 数据集")
    parser.add_argument("--config", type=Path, default=Path("skirl_rl/config.yaml"))
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--auto-sample", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
    args = build_argparser().parse_args()
    root = Path(__file__).resolve().parents[1]
    config = _load_config(_resolve_path(root, args.config))

    stats_cfg = config.get("stats", {}) if isinstance(config.get("stats", {}), dict) else {}
    if not stats_cfg:
        raise ValueError("missing stats config in skirl_rl/config.yaml")

    dataset_dir = _resolve_path(root, stats_cfg["dataset_dir"])
    if args.dataset_dir is not None:
        dataset_dir = _resolve_path(root, args.dataset_dir)

    output_path = _resolve_path(root, stats_cfg["output_path"])
    if args.output is not None:
        output_path = _resolve_path(root, args.output)

    processed_files = resolve_processed_files(config)
    file_keys = stats_cfg.get("dataset_file_keys", ["traj", "rl_prompts", "sft", "event", "pairs"])
    dataset_files = [processed_files[key] for key in file_keys if key in processed_files]
    if not dataset_files:
        raise ValueError("stats.dataset_file_keys is empty or invalid")

    auto_sample = bool(stats_cfg.get("auto_sample", False)) or args.auto_sample

    available_records: Dict[str, List[Dict[str, Any]]] = {}
    missing_files: List[str] = []
    for name in dataset_files:
        path = dataset_dir / name
        if path.exists():
            available_records[name] = _read_jsonl(path)
        else:
            missing_files.append(name)

    sample_paths: Dict[str, Path] | None = None
    if missing_files and auto_sample:
        LOGGER.debug("Missing files %s, generating sample dataset.", missing_files)
        sample_paths = _ensure_sample_dataset(root, stats_cfg, processed_files)
        for name in missing_files:
            sample_key = next((key for key, filename in processed_files.items() if filename == name), None)
            if sample_key:
                sample_path = sample_paths.get(sample_key)
                if sample_path and sample_path.exists():
                    available_records[name] = _read_jsonl(sample_path)

    summary: Dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "used_sample": bool(sample_paths),
        "files": {},
    }

    for name, handler in (
        (processed_files.get("traj"), _trajectory_stats),
        (processed_files.get("pairs"), _pairs_stats),
        (processed_files.get("event"), _event_stats),
        (processed_files.get("sft"), _sft_stats),
        (processed_files.get("rl_prompts"), _rl_prompt_stats),
    ):
        if name and name in available_records:
            summary["files"][name] = handler(available_records[name])

    max_examples = int(stats_cfg.get("max_examples", 2))
    summary["examples"] = _collect_examples(available_records, dataset_files, max_examples=max_examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("统计完成，输出至 %s", output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
