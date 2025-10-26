# -*- coding: utf-8 -*-
"""SKIRL 最大熵奖励回调，用于 PPO/GRPO 外部打分。"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from skirl_rl.irl.maxent_irl import MaxEntIRL

LOGGER = logging.getLogger("skirl_maxent")


class SkirlMaxentRewardCallback:
    """结合最大熵奖励与策略 logprob 的奖励回调。"""

    def __init__(
        self,
        reward_ckpt: Path,
        trajectory_path: Optional[Path] = None,
        alpha: float = 0.6,
        reward_clip: Optional[Tuple[float, float]] = None,
        normalizer: str = "standard",
    ) -> None:
        self.model = MaxEntIRL()
        self.model.load(Path(reward_ckpt))
        traj_path = Path(trajectory_path) if trajectory_path is not None else Path("data/processed/traj.jsonl")
        self.trajectories = self._load_trajectories(traj_path)
        self.alpha = alpha
        self.reward_clip = reward_clip
        self.normalizer = normalizer

    @staticmethod
    def _load_trajectories(path: Path) -> Dict[str, Dict]:
        if not path.exists():
            LOGGER.warning("trajectory file %s not found; reward回调将仅使用 prompt 信息", path)
            return {}
        data: Dict[str, Dict] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                payload = json.loads(line)
                data[payload.get("trajectory_id", f"unknown-{len(data)}")] = payload
        return data

    def _norm(self, values: List[float]) -> List[float]:
        if not values:
            return []
        arr = np.asarray(values, dtype=np.float32)
        if self.normalizer == "minmax":
            vmin, vmax = arr.min(), arr.max()
            if np.isclose(vmax, vmin):
                return [0.0] * len(values)
            return ((arr - vmin) / (vmax - vmin) * 2 - 1).tolist()
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-6:
            return [0.0] * len(values)
        return ((arr - mean) / std).tolist()

    def _clip(self, values: List[float]) -> List[float]:
        if not self.reward_clip:
            return values
        lo, hi = self.reward_clip
        return [float(np.clip(v, lo, hi)) for v in values]

    @staticmethod
    def _summarize_text(text: Optional[str], limit: int = 120) -> str:
        if not text:
            return "<empty>"
        text = str(text).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    @staticmethod
    def _resolve_trajectory_id(meta: Optional[Dict]) -> Optional[str]:
        if not isinstance(meta, dict):
            return None

        for key in ("trajectory_id", "trajectoryId", "trajectory", "traj_id", "trajId"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        prompt_like = (
            meta.get("prompt")
            or meta.get("raw_prompt")
            or meta.get("raw_query")
            or meta.get("input")
        )
        if isinstance(prompt_like, str):
            match = re.search(r"\[TRAJ\]\s*([^\s]+)", prompt_like)
            if match:
                return match.group(1)

        history = meta.get("history")
        if isinstance(history, list):
            for item in history:
                if isinstance(item, str):
                    match = re.search(r"\[TRAJ\]\s*([^\s]+)", item)
                    if match:
                        return match.group(1)

        return None

    def score_batch(
        self,
        metas: Sequence[Dict],
        logprobs: Optional[Sequence[float]] = None,
    ) -> List[float]:
        rewards: List[float] = []
        for meta in metas:
            trajectory_id = self._resolve_trajectory_id(meta)
            trajectory = self.trajectories.get(trajectory_id)
            if trajectory is None:
                meta_keys = sorted(meta.keys()) if isinstance(meta, dict) else "<non-dict>"
                prompt_preview = self._summarize_text(
                    meta.get("raw_prompt") if isinstance(meta, dict) else None
                )
                response_preview = self._summarize_text(
                    meta.get("reference_response") if isinstance(meta, dict) else None
                )
                LOGGER.warning(
                    "trajectory %s 未在缓存中找到，回退为零奖励 (meta keys=%s, prompt≈%s, response≈%s)",
                    trajectory_id,
                    meta_keys,
                    prompt_preview,
                    response_preview,
                )
                rewards.append(0.0)
                continue
            try:
                reward = self.model.score(trajectory)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("评分 trajectory=%s 失败: %s", trajectory_id, exc)
                reward = 0.0
            rewards.append(reward)

        rewards = self._clip(self._norm(rewards))
        if logprobs is None:
            return rewards

        norm_logprobs = self._norm(list(logprobs))
        combined = []
        for r, l in zip(rewards, norm_logprobs):
            combined.append(self.alpha * r + (1 - self.alpha) * l)
        combined = self._clip(combined)
        return combined

    def __call__(
        self,
        sequences: Sequence[Sequence[int]],
        response_scores: Optional[Sequence[float]] = None,
        metas: Optional[Sequence[Dict]] = None,
        logprobs: Optional[Sequence[float]] = None,
    ) -> List[float]:
        """计算奖励。"""

        metas = metas or [{} for _ in sequences]
        try:
            return self.score_batch(metas, logprobs=logprobs)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("SKIRL 回调失败，返回零奖励: %s", exc)
            return [0.0 for _ in sequences]


def build_reward_callback(**kwargs) -> SkirlMaxentRewardCallback:
    return SkirlMaxentRewardCallback(**kwargs)


__all__ = ["SkirlMaxentRewardCallback", "build_reward_callback"]
