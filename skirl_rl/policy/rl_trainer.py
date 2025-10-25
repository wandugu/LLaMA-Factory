# -*- coding: utf-8 -*-
"""Fallback RL trainer for environments without PPO/GRPO support.

This module simulates a PPO loop using heuristic updates so that the demo can run
fully offline.  When LlamaFactory provides PPO/GRPO, prefer invoking the official
CLI with ``configs/ppo_rl.yaml`` instead of this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from irl.maxent_irl import MaxEntIRL  # type: ignore
else:
    from ..irl.maxent_irl import MaxEntIRL


class HeuristicRLTrainer:
    def __init__(self, reward_ckpt: Path, prompts_path: Path, output_dir: Path, alpha: float = 0.6) -> None:
        self.reward_model = MaxEntIRL()
        self.reward_model.load(reward_ckpt)
        self.prompts_path = prompts_path
        self.output_dir = output_dir
        self.alpha = alpha

    def load_prompts(self) -> List[Dict]:
        prompts: List[Dict] = []
        with self.prompts_path.open("r", encoding="utf-8") as f:
            for line in f:
                prompts.append(json.loads(line))
        if not prompts:
            raise ValueError("no RL prompts found")
        return prompts

    def simulate_policy(self, prompt: Dict) -> Dict[str, float]:
        # Score the target response using reward features as a proxy
        traj_stub = {
            "trajectory_id": prompt.get("trajectory_id"),
            "steps": [
                {
                    "skeleton_hits": [prompt.get("response", "EXECUTE")],
                    "delta_days_from_prev": 0,
                    "roles": {"Agent": "P_SIM"},
                    "time": "2014-03-21",
                }
            ],
            "meta": {"graph_nodes": ["P_SIM"], "graph_edges": []},
        }
        reward = self.reward_model.score(traj_stub)
        logprob = np.tanh(reward)
        return {"reward": reward, "logprob": float(logprob)}

    def train(self) -> None:
        prompts = self.load_prompts()
        stats = []
        policy_logprobs = {}
        for prompt in prompts:
            simulation = self.simulate_policy(prompt)
            combined = self.alpha * simulation["reward"] + (1 - self.alpha) * simulation["logprob"]
            policy_logprobs[prompt["trajectory_id"]] = combined
            stats.append(combined)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "policy_logprobs.json").write_text(json.dumps(policy_logprobs, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {"mean_logprob": float(np.mean(stats)), "num_prompts": len(stats)}
        (self.output_dir / "heuristic_rl.stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heuristic PPO trainer (offline demo)")
    parser.add_argument("--reward-ckpt", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.6)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    trainer = HeuristicRLTrainer(args.reward_ckpt, args.prompts, args.output_dir, alpha=args.alpha)
    trainer.train()


if __name__ == "__main__":
    main()
