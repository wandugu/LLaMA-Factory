#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT="maven-irl"
export WANDB_MODE="online"              # offline/online
# 可选：团队与标签
# export WANDB_ENTITY="your_team"
export WANDB_TAGS="sft,qwen3-4b,maven"

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG_PATH="${ROOT_DIR}/configs/ppo_rl.yaml"
PROMPTS_PATH="${ROOT_DIR}/data/processed/rl_prompts.jsonl"
REWARD_CKPT="/root/autodl-tmp/qwen-4b-maven-rm/reward.ckpt"
OUTPUT_DIR="${ROOT_DIR}/outputs/qwen-4b-rl"

cd "${ROOT_DIR}"

if [ ! -f "${PROMPTS_PATH}" ]; then
  echo "[INFO] RL 提示缺失，先生成演示数据"
  exit 1
fi

if [ ! -f "${REWARD_CKPT}" ]; then
  echo "[ERROR] 奖励模型未训练，请先运行 2_train_reward_maxent.sh"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [ "${SKIRL_USE_OFFICIAL_PPO:-0}" -eq 1 ]; then
  echo "[INFO] SKIRL_USE_OFFICIAL_PPO=1，尝试执行 LlamaFactory PPO"
  python -m src.train --config "${CONFIG_PATH}"
else
  echo "[WARN] 未检测到 LlamaFactory PPO 支持，使用启发式策略训练"
  python skirl_rl/policy/rl_trainer.py \
    --reward-ckpt "${REWARD_CKPT}" \
    --prompts "${PROMPTS_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --alpha 0.6
fi
