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
REWARD_CKPT="${ROOT_DIR}/outputs/qwen-4b-rm/reward.ckpt"
OUTPUT_DIR="${ROOT_DIR}/outputs/qwen-4b-rl"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
MODEL_PATH=$(python - <<'PY' "${CONFIG_PATH}" "${ROOT_DIR}"
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
root = Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
model_path = config.get("model_name_or_path", "")
if not model_path:
    print("")
else:
    path = Path(model_path)
    if not path.is_absolute():
        path = (root / path).resolve()
    print(path)
PY
)



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
  if [ -z "${MODEL_PATH}" ]; then
    echo "[ERROR] 配置 ${CONFIG_PATH} 未设置 model_name_or_path"
    exit 1
  fi

  if [ ! -d "${MODEL_PATH}" ] && [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[ERROR] 未找到模型目录 ${MODEL_PATH}，请先完成 1_pretrain_qwen_maven.sh 或更新配置路径"
    exit 1
  fi

  if command -v llamafactory-cli >/dev/null 2>&1; then
    llamafactory-cli train "${CONFIG_PATH}"
  else
    python -m llamafactory.cli train "${CONFIG_PATH}"
  fi
else
  echo "[WARN] 未检测到 LlamaFactory PPO 支持，使用启发式策略训练"
  python skirl_rl/policy/rl_trainer.py \
    --reward-ckpt "${REWARD_CKPT}" \
    --prompts "${PROMPTS_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --alpha 0.6
fi
