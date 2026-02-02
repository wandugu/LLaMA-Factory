#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT="${WANDB_PROJECT:-maven-irl}"
export WANDB_MODE="${WANDB_MODE:-online}"              # offline/online
# 可选：团队与标签
# export WANDB_ENTITY="your_team"
export WANDB_TAGS="${WANDB_TAGS:-sft,qwen3-4b,maven}"

# 屏蔽 transformers 的冗余 WARNING，保持训练日志整洁
export TRANSFORMERS_VERBOSITY="error"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="1"

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG_PATH="${ROOT_DIR}/configs/ppo_rl.yaml"
PROMPTS_PATH="${ROOT_DIR}/data/processed/rl_prompts.jsonl"
REWARD_CKPT="/root/autodl-tmp/qwen-4b-maven-rm/reward.ckpt"
OUTPUT_DIR="/root/autodl-tmp/qwen-4b-rl"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
readarray -t __SKIRL_CONFIG_INFO < <(python - <<'PY' "${CONFIG_PATH}" "${ROOT_DIR}"
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
root = Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

model_path = config.get("model_name_or_path", "")
if model_path:
    path = Path(model_path)
    if not path.is_absolute():
        path = (root / path).resolve()
    print(path)
else:
    print("")

print(config.get("run_name", ""))
PY
)

MODEL_PATH="${__SKIRL_CONFIG_INFO[0]}"
CONFIG_RUN_NAME="${__SKIRL_CONFIG_INFO[1]}"
unset __SKIRL_CONFIG_INFO

if [ -n "${CONFIG_RUN_NAME}" ] && [ -z "${WANDB_NAME:-}" ]; then
  export WANDB_NAME="${CONFIG_RUN_NAME}"
fi



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

has_llamafactory() {
  if command -v llamafactory-cli >/dev/null 2>&1; then
    return 0
  fi

  python - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("llamafactory") is not None else 1)
PY
}

USE_OFFICIAL="${SKIRL_USE_OFFICIAL_PPO:-auto}"
SHOULD_USE_OFFICIAL=0

case "${USE_OFFICIAL}" in
  1|true|TRUE)
    SHOULD_USE_OFFICIAL=1
    ;;
  0|false|FALSE)
    SHOULD_USE_OFFICIAL=0
    ;;
  auto)
    if has_llamafactory; then
      SHOULD_USE_OFFICIAL=1
    fi
    ;;
  *)
    echo "[WARN] 未识别的 SKIRL_USE_OFFICIAL_PPO=${USE_OFFICIAL}，回退为 auto 检测"
    if has_llamafactory; then
      SHOULD_USE_OFFICIAL=1
    fi
    ;;
esac

if [ "${SHOULD_USE_OFFICIAL}" -eq 1 ]; then
  echo "[INFO] 检测到 LlamaFactory PPO，直接执行官方 CLI"
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
