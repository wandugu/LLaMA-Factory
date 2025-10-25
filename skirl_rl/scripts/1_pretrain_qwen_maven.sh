#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT="maven-irl"
export WANDB_MODE="online"              # offline/online
# 可选：团队与标签
# export WANDB_ENTITY="your_team"
# export WANDB_TAGS="sft,qwen3-4b,maven"
# 可选：日志目录
# export WANDB_DIR="$(pwd)/wandb"
# 若之前离线过，这条会切回在线；失败不致命
wandb online >/dev/null 2>&1 || true

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG_PATH="${ROOT_DIR}/configs/pretrain_maven.yaml"

echo "$CONFIG_PATH"; test -f "$CONFIG_PATH"


cd "${ROOT_DIR}"

# if [ ! -f "data/processed/maven_sft_train.jsonl" ]; then
  # echo "[INFO] 数据未找到，自动生成演示样本。"
  # python skirl_rl/scripts/0_convert_maven_to_event_traj.py
# fi

if [ ! -f "requirements.txt" ]; then
  echo "[ERROR] requirements.txt 不存在"
  exit 1
fi

# python -m pip install --upgrade pip >/dev/null
# python -m pip install -r requirements.txt

echo "[INFO] 启动 maven sft"
llamafactory-cli train "${CONFIG_PATH}"

