#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TRAJ_PATH="${ROOT_DIR}/data/processed/traj.jsonl"
PAIR_PATH="${ROOT_DIR}/data/processed/pairs.jsonl"
OUTPUT_DIR="${ROOT_DIR}/outputs/qwen-4b-rm"
OUTPUT_PATH="${OUTPUT_DIR}/reward.ckpt"

cd "${ROOT_DIR}"

if [ ! -f "${TRAJ_PATH}" ]; then
  echo "[INFO] 轨迹文件缺失，先生成演示数据"
fi

mkdir -p "${OUTPUT_DIR}"

python skirl_rl/irl/maxent_irl.py \
  --traj-path "${TRAJ_PATH}" \
  --pair-path "${PAIR_PATH}" \
  --output "${OUTPUT_PATH}" \
  --epochs 150 \
  --lr 0.05
