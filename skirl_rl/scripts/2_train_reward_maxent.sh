#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODE_CONFIG="${ROOT_DIR}/skirl_rl/config.yaml"
eval "$(python "${ROOT_DIR}/skirl_rl/scripts/resolve_mode_env.py" --config "${MODE_CONFIG}")"

TRAJ_PATH="${ROOT_DIR}/${SKIRL_PROCESSED_DIR}/${SKIRL_TRAJ_FILE}"
PAIR_PATH="${ROOT_DIR}/${SKIRL_PROCESSED_DIR}/${SKIRL_PAIRS_FILE}"
OUTPUT_DIR="${SKIRL_REWARD_OUTPUT_DIR}"
OUTPUT_PATH="${SKIRL_REWARD_CKPT}"

cd "${ROOT_DIR}"

if [ ! -f "${TRAJ_PATH}" ]; then
  echo "[INFO] 轨迹文件缺失，先生成演示数据"
fi

echo "[INFO] 当前模式：${SKIRL_MODE}"

mkdir -p "${OUTPUT_DIR}"

python skirl_rl/irl/maxent_irl.py \
  --traj-path "${TRAJ_PATH}" \
  --pair-path "${PAIR_PATH}" \
  --output "${OUTPUT_PATH}" \
  --epochs 150 \
  --lr 0.05
