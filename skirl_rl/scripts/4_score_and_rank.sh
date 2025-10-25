#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUTPUT_DIR="${ROOT_DIR}/outputs/qwen-4b-rl"
REWARD_CKPT="${ROOT_DIR}/outputs/qwen-4b-rm/reward.ckpt"
TRAJ_PATH="${ROOT_DIR}/data/processed/traj.jsonl"
POLICY_LOGPROB="${OUTPUT_DIR}/policy_logprobs.json"

cd "${ROOT_DIR}"

if [ ! -f "${REWARD_CKPT}" ]; then
  echo "[ERROR] 奖励模型缺失，请先运行 2_train_reward_maxent.sh"
  exit 1
fi

if [ ! -f "${POLICY_LOGPROB}" ]; then
  echo "[WARN] 未找到策略 logprob，启用启发式得分"
fi

python skirl_rl/policy/score_policy.py \
  --traj-path "${TRAJ_PATH}" \
  --reward-ckpt "${REWARD_CKPT}" \
  --policy-logprobs "${POLICY_LOGPROB}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
