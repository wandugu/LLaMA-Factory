#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONFIG_PATH="${ROOT_DIR}/configs/pretrain_maven.yaml"

echo "$CONFIG_PATH"; test -f "$CONFIG_PATH" && head -n 5 "$CONFIG_PATH"


cd "${ROOT_DIR}"

if [ ! -f "data/processed/maven_sft.jsonl" ]; then
  echo "[INFO] 数据未找到，自动生成演示样本。"
  python skirl_rl/scripts/0_convert_maven_to_event_traj.py
fi

if [ ! -f "requirements.txt" ]; then
  echo "[ERROR] requirements.txt 不存在"
  exit 1
fi

# python -m pip install --upgrade pip >/dev/null
# python -m pip install -r requirements.txt

echo "[INFO] 启动 LlamaFactory 预训练"
llamafactory-cli train "${CONFIG_PATH}"

