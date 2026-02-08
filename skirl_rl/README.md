# SKIRL-RL 最小演示

本目录提供一个基于 LlamaFactory 的 **SKIRL-RL**（监督预训练 → 最大熵 IRL 奖励 → PPO/GRPO 策略微调 → 轨迹打分与 Top-K 排序）最小可运行流程。所有步骤均可在无网络环境下运行，支持 MAVEN-ERE 与 RAMS 数据集，数据转换脚本分别为 `skirl_rl/scripts/0_convert_maven_to_event_traj.py` 与 `skirl_rl/scripts/0_convert_rams_to_event_traj.py`。

## 目录结构

```text
skirl_rl/
├── irl/
│   ├── features.py            # 轨迹特征构建与单元测试
│   ├── tgn_encoder.py         # 极简 TGN 时序图编码器
│   └── maxent_irl.py          # 最大熵 IRL 训练与推理接口
├── policy/
│   ├── build_prompt.py        # RL/推理阶段 prompt 生成
│   ├── rl_trainer.py          # 无官方 PPO 时的启发式替代器
│   └── score_policy.py        # 轨迹打分、排序与评估
├── scripts/
│   ├── 0_convert_maven_to_event_traj.py
│   ├── 0_convert_rams_to_event_traj.py
│   ├── 1_pretrain_qwen_maven.sh
│   ├── 2_train_reward_maxent.sh
│   ├── 3_train_policy_rl.sh
│   └── 4_score_and_rank.sh
└── README.md
```

## 运行步骤

1. **数据构造**

   ```bash
   python skirl_rl/scripts/0_convert_maven_to_event_traj.py
   ```

   RAMS 数据集则运行：

   ```bash
   python skirl_rl/scripts/0_convert_rams_to_event_traj.py
   ```

   生成的文件位于 `data/processed/`，MAVEN 对应 `event.jsonl`、`traj.jsonl`、`pairs.jsonl`、`maven_sft.jsonl`、`rl_prompts.jsonl`；RAMS 对应 `rams_event.jsonl`、`rams_traj.jsonl`、`rams_pairs.jsonl`、`rams_sft.jsonl`、`rams_rl_prompts.jsonl`。

2. **监督预训练（SFT）**

   ```bash
   bash skirl_rl/scripts/1_pretrain_qwen_maven.sh
   ```

   训练脚本会读取 `skirl_rl/config.yaml` 的 `data.mode` 来自动选择 `configs/pretrain_*.yaml` 与输出目录。

3. **最大熵 IRL 奖励模型**

   ```bash
   bash skirl_rl/scripts/2_train_reward_maxent.sh
   ```

   基于 `traj.jsonl`/`pairs.jsonl`（或 RAMS 对应文件）学习奖励，保存至 `data.mode` 对应的奖励目录。

4. **策略 RL（PPO/GRPO 或离线启发式）**

   ```bash
   bash skirl_rl/scripts/3_train_policy_rl.sh
   ```

   若环境已启用 LlamaFactory PPO/GRPO，将调用官方 CLI；否则使用 `policy/rl_trainer.py` 进行启发式估计，并写出 `policy_logprobs.json`。

5. **打分与 Top-K 排序**

   ```bash
   bash skirl_rl/scripts/4_score_and_rank.sh --k 20
   ```

   输出 `data.mode` 对应的 Top-K 结果与 `reasons.jsonl`，可选 `--eval` 计算 `NDCG@K/MAP@K/Hit@K`。

6. **SKEIN 表格评估（Top-10 指标）**

   ```bash
   python skirl_rl/evaluate_skein.py
   ```

   默认读取 `skirl_rl/config.yaml` 的 `run` 配置，按数据集与 profile 组合输出 `evaluation_*.json`、`topk_*.json` 与 `reasons_*.jsonl`。
   若本地缺少轨迹或奖励模型，会自动生成一条合成样本用于跑通流程（可在 `run.auto_sample` 中控制）。

## 注意事项

- `src/llamafactory/plugins/reward_callbacks/skirl_maxent.py` 提供 PPO/GRPO 外部奖励回调。
- `requirements.txt` 已补充所需依赖（`numpy`、`scipy`、`networkx`、`pydantic`、`einops`）。
- 所有脚本会在目标目录生成统计文件，便于快速验证数据与模型维度。
- 可通过 `python skirl_rl/irl/features.py`、`python skirl_rl/irl/tgn_encoder.py`、`python skirl_rl/irl/maxent_irl.py --help` 进行最小单元校验。
