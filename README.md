# LLaMA-Factory AGPO 训练指南

本仓库实现了在 LLaMA-Factory 框架下的 **AGPO (Adaptive Grouped Policy Optimization)** 强化学习训练流程，用于在奖励模型或外部打分器的指导下对大语言模型进行对齐。本说明文件针对当前代码中的 AGPO 实现进行解读，帮助你快速理解组件、配置项以及训练步骤。

## 核心特性概览

- **基于 PPOTrainer 的自定义训练器**：`CustomAGPOTrainer` 在 Hugging Face TRL 的 `PPOTrainer` 之上扩展，整合了自定义优化器、回调与生成配置，专门用于分组采样和自适应剪切控制的 AGPO 训练。【F:src/llamafactory/train/agpo/trainer.py†L63-L163】
- **探测-训练双阶段采样**：每个提示首先使用基准温度进行探测采样以估计不确定性，再根据不确定性调整温度进行正式训练采样，从而提高样本质量与多样性。【F:src/llamafactory/train/agpo/trainer.py†L467-L507】
- **自适应优势归一化与剪切阈值**：利用可选的稳健离散度指标（标准差/MAD/IQR）归一化奖励，并根据奖励方差与 KL 距离动态调整 PPO 剪切范围，兼顾稳定性与探索性。【F:src/llamafactory/train/agpo/trainer.py†L508-L544】【F:src/llamafactory/hparams/finetuning_args.py†L223-L297】
- **多源奖励与日志统计**：支持本地 reward head、LoRA/Full/API 奖励模型，记录 `agpo/*` 指标（温度、方差、KL、奖励等），便于监控训练健康度。【F:src/llamafactory/train/agpo/trainer.py†L611-L620】【F:src/llamafactory/train/agpo/trainer.py†L538-L563】【F:src/llamafactory/train/agpo/trainer.py†L412-L439】

## 目录结构

| 模块 | 作用 |
| --- | --- |
| `src/llamafactory/train/agpo/workflow.py` | 组装数据集、模型、奖励模型并启动 AGPO 训练主流程。 |【F:src/llamafactory/train/agpo/workflow.py†L31-L74】
| `src/llamafactory/train/agpo/trainer.py` | 定义 AGPO 训练器、采样策略、损失计算与日志逻辑。 |【F:src/llamafactory/train/agpo/trainer.py†L63-L569】
| `src/llamafactory/hparams/finetuning_args.py` | 声明可通过命令行配置的 AGPO 超参数及其默认值。 |【F:src/llamafactory/hparams/finetuning_args.py†L223-L297】
| `src/train.py` | CLI 入口，调用 `run_exp` 进而在 `rl_algo=agpo` 时使用 AGPO 流程。 |【F:src/train.py†L15-L24】【F:src/llamafactory/train/tuner.py†L21-L72】

## 环境准备

1. **安装依赖**：建议使用 Python 3.10+，并根据项目提供的 `requirements.txt` 或 `pyproject.toml` 安装依赖：
   ```bash
   pip install -e .
   ```
2. **准备模型与数据**：确保可以访问基础模型（例如 Hugging Face Hub 上的 LLaMA 检查点）与奖励模型，并准备好 JSON/JSONL 或其他受支持格式的数据集。
3. **可选：DeepSpeed/Ray**：若需分布式训练，可在命令行参数中配置 DeepSpeed 或 Ray；AGPO 训练器会自动适配并在必要时关闭外部 logger 以保持兼容。【F:src/llamafactory/train/agpo/trainer.py†L101-L109】

## 快速上手

下面示例展示如何在单机上启动一次最小化的 AGPO 训练（请根据实际资源调整参数）：

```bash
python src/train.py \
  --stage ppo \
  --rl_algo agpo \
  --model_name_or_path your-base-model \
  --reward_model your-reward-model \
  --dataset your-dataset-id \
  --template llama3 \
  --finetuning_type lora \
  --do_train True \
  --output_dir outputs/agpo-demo \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 100 \
  --report_to tensorboard
```

运行时，`run_exp` 会解析命令行参数、构建回调，并在 `--rl_algo agpo` 时调用 `run_agpo` 完成模型、奖励模型与数据加载后进入训练循环。【F:src/llamafactory/train/tuner.py†L21-L72】【F:src/llamafactory/train/agpo/workflow.py†L31-L74】

训练过程中，终端与日志目录会定期输出如下关键信息：
- `loss`、`reward`、`learning_rate` 等汇总指标。【F:src/llamafactory/train/agpo/trainer.py†L412-L439】
- `agpo/tau`、`agpo/sigma`、`agpo/step_kl`、`agpo/eps` 等自适应控制信号，用于监测温度与剪切阈值的动态变化。【F:src/llamafactory/train/agpo/trainer.py†L538-L563】

## 超参数说明

AGPO 特有超参数均可通过命令行指定，对应字段定义在 `FinetuningArguments` 中：【F:src/llamafactory/hparams/finetuning_args.py†L223-L297】

- **采样控制**
  - `--agpo_group_size`：每个提示的采样响应数量，影响探测与训练阶段的多样性。
  - `--agpo_tau_base` / `--agpo_tau_min` / `--agpo_tau_max` / `--agpo_lambda_temp`：控制温度自适应范围与响应不确定性对温度的放大系数。
- **不确定性度量**
  - `--agpo_use_robust_dispersion`：选择 `std`、`mad` 或 `iqr` 作为奖励离散度指标。
  - `--agpo_w_r`、`--agpo_w_e`、`--agpo_w_k`：分别衡量奖励方差、投票熵、偏度在不确定性中的贡献。【F:src/llamafactory/train/agpo/trainer.py†L474-L505】【F:src/llamafactory/train/agpo/trainer.py†L486-L500】
- **自适应剪切**
  - `--agpo_eps_base`、`--agpo_eps_min`、`--agpo_eps_max`：确定 PPO 剪切阈值的初始值与上下限。
  - `--agpo_alpha_var`、`--agpo_gamma_stepkl`：利用奖励离散度与 KL 偏移调整实际剪切半径。【F:src/llamafactory/train/agpo/trainer.py†L524-L536】
- **正则与日志**
  - `--agpo_beta_ref_kl`：参考模型 KL 正则化系数，用于约束策略更新幅度。【F:src/llamafactory/train/agpo/trainer.py†L542-L544】
  - `--agpo_log_probe_metrics`、`--agpo_count_probe_tokens_in_budget`：控制是否记录探测阶段统计及其在算力预算中的计算方式。【F:src/llamafactory/train/agpo/trainer.py†L558-L563】

## 训练流程细节

1. **加载组件**：`run_agpo` 会加载分词器、模板、数据集、基座模型以及奖励/参考模型，并构造多模态数据整理器与训练器实例。【F:src/llamafactory/train/agpo/workflow.py†L31-L64】
2. **主循环**：训练期间会同步旧策略、按梯度累积步长处理 batch，执行 `_agpo_step` 计算损失与奖励并回传梯度，定期保存检查点与日志。【F:src/llamafactory/train/agpo/trainer.py†L401-L448】
3. **采样与奖励**：`_agpo_step` 先进行探测采样获取奖励统计，再基于自适应温度生成训练样本，通过 `get_rewards` 支持本地或 API 奖励模型估分。【F:src/llamafactory/train/agpo/trainer.py†L467-L507】【F:src/llamafactory/train/agpo/trainer.py†L611-L620】
4. **优势与损失**：根据奖励均值与离散度计算优势，结合动态剪切与参考 KL 生成最终损失，记录关键监控指标。【F:src/llamafactory/train/agpo/trainer.py†L508-L566】
5. **收尾**：训练结束后自动保存模型、状态，并在需要时绘制损失/奖励曲线。【F:src/llamafactory/train/agpo/workflow.py†L66-L74】

## 实践建议

- **奖励模型质量**：AGPO 的稳定性高度依赖奖励信号，建议优先使用经过校准的奖励模型或 API 服务，并充分监控 `agpo/reward/mean` 与 `agpo/sigma` 以评估训练难度。【F:src/llamafactory/train/agpo/trainer.py†L412-L563】
- **温度与组大小**：在资源受限时可先减小 `agpo_group_size` 以及采样温度上限，并观察探测阶段的 `agpo/probe_dispersion` 以决定是否需要更高探索度。【F:src/llamafactory/train/agpo/trainer.py†L474-L563】
- **分布式训练**：当启用 DeepSpeed/FSDP 时，训练器会根据 `training_args` 自动调整加速器参数；记得在配置文件中显式设置 `gradient_accumulation_steps` 与 `per_device_train_batch_size` 以控制总批量。【F:src/llamafactory/train/agpo/trainer.py†L85-L147】

## 更多资源

- 若需导出或推理，请参考 `src/llamafactory/train/tuner.py` 中的 `export_model` 实现，以及 `examples/` 目录内的其他训练/推理脚本模板。
- 如需定制新策略或回调，可继承 `CustomAGPOTrainer` 并复用现有的优化器与调度器构建逻辑。【F:src/llamafactory/train/agpo/trainer.py†L571-L610】

希望本指南能够帮助你快速上手并充分利用 LLaMA-Factory 中的 AGPO 训练能力。
