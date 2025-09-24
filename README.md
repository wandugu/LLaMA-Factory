# PSPO Factory

本仓库是一个专注于**潜势形状化 PPO（PSPO）**、标准 **PPO** 以及 **GRPO** 的强化学习后训练框架。我们基于原 LLaMA-Factory 的 PPO 管线进行了裁剪与强化，只保留与策略优化直接相关的代码，便于在单一代码库内进行策略对比与快速实验。

## 核心特性

- 🚀 **PPO 基线**：复用了经过社区验证的 PPO 数据采样、KL 惩罚与优化流程。
- 🧭 **PSPO 势能整形**：内置可训练的 PotentialNet，支持 token 级势能差分、交替优化与信任域正则。
- 🧮 **GRPO 模式**：提供 critic-free 的归一化回报优化策略，并自动冻结 value head。
- 🧩 **统一配置**：通过统一的参数解析器与 `pspo_args` 数据类即可在 CLI / WebUI 中切换不同策略。

## 快速上手

1. 安装依赖：
   ```bash
   pip install -e .
   ```
2. 准备数据集与奖励模型，确保在配置文件或命令行中指定 `reward_model`。
3. 选择训练阶段并启动 CLI：
   ```bash
   llamafactory-cli \
     --stage pspo \
     --config_file path/to/your.yaml
   ```
   - `--stage pspo`：启用势能整形的 PSPO。
   - `--stage ppo`：运行标准 PPO。
   - `--stage grpo`：启用 GRPO（归一化回报，value head 冻结，VF loss 关闭）。

> **提示**：PSPO 会在采样阶段自动打开隐藏状态与 logits 返回，用于 PotentialNet 计算势能差分。

## 主要配置项

- `pspo.enable`：是否启用势能整形（`stage=pspo` 时自动开启）。
- `pspo.gamma`：势能差分的折扣因子。
- `pspo.shaping_mode`：`token_delta`（逐 token 注入）或 `terminal_add`（终端累加）。
- `pspo.potential_*`：势能网络结构与优化器超参。
- `pspo.altopt_every`：策略与势能交替优化的步频。

GRPO 额外行为：
- 自动将 `vf_coef` 设为 0，并对回报执行 `masked_whiten` 归一化。
- 自动冻结策略模型的 value head。

## 目录结构

```
src/llamafactory/
├── hparams/          # 训练、生成及 PSPO 的参数解析
├── train/
│   ├── ppo/          # PPO / GRPO 训练工作流与自定义 Trainer
│   └── pspo/         # 势能网络与奖励整形实现
└── ...               # 数据、模型加载与通用工具
```

## 许可证

本项目沿用原始 LLaMA-Factory 的 Apache 2.0 许可证。
