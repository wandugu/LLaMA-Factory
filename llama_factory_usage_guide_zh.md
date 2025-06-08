# 如何使用 LLaMA Factory

LLaMA Factory 是一个功能强大且易于上手的大语言模型微调与服务框架。本文档将指导您如何安装、准备数据、快速上手以及使用其各项主要功能。

## 1. 安装 LLaMA Factory

您可以选择从源码或 Docker 镜像进行安装。

### 从源码安装

这是推荐的安装方式，可以确保您使用的是最新的代码。

```bash
# 1. 克隆仓库，--depth 1 表示只克隆最新的 commit 以节省时间
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

# 2. 进入项目目录
cd LLaMA-Factory

# 3. 安装 LLaMA Factory 及其核心依赖
#    .[torch,metrics] 表示同时安装 PyTorch 和评估相关的依赖
#    --no-build-isolation 建议用于解决一些环境下的构建问题
pip install -e ".[torch,metrics]" --no-build-isolation
```

**可选依赖项说明：**

LLaMA Factory 支持许多可选功能，您可以通过在 `pip install -e .[...]` 的方括号中添加相应的依赖名称来安装它们。常用的可选依赖包括：

*   `torch-npu`: 昇腾 NPU 支持。
*   `deepspeed`: 用于大规模模型训练的 DeepSpeed 优化库。
*   `liger-kernel`: Liger 内核，用于加速训练。
*   `bitsandbytes`: 用于模型量化，例如 QLoRA。
*   `hqq`, `eetq`, `gptq`, `aqlm`: 其他不同的模型量化方法。
*   `vllm`, `sglang`: 用于高速推理服务的后端。
*   `galore`, `apollo`, `badam`, `adam-mini`: 高级优化器。
*   `qwen`, `minicpm_v`: 特定模型的支持。
*   `modelscope`, `openmind`: 从 ModelScope 或 OpenMind Hub 下载模型。
*   `swanlab`: SwanLab 实验监控。
*   `dev`: 开发和测试所需的依赖。

例如，如果您需要在昇腾 NPU 上使用 DeepSpeed 和 BitsAndBytes，可以这样安装：
`pip install -e ".[torch-npu,metrics,deepspeed,bitsandbytes]"`

### 从 Docker 镜像安装

如果您希望在隔离的环境中运行 LLaMA Factory，或者希望快速部署，可以使用官方提供的 Docker 镜像。

```bash
# 拉取并运行最新的 Docker 镜像
# --gpus=all: 允许 Docker容器访问所有可用的 GPU
# --ipc=host: 允许容器共享主机的 IPC 命名空间，通常为了 PyTorch 多进程通信
docker run -it --rm --gpus=all --ipc=host hiyouga/llamafactory:latest
```

该默认镜像基于 Ubuntu 22.04、CUDA 12.4、Python 3.11、PyTorch 2.6.0 和 Flash-attn 2.7.4 构建。您可以在 [Docker Hub](https://hub.docker.com/r/hiyouga/llamafactory/tags) 查看所有可用的镜像标签。

### 特殊平台用户指南

#### Windows 用户

1.  **安装 PyTorch**: Windows 用户需要手动安装 GPU 版本的 PyTorch。请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 的指引。例如，针对 CUDA 12.6 的安装命令：
    ```bash
    pip uninstall torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    # 验证安装
    python -c "import torch; print(torch.cuda.is_available())"
    ```
    如果输出 `True` 则表示 PyTorch GPU 版本安装成功。
    如果遇到 `Can't pickle local object` 错误，尝试在配置文件中设置 `dataloader_num_workers: 0`。

2.  **安装 BitsAndBytes (用于 QLoRA)**: 需要安装预编译的 `bitsandbytes`。根据您的 CUDA 版本从 [bitsandbytes-windows-webui releases](https://github.com/jllllll/bitsandbytes-windows-webui/releases/tag/wheels) 下载对应的 whl 文件安装。例如：
    ```bash
    pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
    ```

3.  **安装 Flash Attention-2**: Windows 用户需要自行编译安装，可以参考 [flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel) 提供的脚本。

#### 昇腾 NPU 用户

1.  **Python 版本**: 请确保 Python 版本 >= 3.10。
2.  **安装依赖**: 使用特定依赖项进行安装：`pip install -e ".[torch-npu,metrics]"`。
3.  **CANN Toolkit 与 Kernels**: 必须安装华为的 Ascend CANN Toolkit 和 Kernels。请参考官方[安装教程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/quickstart/quickstart/quickstart_18_0004.html)。安装完成后需要设置环境变量：
    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```
4.  **环境变量**: 使用 `ASCEND_RT_VISIBLE_DEVICES` 而非 `CUDA_VISIBLE_DEVICES` 来指定运算设备。
5.  **BitsAndBytes (用于 QLoRA)**:
    *   手动从源码编译 `bitsandbytes`，参考[官方NPU安装文档](https://huggingface.co/docs/bitsandbytes/installation?backend=Ascend+NPU&platform=Ascend+NPU)。需要 cmake >= 3.22.1 和 g++ >= 12.x。
    *   安装 `transformers` 的 main 分支版本。
    *   在训练参数中设置 `double_quantization: false`。

## 2. 数据准备

高质量的数据是成功微调模型的关键。

### 数据集文件格式

LLaMA Factory 支持多种数据集格式，主要分为 **alpaca** 格式和 **sharegpt** 格式。允许的文件类型包括 `.json`, `.jsonl`, `.csv`, `.parquet`, 和 `.arrow`。

详细的数据集格式说明（包括如何组织指令、输入、输出、历史对话、系统提示、工具调用、多模态数据等）请务必仔细阅读项目根目录下的 `data/README_zh.md` 文件。该文件详细描述了：

*   **Alpaca 格式**:
    *   指令监督微调数据集 (包含 `instruction`, `input`, `output`, `system`, `history` 字段)
    *   预训练数据集 (包含 `text` 字段)
    *   偏好数据集 (包含 `instruction`, `input`, `chosen`, `rejected` 字段，用于 RM, DPO, ORPO, SimPO)
    *   KTO、多模态图像/视频/音频数据集的 Alpaca 格式扩展。
*   **ShareGPT 格式**:
    *   指令监督微调数据集 (使用 `conversations` 列表，支持 `human`, `gpt`, `observation`, `function_call` 等多种角色)
    *   偏好数据集 (在 `conversations` 基础上增加 `chosen` 和 `rejected` 消息对象)
    *   KTO、多模态图像/视频/音频数据集的 ShareGPT 格式扩展。
    *   OpenAI 格式 (作为 ShareGPT 的一种特例)。

### 使用数据集

您可以：

1.  **使用 HuggingFace Hub 上的数据集**: 在 `dataset_info.json` 中配置数据集的 `hf_hub_url`。
2.  **使用 ModelScope Hub 上的数据集**: 在 `dataset_info.json` 中配置数据集的 `ms_hub_url`。需要设置环境变量 `USE_MODELSCOPE_HUB=1`。
3.  **使用 Modelers Hub 上的数据集**: 在 `dataset_info.json` 中配置。需要设置环境变量 `USE_OPENMIND_HUB=1`。
4.  **加载本地数据集**: 将数据集文件放置在 `data` 目录（或自定义的 `dataset_dir` 目录）下，并在 `dataset_info.json` 中配置 `file_name`。

### 自定义数据集

当您使用自己的数据集时，**必须**在 `data/dataset_info.json` 文件中为您的数据集添加描述。`dataset_info.json` 的结构如下：

```json
"您的数据集名称": {
  "hf_hub_url": "Hugging Face 的数据集仓库地址（可选）",
  "ms_hub_url": "ModelScope 的数据集仓库地址（可选）",
  "script_url": "包含数据加载脚本的本地文件夹名称（可选）",
  "file_name": "本地数据集文件名或文件夹名（如果前三者未指定，则此项必需）",
  "formatting": "数据集格式（可选，默认：alpaca，可为 alpaca 或 sharegpt）",
  "ranking": "是否为偏好数据集（可选，默认：False）",
  // ... 其他如 subset, split, folder, num_samples 等参数
  "columns（可选）": { // 定义数据列与标准字段的映射关系
    "prompt": "instruction", // 对于 Alpaca 格式，表示指令的列名
    "query": "input",      // 对于 Alpaca 格式，表示输入的列名
    "response": "output",  // 对于 Alpaca 格式，表示输出的列名
    "messages": "conversations", // 对于 ShareGPT 格式，表示对话列表的列名
    // ... 其他如 system, tools, images, videos, audios, chosen, rejected, kto_tag 等
  },
  "tags（可选，用于 sharegpt 格式）": { // 定义 ShareGPT 格式中消息对象的键名
    "role_tag": "from",
    "content_tag": "value",
    // ... 其他如 user_tag, assistant_tag, observation_tag, function_tag, system_tag
  }
}
```

确保您的数据集描述与实际文件结构和内容相匹配。然后，在训练配置文件 (yaml) 中通过 `dataset: 您的数据集名称` 来指定使用哪个数据集。

对于部分需要登录才能访问的数据集 (例如 Llama 系列模型)，推荐先登录 Hugging Face 账户：
```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

## 3. 快速开始

LLaMA Factory 提供了一个命令行工具 `llamafactory-cli` 来执行各种操作。以下是对 Llama3-8B-Instruct 模型进行 LoRA 微调、推理和合并权重的示例。

配置文件位于 `examples/` 目录下，您可以根据需求复制和修改这些 yaml 文件。

### LoRA 微调 (Supervised Fine-Tuning)

```bash
# 使用 llamafactory-cli train 命令，并指定训练配置文件
# examples/train_lora/llama3_lora_sft.yaml 是一个 LoRA SFT 的示例配置
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```
该命令会根据配置文件中的参数（如模型名称、数据集、学习率、LoRA 模块等）启动微调过程。训练好的 LoRA 权重会保存在指定的输出目录。

### 模型推理 (Chat)

微调完成后，您可以使用 `llamafactory-cli chat` 与模型进行交互式对话。

```bash
# examples/inference/llama3_lora_sft.yaml 是一个推理配置，
# 它会加载基础模型和指定路径下的 LoRA 权重
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
```

### 合并 LoRA 权重

如果您希望将 LoRA 权重合并到基础模型中，以得到一个完整的、可以直接部署的模型，可以使用 `llamafactory-cli export` 命令。

```bash
# examples/merge_lora/llama3_lora_sft.yaml 是一个导出（合并）配置
# 它指定了基础模型、LoRA 权重路径以及导出模型的保存路径和格式
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### 高级用法

`examples/README_zh.md` 文件中包含了更详细和高级的用法示例，例如：

*   多 GPU 微调 (DeepSpeed, FSDP)
*   不同训练方法 (预训练, RM, PPO, DPO, KTO, ORPO)
*   不同量化方案 (QLoRA, GPTQ, AWQ)
*   模型评估
*   使用不同的模型和数据集

> **提示**:
> *   使用 `llamafactory-cli help` 查看所有可用子命令和参数。
> *   遇到问题时，可以先查阅 [常见问题](https://github.com/hiyouga/LLaMA-Factory/issues/4614)。

## 4. LLaMA Board 可视化微调

LLaMA Factory 提供了一个基于 Gradio 的 Web UI，称为 LLaMA Board，允许用户通过浏览器界面进行模型微调和推理。

启动 LLaMA Board:
```bash
llamafactory-cli webui
```
之后在浏览器中打开提示的 URL (通常是 `http://localhost:7860`) 即可使用。

## 5. 构建 Docker

如果您需要自定义 Docker 镜像（例如，包含特定的依赖或代码修改），可以参考 `docker/` 目录下的 Dockerfile。

*   **CUDA 用户**:
    ```bash
    cd docker/docker-cuda/
    docker compose up -d
    docker compose exec llamafactory bash
    ```
    或者不使用 `docker-compose`:
    ```bash
    docker build -f ./docker/docker-cuda/Dockerfile \
        --build-arg PIP_INDEX=https://pypi.org/simple \
        --build-arg EXTRAS=metrics \
        -t llamafactory:latest .
    # 根据需要调整 docker run 命令的参数
    docker run -dit --ipc=host --gpus=all -p 7860:7860 -p 8000:8000 --name llamafactory llamafactory:latest
    docker exec -it llamafactory bash
    ```

*   **昇腾 NPU 用户**:
    ```bash
    cd docker/docker-npu/
    docker compose up -d
    docker compose exec llamafactory bash
    ```
    构建命令类似，但 Dockerfile 和基础镜像不同，并且 `EXTRAS` 参数通常会包含 `torch-npu`。

*   **AMD ROCm 用户**:
    ```bash
    cd docker/docker-rocm/
    docker compose up -d
    docker compose exec llamafactory bash
    ```
    构建命令类似，针对 ROCm 环境。

`README_zh.md` 中还提到了如何使用数据卷 (Volume) 来持久化 Hugging Face 缓存、共享数据和输出模型。

## 6. 利用 vLLM 部署 OpenAI API

LLaMA Factory 支持使用 [vLLM](https://github.com/vllm-project/vllm) 或 [SGLang](https://github.com/sgl-project/sglang) 作为后端，将微调后的模型部署为 OpenAI 兼容的 API 服务，从而可以轻松集成到现有的 ChatGPT 应用生态中。

```bash
# API_PORT=8000 设置 API 服务端口
# examples/inference/llama3.yaml 指定了要加载的模型（可以是合并后的完整模型，或基础模型+LoRA）
# infer_backend=vllm 指定使用 vLLM 作为推理后端
# vllm_enforce_eager=true 是 vLLM 的一个可选参数
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

部署成功后，您可以像调用 OpenAI API 一样，向 `http://localhost:8000/v1/chat/completions` (或其他指定端口) 发送请求。
API 文档可以参考 [OpenAI官方文档](https://platform.openai.com/docs/api-reference/chat/create)。
`scripts/api_example/` 目录下提供了图像理解和工具调用的 API 使用示例。

---

通过以上步骤，您应该能够成功安装并开始使用 LLaMA Factory 进行模型的微调和部署。建议详细阅读项目中的 `README_zh.md` 和 `examples/README_zh.md` 以获取更全面的信息和高级功能介绍。
