"""PSPO 配置项定义。"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class PSPOArguments:
    """潜势形状化 PPO（PSPO）相关的超参数。"""

    enable: bool = field(
        default=False,
        metadata={"help": "是否启用潜势形状化奖励（PSPO）。"},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "形状化奖励中的折扣因子 γ。"},
    )
    shaping_mode: Literal["token_delta", "terminal_add"] = field(
        default="token_delta",
        metadata={
            "help": (
                "潜势差分注入方式：token_delta 会把逐 token 的 ΔΦ 加到 PPO token 奖励里；"
                "terminal_add 会将势能差分在终端标量上累加。"
            )
        },
    )
    potential_model: Literal["mlp"] = field(
        default="mlp",
        metadata={"help": "潜势网络结构，目前支持 mlp。"},
    )
    potential_hidden_size: int = field(
        default=4096,
        metadata={"help": "潜势网络隐藏层维度。"},
    )
    potential_num_layers: int = field(
        default=2,
        metadata={"help": "潜势网络线性层层数（>=2，含输出层）。"},
    )
    potential_dropout: float = field(
        default=0.0,
        metadata={"help": "潜势网络层间 dropout。"},
    )
    potential_lr: float = field(
        default=1e-4,
        metadata={"help": "潜势网络优化器学习率。"},
    )
    potential_weight_decay: float = field(
        default=0.0,
        metadata={"help": "潜势网络优化器的 weight decay。"},
    )
    potential_grad_clip: Optional[float] = field(
        default=1.0,
        metadata={"help": "潜势网络梯度裁剪阈值，None 表示不裁剪。"},
    )
    tr_lambda: float = field(
        default=1e-3,
        metadata={"help": "潜势漂移的软信任域正则系数。"},
    )
    altopt_every: int = field(
        default=1,
        metadata={"help": "潜势网络与策略交替更新的频率（多少个 PPO step 更新一次潜势）。"},
    )
    adv_target: Literal["unshaped_center", "none"] = field(
        default="unshaped_center",
        metadata={"help": "潜势差分的监督目标：未形状化奖励的居中 RTG 或关闭监督。"},
    )
    include_policy_entropy: bool = field(
        default=True,
        metadata={"help": "是否将策略分布熵拼接进潜势特征。"},
    )
    include_attention_entropy: bool = field(
        default=True,
        metadata={"help": "是否将注意力熵拼接进潜势特征。"},
    )
    max_buffer_size: int = field(
        default=4,
        metadata={"help": "潜势优化缓冲区可累积的 batch 数量上限。"},
    )

