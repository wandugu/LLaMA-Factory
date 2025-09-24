"""PSPO 势函数与奖励整形实现。"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from ...extras import logging

if TYPE_CHECKING:
    from ..ppo.trainer import CustomPPOTrainer
    from ...hparams import PSPOArguments


logger = logging.get_logger(__name__)


class PotentialMLP(nn.Module):
    """简单的 MLP 潜势网络。"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("`potential_num_layers` 至少为 2。")

        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1, bias=True)

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向计算潜势值。"""

        hidden = self.backbone(features)
        phi = self.output(hidden).squeeze(-1)
        if mask is not None:
            phi = phi * mask
        return phi


class PSPOHelper:
    """管理势函数网络、奖励整形与潜势阶段优化。"""

    def __init__(self, trainer: "CustomPPOTrainer", args: "PSPOArguments") -> None:
        self.trainer = trainer
        self.args = args
        self.enabled = bool(args.enable)
        self.gamma = args.gamma
        self.device = trainer.current_device

        self.feature_dim: Optional[int] = None
        self.potential_net: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self.buffer: list[dict[str, torch.Tensor]] = []
        self.last_delta_norm: float = 0.0
        self.last_potential_loss: float = 0.0

    def maybe_enable_generation_kwargs(self, generation_config: Any) -> None:
        if not self.enabled:
            return

        generation_config.return_dict_in_generate = True
        generation_config.output_scores = True
        generation_config.output_hidden_states = True
        if self.args.include_attention_entropy:
            generation_config.output_attentions = True

    def _build_model(self, feature_dim: int) -> None:
        if self.potential_net is not None:
            return

        self.feature_dim = feature_dim
        hidden_dim = self.args.potential_hidden_size
        num_layers = self.args.potential_num_layers
        dropout = self.args.potential_dropout
        self.potential_net = PotentialMLP(feature_dim, hidden_dim, num_layers, dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.potential_net.parameters(),
            lr=self.args.potential_lr,
            weight_decay=self.args.potential_weight_decay,
        )
        num_params = sum(p.numel() for p in self.potential_net.parameters())
        logger.info_rank0(f"初始化 PotentialNet（{num_params:,} 参数，输入维度 {feature_dim}）。")

    def _policy_forward(self, model_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        with torch.no_grad():
            outputs = model(
                **model_inputs,
                output_hidden_states=True,
                output_attentions=self.args.include_attention_entropy,
                return_dict=True,
                use_cache=False,
            )
        return {
            "logits": outputs.logits.float(),
            "hidden_states": outputs.hidden_states[-1].float(),
            "attentions": outputs.attentions[-1].float() if outputs.attentions is not None else None,
        }

    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs.clamp_min(1e-12))
        return -(probs * log_probs).sum(dim=-1)

    @staticmethod
    def _entropy_from_attn(attn: torch.Tensor) -> torch.Tensor:
        if attn is None:
            raise ValueError("attention tensor is required to compute entropy")
        probs = attn.mean(dim=1)  # average over heads
        log_probs = torch.log(probs.clamp_min(1e-12))
        return -(probs * log_probs).sum(dim=-1)

    def _build_feature_batch(
        self,
        queries: list[torch.Tensor],
        responses: list[torch.Tensor],
        model_inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        outputs = self._policy_forward(model_inputs)
        hidden_states = outputs["hidden_states"]
        logits = outputs["logits"]
        attention_entropy_full = None
        if self.args.include_attention_entropy and outputs["attentions"] is not None:
            attention_entropy_full = self._entropy_from_attn(outputs["attentions"])

        policy_entropy_full = (
            self._entropy_from_logits(logits) if self.args.include_policy_entropy else torch.zeros_like(logits[..., 0])
        )

        if "input_ids" not in model_inputs or "attention_mask" not in model_inputs:
            raise NotImplementedError("PSPO 当前仅支持 decoder-only 结构的策略模型。")

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        batch_size = input_ids.size(0)
        feature_chunks: list[torch.Tensor] = []
        state_lengths: list[int] = []
        response_lengths: list[int] = []

        for i in range(batch_size):
            prompt_len = int(queries[i].size(0))
            resp_len = int(responses[i].size(0))
            response_lengths.append(resp_len)

            valid_len = int(attention_mask[i].sum().item())
            offset = input_ids.size(1) - valid_len

            state_steps = resp_len + 1 if resp_len > 0 else 1
            start = offset + max(prompt_len - 1, 0)
            end = start + state_steps
            max_index = offset + valid_len - 1
            indices = torch.arange(start, end, device=hidden_states.device)
            if indices.numel() == 0:
                indices = torch.tensor([max(offset, 0)], device=hidden_states.device)
            indices = indices.clamp(max=max_index)
            if indices.numel() < state_steps:
                pad_index = indices[-1]
                pad = pad_index.expand(state_steps - indices.numel())
                indices = torch.cat((indices, pad))

            state_hidden = hidden_states[i, indices, :]
            feature_components = [state_hidden]
            if self.args.include_policy_entropy:
                feature_components.append(policy_entropy_full[i, indices].unsqueeze(-1))
            if self.args.include_attention_entropy and attention_entropy_full is not None:
                feature_components.append(attention_entropy_full[i, indices].unsqueeze(-1))

            features = torch.cat(feature_components, dim=-1)
            feature_chunks.append(features)
            state_lengths.append(features.size(0))

        max_state_len = max(state_lengths)
        feature_dim = feature_chunks[0].size(-1)
        features_tensor = torch.zeros(
            (batch_size, max_state_len, feature_dim), device=self.device, dtype=torch.float32
        )
        state_mask = torch.zeros((batch_size, max_state_len), device=self.device, dtype=torch.float32)

        for i, feats in enumerate(feature_chunks):
            length = feats.size(0)
            features_tensor[i, :length] = feats
            state_mask[i, :length] = 1.0

        if max_state_len > 1:
            delta_mask = state_mask[:, 1:] * state_mask[:, :-1]
        else:
            delta_mask = torch.zeros((batch_size, 0), device=self.device, dtype=torch.float32)

        self._build_model(feature_dim)
        return features_tensor, state_mask, delta_mask, response_lengths

    def shape_rewards(
        self,
        queries: list[torch.Tensor],
        responses: list[torch.Tensor],
        base_rewards: torch.Tensor,
        model_inputs: dict[str, torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[float]]:
        if not self.enabled:
            raise RuntimeError("PSPO 未启用却调用了 shape_rewards。")

        features, state_mask, delta_mask, response_lengths = self._build_feature_batch(queries, responses, model_inputs)

        potential_model = self.potential_net
        assert potential_model is not None and self.optimizer is not None

        phi = potential_model(features, state_mask)
        delta = self.gamma * phi[:, 1:] - phi[:, :-1]
        if delta_mask.numel() > 0:
            delta = delta * delta_mask

        shaped_rewards: list[torch.Tensor] = []
        reward_summaries: list[float] = []

        base = base_rewards.to(self.device, dtype=torch.float32).view(-1)
        self.last_delta_norm = float(delta.abs().mean().detach().cpu()) if delta_mask.numel() > 0 else 0.0

        max_delta_len = delta.size(1) if delta.dim() > 1 else 0
        targets = torch.zeros((delta.size(0), max_delta_len), device=self.device, dtype=torch.float32)
        if self.args.adv_target == "unshaped_center" and max_delta_len > 0:
            centered = base - base.mean()
            for i, length in enumerate(response_lengths):
                if length <= 0:
                    continue
                length = min(length, max_delta_len)
                targets[i, :length] = centered[i]

        for i, length in enumerate(response_lengths):
            base_scalar = base[i].item()
            if length <= 0:
                reward_vec = torch.tensor([base_scalar], dtype=torch.float32)
                shaped_rewards.append(reward_vec)
                reward_summaries.append(float(base_scalar))
                continue

            base_vec = torch.full((length,), base_scalar, device=self.device, dtype=torch.float32)
            delta_vec = delta[i, :length] if max_delta_len > 0 else torch.zeros_like(base_vec)

            if self.args.shaping_mode == "terminal_add":
                shaped_scalar = base_scalar + float(delta_vec.sum().item())
                shaped_vec = torch.full((length,), shaped_scalar, device=self.device, dtype=torch.float32)
            else:
                shaped_vec = base_vec + delta_vec

            shaped_rewards.append(shaped_vec.detach().cpu())
            reward_summaries.append(float(shaped_vec.mean().item()))

        if max_delta_len > 0:
            targets = targets * delta_mask

        batch_record = {
            "features": features.detach().cpu(),
            "state_mask": state_mask.detach().cpu(),
            "delta_mask": delta_mask.detach().cpu(),
            "targets": targets.detach().cpu(),
        }
        self.buffer.append(batch_record)
        if len(self.buffer) > max(1, self.args.max_buffer_size):
            self.buffer.pop(0)

        return shaped_rewards, reward_summaries

    def optimize_if_needed(self, step: int) -> dict[str, torch.Tensor]:
        if not self.enabled or self.potential_net is None or self.optimizer is None:
            return {}
        if not self.buffer or self.args.altopt_every <= 0:
            return {}
        if (step + 1) % self.args.altopt_every != 0:
            return {}

        self.optimizer.zero_grad()
        loss_accum = 0.0
        count = 0

        for payload in self.buffer:
            features = payload["features"].to(self.device)
            state_mask = payload["state_mask"].to(self.device)
            delta_mask = payload["delta_mask"].to(self.device)
            targets = payload["targets"].to(self.device)

            phi = self.potential_net(features, state_mask)
            delta = self.gamma * phi[:, 1:] - phi[:, :-1]
            if delta_mask.numel() == 0:
                continue

            delta = delta * delta_mask
            mse = ((delta - targets) ** 2) * delta_mask
            denom = delta_mask.sum()
            if denom <= 0:
                continue

            loss = mse.sum() / denom
            if self.args.tr_lambda > 0:
                trust = (delta.pow(2) * delta_mask).sum() / denom
                loss = loss + self.args.tr_lambda * trust

            loss.backward()
            loss_accum += float(loss.detach().cpu())
            count += 1

        self.buffer.clear()

        if count == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return {}

        if self.args.potential_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.potential_net.parameters(), self.args.potential_grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.last_potential_loss = loss_accum / count

        stats = {
            "pspo/potential_loss": torch.tensor(self.last_potential_loss, device=self.device),
            "pspo/delta_mean": torch.tensor(self.last_delta_norm, device=self.device),
        }
        return stats

    def save_state(self, output_dir: str) -> None:
        if not self.enabled or self.potential_net is None or self.optimizer is None:
            return

        payload = {
            "model": self.potential_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "pspo_args": asdict(self.args),
        }
        torch.save(payload, os.path.join(output_dir, "potential_model.pt"))
