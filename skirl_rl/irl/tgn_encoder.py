# -*- coding: utf-8 -*-
"""Minimal temporal graph encoder used by IRL features."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn


class SimpleTemporalGraphEncoder(nn.Module):
    """A lightweight encoder inspired by TGN.

    Given ``nodes`` and ``edges`` (``[src, relation, dst, time]``), the encoder assigns
    learned embeddings to nodes and aggregates relation/time information via gated
    updates.  The forward method returns a numpy array with one row per node.
    """

    def __init__(self, node_embedding_dim: int = 16, time_decay: float = 0.1) -> None:
        super().__init__()
        self.node_embedding_dim = node_embedding_dim
        self.time_decay = time_decay
        self.relation_embed = nn.Embedding(64, node_embedding_dim)
        self.node_embed = nn.Embedding(128, node_embedding_dim)
        self.gru = nn.GRUCell(node_embedding_dim, node_embedding_dim)
        self._node_index = {}

        torch.manual_seed(0)
        for module in [self.relation_embed, self.node_embed]:
            nn.init.xavier_uniform_(module.weight)

    def _index(self, key: str) -> int:
        if key not in self._node_index:
            self._node_index[key] = len(self._node_index)
        return self._node_index[key]

    def forward(self, nodes: Sequence[str], edges: Sequence[Sequence[str]]) -> np.ndarray:
        if not nodes:
            raise ValueError("nodes must not be empty")

        node_ids = torch.tensor([self._index(node) % self.node_embed.num_embeddings for node in nodes])
        hidden = self.node_embed(node_ids)

        for edge in edges:
            if len(edge) < 4:
                continue
            src, relation, dst, time_str = edge[:4]
            src_idx = self._index(src) % self.node_embed.num_embeddings
            dst_idx = self._index(dst) % self.node_embed.num_embeddings
            relation_id = hash(relation) % self.relation_embed.num_embeddings

            src_vec = self.node_embed(torch.tensor(src_idx))
            rel_vec = self.relation_embed(torch.tensor(relation_id))
            message = src_vec + rel_vec

            # simple time decay
            decay = torch.exp(torch.tensor(-self.time_decay))
            dst_vec = self.node_embed(torch.tensor(dst_idx)) * decay + message * (1 - decay)
            hidden_dst = self.gru(message.unsqueeze(0), dst_vec.unsqueeze(0))
            self.node_embed.weight.data[dst_idx] = hidden_dst.squeeze(0)

        output = self.node_embed(node_ids).detach().cpu().numpy()
        return output


def _test_encoder() -> None:
    encoder = SimpleTemporalGraphEncoder(node_embedding_dim=8)
    nodes = ["A", "T"]
    edges = [["A", "EXECUTE", "T", "2014-03-21"]]
    emb = encoder(nodes, edges)
    assert emb.shape == (2, 8)


if __name__ == "__main__":
    _test_encoder()
    print("TGN encoder tests passed.")
