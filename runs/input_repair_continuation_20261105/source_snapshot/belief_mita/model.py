"""Small topology-landmark Belief-MiTA network for the original M0 protocol."""

import math

import torch
from torch import nn


class BeliefMiTA(nn.Module):
    def __init__(self, feature_dim=18, d=64, landmarks=16, top_k=12, compressed=4):
        super().__init__()
        if d % 4 or min(feature_dim, d, landmarks, top_k, compressed) <= 0:
            raise ValueError("positive dimensions and d divisible by four are required")
        self.feature_dim = feature_dim
        self.d = d
        self.landmarks = landmarks
        self.top_k = top_k
        self.heads = 4
        self.head_dim = d // self.heads
        self.temporal_encoder = nn.GRU(feature_dim, d, batch_first=True)
        self.cell_norm = nn.LayerNorm(d)
        self.landmark_encoder = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.key_projection = nn.Linear(d, d, bias=False)
        self.value_projection = nn.Linear(d, d)
        self.expert_projection = nn.Linear(d, d)
        self.expert_norm = nn.LayerNorm(d)
        # Pool the full grid independently so sparse routing cannot discard global evidence.
        self.compress_down = nn.Linear(d, d // 4)
        self.compress_tokens = nn.Linear(d // 4, compressed, bias=False)
        self.compress_up = nn.Linear(d // 4, d)
        self.agent_embedding = nn.Parameter(torch.randn(2, d) / math.sqrt(d))
        self.agent_attention = nn.MultiheadAttention(d, self.heads, batch_first=True)
        self.agent_norm = nn.LayerNorm(d)
        self.enemy_embedding = nn.Parameter(torch.randn(2, d) / math.sqrt(d))
        self.belief_query = nn.Linear(d, d)
        self.belief_key = nn.Linear(d, d)
        self.action_head = nn.Linear(d, 5)
        self.first_action_embedding = nn.Embedding(5, d)
        self.conditional_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 5))
        self.value_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1), nn.Tanh())

    def forward(self, x, landmark_indices, agent_indices, cell_mask=None, *, return_attention=False):
        """Return logits; cell_mask=True marks legal cells, including selected indices.

        With return_attention=True, also return per-head landmark-to-cell and
        agent-to-expert probabilities. No target positions enter these queries.
        """
        if x.ndim != 4 or x.shape[1] != 8 or x.shape[-1] != self.feature_dim:
            raise ValueError("x must have shape [batch, 8, cells, feature_dim]")
        batch, _, cells, _ = x.shape
        if cells == 0:
            raise ValueError("each map needs a legal cell")
        if cell_mask is None:
            cell_mask = torch.ones(batch, cells, dtype=torch.bool, device=x.device)
        if cell_mask.shape != (batch, cells) or cell_mask.dtype != torch.bool:
            raise ValueError("cell_mask must be bool [batch, cells]")
        if not cell_mask.any(dim=1).all():
            raise ValueError("each map needs a legal cell")
        for name, indices, count in (("landmark", landmark_indices, self.landmarks),
                                     ("agent", agent_indices, 2)):
            if indices.shape != (batch, count) or indices.dtype != torch.long:
                raise ValueError(f"{name} indices must be long [batch, {count}]")
            if ((indices < 0) | (indices >= cells)).any():
                raise ValueError(f"{name} indices must select existing cells")
            if not cell_mask.gather(1, indices).all():
                raise ValueError(f"{name} indices must select legal cells")

        history = x.masked_fill(~cell_mask[:, None, :, None], 0)
        history = history.permute(0, 2, 1, 3).reshape(batch * cells, 8, self.feature_dim)
        _, hidden = self.temporal_encoder(history)
        tokens = self.cell_norm(hidden[-1].reshape(batch, cells, self.d))
        gather_landmarks = landmark_indices[..., None].expand(-1, -1, self.d)
        queries = self.landmark_encoder(tokens.gather(1, gather_landmarks))
        q = queries.view(batch, self.landmarks, self.heads, self.head_dim).transpose(1, 2)
        keys = self.key_projection(tokens).view(batch, cells, self.heads, self.head_dim).transpose(1, 2)
        values = self.value_projection(tokens).view(batch, cells, self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~cell_mask[:, None, None, :], -torch.inf)
        top_scores, top_indices = scores.topk(min(self.top_k, cells), dim=-1)
        top_weights = top_scores.softmax(-1)
        expanded_values = values[:, :, None].expand(-1, -1, self.landmarks, -1, -1)
        selected_values = expanded_values.gather(
            3, top_indices[..., None].expand(-1, -1, -1, -1, self.head_dim))
        experts = (top_weights[..., None] * selected_values).sum(-2)
        experts = experts.transpose(1, 2).reshape(batch, self.landmarks, self.d)
        experts = self.expert_norm(queries + self.expert_projection(experts))
        narrow = torch.nn.functional.gelu(self.compress_down(tokens))
        global_scores = self.compress_tokens(narrow).transpose(1, 2)
        global_scores = global_scores.masked_fill(~cell_mask[:, None, :], -torch.inf)
        global_weights = global_scores.softmax(-1)
        global_experts = self.compress_up(global_weights @ narrow)
        all_experts = torch.cat((experts, global_experts), dim=1)
        own_queries = tokens.gather(1, agent_indices[..., None].expand(-1, -1, self.d))
        own_queries = own_queries + self.agent_embedding[None]
        reads, agent_weights = self.agent_attention(
            own_queries, all_experts, all_experts, need_weights=return_attention,
            average_attn_weights=False)
        contexts = self.agent_norm(own_queries + reads)
        enemy_queries = self.belief_query(contexts.mean(1, keepdim=True) + self.enemy_embedding[None])
        belief_logits = (enemy_queries @ self.belief_key(tokens).transpose(1, 2)) / math.sqrt(self.d)
        belief_logits = belief_logits.masked_fill(~cell_mask[:, None, :], torch.finfo(x.dtype).min)
        joint_context = contexts.reshape(batch, 2 * self.d)
        first_actions = self.first_action_embedding.weight[None].expand(batch, -1, -1)
        conditional_input = torch.cat((joint_context[:, None].expand(-1, 5, -1), first_actions), dim=-1)
        result = {
            "belief_logits": belief_logits,
            "action_logits": self.action_head(contexts),
            "conditional_action_logits": self.conditional_head(conditional_input),
            "value": self.value_head(joint_context).squeeze(-1),
        }
        if return_attention:
            weights = torch.zeros_like(scores).scatter(-1, top_indices, top_weights)
            result["attention"] = {"landmark_weights": weights, "top_indices": top_indices,
                                   "agent_weights": agent_weights, "global_weights": global_weights}
        return result
