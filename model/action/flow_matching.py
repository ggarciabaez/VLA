"""
Action Expert — complete implementation
========================================
Drop these classes directly into vla.py, replacing the stubs.

Components:
    SinusoidalTimeEmbedding   — encodes t ∈ [0,1] as a sinusoidal vector
    ActionExpertLayer         — one transformer layer: self-attn + cross-attn + FFN
    ActionExpert              — stacks N layers, handles projection in/out
    FlowMatchingHead          — wraps ActionExpert with FM loss and Euler sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import VLAConfig
from .action_expert import ActionExpert


# ── Flow Matching Head ─────────────────────────────────────────────────────────

class FlowMatchingHead(nn.Module):
    """
    Flow matching training loss and Euler inference sampling.

    Flow path: linear interpolation between Gaussian noise x_0 and
    demonstration actions x_1.

        x_t = (1 - t) * x_0 + t * x_1       interpolated point
        v*  = x_1 - x_0                      constant velocity field

    The ActionExpert learns to predict v*, conditioned on context.
    At inference, 10 Euler steps from noise to action.
    """
    def __init__(
        self, cfg: VLAConfig,
    ):
        super().__init__()
        self.chunk_size = cfg.chunk_size
        self.flow_steps = cfg.flow_steps
        self.action_dim = cfg.action_dim

        self.expert = ActionExpert(
            action_dim=cfg.action_dim,
            d_model=cfg.d_model,
            n_heads=cfg.action_heads,
            n_layers=cfg.action_layers,
            ffn_dim=cfg.d_model * 4,
            dropout=cfg.dropout,
        )

    def loss(
        self,
        actions: torch.Tensor,   # (B, C, action_dim), normalised ground truth
        context: torch.Tensor,   # (B, N_ctx, d_model)
    ) -> torch.Tensor:
        B = actions.size(0)

        t     = torch.rand(B, device=actions.device)
        t_exp = t.reshape(B, 1, 1)  # for c, action_dim

        x_0      = torch.randn_like(actions)
        x_t      = (1.0 - t_exp) * x_0 + t_exp * actions
        target_v = actions - x_0

        v_pred = self.expert(x_t, t, context)
        return F.mse_loss(v_pred, target_v)

    @torch.no_grad()
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        B  = context.size(0)
        dt = 1.0 / self.flow_steps

        x_t = torch.randn(B, self.chunk_size, self.action_dim, device=context.device)

        for step in range(self.flow_steps):
            t = torch.full((B,), step * dt, device=context.device)
            v = self.expert(x_t, t, context)
            x_t = x_t + v * dt

        return x_t  # (B, C, action_dim)