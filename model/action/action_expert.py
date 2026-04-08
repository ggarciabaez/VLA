from torch import nn
import torch, math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for sin/cos pairs"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) — continuous float in [0, 1]
        half = self.dim // 2
        # log-spaced frequencies from 1 to 1000
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device)
        )                                           # (half,)
        args = t[:, None] * freqs[None, :]          # (B, half)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return emb


# TODO: cached KV could help compute.
class ActionExpertLayer(nn.Module):
    """
    One transformer layer for the action expert.

    Order of operations (Pre-LN throughout):
        1. LayerNorm → self-attention  → residual   (chunk coherence)
        2. LayerNorm → cross-attention → residual   (observation conditioning)
        3. LayerNorm → FFN             → residual

    Self-attention:  action tokens attend to each other
                     → adjacent actions in the chunk coordinate
                     → produces smooth, consistent trajectories

    Cross-attention: action tokens (Q) attend to context tokens (K, V)
                     → each action step queries the full observation
                     → e.g. grasp step attends to gripper/object tokens
    """
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()

        # self-attention
        self.norm1      = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # cross-attention — action tokens query context tokens
        self.norm2      = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x:       torch.Tensor,   # (B, C, d_model)        — action tokens
        context: torch.Tensor,   # (B, N_ctx, d_model)    — fusion transformer output
    ) -> torch.Tensor:

        # 1. self-attention over action chunk (Pre-LN)
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # 2. cross-attention: action tokens query context tokens (Pre-LN)
        #    Q = action tokens, K = V = context tokens
        #    do NOT swap these — action tokens are asking questions,
        #    context tokens hold the answers
        h = self.norm2(x)
        h, _ = self.cross_attn(h, context, context)
        x = x + h

        # 3. FFN (Pre-LN)
        x = x + self.ffn(self.norm3(x))

        return x  # (B, C, d_model)


class ActionExpert(nn.Module):
    def __init__(
        self, action_dim: int, d_model: int, n_heads: int, n_layers: int, ffn_dim: int, dropout: float = 0.05
    ):
        super().__init__()

        self.action_proj = nn.Linear(action_dim, d_model)
        self.time_emb    = SinusoidalTimeEmbedding(d_model)

        self.layers = nn.ModuleList([
            ActionExpertLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, action_dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_t:     torch.Tensor,   # (B, C, action_dim)
        t:       torch.Tensor,   # (B,)
        context: torch.Tensor,   # (B, N_ctx, d_model)
    ) -> torch.Tensor:

        x = self.action_proj(x_t)  # project onto model space
        # add time embedding — same t for all steps in the chunk
        t_emb = self.time_emb(t).unsqueeze(1)           # (B, 1, d_model)
        x = x + t_emb                                   # broadcasts to (B, C, d_model)

        # run transformer layers
        for layer in self.layers:
            x = layer(x, context)

        # project back to action space
        return self.out_proj(self.norm_out(x))          # (B, C, action_dim)