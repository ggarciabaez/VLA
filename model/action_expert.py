import torch
import math
from torch import nn
from model.utils import VLAConfig
from model.mha_impl import MultiHeadAttention


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for sin/cos pairs"
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device)
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaLNBlock(nn.Module):
    def __init__(self, d, n_heads, kv_div=2, dropout=0.0, is_cross=False, ffn_mult=4, r=256):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = MultiHeadAttention(d, n_heads, kv_div, dropout, is_cross)
        self.ffn = nn.Sequential(nn.Linear(d, d*ffn_mult), nn.GELU(approximate='tanh'), nn.Linear(d*ffn_mult, d))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d, r), nn.SiLU(), nn.Linear(r, 6*d))
        nn.init.zeros_(self.mod[-1].weight); nn.init.zeros_(self.mod[-1].bias)

    def forward(self, x, cond, ctx=None):
        s1, c1, g1, s2, c2, g2 = self.mod(cond).unsqueeze(1).chunk(6, -1)
        h = self.n1(x) * (1 + c1) + s1
        a = self.attn(h, ctx, ctx) if self.attn.is_cross else self.attn(h)
        x = x + g1 * a
        return x + g2 * self.ffn(self.n2(x) * (1 + c2) + s2)


class VelocityGenerator(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        action_dim = cfg.action_dim

        # 1. Embed continuous flow matching time
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(d_model * 4, d_model)
        )

        # 2. Project raw action dimensions to latent sequence tokens
        self.action_proj = nn.Linear(action_dim, d_model)

        # 3. Explicit positional embeddings for sequential action ordering
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.chunk_size, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # 4. Alternating Self-Attention and Cross-Attention blocks
        num_layers = getattr(cfg, "n_action_layers", 4)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "self_attn": AdaLNBlock(
                    d_model, cfg.n_heads, is_cross=False
                ),
                "cross_attn": AdaLNBlock(
                    d_model, cfg.n_heads, is_cross=True
                )
            }))

        # 5. Output velocity mapping
        self.norm_out = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, action_dim)
        self.ctx_norm = nn.LayerNorm(d_model)
        self.cond_proj = nn.Linear(2*d_model, d_model)

        # Zero-initialize the final output to maintain an identity start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, noisy_actions: torch.Tensor, t: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        B, seq_len, _ = noisy_actions.shape

        if t.ndim > 1:
            t = t.squeeze(-1)

        ctx = self.ctx_norm(context_tokens)
        pooled = torch.cat([ctx[:, -1], ctx[:, :self.cfg.lq_size].mean(1)], -1)
        cond = self.time_mlp(t) + self.cond_proj(pooled)
        x = self.action_proj(noisy_actions) + self.pos_emb[:, :seq_len]
        # Iterate through DiT-style alternating attention cascades
        for layer in self.layers:
            x = layer["self_attn"](x, cond)
            x = layer["cross_attn"](x, cond, ctx)

        x = self.norm_out(x)
        return self.output_proj(x)


class ActionExpert(nn.Module):
    def __init__(self, cfg: VLAConfig, return_traj=False):
        super().__init__()
        self.vel = VelocityGenerator(cfg)
        self.cfg = cfg
        self.return_traj = return_traj

    def loss(
            self,
            actions: torch.Tensor,
            reasoning: torch.Tensor,
    ) -> torch.Tensor:
        B = actions.size(0)

        # Logit-Normal time sampling concentrates updates around t=0.5 where physics prediction is most difficult
        t_raw = torch.randn(B, device=actions.device)
        t = torch.sigmoid(t_raw)
        t_exp = t.view(B, 1, 1)

        x_0 = torch.randn_like(actions)
        x_t = (1.0 - t_exp) * x_0 + t_exp * actions
        target_v = actions - x_0

        v_pred = self.vel(x_t, t, reasoning)
        return nn.functional.mse_loss(v_pred, target_v)

    @torch.no_grad()
    def sample(
            self,
            reasoning: torch.Tensor,
    ) -> tuple | torch.Tensor:
        B = reasoning.size(0)
        dt = 1.0 / self.cfg.flow_steps
        x_t = torch.randn(B, self.cfg.chunk_size, self.cfg.action_dim, device=reasoning.device)
        trajectory = [x_t] if self.return_traj else []

        for step in range(self.cfg.flow_steps):
            t = torch.full((B,), step * dt, device=reasoning.device)
            v = self.vel(x_t, t, reasoning)
            x_t = x_t + v * dt
            if self.return_traj:
                trajectory.append(x_t.clone())

        if self.return_traj:
            return x_t, torch.stack(trajectory, dim=0)
        return x_t

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)