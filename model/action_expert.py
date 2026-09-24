import torch
import math
from torch import nn
from model.utils import VLAConfig
from model.mha_impl import TransformerBlock


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
                "self_attn": TransformerBlock(
                    d_model, cfg.n_heads, 2, cfg.dropout, is_cross=False
                ),
                "cross_attn": TransformerBlock(
                    d_model, cfg.n_heads, 2, cfg.dropout, is_cross=True
                )
            }))

        # 5. Output velocity mapping
        self.norm_out = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, action_dim)

        # Zero-initialize the final output to maintain an identity start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, noisy_actions: torch.Tensor, t: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        B, seq_len, _ = noisy_actions.shape

        if t.ndim > 1:
            t = t.squeeze(-1)

        t_embed = self.time_mlp(t).unsqueeze(1)  # (B, 1, d_model)

        # Base token features: Actions + Time + Sequence Position
        x = self.action_proj(noisy_actions) + t_embed + self.pos_emb[:, :seq_len, :]

        # Iterate through DiT-style alternating attention cascades
        for layer in self.layers:
            x = layer["self_attn"](x)
            # Action tokens actively query unpooled multi-modal reasoning tokens
            x = layer["cross_attn"](x, key=context_tokens, value=context_tokens)

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