from torch import nn
import torch, math
from model.utils import VLAConfig
from model.mha_impl import TransformerBlock, MultiHeadAttention

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

class Conv1DBlock(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, ctx_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU(approximate='tanh')
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

        # FiLM projections for time and multimodal context
        self.time_film = nn.Sequential(
            nn.GELU(approximate='tanh'),
            nn.Linear(time_embed_dim, channels * 2)
        )
        self.ctx_film = nn.Sequential(
            nn.GELU(approximate='tanh'),
            nn.Linear(ctx_dim, channels * 2)
        )
        
        nn.init.zeros_(self.time_film[-1].weight)
        nn.init.zeros_(self.time_film[-1].bias)
        nn.init.normal_(self.ctx_film[-1].weight, std=1e-5)
        nn.init.zeros_(self.ctx_film[-1].bias)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, ctx_embed: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        # FiLM modulation
        time_scale, time_shift = self.time_film(t_embed).unsqueeze(-1).chunk(2, dim=1)
        ctx_scale, ctx_shift = self.ctx_film(ctx_embed).unsqueeze(-1).chunk(2, dim=1)

        h = h * (time_scale + 1) + time_shift
        h = h * (ctx_scale + 1) + ctx_shift

        return x + h  # Residual connection


class VelocityGenerator(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        action_dim = cfg.action_dim

        # 1. Embed the flow timestep t
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(approximate='tanh'),
            nn.Linear(d_model * 4, d_model)
        )

        # 2. Context projector for FiLM modulation
        self.ctx_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(approximate='tanh'),
            nn.Linear(d_model * 2, d_model)
        )

        # 3. Input projection (Action chunk -> Latent channels)
        self.input_proj = nn.Conv1d(action_dim, d_model, kernel_size=3, padding=1)

        # 4. Fast temporal 1D Conv Blocks with FiLM conditioning
        self.down_blocks = nn.ModuleList([
            Conv1DBlock(d_model, d_model, d_model),
            Conv1DBlock(d_model, d_model, d_model)
        ])
        self.mid_blocks = nn.ModuleList([
            Conv1DBlock(d_model, d_model, d_model),
            Conv1DBlock(d_model, d_model, d_model)
        ])
        self.up_blocks = nn.ModuleList([
            Conv1DBlock(d_model, d_model, d_model),
            Conv1DBlock(d_model, d_model, d_model)
        ])

        # 5. Output projection to Velocity Field
        self.output_proj = nn.Conv1d(d_model, action_dim, kernel_size=3, padding=1)

    def forward(self, noisy_actions: torch.Tensor, t: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        """
        noisy_actions:  (B, seq_len, action_dim)
        t:              (B,) or (B, 1) - Flow matching time
        context_tokens: (B, num_tokens, d_model)
        """
        if t.ndim > 1:
            t = t.squeeze(-1)

        # Embed time and pool context
        t_embed = self.time_mlp(t)                             # (B, d_model)
        ctx_embed = self.ctx_mlp(context_tokens.mean(dim=1))  # (B, d_model)

        # Transpose actions for Conv1D: (B, action_dim, seq_len)
        x = noisy_actions.transpose(1, 2)
        x = self.input_proj(x)

        # Pass through Conv Blocks with FiLM
        # Save features for skip connections
        skips = []
        for block in self.down_blocks:
            x = block(x, t_embed, ctx_embed)
            skips.append(x)

        for block in self.mid_blocks:
            x = block(x, t_embed, ctx_embed)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x + skip, t_embed, ctx_embed)  # Add or concatenate skip connection

        # Output Velocity Field
        v_t = self.output_proj(x)

        # Transpose back: (B, seq_len, action_dim)
        return v_t.transpose(1, 2)


class ActionExpert(nn.Module):
    def __init__(self, cfg: VLAConfig, return_traj=False):
        super().__init__()
        self.vel = VelocityGenerator(cfg)
        self.cfg = cfg
        self.return_traj = return_traj

    def loss(
            self,
            actions: torch.Tensor,  # (B, C, action_dim) — ground truth
            reasoning: torch.Tensor,  # (B, n_queries, d_model)
    ) -> torch.Tensor:
        B = actions.size(0)

        t = torch.rand(B, device=actions.device)
        t_exp = t.reshape(B, 1, 1)

        x_0 = torch.randn_like(actions)
        x_t = (1.0 - t_exp) * x_0 + t_exp * actions
        target_v = actions - x_0
        v_pred = self.vel(x_t, t, reasoning)
        return nn.functional.mse_loss(v_pred, target_v)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
            self,
            reasoning: torch.Tensor,
    ) -> tuple | torch.Tensor:
        B = reasoning.size(0)
        dt = 1.0 / self.cfg.flow_steps
        x_t = torch.randn(B, self.cfg.chunk_size, self.cfg.action_dim, device=reasoning.device)
        trajectory = [x_t]
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


if __name__ == '__main__':
    cfg = VLAConfig()
    ae = ActionExpert(cfg)
    reasoning = torch.randn(2, cfg.lq_size + 1, cfg.d_model)
    actions = torch.randn(2, cfg.chunk_size, cfg.action_dim)

    loss_val = ae.loss(actions, reasoning)
    sample_val = ae.sample(reasoning)

    print(f"loss: {loss_val.item():.4f}")
    print(f"sample shape : {sample_val.shape}")
    assert sample_val.shape == (2, cfg.chunk_size, cfg.action_dim)
    print("ActionExpert verification passed.")