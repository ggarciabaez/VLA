from torch import nn
import torch, math
from model.mha_impl import MultiHeadAttention
from model.utils import VLAConfig  # full coverage here

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
    def __init__(self, channels, time_embed_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

        # AdaLN projection for time
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_embed_dim, channels * 2)
        )

    def forward(self, x, t_embed):
        # x: (B, C, L)
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # AdaLN: time injection
        time_scale, time_shift = self.time_mlp(t_embed).unsqueeze(-1).chunk(2, dim=1)
        h = h * (time_scale + 1) + time_shift

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return x + h  # Residual connection


class VelocityGenerator(nn.Module):  # TODO: add better configuration / parameters (things like n_conv)
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        action_dim = cfg.action_dim

        # 1. Embed the flow timestep t
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        # 2. Input projection (Action chunk -> Latent channels)
        # Conv1d expects (Batch, Channels, Length), so we will transpose the action chunk
        self.input_proj = nn.Conv1d(action_dim, d_model, kernel_size=3, padding=1)

        # 3. Down-sampling / Feature Extraction (Fast 1D CNNs)
        self.down_blocks = nn.ModuleList([
            Conv1DBlock(d_model, d_model),
            Conv1DBlock(d_model, d_model)
        ])

        # 4. The Semantic Bottleneck (The only expensive part)
        # We project the QFormer context into the velocity space
        self.context_proj = nn.Linear(d_model, d_model)
        self.bottleneck_attn = MultiHeadAttention(d_model, 4, dropout=cfg.dropout, is_cross=True)
        self.bottleneck_norm = nn.LayerNorm(d_model)

        # 5. Up-sampling / Decoding
        self.up_blocks = nn.ModuleList([
            Conv1DBlock(d_model, d_model),
            Conv1DBlock(d_model, d_model)
        ])

        # 6. Output projection to Velocity Field
        self.output_proj = nn.Conv1d(d_model, action_dim, kernel_size=3, padding=1)

    def forward(self, noisy_actions, t, context_tokens):
        """
        noisy_actions:  (B, seq_len, action_dim)
        t:              (B, 1) - Flow matching time
        context_tokens: (B, num_queries+1, d_model), concat the state token to the context tokens
        """
        # Embed time
        t_embed = self.time_mlp(t)  # (B, d_model)

        # Transpose actions for Conv1D: (B, action_dim, seq_len)
        x = noisy_actions.transpose(1, 2)
        x = self.input_proj(x)

        # Pass through Conv Blocks (Fast temporal feature extraction)
        for block in self.down_blocks:
            x = block(x, t_embed)

        # --- BOTTLENECK CROSS-ATTENTION ---
        # Transpose back to sequence format for attention: (B, seq_len, d_model)
        x_seq = x.transpose(1, 2)
        ctx = self.context_proj(context_tokens)

        # Action chunk queries the Context tokens
        attn_out = self.bottleneck_attn(self.bottleneck_norm(x_seq), ctx, ctx)
        x_seq = x_seq + attn_out

        # Transpose back for Conv1D: (B, d_model, seq_len)
        x = x_seq.transpose(1, 2)
        # ----------------------------------

        # Pass through Up Blocks
        for block in self.up_blocks:
            x = block(x, t_embed)

        # Output Velocity Field
        v_t = self.output_proj(x)

        # Transpose back to original shape: (B, seq_len, action_dim)
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
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    ) -> tuple:
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
    chunk = ae.sample(torch.randn(1, cfg.lq_size+1, cfg.d_model))
    print(chunk, chunk.shape)