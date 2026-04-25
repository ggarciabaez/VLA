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

class FiLMLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, d_film: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.film = nn.Linear(d_film, out_dim*2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.linear(x))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        return x + self.act((1 + gamma) * h + beta)  # Now it's residual

class FiLMActionExpert(nn.Module):
    def __init__(
        self,
        cfg: VLAConfig
    ):
        super().__init__()
        self.cfg = cfg

        self.state_enc = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model)
        )
        self.action_proj = nn.Linear(cfg.action_dim, cfg.d_model)
        self.action_deproj = nn.Linear(cfg.d_model, cfg.action_dim)
        nn.init.normal_(self.action_deproj.weight, std=1e-3)
        nn.init.zeros_(self.action_deproj.bias)

        self.mem_latent = nn.Parameter(torch.randn(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mem_latent, std=0.02)

        self.mem_cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, is_cross=True)
        self.norm_q = nn.LayerNorm(cfg.d_model)
        self.norm_kv = nn.LayerNorm(cfg.d_model)

        self.film_proj = nn.Linear(cfg.d_model*2, cfg.d_model)  # let's just make d_film=cfg.d_model
        self.t_embed = SinusoidalTimeEmbedding(cfg.d_model)  # encode the whole chunk with the current timestep.
        self.pos_embed = nn.Embedding(cfg.chunk_size, cfg.d_model)
        self.film_layers = nn.ModuleList([FiLMLayer(cfg.d_model, cfg.d_model, cfg.d_model) for _ in range(cfg.film_layers)])

    def generate_mem(self, reasoning, state):
        B = reasoning.size(0)
        state_enc = self.state_enc(state).unsqueeze(1)
        kv = torch.cat([reasoning, state_enc], dim=1)

        q = self.norm_q(self.mem_latent.expand(B, -1, -1))
        kv = self.norm_kv(kv)
        mem = self.mem_cross_attn(q, kv, kv)

        return mem

    def generate_velocity_field(self, chunk, t, mem):
        B, C, _ = chunk.shape
        pos = torch.arange(C, device=chunk.device)
        x = self.action_proj(chunk) + self.pos_embed(pos)

        x_flat = x.reshape(B * C, -1)  # (B*C, d_model)
        cond = self.film_proj(torch.cat([mem.squeeze(1), self.t_embed(t)], dim=-1))
        cond = cond.unsqueeze(1).expand(-1, C, -1).reshape(B * C, -1)

        for layer in self.film_layers:
            x_flat = layer(x_flat, cond)

        return self.action_deproj(x_flat).reshape(B, C, -1)  # (B, C, action_dim)

    def loss(self, actions, state, reasoning, return_mem=False):
        mem = self.generate_mem(reasoning, state)
        B = actions.size(0)

        t = torch.rand(B, device=actions.device)
        t_exp = t.reshape(B, 1, 1)  # for c, action_dim

        x_0 = torch.randn_like(actions)
        x_t = (1.0 - t_exp) * x_0 + t_exp * actions
        target_v = actions - x_0

        v_pred = self.generate_velocity_field(x_t, t, mem)
        l = torch.functional.F.mse_loss(v_pred, target_v)
        if return_mem:
            return l, mem
        return l

    @torch.no_grad()
    def sample(self, reasoning: torch.Tensor, state: torch.Tensor, return_trajectory: bool = False):
        B = reasoning.size(0)
        mem = self.generate_mem(reasoning, state)

        dt = 1.0 / self.cfg.flow_steps
        x_t = torch.randn(B, self.cfg.chunk_size, self.cfg.action_dim, device=reasoning.device)
        outputs = [x_t]

        for step in range(self.cfg.flow_steps):
            t = torch.full((B,), step * dt, device=reasoning.device)
            v = self.generate_velocity_field(x_t, t, mem)
            x_t = x_t + v * dt
            if return_trajectory:
                outputs.append(x_t.clone())
        if return_trajectory:
            return x_t, mem, torch.stack(outputs).to(reasoning.device)  # (B, C, action_dim)
        return x_t, mem

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

# -------------------The old (new) method------------

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
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        # cross-attention — action tokens query context tokens
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, is_cross=True)

        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
            self,
            x: torch.Tensor,  # (B, C, d_model)     — action tokens
            context: torch.Tensor,  # (B, N_ctx, d_model) — fusion transformer output
    ) -> torch.Tensor:
        # 1. self-attention over action chunk (Pre-LN)
        x = x + self.self_attn(self.norm1(x))

        # 2. cross-attention: action tokens query context tokens (Pre-LN)
        #    Q = action tokens, K = V = context tokens
        #    do NOT swap these — action tokens are asking questions,
        #    context tokens hold the answers
        x = x + self.cross_attn(self.norm2(x), context, context)

        # 3. FFN (Pre-LN)
        x = x + self.ffn(self.norm3(x))

        return x  # (B, C, d_model)


class ActionExpert(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.cfg = cfg
        self.chunk_size = cfg.chunk_size

        self.time_emb = SinusoidalTimeEmbedding(cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.chunk_size, cfg.d_model)

        self.layers = nn.ModuleList([
            ActionExpertLayer(cfg.d_model, cfg.n_heads // 2, 4 * cfg.d_model, cfg.dropout)
            for _ in range(cfg.n_layers // 2)
        ])

        self.state_enc = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        self.norm_out = nn.LayerNorm(cfg.d_model)
        self.action_proj = nn.Linear(cfg.action_dim, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.action_dim)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

        # MEM generation — learned query cross-attends reasoning + state
        self.mem_latent = nn.Parameter(torch.empty(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.mem_latent, std=0.02)

        self.mem_cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, is_cross=True)
        self.norm_q = nn.LayerNorm(cfg.d_model)
        self.norm_kv = nn.LayerNorm(cfg.d_model)

    # ------------------------------------------------------------------
    # MEM generation
    # ------------------------------------------------------------------

    def generate_mem(
            self,
            reasoning: torch.Tensor,  # (B, n_queries, d_model)
            state: torch.Tensor,  # (B, state_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = reasoning.size(0)
        state_enc = self.state_enc(state).unsqueeze(1)  # (B, 1, d_model)
        kv = torch.cat([reasoning, state_enc], dim=1)  # (B, n_queries+1, d_model)

        q = self.norm_q(self.mem_latent.expand(B, -1, -1))  # (B, 1, d_model)
        mem = self.mem_cross_attn(q, self.norm_kv(kv), self.norm_kv(kv))  # (B, 1, d_model)
        return mem, kv  # kv is reused as context for generate_velocity_field

    # ------------------------------------------------------------------
    # Velocity field (flow matching denoiser)
    # ------------------------------------------------------------------

    def generate_velocity_field(
            self,
            x_t: torch.Tensor,  # (B, C, action_dim)
            t: torch.Tensor,  # (B,)
            context: torch.Tensor,  # (B, N_ctx, d_model)
    ) -> torch.Tensor:
        x = self.action_proj(x_t)  # (B, C, d_model)
        C = x.size(1)
        if C > self.chunk_size:
            raise ValueError(f"Chunk length {C} exceeds configured chunk_size={self.chunk_size}")

        pos = torch.arange(C, device=x.device)
        x = x + self.pos_emb(pos).unsqueeze(0)  # chunk positional encoding
        x = x + self.time_emb(t).unsqueeze(1)  # broadcast t across chunk steps

        for layer in self.layers:
            x = layer(x, context)

        return self.out_proj(self.norm_out(x))  # (B, C, action_dim)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def loss(
            self,
            actions: torch.Tensor,  # (B, C, action_dim) — ground truth
            state: torch.Tensor,  # (B, state_dim)
            reasoning: torch.Tensor,  # (B, n_queries, d_model)
            return_mem: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mem, kv = self.generate_mem(reasoning, state)
        B = actions.size(0)

        t = torch.rand(B, device=actions.device)
        t_exp = t.reshape(B, 1, 1)

        x_0 = torch.randn_like(actions)
        x_t = (1.0 - t_exp) * x_0 + t_exp * actions
        target_v = actions - x_0

        v_pred = self.generate_velocity_field(x_t, t, kv)
        l = nn.functional.mse_loss(v_pred, target_v)

        return (l, mem) if return_mem else l

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
            self,
            reasoning: torch.Tensor,
            state: torch.Tensor,
            return_trajectory: bool = False,
    ) -> tuple:
        B = reasoning.size(0)
        mem, kv = self.generate_mem(reasoning, state)

        dt = 1.0 / self.cfg.flow_steps
        x_t = torch.randn(B, self.cfg.chunk_size, self.cfg.action_dim, device=reasoning.device)
        trajectory = [x_t]

        for step in range(self.cfg.flow_steps):
            t = torch.full((B,), step * dt, device=reasoning.device)
            v = self.generate_velocity_field(x_t, t, kv)
            x_t = x_t + v * dt
            if return_trajectory:
                trajectory.append(x_t.clone())

        if return_trajectory:
            return x_t, mem, torch.stack(trajectory, dim=0)  # (steps+1, B, C, action_dim)
        return x_t, mem

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

if __name__ == '__main__':
    cfg = VLAConfig()
    ae = ActionExpert(cfg)
    chunk, mem = ae.sample(torch.randn(1, cfg.lq_size, cfg.d_model), torch.randn(1, cfg.state_dim))
    print(chunk.shape, mem.shape)