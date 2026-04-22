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
        return self.act((1 + gamma) * h + beta)

class ActionExpert(nn.Module):
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

        self.film_proj = nn.Linear(cfg.d_model, cfg.d_model)  # let's just make d_film=cfg.d_model
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

        film_vec = self.film_proj(mem.squeeze(1))
        return mem, film_vec

    def generate_velocity_field(self, chunk, t, film_vec):
        B, C, _ = chunk.shape
        x = self.action_proj(chunk)  # (B, C, d_model)
        pos = torch.arange(C, device=x.device)
        x = x + self.pos_embed(pos)  # broadcast over B
        x = x + self.t_embed(t).unsqueeze(1)  # (B, 1, d_model)

        x_flat = x.reshape(B * C, -1)  # (B*C, d_model)
        cond = film_vec.unsqueeze(1).expand(-1, C, -1).reshape(B * C, -1)

        for layer in self.film_layers:
            x_flat = layer(x_flat, cond)

        return self.action_deproj(x_flat).reshape(B, C, -1)  # (B, C, action_dim)

    def loss(self, actions, state, reasoning, return_mem=False):
        mem, film_vec = self.generate_mem(reasoning, state)
        B = actions.size(0)

        t = torch.rand(B, device=actions.device)
        t_exp = t.reshape(B, 1, 1)  # for c, action_dim

        x_0 = torch.randn_like(actions)
        x_t = (1.0 - t_exp) * x_0 + t_exp * actions
        target_v = actions - x_0

        v_pred = self.generate_velocity_field(x_t, t, film_vec)
        l = torch.functional.F.mse_loss(v_pred, target_v)
        if return_mem:
            return l, mem
        return l

    @torch.no_grad()
    def sample(self, reasoning: torch.Tensor, state: torch.Tensor, return_trajectory: bool = False):
        B = reasoning.size(0)
        mem, film_vec = self.generate_mem(reasoning, state)

        dt = 1.0 / self.cfg.flow_steps
        x_t = torch.randn(B, self.cfg.chunk_size, self.cfg.action_dim, device=reasoning.device)
        outputs = [x_t]

        for step in range(self.cfg.flow_steps):
            t = torch.full((B,), step * dt, device=reasoning.device)
            v = self.generate_velocity_field(x_t, t, film_vec)
            x_t = x_t + v * dt
            if return_trajectory:
                outputs.append(x_t.clone())
        if return_trajectory:
            return x_t, mem, torch.stack(outputs).to(reasoning.device)  # (B, C, action_dim)
        return x_t, mem

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

if __name__ == '__main__':
    ae = ActionExpert(768, 64, 4, 64, 10, 4, 8)
    print(ae.sample(torch.randn(1, 64, 768), torch.randn(1, 64)))