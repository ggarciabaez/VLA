import torch
import torch.nn as nn
from model.utils import VLAConfig
from model.mha_impl import MultiHeadAttention

class LatentFusionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(
            d_model, n_heads, dropout=dropout, is_cross=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, dropout=dropout
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, latents: torch.Tensor, memory: torch.Tensor, memory_pad_mask: torch.Tensor | None = None):
        # Cross attend learned latents with memory
        latents = latents + self.drop(self.cross_attn(self.norm1(latents), memory, memory, attn_mask=memory_pad_mask))
        # Self-attend learned latents
        latents = latents + self.drop(self.self_attn(self.norm2(latents)))
        # FFN
        latents = latents + self.ffn(self.norm3(latents))
        return latents


class FusionTransformer(nn.Module):  # TODO: consider adding memory of sorts
    """
    Lightweight latent-bottleneck fusion:
      1) Build multimodal memory from image/text/state tokens
      2) A small set of learned latent tokens cross-attend into memory
      3) Action head consumes only latent tokens (bounded context length)
    """

    def __init__(self, cfg: VLAConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.embedding = nn.Embedding(len(cfg.type_ids), cfg.d_model, device=device)
        self.register_buffer("iembed_vec", torch.tensor(cfg.type_ids["vision"], device=device))
        self.register_buffer("tembed_vec", torch.tensor(cfg.type_ids["text"], device=device))
        self.register_buffer("sembed_vec", torch.tensor(cfg.type_ids["state"], device=device))

        self.latents = nn.Parameter(
            torch.randn(1, cfg.latent_size, cfg.d_model) * (cfg.d_model ** -0.5)
        )
        self.layers = nn.ModuleList(
            [LatentFusionLayer(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        state: torch.Tensor,
        txt_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        img = img + self.embedding(self.iembed_vec)
        txt = txt + self.embedding(self.tembed_vec)
        state = state + self.embedding(self.sembed_vec)

        # Memory tokens can be long, but latent count stays fixed.
        memory = torch.cat([img, txt, state], dim=1)
        B = memory.size(0)

        memory_pad_mask = None
        if txt_pad_mask is not None:
            img_mask = torch.ones(B, img.size(1), dtype=torch.bool, device=memory.device)
            state_mask = torch.ones(B, state.size(1), dtype=torch.bool, device=memory.device)
            memory_pad_mask = torch.cat([img_mask, txt_pad_mask, state_mask], dim=-1)[:, None, None, :]

        latents = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, memory, memory_pad_mask=memory_pad_mask)

        return self.norm(latents)

if __name__ == "__main__":
    import torch
    from model.utils import VLAConfig
    cfg = VLAConfig()
    model = FusionTransformer(cfg, torch.device("cuda")).to(device=torch.device("cuda"))
    model.eval()
    txt_tensor = torch.randn(1, 10, cfg.d_model, device=torch.device("cuda"))
    img_tensor = torch.randn(1, 24, cfg.d_model, device=torch.device("cuda"))
    state_tensor = torch.randn(1, 39, cfg.d_model, device=torch.device("cuda"))
    out = model(img_tensor, txt_tensor, state_tensor)
    print(out.shape)