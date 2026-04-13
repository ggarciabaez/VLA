import torch
import torch.nn as nn

from model.utils import VLAConfig


class LatentFusionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        latents: torch.Tensor,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm_cross(latents)
        h, _ = self.cross_attn(
            h, memory, memory, key_padding_mask=memory_pad_mask, need_weights=False
        )
        latents = latents + self.drop(h)

        h = self.norm_self(latents)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        latents = latents + self.drop(h)

        latents = latents + self.ffn(self.norm_ffn(latents))
        return latents


class FusionTransformer(nn.Module):
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
            torch.randn(1, cfg.fusion_latents, cfg.d_model) * (cfg.d_model ** -0.5)
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
            img_mask = torch.zeros(B, img.size(1), dtype=torch.bool, device=memory.device)
            state_mask = torch.zeros(B, state.size(1), dtype=torch.bool, device=memory.device)
            memory_pad_mask = torch.cat([img_mask, txt_pad_mask, state_mask], dim=1)

        latents = self.latents.expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, memory, memory_pad_mask=memory_pad_mask)

        return self.norm(latents)
