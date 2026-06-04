import torch
import torch.nn as nn
from model.utils import VLAConfig
from model.mha_impl import MultiHeadAttention, TransformerBlock

class QFormerLayer(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.self_attn = TransformerBlock(cfg.d_model, cfg.n_heads, 2, dropout=cfg.dropout)
        self.cross_attn = TransformerBlock(cfg.d_model, cfg.n_heads, 2, dropout=cfg.dropout, is_cross=True)

        self.txt_film = nn.Linear(cfg.d_model, cfg.d_model * 2)

    def forward(self, lq, txt, img, mask=None):
        # 1. Self attend the text and query tokens. Tokens are allowed to fully attend each other (no mask)
        qt = torch.cat([lq, txt], dim=1)
        qt = self.self_attn(qt, attn_mask=mask)
        q = qt[:, :lq.shape[1], :]
        t = qt[:, lq.shape[1]:, :]

        # Stamp text into queries BEFORE image cross-attention can overwrite it
        txt_pooled = t.mean(1, keepdim=True)  # (B, 1, d_model)
        scale, shift = self.txt_film(txt_pooled).chunk(2, dim=-1)
        q = q * (scale + 1) + shift

        q = self.cross_attn(q, img, img)
        return q, t


class QFormer(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.layers = nn.ModuleList([QFormerLayer(cfg) for _ in range(cfg.n_layers)])
        self.lq = nn.Parameter(torch.empty(1, cfg.lq_size, cfg.d_model))
        nn.init.trunc_normal_(self.lq, std=0.02)

    def forward(self, img, txt, mask=None):  # TODO: we could add memory here, let the tokens attend to past tokens
        B = txt.shape[0]
        lq = self.lq.expand(txt.shape[0], -1, -1)
        if mask is not None:
            mask = torch.cat([torch.ones(B, lq.shape[1], dtype=torch.bool, device=mask.device), mask], dim=1)[:, None, None, :]
        for layer in self.layers:
            lq, txt = layer(lq, txt, img, mask=mask)
        return lq


if __name__ == "__main__":
    B = 4
    d_model = 768
    n_heads = 6
    n_layers = 4
    n_queries = 64
    mem_len = 10

    cfg = VLAConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, lq_size=n_queries)
    model = QFormer(cfg)

    image = torch.randn(B, 196, d_model)
    text = torch.randn(B, 64, d_model)
    mem = torch.randn(B, mem_len, d_model)

    reasoning = model(image, text)

    print(f"image          : {image.shape}")
    print(f"text           : {text.shape}")
    print(f"mem            : {mem.shape}  (slots 3-9 masked)")
    print(f"context        : (B, {196 + 64 + mem_len}, {d_model})")
    print(f"reasoning      : {reasoning.shape}")
    assert reasoning.shape == (B, n_queries, d_model)
    print("Shape check passed.")
    print(torch.isnan(reasoning).any() or torch.isinf(reasoning).any())