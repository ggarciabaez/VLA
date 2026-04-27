import torch
import torch.nn as nn
from model.utils import VLAConfig
from model.mha_impl import MultiHeadAttention

class QFormerLayer(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, is_cross=True)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.n1 = nn.LayerNorm(cfg.d_model)
        self.n2 = nn.LayerNorm(cfg.d_model)
        self.n3 = nn.LayerNorm(cfg.d_model)

    def forward(self, lq, txt, img, mask=None):
        # 1. Self attend the text and query tokens. Tokens are allowed to fully attend each other (no mask)
        qt = torch.cat([lq, txt], dim=1)
        qt = qt + self.self_attn.forward(self.n1(qt), attn_mask=mask)  # pre-LN pattern
        q = qt[:, :lq.shape[1], :]
        t = qt[:, lq.shape[1]:, :]

        # Now cross
        q = q + self.cross_attn(self.n2(q), img, img)
        # Return t so that the next layer can use it
        return q + self.ffn(self.n3(q)), t + self.ffn(self.n3(t))


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