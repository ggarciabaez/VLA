import torch
import torch.nn as nn
from model.utils import VLAConfig
from model.mha_impl import TransformerBlock


class PerceiverResamplerLayer(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.cross_attn = TransformerBlock(cfg.d_model, cfg.n_heads, kv_div=2, dropout=cfg.dropout, is_cross=True)
        self.self_attn = TransformerBlock(cfg.d_model, cfg.n_heads, kv_div=2, dropout=cfg.dropout, is_cross=False)

    def forward(self, lq: torch.Tensor, context: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        lq = self.cross_attn(lq, context, context, attn_mask=mask)
        lq = self.self_attn(lq)
        return lq


class PerceiverResampler(nn.Module):
    def __init__(self, cfg: VLAConfig, n_layers: int = 2):
        super().__init__()
        # Text tokens query image tokens to allow text to highlight elements in the scene
        self.text_cross_attn = TransformerBlock(cfg.d_model, cfg.n_heads, kv_div=2, dropout=cfg.dropout, is_cross=True)

        # 2-layer Perceiver Resampler
        num_layers = n_layers if n_layers is not None else getattr(cfg, "n_layers", 2)
        self.layers = nn.ModuleList([PerceiverResamplerLayer(cfg) for _ in range(num_layers)])
        self.lq = nn.Parameter(torch.empty(1, cfg.lq_size, cfg.d_model))
        nn.init.trunc_normal_(self.lq, std=0.02)

    def forward(self, img: torch.Tensor, txt: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B = txt.shape[0]

        # 1. Text tokens query image tokens (text highlights visual scene elements)
        txt = self.text_cross_attn(txt, img, img)

        # 2. Multimodal context concatenation
        context = torch.cat([img, txt], dim=1)

        # 3. Attention mask handling (unmask image tokens, apply text mask if provided)
        ctx_mask = None
        if mask is not None:
            img_mask = torch.ones(B, img.shape[1], dtype=torch.bool, device=mask.device)
            ctx_mask = torch.cat([img_mask, mask], dim=1)[:, None, None, :]

        # 4. Latent queries resample multimodal context across perceiver layers
        lq = self.lq.expand(B, -1, -1)
        for layer in self.layers:
            lq = layer(lq, context, mask=ctx_mask)
        return lq


# Backward compatibility alias
QFormer = PerceiverResampler


if __name__ == "__main__":
    B = 4
    d_model = 768
    n_heads = 8
    n_layers = 2
    n_queries = 64

    cfg = VLAConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, lq_size=n_queries)
    model = PerceiverResampler(cfg, n_layers=n_layers)

    image = torch.randn(B, 196, d_model)
    text = torch.randn(B, 64, d_model)
    text_mask = torch.ones(B, 64, dtype=torch.bool)
    text_mask[:, 32:] = False

    reasoning = model(image, text, mask=text_mask)

    print(f"image     : {image.shape}")
    print(f"text      : {text.shape}")
    print(f"reasoning : {reasoning.shape}")
    assert reasoning.shape == (B, n_queries, d_model)
    assert not (torch.isnan(reasoning).any() or torch.isinf(reasoning).any())
    print("PerceiverResampler verification passed.")