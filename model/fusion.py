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
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, lq: torch.Tensor, context: torch.Tensor, context_pad_mask: torch.Tensor | None = None):
        """
        Forward pass of a single fusion layer.
        :param lq: The learned queries of shape (B, L, d_model)
        :param context: A tensor of shape (B, N_ctx, d_model) containing the context (image, text, memory) all with type embeddings (and memory with position embeddings)
        :param context_pad_mask: The padding mask for the context tensor, of shape (B, N_ctx) to mask out padding and blank memory
        :return: Attended latent representations of shape (B, L, d_model)
        """
        x = self.norm1(lq)
        x = x + self.self_attn(x)

        x = self.norm2(x)
        x = x + self.cross_attn(x, context, context, context_pad_mask)
        x = x + self.ffn(self.norm3(x))
        return x


class QFormer(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.layers = nn.ModuleList([LatentFusionLayer(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        self.out_norm = nn.LayerNorm(cfg.d_model)

        self.lq = nn.Parameter(torch.empty(1, cfg.lq_size, cfg.d_model))  # TODO: bs 1?
        nn.init.trunc_normal_(self.lq, std=0.02)

        self.type_embed = nn.Embedding(3, cfg.d_model)  # img=0, txt=1, mem=2
        self.mem_pos_embed = nn.Embedding(cfg.mem_len, cfg.d_model)  # position embedding for memory.
        # Memory is arranged left->right :: older->newer

    def _embed_context(self, img, txt, mem):
        B, T, _ = mem.shape
        device = img.device
        img = img + self.type_embed(torch.tensor(0, device=device))
        txt = txt + self.type_embed(torch.tensor(1, device=device))

        recency = torch.arange(T-1, -1, -1, device=device)  # the recency value indicates memory age
        mem = mem + self.type_embed(torch.tensor(2, device=device))
        mem = mem + self.mem_pos_embed(recency).unsqueeze(0)
        return torch.cat([img, txt, mem], dim=1)

    def forward(self, img: torch.Tensor, txt: torch.Tensor, mem: torch.Tensor, txt_mask: torch.Tensor | None = None, mem_mask: torch.Tensor | None = None):
        """

        For masks, bool tensors should have True if that element should participate in attention.
        :param img: (B, N_patches, d_model)
        :param txt: (B, N_tokens, d_model)
        :param mem: (B, N_memories (cfg), d_model)
        :param txt_mask: (B, N_tokens) mask for pad tokens.
        :param mem_mask: (B, N_memories) mask for empty memory.
        :return: (B, lq_size, d_model)
        """
        context = self._embed_context(img, txt, mem)
        if txt_mask is None and mem_mask is None:
            mask = None  # no masks on any, for some reason.
        else:
            if txt_mask is None:
                txt_mask = torch.ones(txt.shape[0], txt.shape[1], dtype=torch.bool, device=txt.device)
            if mem_mask is None:
                mem_mask = torch.ones(mem.shape[0], mem.shape[1], dtype=torch.bool, device=mem.device)
            img_mask = torch.ones(img.shape[0], img.shape[1], dtype=torch.bool, device=img.device)
            mask = torch.cat([img_mask, txt_mask, mem_mask], dim=1)[:, None, None, :]  # noqa

        lq = self.lq.expand(context.shape[0], -1, -1)

        for layer in self.layers:
            lq = layer(lq, context, mask)
        return self.out_norm(lq)


if __name__ == "__main__":
    B = 4
    d_model = 768
    n_heads = 6
    n_layers = 4
    n_queries = 64
    mem_len = 10

    cfg = VLAConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, lq_size=n_queries, mem_len=mem_len)
    model = QFormer(cfg)

    image = torch.randn(B, 196, d_model)
    text = torch.randn(B, 64, d_model)
    mem = torch.randn(B, mem_len, d_model)
    text_pad_mask = torch.zeros(B, 64, dtype=torch.bool)
    mem_pad_mask = torch.zeros(B, mem_len, dtype=torch.bool)

    # Episode t=3: first 3 slots filled, rest empty
    mem_pad_mask[:, 3:] = True

    reasoning = model(image, text, mem, text_pad_mask, mem_pad_mask)

    print(f"image          : {image.shape}")
    print(f"text           : {text.shape}")
    print(f"mem            : {mem.shape}  (slots 3-9 masked)")
    print(f"context        : (B, {196 + 64 + mem_len}, {d_model})")
    print(f"reasoning      : {reasoning.shape}")

    assert reasoning.shape == (B, n_queries, d_model)
    print("Shape check passed.")