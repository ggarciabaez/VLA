from torch import nn, Tensor, split
from torch.nn.functional import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_div=2, dropout: float = 0.0,
                 is_cross: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.n_kv_heads = n_heads//kv_div if kv_div is not None else n_heads
        assert self.n_heads % self.n_kv_heads == 0

        self.dropout = dropout
        self.is_cross = is_cross

        if is_cross:
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
            self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        else:
            self.packed_proj = nn.Linear(
                d_model,
                d_model + 2 * (self.n_kv_heads * self.d_head),
                bias=False
            )

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _reshape_heads(self, x: Tensor, num_heads: int) -> Tensor:
        B, T, _ = x.shape
        return x.view(B, T, num_heads, self.d_head).transpose(1, 2)

    def forward(self, query: Tensor, key: Tensor = None, value: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        B, T, _ = query.shape

        if self.is_cross:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        else:
            packed = self.packed_proj(query)
            q, k, v = split(
                packed,
                [self.n_heads * self.d_head, self.n_kv_heads * self.d_head, self.n_kv_heads * self.d_head],
                dim=-1
            )

        q = self._reshape_heads(q, self.n_heads)
        k = self._reshape_heads(k, self.n_kv_heads)
        v = self._reshape_heads(v, self.n_kv_heads)

        out = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            enable_gqa=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_div: int = 2, dropout: float = 0.0,
                 is_cross: bool = False, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, kv_div, dropout, is_cross)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(approximate='tanh'),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, key: Tensor = None, value: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        # Pre-LN pattern
        # Reminder that the encoder is the smaller structure on the left of the illustration, NOT the bigger decoder
        x = self.norm1(x)
        if self.attn.is_cross:  # doesn't really make sense if the model
            attn = self.attn(x, key, value, attn_mask)
        else:
            attn = self.attn(x, attn_mask=attn_mask)
        x = x + self.drop(attn)
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == "__main__":
    import torch
    q = torch.randn(1, 2, 8)
    kv = torch.ones(1, 3, 8)
    mha = MultiHeadAttention(8, 8, 2, is_cross=True)
    mask = torch.ones(1, 3, dtype=torch.bool, device=q.device)
    mask[:, :2] = False
    out = mha(q, kv, torch.ones(1, 3, 8), mask)[0]
    print(*q[0].numpy().tolist(), sep="\n")
    print(*out.detach().numpy().tolist(), sep="\n")
    print(*kv[0].numpy().tolist(), sep="\n")
