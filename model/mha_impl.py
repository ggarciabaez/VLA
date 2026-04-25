from torch import nn, Tensor, chunk, backends
from torch.nn.functional import scaled_dot_product_attention
import math
import torch
backends.cuda.enable_flash_sdp(True)

class MultiHeadAttention(nn.Module):
    """
    SDPA-based MHA. Uses packed projection for self-attention (Q=K=V),
    separate projections for cross-attention.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, is_cross: bool = False, gqa: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.is_cross = is_cross
        self.gqa = gqa

        if is_cross:
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
        else:
            self.packed_proj = nn.Linear(d_model, d_model * 3)

        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        # (B, T, d_model) -> (B, n_heads, T, d_head)
        return x.unflatten(-1, [self.n_heads, self.d_head]).transpose(1, 2)

    def forward(self, query: Tensor, key: Tensor = None,
                value: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        if self.is_cross:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        else:
            q, k, v = chunk(self.packed_proj(query), 3, dim=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        out = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=True  # self.gqa  # TODO: model won't export to onnx with gqa
        )

        # (B, n_heads, T, d_head) -> (B, T, d_model)
        return self.out_proj(out.transpose(1, 2).flatten(-2))

    def oldforward(self, query: Tensor, key: Tensor = None,
                value: Tensor = None, attn_mask: Tensor = None,
                return_weights: bool = False) -> Tensor:

        if self.is_cross:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        else:
            q, k, v = torch.chunk(self.packed_proj(query), 3, dim=-1)

        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        if return_weights:
            # --- MANUAL ATTENTION (For Visualization) ---
            # q, k, v shape: (B, n_heads, T, d_head)

            # 1. Calculate scores: Q * K^T / sqrt(d)
            scale = math.sqrt(self.d_head)
            # Transpose the last two dimensions of K
            scores = (q @ k.transpose(-2, -1)) / scale

            if attn_mask is not None:
                scores = scores + attn_mask

            # 2. Apply Softmax to get the weights
            attn_weights = torch.softmax(scores, dim=-1)

            # Apply dropout if training
            if self.training and self.dropout > 0.0:
                attn_weights = torch.dropout(attn_weights, p=self.dropout, train=self.training)

            # 3. Multiply weights by V
            out = attn_weights @ v

        else:
            # --- FUSED ATTENTION (For Speed/Training) ---
            attn_weights = None
            out = scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                enable_gqa=self.gqa
            )

        # (B, n_heads, T, d_head) -> (B, T, d_model)
        out = self.out_proj(out.transpose(1, 2).flatten(-2))

        if return_weights:
            return out, attn_weights
        return out

if __name__ == "__main__":
    import torch
    q = torch.randn(1, 64, 128)
    kv = torch.randn(1, 205, 128)
    mha = MultiHeadAttention(128, 8, is_cross=True)
    out, wt = mha(q, kv, kv, return_weights=True)
    print(out.shape, wt.shape)
