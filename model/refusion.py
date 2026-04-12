import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import VLAConfig
torch.backends.cuda.enable_flash_sdp(True)


class MultiHeadAttention(nn.Module):
    """
    SDPA-based MHA. Uses packed projection for self-attention (Q=K=V),
    separate projections for cross-attention.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, is_cross: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.is_cross = is_cross

        if is_cross:
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
        else:
            self.packed_proj = nn.Linear(d_model, d_model * 3)

        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, n_heads, T, d_head)
        return x.unflatten(-1, [self.n_heads, self.d_head]).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor = None,
                value: torch.Tensor = None, attn_mask: torch.Tensor = None) -> torch.Tensor:
        if self.is_cross:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        else:
            q, k, v = torch.chunk(self.packed_proj(query), 3, dim=-1)

        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # (B, n_heads, T, d_head) -> (B, T, d_model)
        return self.out_proj(out.transpose(1, 2).flatten(-2))


class FusionLayer(nn.Module):
    """
    Single fusion layer (Pre-LN throughout):
      1. Self-attention on text tokens
      2. Cross-attention: text queries -> vision+state KV
      3. FFN
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1    = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, is_cross=False)

        self.norm2      = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, is_cross=True)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, txt: torch.Tensor, kv: torch.Tensor,
                self_attn_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Self-attention on text
        x = txt + self.drop(self.self_attn(self.norm1(txt), attn_mask=self_attn_mask))
        # 2. Cross-attention: text reads from vision + state
        x = x   + self.drop(self.cross_attn(self.norm2(x), kv, kv))
        # 3. FFN
        x = x   + self.ffn(self.norm3(x))
        return x


class FusionTransformer(nn.Module):
    def __init__(self, cfg: VLAConfig, device: torch.device):
        super().__init__()
        self.cfg    = cfg
        self.device = device

        self.layers = nn.ModuleList([
            FusionLayer(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

        self.embedding  = nn.Embedding(len(cfg.type_ids), cfg.d_model, device=device)
        self.iembed     = torch.tensor(cfg.type_ids["vision"], device=device)
        self.tembed     = torch.tensor(cfg.type_ids["text"],   device=device)
        self.sembed     = torch.tensor(cfg.type_ids["state"],  device=device)
        self.register_buffer("iembed_vec", self.embedding(self.iembed))
        self.register_buffer("tembed_vec", self.embedding(self.tembed))
        self.register_buffer("sembed_vec", self.embedding(self.sembed))

    def _build_self_attn_mask(self, pad_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Convert (B, T) boolean padding mask to (B, 1, T, T) additive mask for SDPA.
        Masked positions get -inf so they're zeroed out after softmax.
        """
        B, T = pad_mask.shape
        mask = torch.zeros(B, 1, T, T, dtype=dtype, device=pad_mask.device)
        # mask padding keys (columns) for all queries
        mask.masked_fill_(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        return mask

    def _fast_self_attn_mask(self, pad_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return pad_mask[:, None, None, :]

    def forward(self, img: torch.Tensor, txt: torch.Tensor, state: torch.Tensor,
                txt_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param img:          (B, P_img, d_model)
        :param txt:          (B, T, d_model)
        :param state:        (B, 1, d_model)
        :param txt_pad_mask: (B, T) bool, True = padding token
        :return:             (B, T, d_model)
        """
        img = img + self.iembed_vec
        txt = txt + self.tembed_vec
        state = state + self.sembed_vec

        # KV memory: everything text should read from
        kv = torch.cat([img, state], dim=1)  # (B, P_img + 1, d_model)

        self_attn_mask = (
            self._fast_self_attn_mask(txt_pad_mask, txt.dtype)
            if txt_pad_mask is not None else None
        )

        x = txt
        for layer in self.layers:
            x = layer(x, kv, self_attn_mask)

        return torch.cat([self.norm(x), kv], dim=1)  # (B, T+P+1, d_model)