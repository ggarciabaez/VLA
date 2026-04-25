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

    def oldforward(self, lq: torch.Tensor, context: torch.Tensor,
                context_pad_mask: torch.Tensor | None = None,
                return_weights: bool = False):
        x = self.norm1(lq)
        x = x + self.self_attn(x)
        x = self.norm2(x)

        if return_weights:
            cross_out, attn_weights = self.cross_attn(
                x, context, context, context_pad_mask, return_weights=True
            )
            x = x + cross_out
            x = x + self.ffn(self.norm3(x))
            return x, attn_weights
        else:
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

    def oldforward(self, img: torch.Tensor, txt: torch.Tensor, mem: torch.Tensor,
                txt_mask: torch.Tensor | None = None, mem_mask: torch.Tensor | None = None,
                return_weights: bool = False):

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

        all_layer_weights = []

        for layer in self.layers:
            if return_weights:
                lq, weights = layer(lq, context, mask, return_weights=True)
                all_layer_weights.append(weights)
            else:
                lq = layer(lq, context, mask)

        out = self.out_norm(lq)

        if return_weights:
            # Stack weights: (n_layers, B, n_heads, lq_size, N_ctx)
            return out, torch.stack(all_layer_weights)
        return out

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

    reasoning, weights = model(image, text, mem, text_pad_mask, mem_pad_mask, return_weights=True)

    print(f"image          : {image.shape}")
    print(f"text           : {text.shape}")
    print(f"mem            : {mem.shape}  (slots 3-9 masked)")
    print(f"context        : (B, {196 + 64 + mem_len}, {d_model})")
    print(f"reasoning      : {reasoning.shape}")
    print(f"weights        : {weights.shape}")
    assert reasoning.shape == (B, n_queries, d_model)
    print("Shape check passed.")

    import math, numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns


    def plot_qformer_attention(
            attn_weights: torch.Tensor,
            batch_idx: int = 0,
            layer_idx: int = -1,
            query_idx: int = 0,
            img_seq_len: int = 196,
            txt_seq_len: int = 64,
            mem_seq_len: int = 10
    ):
        """
        Slices and plots the cross-attention heatmap for a specific learned query.

        :param attn_weights: Tensor of shape (n_layers, B, n_heads, lq_size, N_ctx)
        :param batch_idx: Which batch item to visualize
        :param layer_idx: Which QFormer layer to visualize (default: -1, the last layer)
        :param query_idx: Which of the 64 learned queries to visualize
        """
        # 1. Select the specific layer, batch, and query
        # Shape becomes: (n_heads, N_ctx)
        query_weights = attn_weights[layer_idx, batch_idx, :, query_idx, :]

        # 2. Average across all attention heads to get the consensus view
        # Shape becomes: (N_ctx,)
        avg_weights = query_weights.mean(dim=0).detach().cpu().numpy()

        # 3. Slice the context into its respective modalities
        idx_img_end = img_seq_len
        idx_txt_end = idx_img_end + txt_seq_len

        img_attn = avg_weights[:idx_img_end]
        txt_attn = avg_weights[idx_img_end:idx_txt_end]
        mem_attn = avg_weights[idx_txt_end:]

        # 4. Reshape Image Attention to 2D grid
        # Assuming square aspect ratio (e.g., 196 patches -> 14x14)
        grid_size = int(math.sqrt(img_seq_len))
        assert grid_size * grid_size == img_seq_len, "Image sequence length must be a perfect square for 2D reshaping."
        img_attn_2d = img_attn.reshape(grid_size, grid_size)

        # --- Plotting ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [1.5, 3, 1]})
        fig.suptitle(f'Cross-Attention Weights (Layer {layer_idx}, Query {query_idx})', fontsize=16)

        # Plot Image Attention
        sns.heatmap(img_attn_2d, ax=axes[0], cmap='viridis', cbar=True, square=True)
        axes[0].set_title('Image Patches Spatial Attention')
        axes[0].axis('off')

        # Plot Text Attention
        # Reshaping to 1D heatmap (1 x txt_seq_len) for better visibility
        sns.heatmap(txt_attn[np.newaxis, :], ax=axes[1], cmap='viridis', cbar=True, yticklabels=False)
        axes[1].set_title('Text Token Attention')
        axes[1].set_xlabel('Token Index')

        # Plot Memory Attention
        sns.heatmap(mem_attn[np.newaxis, :], ax=axes[2], cmap='viridis', cbar=True, yticklabels=False)
        axes[2].set_title('Memory Slot Attention')
        axes[2].set_xlabel('Memory Recency (Older -> Newer)')

        plt.tight_layout()
        plt.show()

    plot_qformer_attention(weights, batch_idx=0, layer_idx=0, query_idx=0)
    # --- Example Usage ---
    # reasoning, all_weights = model(image, text, mem, text_pad_mask, mem_pad_mask, return_weights=True)
    # plot_qformer_attention(all_weights, query_idx=0, img_seq_len=196, txt_seq_len=64, mem_seq_len=10)