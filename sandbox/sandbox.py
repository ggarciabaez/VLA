import torch
import math

torch.manual_seed(0)

B, H, d = 1, 1, 4
N_txt, N_mem = 4, 4
L = S = N_txt + N_mem

q = torch.randn(B, H, L, d)
k = torch.randn(B, H, L, d)
v = torch.randn(B, H, L, d)

# ---- Your masks (True = keep, False = mask) ----
txt_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
mem_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)

mask_1d = torch.cat([txt_mask, mem_mask], dim=1)  # (B, S)

# Expand like your code
attn_mask = mask_1d[:, None, None, :]  # (B, 1, 1, S)

# --- Reproduce SDPA internals ---
scale_factor = 1 / math.sqrt(d)

attn_bias = torch.zeros(L, S)

# Apply mask EXACTLY like SDPA
attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
attn_bias = attn_bias.masked_fill(~attn_mask, float("-inf"))

# Compute attention
scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
scores = scores + attn_bias
attn_weights = torch.softmax(scores, dim=-1)

print("=== 1D mask (True = allowed) ===")
print(mask_1d[0].int())

print("\n=== Expanded mask applied to keys (1 = masked) ===")
print((~attn_mask.expand(B, H, L, S))[0, 0].int())

print("\n=== Attention weights ===")
print(attn_weights[0, 0])