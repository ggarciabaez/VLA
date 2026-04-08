# MT50 VLA — Architecture & Data Flow

---

## Overview

The model is a Vision-Language-Action (VLA) policy that maps raw observations
(images, proprioceptive state, language instruction) to a chunk of future actions
via conditional flow matching. It is designed to be compact enough to run on a
Jetson Orin Nano at ~4Hz inference with 30Hz control, using a C=8 action chunk.

```
┌─────────────┐  ┌─────────────┐  ┌──────────────┐
│  Image(s)   │  │    Text     │  │    State     │
│ (H,224,224) │  │  (token IDs)│  │  (H, s_dim) │
└──────┬──────┘  └──────┬──────┘  └──────┬───────┘
       │                │                │
  SigLIP ViT      SigLIP Text       Linear +
  (partial FT)    (partial FT)      LayerNorm
       │                │                │
  patch tokens    text tokens       state tokens
  (H, P, d)       (T, d)            (H, 1, d)
       │                │                │
       └────────────────┴────────────────┘
                        │
              Token assembly + type/history embeddings
                        │
              ┌─────────▼──────────┐
              │  Fusion Transformer │
              │  4 layers, 6 heads  │
              │  d_model = 384      │
              └─────────┬──────────┘
                        │
                context tokens (B, N_ctx, 384)
                        │
              ┌─────────▼──────────┐    ← noisy action chunk (B, C, a_dim)
              │   Action Expert     │    ← time embedding t ∈ [0,1]
              │  Transformer, 4L   │
              │  self-attn + xattn │
              └─────────┬──────────┘
                        │
                velocity field v̂ (B, C, a_dim)
                        │
              Euler integration (10 steps)
                        │
                action chunk (B, C, a_dim)
```

---

## Components

### 1. Vision Encoder

**Module**: `HFSiglipViT` (SigLIP ViT-B/16)  
**Pretrained**: `google/siglip-base-patch16-224`  
**Trainable**: Last 2 encoder layers + projection. All earlier layers frozen.

Accepts a raw uint8 image, normalizes it (mean=0.5, std=0.5 per channel),
and runs it through the SigLIP vision transformer.

**Two output modes** (selected by `use_pooled`):

| Mode | Output shape | When to use |
|---|---|---|
| Pooled (CLS) | `(B, d_hidden)` | Faster, lower memory |
| Unpooled (patches) | `(B, 196, d_hidden)` | Richer spatial info |

`d_hidden = 768` for ViT-B/16. A linear projection maps this to `d_model = 384`.

**With history** (H frames): images are encoded independently then stacked.
```
(B, H, 3, 224, 224) → flatten to (B*H, 3, 224, 224) → encode → reshape to (B, H, [P,] d_model)
```

**With multiple cameras**: encode each camera independently, producing separate
token sequences. Each gets its own type embedding ID in the fusion transformer.
The sequence length grows but no architectural change is needed.

---

### 2. Text Encoder

**Module**: `SiglipTextModel`  
**Pretrained**: `google/siglip-base-patch16-224` (text tower)  
**Trainable**: Last 2 encoder layers + projection. All earlier layers frozen.

Accepts tokenized text (int64 token IDs + attention mask). Outputs either:
- Pooled: `(B, d_hidden)` — the CLS/pooler token
- Unpooled: `(B, T, d_hidden)` — full token sequence, where T varies with prompt length

**Variable token length**: T varies with the prompt. Use the unpooled output
and pass the attention mask through to the fusion transformer. Padding positions
are masked out in the attention computation.

A linear projection maps `d_hidden → d_model`. Text is encoded once per episode
step (not per chunk step).

---

### 3. State Encoder

**Module**: `StateEncoder`  
**Architecture**: `nn.Linear(state_dim, d_model)` + `nn.LayerNorm(d_model)`  
**Trainable**: Always fully trainable.

The proprioceptive state (joint positions, velocities, gripper state) is projected
directly to a single token per timestep. No activation — the transformer layers
provide all necessary nonlinearity.

```
(B, H, state_dim) → (B, H, d_model)     # H tokens, one per history frame
```

State tokens share the history embedding with image tokens, so the transformer
knows which timestep each state token corresponds to.

---

### 4. Token Assembly

Before entering the fusion transformer, all modality tokens are assembled into
a single sequence. Each token receives two additive embeddings:

**Type embedding** — learned `nn.Embedding(N_types, d_model)`, identifies the modality:

| ID | Modality |
|---|---|
| 0 | Image (camera 0) |
| 1 | Text |
| 2 | State |
| 3 | Image (camera 1) — if present |
| 4 | Image (wrist cam) — if present |
| ... | extensible |

**History embedding** — learned `nn.Embedding(H, d_model)`, identifies the timestep
within the observation window. Applied to image and state tokens only (broadcast
over patch tokens). Text tokens have no history embedding (same prompt all steps).

Final sequence (pooled mode, H=1, 1 camera):
```
[ img_token(t) | txt_tokens(1..T) | state_token(t) ]
  shape: (B, 1 + T + 1, d_model)
```

Final sequence (unpooled mode, H=2, 2 cameras):
```
[ img0_patches(t-1, 1..196) | img0_patches(t, 1..196)
| img1_patches(t-1, 1..196) | img1_patches(t, 1..196)
| txt_tokens(1..T)
| state(t-1) | state(t) ]
  shape: (B, 196*4 + T + 2, d_model)
```

**Padding for TensorRT**: at inference, pad all sequences to a fixed `max_tokens`
with a boolean padding mask. The transformer ignores padded positions. This keeps
the TensorRT engine shape static while allowing variable-length inputs at the
Python/logic level.

---

### 5. Fusion Transformer

**Architecture**: `nn.TransformerEncoder` with Pre-LN layers  
**Hyperparameters**: `d_model=384`, `n_layers=4`, `nhead=6`, `dim_feedforward=1536`, `dropout=0.05`  
**Trainable**: Always fully trainable.

Standard bidirectional transformer encoder. All tokens attend to all other tokens
(full attention), so image patches can directly attend to text tokens and vice versa.
This is where language grounding actually happens.

```
input:  (B, N_tokens, d_model)   + padding_mask (B, N_tokens)
output: (B, N_tokens, d_model)   ← full sequence, no pooling
```

The output sequence is passed directly to the action expert as context tokens.
There is no pooling step — the action expert consumes the full sequence via
cross-attention, letting it selectively attend to whichever tokens are relevant
for each action in the chunk.

---

### 6. Action Expert

**Architecture**: Small transformer with interleaved self-attention and cross-attention  
**Hyperparameters**: `d_model=384`, `n_layers=4`, GELU activations  
**Trainable**: Always fully trainable.

This is the flow matching velocity network. It takes:
- Noisy action chunk `x_t` of shape `(B, C, action_dim)` at interpolation time `t`
- Time embedding `t ∈ [0,1]` encoded as a sinusoidal vector, broadcast over the chunk
- Context tokens from the fusion transformer `(B, N_ctx, d_model)`

And predicts the velocity field `v̂(x_t, t | context)`.

**Per-layer structure**:
```
action_tokens = self_attention(action_tokens)          # chunk coherence
                                                       # adjacent actions coordinate
action_tokens = cross_attention(action_tokens,         # conditioning on observation
                                context_tokens)
action_tokens = ffn(action_tokens)
action_tokens = layer_norm(action_tokens)
```

Self-attention allows the model to enforce temporal consistency across the chunk
(e.g., smooth trajectories, coordinated joint motion). Cross-attention lets each
action token selectively query the full context — the reaching action attends to
the object position, the grasp action attends to the gripper state token, etc.

**Input/output projection**:
```
(B, C, action_dim) → Linear → (B, C, d_model)   # project up
+ time_emb (B, d_model) broadcast to (B, C, d_model)
→ N layers of self-attn + cross-attn + FFN
→ Linear → (B, C, action_dim)                    # project back down = velocity v̂
```

---

## Flow Matching

The action expert is the velocity network in a conditional flow matching framework.
Training uses a linear interpolation path between Gaussian noise and demonstrations.

### Training

```python
# actions: (B, C, action_dim) — normalized ground truth chunk
t     = torch.rand(B)                          # uniform in [0, 1]
t_exp = t.reshape(B, 1, 1)                    # broadcast over C, action_dim

x_0 = torch.randn_like(actions)               # source: Gaussian noise
x_t = (1 - t_exp) * x_0 + t_exp * actions    # interpolated point
v_target = actions - x_0                      # constant velocity field

v_pred = action_expert(x_t, t, context_tokens)
loss   = F.mse_loss(v_pred, v_target)
```

The model learns to predict the straight-line velocity from wherever `x_t` is
to the target action, conditioned on the current observation context.

### Inference (Euler integration, 10 steps)

```python
x_t = torch.randn(B, C, action_dim)           # start at noise
dt  = 1.0 / n_steps                           # = 0.1

for step in range(n_steps):                   # 10 steps
    t = torch.full((B,), step * dt)
    v = action_expert(x_t, t, context_tokens)
    x_t = x_t + v * dt

action_chunk = denormalize(x_t)               # (B, C, action_dim)
```

10 Euler steps is sufficient for this task distribution. 32 steps (MT10) was
conservative — the linear path means fewer steps are needed than in diffusion.

---

## Full Forward Pass — Data Flow

```
INPUTS (per observation step):
  image:    (1, H, 3, 224, 224)  uint8
  text_ids: (1, T)               int64    ← T varies with prompt
  attn_mask:(1, T)               int64
  state:    (1, H, state_dim)    float32  ← normalized

ENCODE:
  img_tokens  = siglip_vision(image)               → (1, H, [P,] d_model)
  txt_tokens  = siglip_text(text_ids, attn_mask)   → (1, T,      d_model)
  state_token = state_encoder(state)               → (1, H,      d_model)

ASSEMBLE:
  tokens = concat([img_tokens, txt_tokens, state_token], dim=1)
           + type_embeddings + history_embeddings
  → (1, N_tokens, d_model)

FUSE:
  context = fusion_transformer(tokens, padding_mask)
  → (1, N_tokens, d_model)

SAMPLE (flow matching, 10 Euler steps):
  x_t = randn(1, C, action_dim)
  for step in 0..9:
      v   = action_expert(x_t, t, context)
      x_t = x_t + v * dt
  → (1, C, action_dim)

DENORMALIZE:
  action_chunk = x_t * action_std + action_mean
  → (1, C, action_dim)

EXECUTE:
  for each of C steps: env.step(action_chunk[step])
  re-run inference before chunk exhausts
```

---

## Extending the Model

Because the fusion transformer is purely sequence-based, adding new input
modalities requires only:

1. An encoder that maps the new input to tokens of shape `(B, N_new, d_model)`
2. A new type embedding ID
3. Concatenating the new tokens into the assembly step
4. Updating `max_tokens` for the TensorRT padding budget

**Examples**:
- **Second camera**: encode with the same SigLIP encoder (shared or separate weights), type ID = 3
- **Wrist camera**: same, type ID = 4. Patch tokens give the action expert fine-grained spatial info for grasping
- **Force/torque**: linear projection to a single token per timestep, type ID = 5
- **Task phase / subtask**: embed a discrete phase index, type ID = 6 — useful for long-horizon MT50 tasks

No changes to the fusion transformer or action expert are needed in any of these cases.

---

## Parameter Count (estimates)

| Component | Parameters |
|---|---|
| SigLIP vision encoder (frozen) | ~86M |
| SigLIP vision encoder (unfrozen last 2L) | ~7M trainable |
| SigLIP text encoder (frozen) | ~35M |
| SigLIP text encoder (unfrozen last 2L) | ~3M trainable |
| Projection layers (img, txt, state) | ~0.4M |
| Fusion transformer (4L, d=384, h=6) | ~9M |
| Action expert (4L, d=384, xattn) | ~12M |
| **Total trainable** | **~31M** |
| **Total (incl. frozen)** | **~152M** |

At FP16, the full model is ~300MB — well within the Orin Nano's 8GB unified memory.

---

## Constants (set before training, fixed for TensorRT)

| Symbol | Value | Description |
|---|---|---|
| `d_model` | 384 | Shared model dimension |
| `H` | 1–3 | Observation history length |
| `C` | 8 | Action chunk size |
| `P` | 196 | Patches per image (14×14, ViT-B/16) |
| `max_tokens` | 512 | Padding target for TensorRT |
| `n_flow_steps` | 10 | Euler steps at inference |
| `action_dim` | 4 | MetaWorld: (x, y, z, gripper) |
| `state_dim` | varies | Task-dependent proprioception dim |