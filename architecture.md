# VLA Architecture

## Overview

The current codebase implements a single-frame Vision-Language-Action model trained on MetaWorld MT50
demonstrations with flow matching.

The live model is centered on four pieces:

1. `VisionEncoder` wraps `SiglipVisionModel` and projects patch tokens into `d_model`.
2. `TextEncoder` wraps `SiglipTextModel`, projects token features into `d_model`, and returns a padding mask.
3. `QFormer` uses learned query tokens plus text self-attention, then cross-attends those queries into image tokens.
4. `ActionExpert` predicts a continuous action-chunk velocity field with a Conv1D + cross-attention flow model.

This is the architecture in the current Python modules under `model/`. The older memory-stack design described
previously is not present in the current implementation.

---

## Top-Level Data Flow

```text
image -----------------> VisionEncoder ----\
                                            \
text tokens -----------> TextEncoder ------- +--> QFormer --> reasoning queries
                                                                          |
state -----------------> MLP -------------------------------------------- + append state token
                                                                          |
                                                                     ActionExpert
                                                                          |
                                                           flow-matched action chunk
```

At training time, the model receives:

- `img`: `(B, 3, H, W)` uint8 or float image tensors
- `txt`: `(B, seq_len)` token ids
- `state`: `(B, state_dim)` robot state
- `action`: `(B, chunk_size, action_dim)` normalized target action chunks

`VLA.encode()` returns `torch.cat([reasoning_queries, state_token], dim=1)`, so the action model consumes
`lq_size + 1` context tokens.

---

## Module Breakdown

### 1. Vision Encoder

Implemented in `model/heads.py` as `VisionEncoder`.

- Backbone: `google/siglip2-base-patch16-224`
- Input normalization: `(x / 255 - 0.5) / 0.5` when the input is `uint8`
- Resize: bilinear resize to `cfg.img_size` if needed
- Output: `last_hidden_state` projected from SigLIP hidden size into `cfg.d_model`

Default training config in `trainer.py` sets:

- `d_model = 1024`
- `img_size = 224`
- `n_trainable = 2`

For a single image per sample, the output is effectively:

- image tokens: `(B, N_img, d_model)`

For SigLIP B/16 at 224x224 this is typically 196 patch tokens plus the model's sequence convention from the
backbone output.

### 2. Text Encoder

Implemented in `model/heads.py` as `TextEncoder`.

- Backbone: `SiglipTextModel`
- Tokenizer: `AutoTokenizer.from_pretrained(cfg.siglip_model_id)`
- Projection: SigLIP hidden size -> `cfg.d_model`
- Masking: returns `~torch.isin(tokens, special_ids)`

Outputs:

- text tokens: `(B, N_txt, d_model)`
- text mask: `(B, N_txt)` where `True` means "real token" and `False` means special/padding token

The trainer pre-tokenizes prompt variants once at dataset construction time and samples one prompt variant
per task batch.

### 3. State Encoder

Implemented directly in `model/vla.py`.

```python
nn.Sequential(
    nn.Linear(state_dim, d_model),
    nn.GELU(),
    nn.Linear(d_model, d_model),
)
```

Output:

- state token: `(B, d_model)` then unsqueezed to `(B, 1, d_model)`

The state token does not participate in the Q-Former. It is appended after fusion and consumed only by the
action model.

### 4. Q-Former

Implemented in `model/fusion.py`.

The current Q-Former is not BLIP-2 style full multimodal cross-attention over image, text, and memory.
Its actual pattern is:

1. Start from learned queries `self.lq` of shape `(1, lq_size, d_model)`.
2. Concatenate learned queries with text tokens.
3. Run self-attention over `[queries | text]`.
4. Split them back apart.
5. Run cross-attention from queries into image tokens only.
6. Apply the FFN and repeat for `n_layers`.

Per layer:

```text
qt = concat(learned_queries, text_tokens)
qt = qt + self_attn(LN(qt), mask=[queries valid][text mask])
q, t = split(qt)
q = q + cross_attn(LN(q), image_tokens, image_tokens)
q = q + ffn(LN(q))
t = t + ffn(LN(t))
```

Important details:

- Text participates in self-attention with the learned queries.
- Image tokens are only used as keys/values in cross-attention.
- The text stream is updated layer-by-layer and fed into the next layer.
- There is no episodic memory mechanism in the current `QFormer`.

Default trainer config:

- `n_layers = 8`
- `n_heads = 8`
- `lq_size = 64`
- `d_model = 1024`

Output:

- reasoning queries: `(B, lq_size, d_model)`

### 5. Attention Implementation

Implemented in `model/mha_impl.py` as `MultiHeadAttention`.

- Uses `torch.nn.functional.scaled_dot_product_attention`
- Uses packed QKV projection for self-attention
- Uses separate Q/K/V projections for cross-attention
- Enables Flash SDP on CUDA
- Keeps the mask in SDPA form rather than building explicit attention matrices

This module is shared by both the Q-Former and the action bottleneck attention.

### 6. Action Expert

Implemented in `model/action_expert.py`.

The current action head is not FiLM-conditioned on a memory token. Instead it is a flow-matching model with:

- time embedding MLP
- Conv1D temporal stack
- a semantic bottleneck where action-sequence features attend to fused context tokens
- Conv1D decoder back to action-space velocities

#### Velocity Generator

Inputs:

- `noisy_actions`: `(B, chunk_size, action_dim)`
- `t`: `(B,)`
- `context_tokens`: `(B, lq_size + 1, d_model)`

Pipeline:

1. `t` -> sinusoidal embedding -> MLP -> `(B, d_model)`
2. action chunk -> transpose to `(B, action_dim, chunk_size)` -> `Conv1d` input projection
3. two residual `Conv1DBlock`s with AdaLN-style time conditioning
4. transpose to sequence form and cross-attend action features into projected context tokens
5. two more residual `Conv1DBlock`s
6. output `Conv1d` -> velocity field `(B, chunk_size, action_dim)`

The bottleneck cross-attention is:

```text
queries = action-sequence latents
keys    = projected context tokens
values  = projected context tokens
```

So the action chunk queries the fused reasoning/state context.

#### Flow-Matching Objective

Training in `ActionExpert.loss()`:

```text
x0      ~ N(0, I)
t       ~ Uniform(0, 1)
xt      = (1 - t) * x0 + t * actions
target  = actions - x0
loss    = MSE(velocity_model(xt, t, context), target)
```

Inference in `ActionExpert.sample()`:

```text
x <- N(0, I)
repeat flow_steps times:
    t = step / flow_steps
    v = velocity_model(x, t, context)
    x = x + v * dt
return x
```

Default trainer config:

- `chunk_size = 32`
- `action_dim = 4`
- `flow_steps = 10`

---

## Training/Data Pipeline

The training entry point is `trainer.py`.

### Dataset Layout

`MT50Dataset` loads merged episode shards from:

- `data/dataset_shards/checkpoints`

Expected per-episode arrays:

- `images`: `(T, N, 3, H, W)`
- `states`: `(T, N, state_dim_raw)`
- `actions`: `(T, N, action_dim)`
- `chunk_indices`: `(T, chunk_size)`
- `task_names`: `(N,)`

At load time the trainer:

1. loads every shard into RAM
2. normalizes actions using `norm_stats.npz`
3. slices states down to `cfg["state_dim"]`
4. preindexes chunk targets with `actions_t[chunk_indices]`
5. tokenizes all prompt variants from `task_prompts.json`

The hot training loop then uses direct tensor slicing rather than a `DataLoader`.

### Episode-Level Training Loop

For each batch of task columns:

1. sample one prompt variant per task batch
2. iterate through all timesteps in the episode
3. slice current image/state/action chunk for that timestep
4. run `model.loss(img_t, txt, state_t, action_t)`
5. accumulate gradients for `optimizer_stride` timesteps
6. step AdamW with cosine decay

This means the current trainer behaves like sequential episode playback, but the live model itself is still
single-step and stateless.

---

## Current Config Surface

There are two sources of defaults:

- `model/utils.py::VLAConfig` defines model-level defaults
- `trainer.py::CFG` overrides them for training

The effective training architecture currently comes from `trainer.py`, not the dataclass defaults alone.

Key active trainer values:

| Key | Value |
|---|---:|
| `siglip_model_id` | `google/siglip2-base-patch16-224` |
| `d_model` | `1024` |
| `n_heads` | `8` |
| `n_layers` | `8` |
| `lq_size` | `64` |
| `state_dim` | `4` |
| `action_dim` | `4` |
| `chunk_size` | `32` |
| `flow_steps` | `10` |
| `film_layers` | `4` |
| `dropout` | `0.1` |
| `n_trainable` | `2` |

Note: `film_layers` remains in config, but the current `VelocityGenerator` does not build a FiLM MLP stack from it.

---

## Known Drift In Repo

Some files still reflect an older memory-based architecture and are not consistent with the current `model/`
implementation:

- the previous `architecture.md` described a memory stack and FiLM-conditioned MEM token path
- `trainer.py` still calls `model.reset()` and `update_memory=True`, but `model/vla.py` does not implement them
- `scripts/evaluate_model.py` also assumes memory/update APIs that are not present in the live model

So the source of truth for the current architecture is:

- `model/vla.py`
- `model/heads.py`
- `model/fusion.py`
- `model/action_expert.py`
- `model/mha_impl.py`

---

## Summary

The current project is a stateless, single-frame VLA with:

- SigLIP image and text backbones
- learned-query fusion where text interacts through self-attention and images through cross-attention
- a separate state token appended after fusion
- a Conv1D flow-matching action generator conditioned through bottleneck cross-attention

In the future, we will implement the HAMLET memory component for long-term memory.
