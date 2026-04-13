# VLA Architecture (Current)

## Goal
A lightweight Vision-Language-Action policy for MetaWorld-style control that:
- keeps context computation bounded for edge deployment,
- preserves multimodal conditioning,
- predicts action chunks with conditional flow matching.

## High-Level Pipeline

```text
image(s), text tokens, state
  -> modality encoders (vision/text/state)
  -> latent-bottleneck fusion transformer
  -> flow-matching action expert
  -> action chunk (C x action_dim)
```

Model entrypoint: `model/vla.py` (`VLA` class).

---

## Inputs

### Runtime inference inputs
- `img`: `(B, 3, H, W)` or `(B, N_img, 3, H, W)`
- `txt`: `(B, T)` token IDs
- `state`: `(B, state_dim)` or `(B, 1, state_dim)`

### Training data layout (episode shards)
Each episode `.npz` stores:
- `images`: `(steps, tasks, ...)`
- `states`: `(steps, tasks, state_dim)`
- `actions`: `(steps, tasks, action_dim)`
- `chunk_indices`: `(steps, chunk_size)`
- `task_names`: `(tasks,)` (or legacy `task_name`)

Chunk targets are reconstructed as:
- `chunk = actions[chunk_indices[t], task_id]` -> `(chunk_size, action_dim)`

---

## Encoders

### Vision encoder (`model/heads/vision_encoder.py`)
- Backbone: `SiglipVisionModel` (`google/siglip-base-patch16-224`)
- Output projection: hidden size -> `d_model`
- Supports single or multiple images per sample.
- Normalization:
  - `uint8` is scaled to `[0,1]`
  - `float` inputs are treated as normalized; if range looks like `0..255`, they are auto-scaled.

Output shape:
- single image: `(B, P, d_model)`
- multi-image: `(B, N_img * P, d_model)`

### Text encoder (`model/heads/text_encoder.py`)
- Backbone: `SiglipTextModel`
- Output projection: hidden size -> `d_model`
- Returns:
  - text tokens: `(B, T, d_model)`
  - text padding mask: `(B, T)` (`True` where token equals pad token id `1`)

### State encoder (`model/heads/state_encoder.py`)
- `Linear(state_dim, d_model) + LayerNorm`
- Output: `(B, 1, d_model)`

---

## Fusion Core: Latent Bottleneck

File: `model/refusion.py`

### Core idea
Instead of letting downstream modules attend over all multimodal tokens directly, use a fixed number of learned latent tokens (`fusion_latents`) that compress image/text/state information.

Benefits:
- fixed context length for action head,
- lower and predictable memory/latency,
- better edge deployment characteristics.

### Flow
1. Add type embeddings to each modality (`vision`, `text`, `state`).
2. Build memory sequence:
   - `memory = concat([img_tokens, txt_tokens, state_tokens], dim=1)`
3. Initialize learned latents:
   - `(1, fusion_latents, d_model)` -> broadcast to `(B, fusion_latents, d_model)`
4. For each fusion layer:
   - latent cross-attention to memory,
   - latent self-attention,
   - FFN.
5. Return normalized latents.

Fusion output shape:
- `(B, fusion_latents, d_model)`

---

## Action Head: Flow Matching Expert

Files:
- `model/action/action_expert.py`
- `model/action/flow_matching.py`

### Core idea
Predict velocity field on a continuous noise->action path.

### ActionExpert details
Input:
- noisy chunk `x_t`: `(B, C, action_dim)`
- diffusion time `t`: `(B,)`
- fused context latents: `(B, fusion_latents, d_model)`

Processing:
1. `action_proj`: `action_dim -> d_model`
2. add learned chunk positional embedding (order-aware across chunk steps)
3. add sinusoidal time embedding
4. `N` transformer blocks:
   - self-attention over action tokens,
   - cross-attention from action tokens to fused latents,
   - FFN
5. project back to action space

Output:
- predicted velocity `v_pred`: `(B, C, action_dim)`

### Flow-matching objective
Given normalized target actions `x_1` and noise `x_0 ~ N(0, I)`:
- `x_t = (1 - t) * x_0 + t * x_1`
- `v_target = x_1 - x_0`
- train with `MSE(v_pred, v_target)`

### Sampling
- Initialize `x_t` as Gaussian noise.
- Euler integrate for `flow_steps`:
  - `x_t = x_t + v(x_t, t, context) * dt`
- Return final normalized chunk.
- De-normalization is applied outside the model with dataset stats.

---

## Configuration (VLAConfig)

File: `model/utils.py`

Key fields:
- `d_model`, `n_heads`, `n_layers`
- `fusion_latents`
- `action_layers`, `action_heads`, `chunk_size`, `flow_steps`
- `state_dim`, `action_dim`
- `siglip_model_id`, `n_trainable`

---

## Design Rationale

### 1) Bounded multimodal context
A fixed latent bottleneck prevents context length explosion when adding text tokens or extra image views.

### 2) Better action ordering
Chunk positional embeddings in the action expert make per-step order explicit, improving temporal structure in predicted chunks.

### 3) Lightweight but expressive
Most heavy lifting stays in pretrained SigLIP towers; trainable task-specific modules are compact (`state encoder`, `fusion`, `action head`).

### 4) Edge readiness
Static-friendly dimensions (`img_size`, token length, chunk size, latent count) make the graph easier to optimize for deployment runtimes later.

---

## End-to-End Forward Contract

```text
img -> VisionEncoder -> img_tokens
txt -> TextEncoder -> txt_tokens, txt_pad_mask
state -> StateEncoder -> state_tokens

(img_tokens, txt_tokens, state_tokens, txt_pad_mask)
  -> FusionTransformer
  -> context_latents

(context_latents, noisy_action_chunk, t)
  -> ActionExpert
  -> velocity

FlowMatchingHead integrates velocity over time
  -> normalized action chunk
```

This is the architecture implemented by the current codebase.
