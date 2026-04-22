# MT50 VLA Architecture

## Overview

A Vision-Language-Action model for multi-task robotic manipulation across 50 MetaWorld tasks.
Perception and language are compressed into reasoning tokens via a Q-Former. An action expert
generates a MEM token summarizing the current timestep, uses it as a FiLM conditioning vector
to produce a velocity field via flow matching, and pushes the MEM token onto a frozen episodic
memory stack for future timesteps.

---

## Data Flow

```
[image] [text*] [MEM stack]
    |       |       |
    +-------+-------+
            |
        Q-Former                     * cached per episode
            |
     reasoning tokens (B, 64, 768)
            |
    +-------+--------+
    |                |
 [state]     learned_mem_latent
    |                |
    +-------+--------+
            |
         cross-attn
            |
        new_mem_token (B, 1, 768) -----> pushed onto MEM stack
            |
         FiLM MLP
            |
      velocity field (B, C, action_dim)
            |
      Euler integration
            |
      action chunk (B, C, action_dim)
```

---

## Stage 1 — Encoders

| Stream | Module | Output shape | Notes |
|---|---|---|---|
| Current image | SigLIP ViT B/16 | (B, 196, 768) | Last 2 layers unfrozen |
| Task instruction | SigLIP Text | (B, 64, 768) | Cached per episode |
| Robot state | Linear + LayerNorm | (B, 1, 768) | Action expert only |
| MEM stack | Frozen past tokens | (B, T, 768) | T ≤ 10, zero-padded early in episode |

---

## Stage 2 — Q-Former (Reasoning Head)

Goal: compress current observations + episodic memory into 64 dense reasoning tokens.

### Embeddings applied to context before cross-attention

SigLIP already encodes positional information internally into image patches and text tokens.
Adding position embeddings on top would be redundant. Type embeddings are applied to all three
modalities so queries know what they are attending to. MEM additionally receives recency
positional embeddings since it has no backbone-level encoding.

| Modality | Type embedding | Position embedding |
|---|---|---|
| Image (196 tokens) | ✓ `type_embed(IMAGE)` | ✗ already in SigLIP |
| Text (64 tokens) | ✓ `type_embed(TEXT)` | ✗ already in SigLIP |
| MEM (T tokens) | ✓ `type_embed(MEM)` | ✓ `recency_pos_embed(0..T-1)` |

```python
# Embedding vocabularies
type_embed      = nn.Embedding(3, d_model)   # IMAGE=0, TEXT=1, MEM=2
recency_pos_embed = nn.Embedding(mem_len, d_model)  # 0=most recent, 9=oldest
```

### Attention structure

```
image   = image + type_embed(IMAGE)
text    = text  + type_embed(TEXT)
mem     = mem   + recency_pos_embed(0..T-1) + type_embed(MEM)

context = concat(image, text, mem)        # (B, 196 + 64 + T, 768)  K and V
queries = learned_queries                 # (B, 64, 768)              Q only

for each layer:
    queries = self_attn(Q=queries, K=queries, V=queries)
    queries = cross_attn(Q=queries, K=context, V=context)

reasoning_tokens = queries                # (B, 64, 768)
```

### Attention table

| Operation | Q | K | V | Purpose |
|---|---|---|---|---|
| Self-attention | queries | queries | queries | Queries coordinate before reading context |
| Cross-attention | queries | [image \| text \| MEM] | [image \| text \| MEM] | Queries read current obs + frozen past |

### Key properties

- MEM tokens appear only as K and V — read-only, never act as Q
- Text padding mask and MEM zero-padding mask concatenated into a single `key_padding_mask`
- Text embeddings computed once at episode start and reused every timestep

### Config

```python
n_queries        = 64
n_qformer_layers = 4
n_heads          = 6      # head_dim = 128, Flash Attention eligible
mem_len          = 10     # ~10 seconds of episodic memory
```

---

## Stage 3 — Action Expert

Goal: produce a new MEM token and a velocity field for flow matching.
The MEM token is the **only** information bridge between perception and action.
Training loss propagates through it directly, forcing meaningful compression.

### Step 1 — MEM token generation

```
Q = learned_mem_latent                         # (B, 1, 768)  trained parameter
K = V = concat(reasoning_tokens, state_token)  # (B, 65, 768)

new_mem_token = cross_attn(Q, K, V)            # (B, 1, 768)
```

### Step 2 — FiLM conditioning

```
film_vector = Linear(new_mem_token.squeeze(1))  # (B, d_film)
```

MEM token and film_vector are constant across all Euler steps —
recompute once per chunk, not per integration step.

### Step 3 — Velocity field (per chunk step)

Each chunk step receives three inputs before entering the FiLM MLP:
- Noisy action at that step: `(B, action_dim)`
- Flow timestep embedding: `(B, t_dim)` — drives time-varying denoising dynamics
- Chunk position embedding: `(B, chunk_pos_dim)` — tells the MLP which step it is computing

```python
chunk_pos_embed = nn.Embedding(chunk_size, chunk_pos_dim)  # 0..C-1

for step in range(C):
    x_in = concat(
        noisy_chunk[:, step, :],    # (B, action_dim)
        t_embedding,                # (B, t_dim)        — constant within Euler step
        chunk_pos_embed(step),      # (B, chunk_pos_dim)— constant within Euler loop
    )                               # (B, action_dim + t_dim + chunk_pos_dim)
    velocity[:, step, :] = FiLM_MLP(x_in, film_vector)
```

Chunk steps are processed independently. Trajectory smoothness is a learned property
enforced by the flow matching objective on smooth expert demonstrations, not by
explicit cross-step coupling in the architecture.

### Division of labor

| Signal | Answers |
|---|---|
| `film_vector` (from MEM) | What motion to generate — task, object, state, history |
| `t_embedding` | How to denoise at this noise level — time-varying dynamics |
| `chunk_pos_embed` | Which step of the chunk — step-specific trajectory shaping |

### Attention table

| Operation | Q | K | V | Purpose |
|---|---|---|---|---|
| Cross-attention | learned_mem_latent | [reasoning \| state] | [reasoning \| state] | Compress perception + state into MEM token |

### Config

```python
chunk_size    = 8     # C
action_dim    = 4     # MetaWorld: x, y, z, gripper
t_dim         = 64    # sinusoidal flow timestep embedding dim
chunk_pos_dim = 64    # chunk step position embedding dim
d_film        = 768
n_film_layers = 4
euler_steps   = 10
```

---

## Stage 4 — Flow Matching

```python
x = randn(B, C, action_dim)
dt = 1.0 / euler_steps
for i in range(euler_steps):
    t = full((B,), i / euler_steps)
    v = expert.compute_velocity(x, t, film_vector)
    x = x + v * dt
action_chunk = x    # (B, C, action_dim)
```

MEM token and film_vector are constant across all Euler steps — KV cache
of the MEM cross-attention is pre-computable before the integration loop.

---

## Memory Stack Protocol

```python
# Episode start
MEM          = zeros(B, mem_len, d_model)
mem_pad_mask = ones(B, mem_len, dtype=bool)    # all slots empty

# After each chunk
MEM          = concat(new_mem_token, MEM[:, :mem_len-1, :], dim=1)
mem_pad_mask = concat([False], mem_pad_mask[:, :mem_len-1])
```

MEM tokens are never modified after creation. Each token implicitly encodes:
- What was observed (via reasoning tokens)
- What the robot state was (via state token)
- What action followed (via flow matching loss backpropagating through MEM)

---

## Embedding Vocabulary Summary

| Embedding | Type | Size | Applied to |
|---|---|---|---|
| `type_embed` | `nn.Embedding(3, 768)` | 3 entries | All context modalities |
| `recency_pos_embed` | `nn.Embedding(10, 768)` | 10 entries | MEM stack slots |
| `chunk_pos_embed` | `nn.Embedding(8, 64)` | 8 entries | Action chunk steps |

---

## Auxiliary Training Opportunities

| Signal | Source | Benefit |
|---|---|---|
| Task phase classification | Linear probe on MEM tokens | Forces semantically ordered memory |
| Binary success prediction | Linear probe on final MEM token | Value function for RL finetuning |
| Attention weight logging | Q-Former cross-attn | Which image regions / MEM slots drive each query |

---

## Parameter Budget (approximate)

| Component | Parameters | Trainable |
|---|:-:|---|
| SigLIP vision (2 layers unfrozen) | ~86M | ~10M |
| SigLIP text (2 layers unfrozen) | ~39M | ~5M |
| State encoder | ~1M | ~1M |
| Q-Former (4 layers) | ~28M | ~28M |
| Action expert (cross-attn + FiLM MLP) | ~12M | ~12M |
| **Total** | **~166M** | **~56M** |

