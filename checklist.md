This paper is very relevant — and notably, your architecture already has the right bones. The gap is mostly in two things: how memory is stored and where it's injected, and whether text actually conditions image feature extraction or just rides alongside it.

## Yes, pre-LN residuals

That's exactly what I meant. Current code:

```python
x = self.norm1(lq)
x = x + self.self_attn(x)   # lq is gone, residual skips nothing
```

Fix:
```python
x = lq + self.self_attn(self.norm1(lq))
x = x + self.cross_attn(self.norm2(x), context, context, mask)
x = x + self.ffn(self.norm3(x))
```

## The bigger problem: text isn't actually conditioning image extraction

This is probably your main issue given that text is the only task differentiator in shared environments. In your current QFormer, text and image are concatenated into `context` and the learned queries cross-attend to all of it simultaneously. The queries have no reason to extract text-relevant image features — they can just ignore text in the context and attend to salient image patches regardless of instruction.

The BLIP-2 QFormer solves this with a specific attention pattern:

```
Learned queries ──┐
                  ├── Self-attention together (queries become text-aware)
Text tokens ──────┘

Text-aware queries ──── Cross-attention ──── Image patches
                        (text can't see image directly,
                         only through what queries ask for)
```

The queries pick up text conditioning through self-attention with text, *then* use that conditioning to selectively query image features. Image extraction becomes text-conditional. Your current design doesn't have this property.

Concretely, the attention mask for the self-attention block in faithful BLIP-2:

```python
# queries attend to queries + text; text is causal (attends to past text only)
# shape: (lq_size + txt_len, lq_size + txt_len)
mask = torch.zeros(L, L)
mask[lq_size:, :lq_size] = float('-inf')   # text tokens cannot attend to queries
# (queries attend to everything, text attends to text only)
```

Then image cross-attention is applied only to the query rows, not text rows. Memory can then be injected separately — either into the cross-attention context alongside image, or as HAMLET does it, downstream at the action expert.

## On HAMLET adaptation for your model

The key insight you should steal isn't the memory architecture (you already have that) — it's **what you store** and **where you inject it**.

**What you store:** Right now you're storing the full `lq_size × d_model` QFormer output as memory. That's 64 × 768 = 49k values per timestep, largely redundant and noisy. HAMLET stores only 4 learned moment tokens per step — a compressed, TCL-trained summary. The contrastive training forces these tokens to be temporally discriminative (suppress static background, emphasize what changed), which is exactly what a memory module needs. Without it, your memory likely fills with redundant "robot is above table" information at every step.

**Where you inject it:** You currently feed memory back into the QFormer context, which creates the error cascade I mentioned — corrupted memory biases QFormer output, which corrupts next memory slot, etc. HAMLET injects history at the action expert instead, leaving the QFormer as a clean single-frame feature extractor. This preserves the generalization you get from single-frame inference while adding history only where it conditions action selection.

## Concrete rebuild plan

Rather than a full rewrite, the minimal impactful changes in order of expected gain:

**1. Fix the QFormer to be text-conditional.** Split the current single cross-attention into two sub-layers: queries self-attend with text tokens (with the mask above), then queries cross-attend with image (and optionally memory). This directly addresses the text-not-working problem.

**2. Replace memory storage with moment tokens.** Add a small parameter `moment_tokens: nn.Parameter(1, n_moments, d_model)` — say 4–8 tokens. After each QFormer forward, run a tiny projection head on the moment token outputs and store those instead of the full latent. Pre-train them with TCL before the main training run.

**3. Move memory injection to the action expert.** The action expert already takes `(reasoning, state)` — extend it to `(reasoning, memory_feature, state)` where `memory_feature` comes from a shallow transformer over the last T moment token snapshots. This decouples the memory error cascade from the perception pipeline entirely.

**4. Add TCL pre-training.** Before main training: freeze everything, train only the moment tokens with the contrastive objective (same frame + augmentation = positive, frame > 16 steps away = negative). This takes a few hours but meaningfully helps the memory module know what to attend to, as shown in HAMLET's ablation Table 5a where removing TCL costs ~0.6% and removing the memory module costs ~1.7%.

The architecture is sound — it's closer to HAMLET than it might look. The residual bug, text-conditioning gap, and memory storage format are the practical gaps.