# MT50 VLA — Project Checklist

Architecture, training pipeline, data collection, and deployment.
Work top-to-bottom within each section — later items depend on earlier ones.

---

## 1. Codebase

> Do this first. Everything else is easier in a proper package than a notebook.

- [ ] **Restructure into a Python package**
  - Break the monolithic notebook into a `vla/` package with the following layout:
    ```
    vla/
      encoders/   vision.py, text.py
      model/      fusion.py, action_expert.py, vla.py
      data/       dataset.py, sampler.py, transforms.py
      train/      trainer.py, config.py, losses.py
      eval/        rollout.py, metrics.py
      scripts/    train.py, evaluate.py, collect.py
    configs/
      mt50.yaml
    notebooks/
      launcher.ipynb   ← Colab entry point, calls scripts only
    ```
  - Config moves to a dataclass or YAML file. No more bare `CFG = dict(...)`.
  - Each module is independently importable and testable.

- [ ] **Set up GitHub + Colab workflow**
  - Push the package to a GitHub repo.
  - The Colab notebook becomes a thin launcher only:
    ```python
    !git clone https://github.com/yourname/vla-mt50.git
    %cd vla-mt50
    !pip install -e . -q
    from google.colab import drive
    drive.mount("/content/drive")
    # data and checkpoints live on Drive, code lives in the repo
    ```
  - Iterate locally, push, `!git pull` in Colab to pick up changes. All real logic is in `.py` files.

---

## 2. Data Collection

> Runs independently. Start this early — it takes time and produces the foundation for everything else.

- [ ] **MT50 scripted expert collection**
  - Collect 10 episodes per task across all 50 MetaWorld tasks = 500 episodes total.
  - Use MetaWorld's built-in scripted expert policies — no manual control needed.
  - Vary random seeds per episode for object position diversity.
  - Record at every timestep: raw image (224×224 uint8), joint state, action.
  - Store a `success` flag per episode; filter out failed episodes from the index.
    ```python
    for task in mt50.train_tasks:
        env = task()
        policy = get_scripted_policy(task)
        for seed in range(n_episodes):
            obs = env.reset_with_seed(seed)
            record_episode(env, policy, obs, seed)
    ```

- [ ] **Storage format — memory-mapped numpy arrays**
  - Store each modality as a flat `.npy` file per task (or globally), memory-mapped at training time.
  - Images stored as uint8 to keep disk usage manageable (~15GB total at 224×224).
  - Decode and normalize images in the DataLoader collate function, not in the model.
    ```python
    # collection time
    np.save("images.npy", all_images)   # (N, 224, 224, 3) uint8

    # training time — nothing loaded into RAM
    images = np.load("images.npy", mmap_mode="r")
    img = images[idx]  # OS pages in just this slice
    ```
  - No HDF5 — file handles can't be safely shared across DataLoader workers.

- [ ] **Episode index file**
  - Build a flat index of valid sample positions as a `.json` or `.npz`.
  - A position `(episode_id, t)` is valid if: episode succeeded, and `t + chunk_size C ≤ episode_length`.
  - This handles episode boundaries cleanly without touching the image array at sample time.
  - Also store global normalization stats (mean, std per action/state dimension) computed over the full dataset.
    ```json
    {
      "samples": [[episode_id, timestep], ...],
      "action_mean": [...],
      "action_std": [...],
      "state_mean": [...],
      "state_std": [...]
    }
    ```

- [ ] **Language annotation file**
  - Write 3–5 phrasings per task for all 50 tasks, stored in a single JSON.
  - Sample randomly during training to prevent the text encoder from ignoring language.
    ```json
    {
      "reach-v2": [
        "reach the red sphere",
        "move the end effector to the target",
        "touch the goal position"
      ]
    }
    ```

---

## 3. Model Architecture

> Implement and unit-test each component in isolation before assembling the full VLA.

- [ ] **Remove offline pre-encoding path**
  - Delete `PrecomputedDataset` and the `_encode_obs_precompute` branch.
  - `encode()` always calls `_compute_embeddings()`. Remove the `precompute_mode` flag entirely.
  - Vision and text encoders now run in the forward pass on raw inputs every step.

- [ ] **Partial SigLIP unfreezing**
  - Freeze all SigLIP layers except the last 2 encoder layers, for both vision and text towers.
  - Projection layers and pooler always remain trainable.
    ```python
    all_layers = list(self.backbone.encoder.layers)
    for layer in all_layers[:-2]:
        for p in layer.parameters():
            p.requires_grad = False
    ```
  - Start with `unfreeze_n=2`. Can increase to 4 if validation loss plateaus.

- [ ] **Scale fusion transformer**
  - `d_model`: 128 → 384
  - Layers: 2 → 4
  - Heads: 4 → 6
  - `dim_feedforward`: stays at `4 × d_model` = 1536
  - Dropout: 0.1 → 0.05
  - Update `img_proj`, `txt_proj`, and `StateEncoderMLP` to target new `d_model`.

- [ ] **Fusion transformer outputs token sequence**
  - Remove the final `projector` (Linear + LayerNorm) and mean/CLS pooling from `FusionTransformer.forward`.
  - Return the full `out` tensor of shape `(B, N_tokens, d_model)`.
  - These context tokens are the key/value input to the action expert's cross-attention layers.

- [ ] **Simplify state encoder**
  - Replace the 2-layer MLP with `nn.Linear(state_dim, d_model)` + `nn.LayerNorm(d_model)`.
  - Output shape `(B, H, d_model)` — H tokens fed directly into the fusion transformer alongside image tokens. No activation.

- [ ] **Transformer action expert**
  - Replace `FlowMatchingModel` / `ResidualActionMLP` with a small transformer action expert.
  - Takes noisy action chunk `(B, C, action_dim)`, time embedding, and context tokens `(B, N_ctx, d_model)`.
  - Projects actions up to `d_model`, adds sinusoidal time embedding broadcast over chunk, then runs N layers of self-attention → cross-attention → FFN, then projects back down.
  - Use GELU activation (not SiLU) throughout the action expert for cleaner quantization later.
    ```python
    class ActionExpertLayer(nn.Module):
        # self-attn over the action chunk (inter-action coherence)
        # cross-attn to context_tokens (conditioning on observation)
        # ffn

    class ActionExpert(nn.Module):
        def __init__(self, action_dim, d_model, n_layers, chunk_size):
            self.action_proj  = nn.Linear(action_dim, d_model)
            self.time_emb     = SinusoidalTimeEmbedding(d_model)
            self.layers       = nn.ModuleList([ActionExpertLayer(...)])
            self.out_proj     = nn.Linear(d_model, action_dim)
    ```
  - Flow matching loss and sampling logic stays the same — just operates on `(B, C, action_dim)` tensors throughout.

---

## 4. Training Pipeline

- [ ] **Rewrite dataset to serve raw data**
  - New `EpisodeDataset` class backed by memory-mapped arrays.
  - `__getitem__` loads a window of H history frames + C chunk frames and returns raw tensors.
  - Image decode (uint8 → float, resize if needed, normalize) happens in the DataLoader collate function.
    ```python
    item = {
        "images":   (H, 224, 224, 3),   # uint8
        "states":   (H, state_dim),      # float32, already normalized
        "text_ids": (seq_len,),          # int64
        "attn_mask":(seq_len,),          # int64
        "actions":  (C, action_dim),     # float32, already normalized
    }
    ```

- [ ] **Update flow matching loss for action chunking**
  - `actions` is now `(B, C, action_dim)`. Broadcast time `t` over the chunk dimension.
    ```python
    t     = torch.rand(B, device=actions.device)
    t_exp = t.reshape(B, 1, 1)          # broadcasts over C and action_dim
    x_0   = torch.randn_like(actions)
    x_t   = (1 - t_exp) * x_0 + t_exp * actions
    target_v = actions - x_0
    v_pred   = self.model(x_t, t, context_tokens)
    loss     = F.mse_loss(v_pred, target_v)
    ```

- [ ] **Differential learning rates**
  - Two parameter groups: unfrozen encoder layers at low lr, everything else at main lr.
    ```python
    optimizer = AdamW([
        {"params": unfrozen_encoder_params, "lr": 1e-5},
        {"params": fusion_and_expert_params, "lr": 3e-4},
    ])
    ```

- [ ] **Gradient clipping**
  - `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` every step.
  - Keeps weight magnitudes bounded, which directly improves INT8 quantization quality later.

- [ ] **Mixed precision + gradient checkpointing**
  - Wrap the training step in `torch.cuda.amp.autocast()`.
  - Enable gradient checkpointing on the SigLIP backbone to trade compute for memory when fine-tuning encoder layers.

- [ ] **Fix all inference shapes at training time**
  - Decide and hardcode: batch size = 1 at inference, history length H, chunk size C, image size 224×224.
  - Write `act()` with these constants, not variable dimensions. TensorRT requires static shapes.

---

## 5. Deployment (Jetson Orin Nano + TensorRT)

> Do this after training converges. Build the engine on the Orin itself, not on a desktop GPU.

- [ ] **Ensure forward pass is ONNX-exportable**
  - No Python control flow that depends on tensor values.
  - No custom ops without TensorRT plugins.
  - Use standard `nn.MultiheadAttention` and `nn.TransformerEncoderLayer` throughout — these have known TensorRT fusion patterns.
  - Test export before committing to the final architecture.

- [ ] **Export to ONNX**
    ```python
    torch.onnx.export(
        model,
        (image, state, text_ids),
        "vla.onnx",
        input_names=["image", "state", "text_ids"],
        output_names=["action_chunk"],
        opset_version=17,
    )
    ```

- [ ] **Build TensorRT engine on the Orin**
  - Start with FP16 — native on Orin's Ampere GPU, roughly 2× throughput, near-zero accuracy loss.
    ```bash
    trtexec --onnx=vla.onnx --saveEngine=vla.engine --fp16
    ```
  - If FP16 isn't fast enough, try INT8 PTQ with a calibration dataset (~500 representative frames):
    ```bash
    trtexec --onnx=vla.onnx --saveEngine=vla.engine --int8 \
            --calib=calibration_cache.bin
    ```
  - QAT (quantization-aware training) is a last resort — only pursue if INT8 PTQ causes measurable accuracy loss.
  - Note: TensorRT-LLM is not relevant here. It targets autoregressive LLM decoding with KV caching. Use standard TensorRT.

- [ ] **Runtime inference loop**
    ```python
    import tensorrt as trt
    import pycuda.driver as cuda

    # --- startup (once) ---
    with open("vla.engine", "rb") as f:
        engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    # allocate GPU buffers once, reuse every frame

    # --- per-observation (at ~4Hz for 30Hz control with C=8) ---
    def infer(image, state, text_ids):
        # copy inputs to GPU
        context.execute_v2(bindings)
        # copy action_chunk output back to CPU
        return action_chunk  # (C, action_dim)

    # --- control loop ---
    chunk = infer(obs)
    for step in range(C):
        action = denormalize(chunk[step])
        env.step(action)
        # re-run infer slightly before chunk exhausts to avoid stutter
    ```

---

## Constants to decide before training

| Parameter | Suggested value | Notes |
|---|---|---|
| `d_model` | 384 | Up from 128 |
| Fusion layers | 4 | Up from 2 |
| Fusion heads | 6 | Up from 4 |
| Unfrozen encoder layers | 2 | Per tower, start conservative |
| History length H | 1–3 | Same as MT10 to start |
| Chunk size C | 8 | ~267ms of actions at 30Hz |
| Action expert layers | 4 | Self-attn + cross-attn per layer |
| Flow matching steps | 10 | Reduced from 32, fine for inference |
| Encoder lr | 1e-5 | |
| Main lr | 3e-4 | |
| Batch size | 256 | Reduced from 512 due to online encoding cost |
| Epochs | 200 | |