"""
EpisodeMemory: GRU-based recurrent memory for multi-step task completion.

How it fits in the forward pass:
  fusion_latents (B, L, d_model)
      |
      v
  mean-pool --> (B, d_model) --> GRUCell --> h_t (B, d_model)
                                    ^
                                    h_{t-1} (B, d_model)

  h_t is unsqueezed and prepended to fusion_latents:
  context = cat([h_t.unsqueeze(1), fusion_latents], dim=1)  # (B, L+1, d_model)

  ActionExpert cross-attends to `context` instead of `fusion_latents` directly.
  No other changes to ActionExpert needed — it already handles variable context length.

Training:
  - Sample windows of length W consecutive steps from the same episode.
  - Roll the GRU through the window, accumulating loss at each step.
  - Detach hidden state across window boundaries (TBPTT).
  - Reset hidden state when a done=True flag appears in the window.

Inference:
  - Call reset() at episode start.
  - Call update() before each action expert forward pass.
  - The returned context token is passed directly to the action expert.
"""

import torch
import torch.nn as nn
from model.utils import VLAConfig


class EpisodeMemory(nn.Module):
    """
    Lightweight GRU memory that converts a sequence of fusion latent summaries
    into a single recurrent context token, injected into the action expert.

    Parameters
    ----------
    cfg : VLAConfig
    n_layers : int
        Number of stacked GRU layers. 1 is usually sufficient; 2 adds capacity
        for longer-horizon tasks at minimal cost (~2M params for d_model=768).
    """

    def __init__(self, cfg: VLAConfig, n_layers: int = 1):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_layers = n_layers

        # Project mean-pooled latents → GRU input
        # (identity if you trust the latents directly, but a learned gate helps)
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        self.gru = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=n_layers,
            batch_first=True,
        )
        nn.init.orthogonal_(self.gru.weight_hh_l0)

        # Output projection back to token space
        # Initialize near-zero so memory starts with small influence and grows
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Layer norm on the output token
        self.out_norm = nn.LayerNorm(cfg.d_model)
        nn.init.ones_(self.out_norm.weight)
        nn.init.zeros_(self.out_norm.bias)
        # Hidden state buffer: not a Parameter, managed manually
        # Shape: (n_layers, B, d_model) — registered as buffer with None default
        self._hidden: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, batch_size: int, device: torch.device, dtype: torch.dtype | None = None):
        """
        Reset hidden state to zeros. Call at the start of each new episode.
        Safe to call mid-batch when only some items in the batch end an episode —
        use reset_rows() for that.
        """
        param_dtype = self.input_proj[0].weight.dtype
        # Keep the recurrent state in module precision. Low-precision hidden
        # states under autocast are prone to dtype mismatches and instability.
        if dtype is None or dtype in (torch.float16, torch.bfloat16):
            dtype = param_dtype
        self._hidden = [
            torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
            for _ in range(self.n_layers)
        ]

    def reset_rows(self, done_mask: torch.Tensor):
        """
        Selectively reset hidden state for finished episodes within a batch.

        Args:
            done_mask: (B,) bool tensor, True where episode just ended.
        """
        if self._hidden is None:
            return
        for i in range(self.n_layers):
            # Zero out rows where done=True, keep others unchanged
            self._hidden[i] = self._hidden[i].masked_fill(done_mask.unsqueeze(-1), 0.0)

    def detach(self):
        """
        Detach hidden state from the computation graph.
        Call between TBPTT windows to prevent gradient flow across window boundaries.
        """
        if self._hidden is not None:
            self._hidden = [h.detach() for h in self._hidden]

    @property
    def is_initialized(self) -> bool:
        return self._hidden is not None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, fusion_latents: torch.Tensor) -> torch.Tensor:
        """
        Update memory with current fusion latents and return a memory token.

        Args:
            fusion_latents: (B, L, d_model) — output of FusionTransformer

        Returns:
            memory_token: (B, 1, d_model) — prepend this to fusion_latents
                          before passing context to ActionExpert

        Side effect:
            Updates self._hidden in-place.
        """
        B, L, D = fusion_latents.shape
        assert D == self.d_model, f"d_model mismatch: got {D}, expected {self.d_model}"
        input_dtype = fusion_latents.dtype
        work_dtype = self.input_proj[0].weight.dtype

        # Lazily initialize if not done yet (e.g., single-step inference)
        if self._hidden is None:
            self.reset(B, fusion_latents.device, work_dtype)
        elif any(h.dtype != work_dtype for h in self._hidden):
            self._hidden = [h.to(dtype=work_dtype) for h in self._hidden]

        if fusion_latents.dtype != work_dtype:
            fusion_latents = fusion_latents.to(dtype=work_dtype)

        x = self.input_proj(fusion_latents.mean(dim=1)).unsqueeze(1)  # (B, 1, d_model)

        if self._hidden is None:
            h0 = torch.zeros(self.n_layers, B, self.d_model, device=x.device, dtype=x.dtype)
        else:
            h0 = torch.stack(self._hidden, dim=0)  # (n_layers, B, d_model)

        out, h_n = self.gru(x, h0)

        self._hidden = list(h_n)  # unpack back into list

        token = self.out_norm(self.out_proj(h_n[-1]))
        return token.unsqueeze(1).to(dtype=input_dtype)


def inject_memory(memory_module: EpisodeMemory,
                  fusion_latents: torch.Tensor) -> torch.Tensor:
    """
    Convenience wrapper: update memory and prepend token to fusion_latents.

    Usage in vla.py forward():
        from model.memory import inject_memory
        context = inject_memory(self.memory, fusion_latents)  # (B, L+1, d_model)
        v_pred = self.action_expert(x_t, t, context)

    Returns:
        context: (B, L+1, d_model)
    """
    mem_token = memory_module(fusion_latents)                    # (B, 1, d_model)
    return torch.cat([mem_token, fusion_latents], dim=1)         # (B, L+1, d_model)

if __name__ == "__main__":
    import torch
    from model.utils import VLAConfig
    cfg = VLAConfig()
    model = EpisodeMemory(cfg)
    for i in range(4):
        tok = inject_memory(model, torch.randn(1, 64, cfg.d_model))
        print(tok.shape)
    print(inject_memory(model, torch.randn(1, 64, cfg.d_model)))
