from torch import nn
import torch
from model.heads import *
from model.refusion import FusionTransformer
from model.action import FlowMatchingHead
from model.utils import VLAConfig
from model.memory import EpisodeMemory, inject_memory


class VLA(nn.Module):
    def __init__(self, cfg: VLAConfig, device: torch.device = torch.device("cuda")):
        super(VLA, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_encoder = VisionEncoder(cfg)
        self.txt_encoder = TextEncoder(cfg)
        if self.cfg.d_model <= 0:
            self.cfg.d_model = self.img_encoder.hidden_size

        self.state_encoder = StateEncoder(cfg)
        self.fusion_core = FusionTransformer(cfg, device)

        # Added memory module
        self.memory = EpisodeMemory(cfg)
        self.action_head = FlowMatchingHead(cfg)

    def encode_features(self, img, txt, state):
        """Encodes modalities and fuses them, strictly without memory injection."""
        img_enc = self.img_encoder(img)
        txt_enc, mask = self.txt_encoder(txt)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        state_enc = self.state_encoder(state)
        return self.fusion_core.forward(img_enc, txt_enc, state_enc, mask)

    def encode(self, img, txt, state):
        """Standard encode for single-step inference during rollouts."""
        fusion_latents = self.encode_features(img, txt, state)
        return inject_memory(self.memory, fusion_latents)

    def loss_seq(self, img_seq, txt, state_seq, action_seq):
        """
        Processes a sequence of length W for Stateful TBPTT.
        img_seq: (B, W, C, H, W)
        txt: (B, D)
        """
        B, W = img_seq.shape[:2]

        # 1. Flatten the batch and window dimensions for parallel encoding
        img_flat = img_seq.flatten(0, 1)  # (B*W, C, H, W)
        state_flat = state_seq.flatten(0, 1)  # (B*W, d_state)
        txt_flat = txt.repeat_interleave(W, dim=0)  # Match B*W

        # 2. Get fusion latents for the whole sequence at once
        fusion_flat = self.encode_features(img_flat, txt_flat, state_flat)
        _, L, D = fusion_flat.shape

        # 3. Reshape back to sequence format
        fusion_seq = fusion_flat.view(B, W, L, D)

        total_loss = 0
        # 4. Roll through time chronologically
        for t in range(W):
            context = inject_memory(self.memory, fusion_seq[:, t])
            total_loss += self.action_head.loss(action_seq[:, t], context)

        # Average loss over the window length
        return total_loss / W

    def act(self, img, txt, state, return_trajectory=False):
        context = self.encode(img, txt, state)
        return self.action_head.sample(context, return_trajectory)

def print_model_counts(model):
    sum_total = 0
    sum_trainable = 0
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        sum_total += total
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        sum_trainable += trainable
        print(f"{name:20s}  total={total:>12,}  trainable={trainable:>12,}")
    return sum_total, sum_trainable

if __name__ == "__main__":
    from contextlib import nullcontext
    device = torch.device("cuda")
    cfg = VLAConfig()
    vla = VLA(cfg, device).to(device)
    B = 3
    img = torch.randn(B, 3, 224, 224, device=device)
    txt = torch.tensor([[262, 266, 1357, 267, 262, 266, 1571, 1]]*B, device=device)
    state = torch.randn(B, 39, device=device)
    with nullcontext():
        encoded = vla.encode(img, txt, state)
        print("Encoded state shape:", encoded.shape)
        import time
        s = time.perf_counter()
        for i in range(10):
            loss = vla.loss_seq(img.unsqueeze(1), txt, state.unsqueeze(1), torch.randn(B, 1, 32, cfg.action_dim, device=device))
            print(loss)
        for i in range(100):
            out = vla.act(img, txt, state)
        print(f"Time: {100/(time.perf_counter() - s)}")

        total, trainable = print_model_counts(vla)
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params:    {total - trainable:,}")
        print(f"Total flops:      {total * 2 * 1e-9:.2f} GFLOPs")
        print(out.shape)
