from torch import nn
import torch
from model.heads import *
from model.refusion import FusionTransformer
from model.action import FlowMatchingHead
from model.utils import VLAConfig
from model.memory import EpisodeMemory, inject_memory


class VLAFusion(nn.Module):
    def __init__(self, cfg: VLAConfig, device: torch.device = torch.device("cuda")):
        super(VLAFusion, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_encoder = VisionEncoder(cfg)
        self.txt_encoder = TextEncoder(cfg)
        if self.cfg.d_model <= 0:
            self.cfg.d_model = self.img_encoder.hidden_size

        self.state_encoder = StateEncoder(cfg)
        self.fusion_core = FusionTransformer(cfg, device)

    def forward(self, img, txt, state):
        """Encodes modalities and fuses them, strictly without memory injection."""
        img_enc = self.img_encoder(img)
        txt_enc, mask = self.txt_encoder(txt)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        state_enc = self.state_encoder(state)
        return self.fusion_core.forward(img_enc, txt_enc, state_enc, mask)

    def encode(self, img, txt, state):
        """Encodes modalities and fuses them, optionally with memory injection. Runs using cuda streams to parallelize."""
        img_stream = torch.cuda.Stream()
        txt_stream = torch.cuda.Stream()
        with torch.cuda.stream(img_stream):
            img_enc = self.img_encoder(img)
        with torch.cuda.stream(txt_stream):
            txt_enc, mask = self.txt_encoder(txt)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        state_enc = self.state_encoder(state)
        torch.cuda.current_stream().wait_stream(img_stream)
        torch.cuda.current_stream().wait_stream(txt_stream)
        return self.fusion_core.forward(img_enc, txt_enc, state_enc, mask)


class VLA(nn.Module):
    def __init__(self, fusion_module: VLAFusion, action_expert: FlowMatchingHead, use_memory: bool = True):
        super(VLA, self).__init__()
        self.cfg = fusion_module.cfg
        self.fusion = fusion_module
        self.action = action_expert
        self.memory = EpisodeMemory(self.cfg) if use_memory else None

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
        fusion_flat = self.fusion.encode(img_flat, txt_flat, state_flat)
        _, L, D = fusion_flat.shape

        # 3. Reshape back to sequence format
        fusion_seq = fusion_flat.view(B, W, L, D)

        total_loss = 0
        # 4. Roll through time chronologically
        for t in range(W):
            context = inject_memory(self.memory, fusion_seq[:, t]) if self.memory is not None else fusion_seq[:, t]
            total_loss += self.action.loss(action_seq[:, t], context)

        # Average loss over the window length
        return total_loss / W

    def act(self, img, txt, state, return_trajectory=False):
        context = self.fusion(img, txt, state)
        context = inject_memory(self.memory, context) if self.memory is not None else context
        return self.action(context, return_trajectory)

    def forward(self, img, txt, state, return_trajectory=False):
        return self.act(img, txt, state, return_trajectory)