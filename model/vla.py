from torch import nn
import torch
from heads import *
from refusion import FusionTransformer
from action import FlowMatchingHead
from utils import VLAConfig

class VLA(nn.Module):
    def __init__(self, cfg: VLAConfig, device: torch.device = torch.device("cuda")):
        super(VLA, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_encoder = VisionEncoder(cfg)
        self.txt_encoder = TextEncoder(cfg)
        if self.cfg.d_model <= 0:
            self.cfg.d_model = self.img_encoder.hidden_size
        # assert self.cfg.d_model == self.txt_encoder.hidden_size, "An error occurred. d_model for img and txt are mismatched."

        self.state_encoder = StateEncoder(cfg)

        self.fusion_core = FusionTransformer(cfg, device)

        self.action_head = FlowMatchingHead(cfg)


    def encode(self, img, txt, state):
        """
        :param img: (B, C, H, W) tensor containing a normalized image
        :param txt: (B, T) tensor containing a tokenized string
        :param state: (B, 1, d_state) tensor containing the robot state
        :return: (B, P+T+1, d_model) tensor containing the encoded representation with attended text tokens for language grounding.
        """
        img_enc = self.img_encoder(img)
        txt_enc, mask = self.txt_encoder(txt)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        state_enc = self.state_encoder(state)
        return self.fusion_core.forward(img_enc, txt_enc, state_enc, mask)

    # the arch. good f*cking update.
    def loss(self, img, txt, state, action):
        context = self.encode(img, txt, state)
        return self.action_head.loss(action, context)

    def act(self, img, txt, state, return_trajectory=False):
        context = self.encode(img, txt, state)
        return self.action_head.sample(context, return_trajectory)  # Remember to de-normalise!

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
    device = torch.device("cuda")
    cfg = VLAConfig(n_layers=8)
    vla = VLA(cfg).to(device).eval()
    B = 3
    img = torch.randn(B, 3, 224, 224, device=device)
    txt = torch.tensor([[262, 266, 1357, 267, 262, 266, 1571, 1]]*B, device=device)
    state = torch.randn(B, 1, 39, device=device)
    with torch.inference_mode():
        encoded = vla.encode(img, txt, state)
        print("Encoded state shape:", encoded.shape)
        # a bit of warmup
        for i in range(10):
            out = vla.loss(img, txt, state, torch.randn(B, cfg.chunk_size, 4, device=device))
        print(out)
        import time
        s = time.perf_counter()
        for i in range(100):
            out = vla.act(img, txt, state)
        print(f"Time: {100/(time.perf_counter() - s)}")

        total, trainable = print_model_counts(vla)
        print(f"Total params:     {total:,}")
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params:    {total - trainable:,}")
        print(f"Total flops:      {total * 2 * 1e-9:.2f} GFLOPs")
        print(out.shape)