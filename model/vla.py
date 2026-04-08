from torch import nn
import torch
from heads import *
from fusion import FusionTransformer
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
        assert self.cfg.d_model == self.txt_encoder.hidden_size, "An error occurred. d_model for img and txt are mismatched."

        self.state_encoder = StateEncoder(cfg)

        self.fusion_core = FusionTransformer(cfg, device)

        self.action_head = FlowMatchingHead(cfg)


    def encode(self, img, txt, state):
        """
                :param img: (B, C, H, W) tensor containing a normalized image
                :param txt: (B, T) tensor containing a tokenized string
                :param state: (B, 1, d_state) tensor containing the robot state
                :return: something ig
                """
        img_enc = self.img_encoder(img)
        txt_enc, mask = self.txt_encoder(txt)
        state_enc = self.state_encoder(state)

        fused = self.fusion_core.forward(img_enc, txt_enc, state_enc, mask)
        return fused

    # the arch. good f*cking update.
    def loss(self, img, txt, state, action):
        context = self.encode(img, txt, state)
        return self.action_head.loss(action, context)

    def act(self, img, txt, state):
        context = self.encode(img, txt, state)
        return self.action_head.sample(context)  # Remember to de-normalise!

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
    cfg = VLAConfig()
    vla = VLA(cfg).to(device)
    img = torch.randn(1, 3, 224, 224, device=device)
    txt = torch.tensor([[262, 266, 1357, 267, 262, 266, 1571, 1]], device=device)
    state = torch.randn(1, 1, 39, device=device)
    out = vla.act(img, txt, state)
    print(out.size())

    total, trainable = print_model_counts(vla)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {total - trainable:,}")
