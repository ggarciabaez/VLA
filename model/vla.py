# TODO: add documentation across the model
# TODO: use direct parameter configuration instead of sending options through the call stack (see use_mask)
from model.utils import VLAConfig
from model.heads import TextEncoder, VisionEncoder
from model.fusion import QFormer
from model.action_expert import ActionExpert
import torch
from torch import nn

class VLA(nn.Module):
    def __init__(self, cfg: VLAConfig, use_mask=True):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.qformer = QFormer(cfg)
        self.action_expert = ActionExpert(cfg)
        self.use_mask = use_mask


    def encode(self, img: torch.Tensor, txt: torch.Tensor, state: torch.Tensor):
        """  TODO: add a text cache later
        :param img: Image tensor of shape (B, 3, H, W) or (B, N, 3, H, W)
        :param txt: Text token tensor of shape (B, 64)
        :param state: State tensor of shape (B, state_dim)
        :param use_mask: Whether to use the text mask for padding tokens
        :return: (B, lq_size, d_model) reasoning tokens (learned queries)
        """
        img_enc, txt_enc, state_enc, txt_mask = self._get_encodings(img, txt, state)
        return torch.cat([self.qformer.forward(img_enc, txt_enc, txt_mask if self.use_mask else None),
                          state_enc.unsqueeze(1)], dim=1)

    def loss(self, action, img, txt, state):
        reasoning = self.encode(img, txt, state)
        return self.action_expert.loss(action, reasoning)

    @torch.no_grad()
    def act(self, img, txt, state):
        reasoning = self.encode(img, txt, state)
        return self.action_expert.sample(reasoning)

    @torch.no_grad()
    def generate_dummy_inputs(self, B=1, device=torch.device("cpu")):
        img = (torch.rand(B, 3, 224, 224, device=device) * 255).to(torch.uint8)
        txt = self.text_encoder.tokenize(["A picture of a dog"] * B).to(device)
        state = torch.randn(B, self.cfg.state_dim, device=device)
        return img, txt, state

    def forward(self, img, txt, state):
        return self.act(img, txt, state)

    def _get_encodings(self, img, txt, state, stream=True):
        if stream:
            img_s = torch.cuda.Stream()
            txt_s = torch.cuda.Stream()

            with torch.cuda.stream(img_s):
                img_enc = self.vision_encoder(img)
            with torch.cuda.stream(txt_s):
                txt_enc, txt_mask = self.text_encoder(txt)
            state_enc = self.state_encoder(state)

            torch.cuda.current_stream().wait_stream(img_s)
            torch.cuda.current_stream().wait_stream(txt_s)
        else:
            img_enc = self.vision_encoder(img)
            txt_enc, txt_mask = self.text_encoder(txt)
            state_enc = self.state_encoder(state)
        return img_enc, txt_enc, state_enc, txt_mask

    def loss_enc(self, action, img_enc, txt_enc, state_enc, txt_mask=None):
        # For pre-encoded inputs!
        reasoning = torch.cat([self.qformer.forward(img_enc, txt_enc, txt_mask), state_enc.unsqueeze(1)], dim=1)
        return self.action_expert.loss(action, reasoning)


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
    B = 3
    img, txt, state = vla.generate_dummy_inputs(B, device)
    action = torch.randn(B, cfg.chunk_size, cfg.action_dim, device=device)
    total, trainable = print_model_counts(vla)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {total - trainable:,}")
    print(f"Total flops:      {total * 2 * 1e-9:.2f} GFLOPs")
    with torch.inference_mode():
        print(vla.loss(action, img, txt, state))

        import time
        s = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(100):
                out = vla.act(img, txt, state)
        print(f"Time: {100/(time.perf_counter() - s)}")

