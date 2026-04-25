# TODO: add documentation across the model
from model.utils import VLAConfig
from model.heads import TextEncoder, VisionEncoder
from model.fusion import QFormer
from model.action_expert import ActionExpert
import torch
from torch import nn

class VLA(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        self.qformer = QFormer(cfg)
        self.action_expert = ActionExpert(cfg)

        self._used_mem = 0
        self._mem_buf = None
        self.mem_mask = None
        self._txt_cache = None

    @torch.no_grad()
    def insert_memory(self, mem: torch.Tensor):
        B = mem.shape[0]
        if self._mem_buf is None:
            self._mem_buf = torch.zeros(B, self.cfg.mem_len, self.cfg.d_model, device=mem.device)
            self.mem_mask = torch.zeros(B, self.cfg.mem_len, dtype=torch.bool, device=mem.device)
        # Shift right, insert at front (index 0 = most recent)
        self._mem_buf = torch.roll(self._mem_buf, 1, dims=1)
        self._mem_buf[:, 0, :] = mem.squeeze(1).detach()
        self._used_mem = min(self._used_mem + 1, self.cfg.mem_len)
        self.mem_mask[:, self._used_mem-1] = True  # mark empty slots


    @torch.no_grad()
    def get_memory(self, B=None, device=None):
        if self._mem_buf is None:
            self._mem_buf = torch.zeros(B, self.cfg.mem_len, self.cfg.d_model, device=device)
            self.mem_mask = torch.zeros(B, self.cfg.mem_len, dtype=torch.bool, device=device)
        return self._mem_buf, self.mem_mask

    def reset_memory(self):
        self._mem_buf = None
        self.mem_mask = None
        self._used_mem = 0

    def reset(self):
        self._txt_cache = None
        self.reset_memory()

    def encode(self, img: torch.Tensor, txt: torch.Tensor, cache_txt: bool = False):
        """
        :param img: Image tensor of shape (B, 3, H, W) or (B, N, 3, H, W)
        :param txt: Text tensor of shape (B, 64)
        :param cache_txt: Whether to cache the text embeddings for faster encoding. Usually for inference.
        :return: (B, lq_size, d_model) reasoning tokens (learned queries)
        """
        img_enc = self.vision_encoder(img)
        # Reusing cached text features during training would retain autograd state
        # across timesteps, which breaks repeated backward passes in the trainer loop.
        use_txt_cache = cache_txt and not self.training
        if self._txt_cache is None or torch.any(self._txt_cache[2] != txt) or not use_txt_cache:
            txt_enc, txt_mask = self.text_encoder(txt)
            if use_txt_cache:
                self._txt_cache = (txt_enc, txt_mask, txt)
        else:
            txt_enc, txt_mask = self._txt_cache[0], self._txt_cache[1]
        return self._fuse(img_enc, txt_enc, txt_mask)

    def _get_encodings(self, img, txt):
        img_enc = self.vision_encoder(img)
        txt_enc, txt_mask = self.text_encoder(txt)
        return img_enc, txt_enc, txt_mask

    def _fuse(self, img_enc, txt_enc, txt_mask, return_weights=False):
        """
        Takes encoded features and fuses them with memory into the reasoning tokens.
        :param img_enc: Encoded image features from SigLIP, translated into d_model.
        :param txt_enc: Encoded text features from SigLIP, translated into d_model.
        :param txt_mask: Attention mask to ignore padding tokens in the text features.
        :return: Reasoning tokens (learned queries) of shape (B, lq_size, d_model)
        """
        mem, mem_mask = self.get_memory(img_enc.shape[0], img_enc.device)
        return self.qformer(img_enc, txt_enc, mem, txt_mask, mem_mask, return_weights=return_weights)

    def loss(self, img, txt, state, action, update_memory=True):
        reasoning = self.encode(img, txt)
        l, mem = self.action_expert.loss(action, state, reasoning, True)
        if update_memory:
            self.insert_memory(mem)
        return l

    @torch.no_grad()
    def act(self, img, txt, state, update_memory=True, return_trajectory=False):
        reasoning = self.encode(img, txt)
        out = self.action_expert.sample(reasoning, state, return_trajectory)
        actions, mem, traj = out[0], out[1], out[2] if return_trajectory else None
        if update_memory:
            self.insert_memory(mem)
        if return_trajectory:
            return actions, traj
        return actions

    def forward(self, img, txt, state, update_memory=True):
        return self.act(img, txt, state, update_memory)

    @torch.no_grad()
    def generate_dummy_inputs(self, B=1, device=torch.device("cpu")):
        img = (torch.rand(B, 3, 224, 224, device=device) * 255).to(torch.uint8)
        txt = self.text_encoder.tokenize(["A picture of a dog"] * B).to(device)
        state = torch.randn(B, 39, device=device)
        return img, txt, state


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
    import torch_tensorrt
    device = torch.device("cuda")
    cfg = VLAConfig()
    vla = VLA(cfg).to(device)
    B = 3
    img, txt, state = vla.generate_dummy_inputs(B, device)
    # ['aot_torch_tensorrt_aten', 'cudagraphs', 'inductor', 'openxla', 'tensorrt', 'torch_tensorrt', 'tvm']
    vla = torch.compile(vla, mode="max-autotune", fullgraph=True)
    total, trainable = print_model_counts(vla)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {total - trainable:,}")
    print(f"Total flops:      {total * 2 * 1e-9:.2f} GFLOPs")
    with torch.inference_mode():
        # No perf boost: 8.5
        # cache: 16.x
        # autocast: 21.0
        # autocast + compile: 21.3
        # autocast + cache: 26.4
        # autocast + cache + compile: 26.2
        encoded = vla.encode(img, txt)
        vla.reset()
        print("Encoded state shape:", encoded.shape)
        vla.act(img, txt, state)
        import time
        s = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(100):
                out = vla.act(img, txt, state)
        print(f"Time: {100/(time.perf_counter() - s)}")
        print(out.shape)
