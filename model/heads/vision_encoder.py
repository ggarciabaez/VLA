from transformers import SiglipVisionModel, logging
logging.set_verbosity_error()
from torch import nn
import torch
from torch.functional import F
from utils import VLAConfig, freeze_except_last_n_layers

class VisionEncoder(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super(VisionEncoder, self).__init__()
        self.cfg = cfg
        self.backbone = SiglipVisionModel.from_pretrained(cfg.siglip_model_id)

        self.hidden_size = int(self.backbone.config.hidden_size)

        # normalize
        self.register_buffer("mean", torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.image_size = int(cfg.img_size or getattr(self.backbone.config, "image_size", 224))
        if cfg.d_model <= 0:
            cfg.d_model = self.hidden_size
        self.proj = nn.Linear(self.hidden_size, cfg.d_model)
        self.backbone = freeze_except_last_n_layers(self.backbone, cfg.n_trainable, model_type="vision")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()

        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        x = (x - self.mean) / self.std  # normalize
        out = self.backbone(pixel_values=x, return_dict=True)
        return self.proj(out.last_hidden_state)

if __name__ == "__main__":
    # test the model
    import cv2
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    cfg = VLAConfig()
    encoder = VisionEncoder(cfg)
    out = encoder.forward(
        torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(torch.float32)/255.0
    )
    print(out.shape)
