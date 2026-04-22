from transformers import SiglipVisionModel, logging
logging.set_verbosity_error()
from torch import nn
import torch
from torch.functional import F
from model.utils import VLAConfig, freeze_except_last_n_layers

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
        x = (x - self.mean) / self.std  # normalize
        if x.ndim == 4:
            return self.singleforward(x)
        elif x.ndim == 5:
            return self.multiforward(x)
        else:
            raise ValueError("Unsupported amount of dimensions.")

    def singleforward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        out = self.backbone(pixel_values=x, return_dict=True)
        return self.proj(out.last_hidden_state)

    def multiforward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x.shape[0], 0, self.cfg.d_model, device=x.device)
        rsz = x[:, 0].shape[-2:] != (self.image_size, self.image_size)
        for i in range(x.shape[1]):
            if rsz:
                img = F.interpolate(x[:, i], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            else:
                img = x[:, i]
            out = torch.cat([out, self.proj(self.backbone(pixel_values=img, return_dict=True).last_hidden_state)], axis=1)

        return out

if __name__ == "__main__":
    # test the model
    import cv2
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    ret, frame2 = cam.read()
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
    frame2 = torch.from_numpy(frame2).permute(2, 0, 1).unsqueeze(0)
    frames = torch.cat([frame.unsqueeze(1), frame2.unsqueeze(1)], axis=1)
    print(frames.shape)
    cam.release()
    cfg = VLAConfig()
    encoder = VisionEncoder(cfg)
    out = encoder.forward(
        frame
    )
    print(out.shape)
