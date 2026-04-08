from torch import nn
from utils import VLAConfig

class StateEncoder(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super(StateEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model)
        )

    def forward(self, x):
        return self.net(x)