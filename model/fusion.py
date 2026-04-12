import torch
from torch import nn
from utils import VLAConfig


class FusionTransformer(nn.Module):
    def __init__(self, cfg: VLAConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.tf_layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads, dropout=cfg.dropout,
                                                   dim_feedforward=cfg.d_model * 4, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(self.tf_layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.embedding = nn.Embedding(len(cfg.type_ids), cfg.d_model)

        self.iembed = torch.tensor(self.cfg.type_ids["vision"], device=device)
        self.tembed = torch.tensor(self.cfg.type_ids["text"], device=device)
        self.state_embed = torch.tensor(self.cfg.type_ids["state"], device=device)


    def forward(self, img: torch.Tensor, txt: torch.Tensor, state: torch.Tensor, txt_pad_mask=None):
        """
        Fuses all the relevant information into a general representation embedding, or something like that.
        :param img: Encoded image embeddings of shape (B, P, d_model), where P is patches
        :param txt: Encoded text embeddings of shape (B, T, d_model)
        :param state: Encoded state embeddings of shape (B, 1, d_model)
        :param txt_pad_mask: Optional padding mask for the text embeddings.
        TODO: add an extra field for new data or more images or stuff. anyways.
        :return:
        """

        B = img.size(0)
        P = img.size(1)

        img += self.embedding(self.iembed)
        txt += self.embedding(self.tembed)
        state += self.embedding(self.state_embed)

        tokens = torch.cat([img, txt, state], dim=1)
        if txt_pad_mask is not None:
            img_mask = torch.zeros(B, P, dtype=torch.bool, device=tokens.device)
            state_mask = torch.zeros(B, 1, dtype=torch.bool, device=tokens.device)
            pad_mask = torch.cat([img_mask, txt_pad_mask, state_mask], dim=1)
        else:
            pad_mask = None

            # --- transformer ---
        out = self.tf(tokens, src_key_padding_mask=pad_mask)
        return out  # (B, N_tokens, d_model)


class CrossAttnFusion(nn.Module):
    def __init__(self, cfg: VLAConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.tf = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads, dropout=cfg.dropout,
                dim_feedforward=cfg.d_model * 4, batch_first=True, norm_first=True
            ),
            num_layers=cfg.n_layers,
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.embedding = nn.Embedding(len(cfg.type_ids), cfg.d_model)

        self.iembed = torch.tensor(self.cfg.type_ids["vision"], device=device)
        self.tembed = torch.tensor(self.cfg.type_ids["text"], device=device)
        self.state_embed = torch.tensor(self.cfg.type_ids["state"], device=device)

    def forward(self, img: torch.Tensor, txt: torch.Tensor, state: torch.Tensor, txt_pad_mask=None):
        """
        Fuses all the relevant information into a general representation embedding, or something like that.
        :param img: Encoded image embeddings of shape (B, P, d_model), where P is patches
        :param txt: Encoded text embeddings of shape (B, T, d_model)
        :param state: Encoded state embeddings of shape (B, 1, d_model)
        :param txt_pad_mask: Optional padding mask for the text embeddings.
        TODO: add an extra field for new data or more images or stuff. anyways.
        :return:
        """

        B = img.size(0)
        P = img.size(1)

        img += self.embedding(self.iembed)
        txt += self.embedding(self.tembed)
        state += self.embedding(self.state_embed)

        kv = torch.cat([img, state], dim=1)  # TODO: we could pass the state to the AE, not here.
            # --- transformer ---
        out = self.tf.forward(tgt=txt, memory=kv, tgt_key_padding_mask=txt_pad_mask)
        return self.norm(out)  # (B, N_tokens, d_model)