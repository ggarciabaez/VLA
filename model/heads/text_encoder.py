from transformers import SiglipTextModel
from torch import nn
from utils import VLAConfig, freeze_except_last_n_layers

class TextEncoder(nn.Module):
    def __init__(self, cfg: VLAConfig):
        super(TextEncoder, self).__init__()
        self.cfg = cfg
        self.backbone = SiglipTextModel.from_pretrained(cfg.siglip_model_id)
        self.hidden_size = int(self.backbone.config.hidden_size)
        # TODO: could be a good idea to just always have a projection for a new embedding space
        self.proj = nn.Identity() if self.cfg.d_model <= 0 or self.cfg.d_model == self.hidden_size else nn.Linear(self.hidden_size, self.cfg.d_model)
        self.backbone = freeze_except_last_n_layers(self.backbone, cfg.n_trainable, model_type="text")


    def forward(self, tokens):
        """
        Encodes text tokens into (B, T, d_model) tensors. T is variable depending on string length.
        :param tokens: Tokenized strings as a tensor. Remember to move to the proper device!
        :return: Text embeddings of dimension d_model, as well as the padding mask.
        """
        return self.proj(self.backbone(tokens).last_hidden_state), tokens==1


if __name__ == "__main__":
    from transformers import SiglipTokenizer
    cfg = VLAConfig()
    txt = TextEncoder(cfg)

    tknzr: SiglipTokenizer = SiglipTokenizer.from_pretrained(cfg.siglip_model_id)
    strings = ["a picture of a dog", "a picture of a cat", "a picture of a person in a hoodie", "a very long instruction word consisting of various connectors and some redundant explanations", "short"]
    tokens = tknzr.encode(strings, padding="max_length", return_tensors="pt")
    print(txt.forward(tokens)[0].size())