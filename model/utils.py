from dataclasses import dataclass, field

@dataclass
class VLAConfig:
    model_name: str = "vla"  # not used anywhere, still nice for metadata
    siglip_model_id: str = "google/siglip-base-patch16-224"
    n_trainable: int = 4
    dropout: float = 0.1

    # Fusion transformer
    d_model: int = 768  # set this at 0 to use siglip default
    n_heads: int = 6
    n_layers: int = 8
    # ffn_dim is d_model * 4

    # Action expert
    action_layers: int = 4
    chunk_size: int = 10
    flow_steps: int = 4
    flow_dim: int = 256
    action_heads: int = 4

    state_dim: int = 39  # your input twin
    action_dim: int = 4  # your output twin

    # head configs
    img_size: int = 224

    # normalization stats, make sure to catch these before training
    action_mean: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    action_std: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    type_ids: dict = field(default_factory=lambda: {
        "vision": 0,
        "text": 1,
        "state": 2,
        # other types of data get added here, things like oh idk, depth?? wink wink
    })



def freeze_except_last_n_layers(model, n_unfrozen_layers, model_type="vision"):
    """
    Freezes all layers in a SigLIP model except the last `n_unfrozen_layers`.
    """
    # 1. Freeze the entire model first
    for param in model.parameters():
        param.requires_grad = False

    # 2. Locate the encoder layers and the final LayerNorm based on model type
    if model_type == "vision":
        layers = model.vision_model.encoder.layers
        post_layernorm = model.vision_model.post_layernorm
    elif model_type == "text":
        layers = model.text_model.encoder.layers
        post_layernorm = model.text_model.final_layer_norm
    else:
        raise ValueError("model_type must be 'vision' or 'text'")

    # 3. Unfreeze the last n encoder layers
    if n_unfrozen_layers > 0:
        for layer in layers[-n_unfrozen_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    # 4. Unfreeze the post_layernorm (Recommended)
    # If you are fine-tuning the final layers, you almost always want
    # to fine-tune the final normalization layer as well.
    for param in post_layernorm.parameters():
        param.requires_grad = True

    return model
