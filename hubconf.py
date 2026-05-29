"""Torch Hub entrypoints for FoL.

Usage examples:
    # 1. Default FoL model (ViT-L)
    model = torch.hub.load("chenshunpeng/FoL", "FoL", pretrained=True, trust_repo=True)

    # 2. Explicit entrypoints
    model = torch.hub.load("chenshunpeng/FoL", "fol_vitl14", pretrained=True, trust_repo=True)
    model = torch.hub.load("chenshunpeng/FoL", "fol_vitb14", pretrained=True, trust_repo=True)
"""

from typing import Dict, Any
import torch
from network_FoL import FoLNet

dependencies = ["torch", "torchvision", "util"]

_WEIGHT_URLS: Dict[str, str] = {
    "vitl": "https://huggingface.co/shunpeng/FoL/resolve/main/FoL_large.pth",
    "vitb": "https://huggingface.co/shunpeng/FoL/resolve/main/FoL_base.pth",
}

_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "vitl": {"model_name": "dinov2_vitl14", "num_channels": 1024},
    "vitb": {"model_name": "dinov2_vitb14", "num_channels": 768},
}

def _build_fol(arch: str, pretrained: bool = True, progress: bool = True, map_location="cpu", **kwargs) -> FoLNet:
    if arch not in _MODEL_CONFIGS:
        raise ValueError(f"Unsupported architecture: {arch}. Choose from {list(_MODEL_CONFIGS.keys())}")

    config = _MODEL_CONFIGS[arch]
    model = FoLNet(
        num_channels=config["num_channels"],
        model_name=config["model_name"],
        num_trainable_blocks=kwargs.get("num_trainable_blocks", 4)
    )  

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            _WEIGHT_URLS[arch], map_location=map_location, progress=progress
        )
        # Use loose loading by default to prevent script crashes during user fine-tuning

        model_state_dict = checkpoint["model_state_dict"]
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

        strict = kwargs.get("strict", False)
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=strict)
        if strict and (missing_keys or unexpected_keys):
           raise RuntimeError(f"Strict loading failed. Missing: {missing_keys}; Unexpected: {unexpected_keys}")

    return model

def FoL(pretrained: bool = True, backbone: str = "vitl", progress: bool = True, map_location="cpu", **kwargs) -> FoLNet:
    """Universal factory entrypoint for FoL models."""
    return _build_fol(backbone, pretrained=pretrained, progress=progress, map_location=map_location, **kwargs)

def fol_vitl14(pretrained: bool = True, progress: bool = True, map_location="cpu", **kwargs) -> FoLNet:
    """FoL model with ViT-L/14 backbone."""
    return _build_fol("vitl", pretrained=pretrained, progress=progress, map_location=map_location, **kwargs)

def fol_vitb14(pretrained: bool = True, progress: bool = True, map_location="cpu", **kwargs) -> FoLNet:
    """FoL model with ViT-B/14 backbone."""
    return _build_fol("vitb", pretrained=pretrained, progress=progress, map_location=map_location, **kwargs)