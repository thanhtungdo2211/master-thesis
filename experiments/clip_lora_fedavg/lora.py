"""Generic LoRA utilities: layer wrapper, injection, and trainable-state helpers.

These helpers are model-agnostic; CLIP-specific wiring lives in clip_model.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """out = W x + (B A) x * (alpha / r). A is kaiming-init, B is zero-init."""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return out + delta.to(out.dtype) * self.scaling


def inject_lora(model, rank, alpha, targets):
    """Wrap every nn.Linear named in `targets` with LoRALinear, anywhere in `model`."""
    targets = [t.strip() for t in targets if t.strip()]
    n_injected = 0
    for module in model.modules():
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
                n_injected += 1
    return n_injected


def mark_trainable(model, tuning):
    if tuning == "full":
        for p in model.parameters():
            p.requires_grad = True
        return
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_A" in n) or ("lora_B" in n)


def trainable_state_dict(model):
    """State dict containing ONLY trainable params — this is what gets sent to the server."""
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in keep}


def load_trainable_state_dict(model, state):
    own = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
