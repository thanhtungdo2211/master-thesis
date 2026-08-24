"""CLIP ViT-B/16 wiring: non-square position embeddings + LoRA/full-FT setup.

Only LoRA params (or all params, for full fine-tuning) end up trainable,
so `lora.trainable_state_dict` decides exactly what gets transmitted in FL.
"""
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from .lora import count_params, inject_lora, mark_trainable


def _patched_vision_embed_forward(self, pixel_values, *args, **kwargs):
    """Skip HF's square-image assertion so arbitrary HxW input is accepted."""
    target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
    class_embeds = self.class_embedding.expand(pixel_values.shape[0], 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    return embeddings + self.position_embedding(self.position_ids)


def resize_position_embedding(clip_model, img_h, img_w, patch=16):
    """Bicubic-interpolate the 14x14 position embedding grid to (img_h/16) x (img_w/16).

    For 384x128 this yields a 24x8 = 192-patch grid (originally 196).
    """
    emb = clip_model.vision_model.embeddings
    old = emb.position_embedding.weight.data
    cls_pos, grid_pos = old[:1], old[1:]
    old_side = int(round(math.sqrt(grid_pos.shape[0])))
    dim = grid_pos.shape[1]

    gh, gw = img_h // patch, img_w // patch
    if old_side == gh and old_side == gw:
        emb.forward = types.MethodType(_patched_vision_embed_forward, emb)
        return gh * gw

    grid = grid_pos.reshape(1, old_side, old_side, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(gh, gw), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(gh * gw, dim)
    new_w = torch.cat([cls_pos, grid], dim=0)

    new_emb = nn.Embedding(new_w.shape[0], dim)
    new_emb.weight.data.copy_(new_w)
    emb.position_embedding = new_emb
    emb.num_patches = gh * gw
    emb.num_positions = gh * gw + 1
    emb.register_buffer(
        "position_ids", torch.arange(emb.num_positions).expand((1, -1)), persistent=False
    )
    emb.forward = types.MethodType(_patched_vision_embed_forward, emb)
    return gh * gw


def build_model(args):
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_name)
    model = CLIPModel.from_pretrained(args.clip_name)

    n_patches = resize_position_embedding(model, args.img_h, args.img_w)

    for p in model.parameters():
        p.requires_grad = False

    n_lora_modules = 0
    if args.tuning == "lora":
        n_lora_modules = inject_lora(model, args.lora_rank, args.lora_alpha,
                                     args.lora_targets.split(","))
    mark_trainable(model, args.tuning)

    total, trainable = count_params(model)
    info = {
        "n_patches": n_patches,
        "n_lora_modules": n_lora_modules,
        "total_params": total,
        "trainable_params": trainable,
        "uplink_mb_fp32": trainable * 4 / (1024 ** 2),
    }
    return model, tokenizer, info


@torch.no_grad()
def encode_image(model, pixel_values):
    f = model.get_image_features(pixel_values=pixel_values)
    return f / f.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_text(model, input_ids, attention_mask):
    f = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
    return f / f.norm(dim=-1, keepdim=True)
