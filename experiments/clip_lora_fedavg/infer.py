"""Single (image, text) inference with a trained LoRA checkpoint.

LoRA does not need a special "decode" step at inference time: LoRALinear.forward
already adds (B @ A) * scaling to the base Linear's output during the forward
pass. The only requirement is rebuilding the SAME architecture used during
training (LoRA injected at the same target modules, same rank/alpha) BEFORE
loading the checkpoint, since the state dict only contains LoRA parameter
names (lora_A/lora_B) — it will not load into a plain CLIPModel.

Usage:
    python -m clip_lora_fedavg.infer \\
        --run_dir runs/B5_camera_lora4 --ckpt best.pt \\
        --image /path/to/some_person.jpg \\
        --text "A man in a grey jacket and black trousers."
"""
import argparse
import json
import os
from types import SimpleNamespace

import torch
from PIL import Image

from .clip_model import build_model
from .data import build_transforms
from .lora import load_trainable_state_dict


def get_infer_args(argv=None):
    p = argparse.ArgumentParser("Single-pair inference with a clip_lora_fedavg checkpoint")
    p.add_argument("--run_dir", type=str, required=True,
                   help="Run output dir containing args.json and the checkpoint "
                        "(e.g. runs/B5_camera_lora4)")
    p.add_argument("--ckpt", type=str, default="best.pt",
                   help="Checkpoint filename inside --run_dir: 'best.pt' or 'checkpoint.pt'")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--device", type=str, default=None,
                   help="Defaults to cuda if available, else cpu")
    return p.parse_args(argv)


def load_model_args(run_dir):
    """Rebuild the exact architecture used for training from the run's args.json."""
    with open(os.path.join(run_dir, "args.json")) as f:
        saved = json.load(f)
    # build_model only reads these fields; keep it minimal and explicit.
    return SimpleNamespace(
        clip_name=saved["clip_name"],
        tuning=saved["tuning"],
        lora_rank=saved["lora_rank"],
        lora_alpha=saved["lora_alpha"],
        lora_targets=saved["lora_targets"],
        img_h=saved["img_h"],
        img_w=saved["img_w"],
        text_len=saved["text_len"],
    )


def load_checkpoint_state(run_dir, ckpt_name):
    ckpt = torch.load(os.path.join(run_dir, ckpt_name), map_location="cpu", weights_only=False)
    # main.py saves best.pt under key "state" and checkpoint.pt under "global_state".
    return ckpt.get("state", ckpt.get("global_state"))


def main(argv=None):
    args = get_infer_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_args = load_model_args(args.run_dir)
    model, tokenizer, info = build_model(model_args)
    state = load_checkpoint_state(args.run_dir, args.ckpt)
    load_trainable_state_dict(model, state)
    model.to(device).eval()
    print(f"[model] {model_args.clip_name} + LoRA r={model_args.lora_rank} "
          f"({info['trainable_params']:,} trainable params) on {device}")

    transform = build_transforms(model_args.img_h, model_args.img_w, is_train=False)
    image = Image.open(args.image).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device)

    tok = tokenizer(args.text, padding="max_length", truncation=True,
                    max_length=model_args.text_len, return_tensors="pt")

    with torch.no_grad():
        image_feat = model.get_image_features(pixel_values=pixel_values)
        text_feat = model.get_text_features(
            input_ids=tok["input_ids"].to(device),
            attention_mask=tok["attention_mask"].to(device),
        )
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        cosine_sim = (image_feat @ text_feat.t()).item()

    print(f"[result] cosine similarity = {cosine_sim:.4f}")
    print("(for reference: matching RSTPReid pairs from this pipeline typically "
          "score noticeably higher than random image/caption pairs; there is no "
          "fixed threshold — compare relative scores across candidates instead.)")


if __name__ == "__main__":
    main()
