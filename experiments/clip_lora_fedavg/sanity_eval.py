"""Memorization sanity check: evaluate a trained checkpoint on a TRAIN sample,
using the exact same evaluate() code path as the real (test-set) evaluation.

This directly tests for overfitting/memorization: if the model has memorized
the training set rather than learned to generalize, R@1 on a train sample
will be dramatically higher than on the held-out test set (e.g. 90%+ vs 57%).
If the two numbers are close, that is evidence AGAINST memorization.

Usage:
    python -m clip_lora_fedavg.sanity_eval \\
        --run_dir runs/B5_camera_lora4 --ckpt best.pt \\
        --root /path/to/RSTPReid --n_ids 200
"""
import argparse
import random

import numpy as np
import torch

from .data import build_transforms, load_annotations
from .evaluate import evaluate
from .infer import load_checkpoint_state, load_model_args
from .clip_model import build_model
from .lora import load_trainable_state_dict


def get_args(argv=None):
    p = argparse.ArgumentParser("Train-sample sanity eval for a clip_lora_fedavg checkpoint")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="best.pt")
    p.add_argument("--root", type=str, required=True,
                   help="RSTPReid folder (imgs/ + data_captions.json)")
    p.add_argument("--n_ids", type=int, default=200,
                   help="Number of train identities to sample, matched to the "
                        "test set's 200 identities so the comparison is apples-to-apples")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def sample_train_subset(train_items, n_ids, seed):
    rng = random.Random(seed)
    ids = sorted({it["pid"] for it in train_items})
    chosen = set(rng.sample(ids, min(n_ids, len(ids))))
    return [it for it in train_items if it["pid"] in chosen]


def main(argv=None):
    args = get_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_args = load_model_args(args.run_dir)
    model, tokenizer, info = build_model(model_args)
    state = load_checkpoint_state(args.run_dir, args.ckpt)
    load_trainable_state_dict(model, state)
    model.to(device).eval()

    splits = load_annotations(args.root)
    train_sample = sample_train_subset(splits["train"], args.n_ids, args.seed)
    n_images = len(train_sample)
    print(f"[sanity_eval] sampled {len(set(it['pid'] for it in train_sample))} train identities "
          f"/ {n_images} train images (model WAS trained on these exact images)")

    eval_tf = build_transforms(model_args.img_h, model_args.img_w, is_train=False)
    metrics = evaluate(model, train_sample, args.root, eval_tf, tokenizer,
                       model_args.text_len, device)

    print(f"[sanity_eval] R@1 on TRAIN sample = {metrics['R@1']:.2f} "
          f"(R@5 {metrics['R@5']:.2f}, mAP {metrics['mAP']:.2f})")
    print("Compare this to the checkpoint's R@1 on the real held-out test set "
          "(see log.csv in --run_dir). If the train-sample number is dramatically "
          "higher (e.g. 90%+ vs the test R@1), that is evidence of memorization. "
          "If the two are close, that argues against overfitting.")


if __name__ == "__main__":
    main()
