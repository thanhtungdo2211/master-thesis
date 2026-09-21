"""Local training + FedAvg.

Only ONE model lives on the GPU. Each round, per client:
  1. load global weights into the model
  2. train locally for E epochs
  3. pull the trainable state dict back to CPU
  4. once all clients are done, weighted-average by n_k

Note on LoRA: naive FedAvg averages A and B SEPARATELY, whereas
  Avg(B) @ Avg(A) != Avg(B @ A)
This is a known aggregation bias (see FedIT). This baseline accepts it to
establish a reference point, and records it as a limitation.
"""
import copy
import math

import torch
from torch.utils.data import DataLoader

from .loss import sdm_loss
from .lora import load_trainable_state_dict, trainable_state_dict


def fedavg(states, weights):
    """states: list[dict[str, Tensor]] (CPU). weights: list[float]."""
    total = float(sum(weights))
    out = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in states[0].items()}
    for st, w in zip(states, weights):
        coef = w / total
        for k in out:
            out[k] += st[k].float() * coef
    return {k: v.to(states[0][k].dtype) for k, v in out.items()}


def round_lr(args, rnd):
    """LR for global round `rnd` (1-indexed), computed SERVER-SIDE.

    Every client selected in a round trains at the same lr, which is what keeps the
    centralized baseline and the FL runs on an identical lr trajectory -- the only
    difference between them stays the partitioning.
    """
    if args.lr_schedule == "constant":
        return args.lr
    w = max(args.warmup_rounds, 0)
    if w and rnd <= w:
        return args.lr * rnd / w
    progress = (rnd - w) / max(args.rounds - w, 1)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * (args.lr_min_ratio + (1.0 - args.lr_min_ratio) * cos)


def build_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def train_one_client(model, global_state, dataset, args, device, scaler=None,
                     lr=None, optimizer=None):
    """Returns (trainable state dict, n_samples, avg_loss).

    Two optimizer regimes, and which one you get is the difference between the two settings:

    optimizer=None  -- FL path (and the FL-symmetric centralized baseline). A fresh AdamW
        is built here and discarded at the end. This is correct FedAvg: clients are
        stateless, a client may be skipped for several rounds, and the global weights moved
        underneath it, so last round's moments point the wrong way.
    optimizer=<AdamW> -- true centralized path (--persistent_optimizer, 1 client only).
        The caller owns one optimizer that lives across all rounds, so Adam's moments
        accumulate over every step exactly as in ordinary non-federated training.

    Note the persistent case is only sound because `load_trainable_state_dict` copies
    IN-PLACE into the same Parameter objects the optimizer holds references to.
    """
    load_trainable_state_dict(model, global_state)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    owns_optimizer = optimizer is None
    if owns_optimizer:
        optimizer = build_optimizer(model, args)
    if lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = lr

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    logit_scale = 1.0 / args.temperature

    total_loss, n_step = 0.0, 0
    for _ in range(args.local_epochs):
        for step, batch in enumerate(loader):
            if args.debug_steps and step >= args.debug_steps:
                break
            pixel = batch["pixel_values"].to(device, non_blocking=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            pid = batch["pid"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    i_f = model.get_image_features(pixel_values=pixel)
                    t_f = model.get_text_features(input_ids=ids, attention_mask=mask)
                    loss = sdm_loss(i_f.float(), t_f.float(), pid, logit_scale)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                i_f = model.get_image_features(pixel_values=pixel)
                t_f = model.get_text_features(input_ids=ids, attention_mask=mask)
                loss = sdm_loss(i_f, t_f, pid, logit_scale)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()

            total_loss += loss.item()
            n_step += 1

    if owns_optimizer:
        del optimizer
    state = trainable_state_dict(model)
    avg_loss = total_loss / max(n_step, 1)
    return state, len(dataset), avg_loss


def clone_state(state):
    return {k: v.clone() for k, v in state.items()}


__all__ = ["fedavg", "train_one_client", "clone_state", "round_lr",
           "build_optimizer", "copy"]
