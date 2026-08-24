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


def train_one_client(model, global_state, dataset, args, device, scaler=None):
    """Returns (trainable state dict, n_samples, avg_loss)."""
    load_trainable_state_dict(model, global_state)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

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

    del optimizer
    state = trainable_state_dict(model)
    avg_loss = total_loss / max(n_step, 1)
    return state, len(dataset), avg_loss


def clone_state(state):
    return {k: v.clone() for k, v in state.items()}


__all__ = ["fedavg", "train_one_client", "clone_state", "copy"]
