"""Entry point: FedAvg + LoRA + CLIP on RSTPReid."""
import csv
import json
import os
import random
import time

import numpy as np
import torch

from .clip_model import build_model
from .config import get_args
from .data import (build_partition, build_transforms, load_annotations,
                   partition_stats, TrainPairDataset)
from .evaluate import evaluate
from .federated import fedavg, train_one_client
from .lora import load_trainable_state_dict, trainable_state_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv=None):
    args = get_args(argv)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(args.out_dir, "checkpoint.pt")
    resume = (not args.fresh) and os.path.exists(ckpt_path)

    # ---------------- data ----------------
    splits = load_annotations(args.root)
    train_items = splits["train"]
    test_items = splits.get("test", splits.get("val"))
    print(f"[data] train {len(train_items)} images | test {len(test_items)} images")

    if resume:
        # Reuse the exact partition from the interrupted run instead of rebuilding it,
        # so resumed training sees exactly the same clients/data as before.
        with open(os.path.join(args.out_dir, "partition.json")) as f:
            partition = json.load(f)
        print(f"[resume] loaded existing partition from {args.out_dir}/partition.json "
              f"({len(partition)} clients)")
    elif args.partition_file:
        with open(args.partition_file) as f:
            partition = {k: list(v) for k, v in json.load(f).items()}
    else:
        partition = build_partition(train_items, args.partition, args.num_clients,
                                    args.dirichlet_alpha, args.seed)

        if args.select_clients:
            keep = [c.strip() for c in args.select_clients.split(",") if c.strip()]
            missing = [c for c in keep if c not in partition]
            if missing:
                raise ValueError(
                    f"--select_clients references unknown client ids: {missing}. "
                    f"Valid clients: {sorted(partition.keys())}"
                )
            dropped = set(partition.keys()) - set(keep)
            partition = {c: partition[c] for c in keep}
            print(f"[select_clients] keeping {keep}, dropping {len(dropped)} clients "
                  f"({sorted(dropped)}) -> {sum(len(v) for v in partition.values())} train images")

        rows, summary = partition_stats(partition, train_items)
        print(f"[partition] {args.partition} -> {summary['n_clients']} clients | "
              f"images/client {summary['images_min']}-{summary['images_max']} "
              f"(mean {summary['images_mean']:.0f}, CV {summary['cv_images']:.3f}) | "
              f"mean pairwise pid Jaccard {summary['mean_pairwise_jaccard_pid']:.3f}")
        with open(os.path.join(args.out_dir, "partition_stats.json"), "w") as f:
            json.dump({"summary": summary, "clients": rows}, f, indent=2)
        with open(os.path.join(args.out_dir, "partition.json"), "w") as f:
            json.dump(partition, f)

    # ---------------- model ----------------
    model, tokenizer, info = build_model(args)
    model.to(device)
    print(f"[model] {info['n_patches']} patches | LoRA modules {info['n_lora_modules']} | "
          f"trainable {info['trainable_params']:,} / {info['total_params']:,} "
          f"({100*info['trainable_params']/info['total_params']:.2f}%) | "
          f"uplink {info['uplink_mb_fp32']:.2f} MB/client/round")

    train_tf = build_transforms(args.img_h, args.img_w, is_train=True)
    eval_tf = build_transforms(args.img_h, args.img_w, is_train=False)

    client_ids = sorted(partition.keys())
    client_sets = {
        cid: TrainPairDataset(train_items, partition[cid], args.root,
                              train_tf, tokenizer, args.text_len)
        for cid in client_ids
    }

    scaler = torch.amp.GradScaler("cuda") if (args.amp and device == "cuda") else None

    # ---------------- logging / resume state ----------------
    log_path = os.path.join(args.out_dir, "log.csv")
    client_log_path = os.path.join(args.out_dir, "client_loss.csv")
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump({**vars(args), **info}, f, indent=2)

    if resume:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        global_state = ckpt["global_state"]
        start_round = ckpt["round"] + 1
        best_r1 = ckpt["best_r1"]
        cum_uplink = ckpt["cum_uplink"]
        t0 = time.time() - ckpt["elapsed_s"]
        print(f"[resume] continuing from round {start_round}/{args.rounds} "
              f"(best R@1 so far {best_r1:.2f}, uplink so far {cum_uplink/1024:.2f} GB)")
    else:
        global_state = trainable_state_dict(model)
        start_round, best_r1, cum_uplink, t0 = 1, 0.0, 0.0, time.time()
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["round", "loss", "R@1", "R@5", "R@10", "mAP", "mINP",
                 "cum_uplink_MB", "elapsed_s"])
        with open(client_log_path, "w", newline="") as f:
            csv.writer(f).writerow(["round", "client_id", "n_samples", "loss"])

    n_per_round = max(1, int(round(args.client_fraction * len(client_ids))))
    if args.max_clients_per_round:
        n_per_round = min(n_per_round, args.max_clients_per_round)

    # ---------------- FedAvg loop ----------------
    for rnd in range(start_round, args.rounds + 1):
        selected = (client_ids if n_per_round >= len(client_ids)
                    else random.sample(client_ids, n_per_round))

        states, weights, losses = [], [], []
        for cid in selected:
            st, n, ls = train_one_client(model, global_state, client_sets[cid],
                                         args, device, scaler)
            states.append(st)
            weights.append(n)
            losses.append(ls)

        with open(client_log_path, "a", newline="") as f:
            w = csv.writer(f)
            for cid, n, ls in zip(selected, weights, losses):
                w.writerow([rnd, cid, n, f"{ls:.4f}"])

        global_state = fedavg(states, weights)
        cum_uplink += info["uplink_mb_fp32"] * len(selected)
        mean_loss = float(np.mean(losses))

        line = f"[round {rnd:03d}/{args.rounds}] loss {mean_loss:.4f} | " \
               f"cumulative uplink {cum_uplink/1024:.2f} GB | {time.time()-t0:.0f}s"

        metrics = {k: "" for k in ("R@1", "R@5", "R@10", "mAP", "mINP")}
        if rnd % args.eval_every == 0 or rnd == args.rounds:
            load_trainable_state_dict(model, global_state)
            metrics = evaluate(model, test_items, args.root, eval_tf, tokenizer,
                               args.text_len, device, num_workers=args.num_workers)
            line += (f" | R@1 {metrics['R@1']:.2f} R@5 {metrics['R@5']:.2f} "
                     f"R@10 {metrics['R@10']:.2f} mAP {metrics['mAP']:.2f} "
                     f"mINP {metrics['mINP']:.2f}")
            if metrics["R@1"] > best_r1:
                best_r1 = metrics["R@1"]
                torch.save({"round": rnd, "state": global_state, "metrics": metrics},
                           os.path.join(args.out_dir, "best.pt"))
                line += "  <- best"
        print(line, flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                rnd, f"{mean_loss:.4f}",
                metrics["R@1"], metrics["R@5"], metrics["R@10"],
                metrics["mAP"], metrics["mINP"],
                f"{cum_uplink:.1f}", f"{time.time()-t0:.0f}"])

        if rnd % args.ckpt_every == 0 or rnd == args.rounds:
            torch.save({
                "round": rnd,
                "global_state": global_state,
                "best_r1": best_r1,
                "cum_uplink": cum_uplink,
                "elapsed_s": time.time() - t0,
            }, ckpt_path)

    print(f"\n[done] best R@1 = {best_r1:.2f} | total uplink {cum_uplink/1024:.2f} GB "
          f"({args.rounds} rounds x {len(client_ids)} clients)")


if __name__ == "__main__":
    main()
