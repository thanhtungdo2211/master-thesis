"""Plot per-client loss curves from a run's client_loss.csv.

Usage:
    python -m clip_lora_fedavg.plot_client_loss --run_dir runs/B5_camera_lora4
"""
import argparse
import csv
import os
from collections import defaultdict


def get_args(argv=None):
    p = argparse.ArgumentParser("plot per-client FedAvg loss")
    p.add_argument("--run_dir", type=str, required=True,
                   help="Run directory containing client_loss.csv")
    p.add_argument("--out", type=str, default=None,
                   help="Output image path. Default: <run_dir>/client_loss.png")
    return p.parse_args(argv)


def load_client_loss(path):
    """Returns dict[client_id] -> list[(round, loss)], sorted by round."""
    by_client = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_client[row["client_id"]].append(
                (int(row["round"]), float(row["loss"])))
    for cid in by_client:
        by_client[cid].sort(key=lambda t: t[0])
    return dict(by_client)


def main(argv=None):
    import matplotlib.pyplot as plt

    args = get_args(argv)
    csv_path = os.path.join(args.run_dir, "client_loss.csv")
    out_path = args.out or os.path.join(args.run_dir, "client_loss.png")

    by_client = load_client_loss(csv_path)
    if not by_client:
        raise SystemExit(f"no rows found in {csv_path}")

    by_round = defaultdict(list)
    for cid, points in by_client.items():
        for rnd, loss in points:
            by_round[rnd].append(loss)
    rounds = sorted(by_round)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    show_legend = len(by_client) <= 15
    for cid, points in sorted(by_client.items()):
        xs = [r for r, _ in points]
        ys = [l for _, l in points]
        ax[0].plot(xs, ys, marker="o", markersize=3, label=cid if show_legend else None)
    ax[0].set_xlabel("round")
    ax[0].set_ylabel("loss")
    ax[0].set_title(f"Per-client loss ({os.path.basename(args.run_dir)})")
    ax[0].grid(alpha=.3)
    if show_legend:
        ax[0].legend(fontsize=7, ncol=2)

    ax[1].boxplot([by_round[r] for r in rounds], positions=rounds, widths=0.6)
    ax[1].set_xlabel("round")
    ax[1].set_ylabel("loss spread across clients")
    ax[1].set_title("Client loss heterogeneity per round")
    ax[1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot_client_loss] saved {out_path}")

    last_round = rounds[-1]
    last = {cid: dict(points)[last_round] for cid, points in by_client.items()
            if last_round in dict(points)}
    worst = max(last, key=last.get)
    best = min(last, key=last.get)
    losses = list(last.values())
    mean = sum(losses) / len(losses)
    std = (sum((l - mean) ** 2 for l in losses) / len(losses)) ** 0.5
    print(f"[plot_client_loss] round {last_round}: "
          f"highest loss {worst} ({last[worst]:.4f}), "
          f"lowest loss {best} ({last[best]:.4f}), "
          f"std across clients {std:.4f}")


if __name__ == "__main__":
    main()
