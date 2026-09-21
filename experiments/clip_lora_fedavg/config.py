"""Configuration for the FedAvg + LoRA + CLIP pipeline on RSTPReid."""
import argparse


def get_args(argv=None):
    p = argparse.ArgumentParser("clip_lora_fedavg: FedAvg + LoRA + CLIP on RSTPReid")

    # ---------- Paths ----------
    p.add_argument("--root", type=str, required=True,
                   help="RSTPReid folder, must contain imgs/ and data_captions.json")
    p.add_argument("--out_dir", type=str, default="./runs/exp")
    p.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch16")

    # ---------- Federated partitioning ----------
    p.add_argument("--partition", type=str, default="camera",
                   choices=["camera", "dirichlet", "iid", "centralized"],
                   help="'centralized' = 1 client holding the whole train set, i.e. the "
                        "non-FL LoRA baseline (FedAvg over a single client is a no-op, so "
                        "rounds x local_epochs is just the total number of epochs).")
    p.add_argument("--num_clients", type=int, default=15,
                   help="Only used for dirichlet/iid. For camera, #clients = #cameras present. "
                        "Ignored for centralized (always 1).")
    p.add_argument("--dirichlet_alpha", type=float, default=0.5)
    p.add_argument("--partition_file", type=str, default=None,
                   help="If set, load the partition from this JSON instead of generating one. "
                        "Format: {client_id: [image_index, ...]}")
    p.add_argument("--select_clients", type=str, default=None,
                   help="Comma-separated client ids (partition keys) to KEEP; all other clients' "
                        "data is dropped entirely (reduces total training volume, unlike "
                        "partial participation via --max_clients_per_round). "
                        "E.g. with --partition camera: '1,2,3' keeps only those 3 cameras.")

    # ---------- Federated ----------
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--client_fraction", type=float, default=1.0)

    # ---------- PEFT ----------
    p.add_argument("--tuning", type=str, default="lora", choices=["lora", "full"])
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,out_proj",
                   help="Linear layer names inside MHA to wrap with LoRA. "
                        "UP-Person only uses 'k_proj,v_proj'.")

    # ---------- Training ----------
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Peak lr (reached at the end of warmup). 1e-4 for LoRA, 1e-5 for full FT.")
    p.add_argument("--lr_schedule", type=str, default="cosine",
                   choices=["constant", "cosine"],
                   help="GLOBAL schedule over ROUNDS. The server computes the round's lr and "
                        "every selected client trains with it, so FL and the centralized "
                        "baseline see the same lr trajectory. 'constant' reproduces runs made "
                        "before this flag existed (no warmup, no decay).")
    p.add_argument("--warmup_rounds", type=int, default=3,
                   help="Linear warmup: round r <= N uses lr * r/N. Ignored when "
                        "--lr_schedule constant.")
    p.add_argument("--lr_min_ratio", type=float, default=0.01,
                   help="Cosine decays to lr * this ratio at the final round.")
    p.add_argument("--weight_decay", type=float, default=0.02)
    p.add_argument("--persistent_optimizer", action="store_true",
                   help="CENTRALIZED BASELINE ONLY -- requires exactly 1 client. Keeps ONE "
                        "AdamW alive across all rounds, i.e. real non-FL training where Adam's "
                        "moments accumulate over every step. Rejected with >1 client: FedAvg "
                        "clients are stateless (a client may sit out rounds, and the global "
                        "weights move under it), so sharing moments across clients is not "
                        "FedAvg. Without this flag a centralized run still goes through the FL "
                        "code path and rebuilds AdamW each round -- valid, and the right choice "
                        "when you want C1 vs FL to differ ONLY in partitioning.")
    p.add_argument("--temperature", type=float, default=0.02, help="SDM temperature")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")

    # ---------- Image / text ----------
    p.add_argument("--img_h", type=int, default=384)
    p.add_argument("--img_w", type=int, default=128)
    p.add_argument("--text_len", type=int, default=77)

    # ---------- Eval / logging ----------
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--ckpt_every", type=int, default=1,
                   help="Save a resumable checkpoint every this many rounds. Checkpoint size "
                        "= uplink_mb_fp32 (tiny for LoRA, ~model size for full fine-tuning).")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing checkpoint.pt in --out_dir and start from round 1. "
                        "Without this flag, if --out_dir already has a checkpoint, training "
                        "AUTO-RESUMES from the last saved round (useful after a Colab disconnect: "
                        "just rerun the exact same command).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_clients_per_round", type=int, default=0,
                   help="0 = no limit. Set >0 for fast debugging (random subsample each round).")
    p.add_argument("--debug_steps", type=int, default=0,
                   help=">0: each client only runs this many steps. For smoke testing.")

    return p.parse_args(argv)
