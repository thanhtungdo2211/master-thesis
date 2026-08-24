"""RSTPReid loading, federated partitioning, and Dataset classes for train/eval."""
import json
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# CLIP normalization stats
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------- annotations
def load_annotations(root):
    """Return dict split -> list of {img_path, pid, captions, cam_id}."""
    ann_path = os.path.join(root, "data_captions.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"{ann_path} not found. Check --root: it must contain "
            "imgs/ and data_captions.json"
        )
    with open(ann_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    splits = defaultdict(list)
    for entry in raw:
        img_path = entry["img_path"]
        item = {
            "img_path": img_path,
            "pid": int(entry["id"]),
            "captions": entry["captions"],
            "cam_id": parse_camera_id(img_path),
        }
        splits[entry["split"]].append(item)

    for k in splits:
        splits[k].sort(key=lambda x: x["img_path"])
    return dict(splits)


def parse_camera_id(img_path):
    """Extract the camera id from a filename.

    RSTPReid uses the Market-1501 format: pid_c{camid}_seq.jpg
    Fallback: MSMT17-style pid_xxx_camid_...
    Returns -1 if it cannot be parsed.
    """
    name = os.path.basename(img_path)
    m = re.search(r"_c(\d+)", name)
    if m:
        return int(m.group(1))
    parts = name.split("_")
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    return -1


# ---------------------------------------------------------------- partitions
def build_partition(train_items, strategy, num_clients, alpha, seed):
    """Return dict {client_id(str): [index into train_items]}."""
    rng = np.random.RandomState(seed)
    n = len(train_items)

    if strategy == "camera":
        buckets = defaultdict(list)
        for i, it in enumerate(train_items):
            buckets[it["cam_id"]].append(i)
        if -1 in buckets:
            raise ValueError(
                "Could not parse a camera id from some filenames. Check "
                "parse_camera_id, or use --partition_file with your own partition."
            )
        return {str(cam): idxs for cam, idxs in sorted(buckets.items())}

    if strategy == "iid":
        perm = rng.permutation(n)
        chunks = np.array_split(perm, num_clients)
        return {str(c): chunk.tolist() for c, chunk in enumerate(chunks)}

    if strategy == "dirichlet":
        # Label (pid) skew
        pid_to_idx = defaultdict(list)
        for i, it in enumerate(train_items):
            pid_to_idx[it["pid"]].append(i)
        pids = sorted(pid_to_idx.keys())
        rng.shuffle(pids)
        client_idx = [[] for _ in range(num_clients)]
        for pid in pids:
            p = rng.dirichlet(np.repeat(alpha, num_clients))
            idxs = pid_to_idx[pid]
            rng.shuffle(idxs)
            cuts = (np.cumsum(p) * len(idxs)).astype(int)[:-1]
            for c, part in enumerate(np.split(np.array(idxs), cuts)):
                client_idx[c].extend(part.tolist())
        return {str(c): sorted(v) for c, v in enumerate(client_idx) if len(v) > 0}

    raise ValueError(f"Unknown partition strategy: {strategy}")


def partition_stats(partition, train_items):
    """Compute partition statistics for logging. Returns (per-client rows, summary dict)."""
    rows = []
    for cid, idxs in partition.items():
        pids = {train_items[i]["pid"] for i in idxs}
        rows.append({
            "client": cid,
            "n_images": len(idxs),
            "n_ids": len(pids),
            "n_captions": sum(len(train_items[i]["captions"]) for i in idxs),
        })
    counts = np.array([r["n_images"] for r in rows], dtype=float)
    cv = counts.std() / counts.mean() if counts.mean() > 0 else 0.0

    # Average pairwise identity overlap (Jaccard) between clients
    pid_sets = [{train_items[i]["pid"] for i in idxs} for idxs in partition.values()]
    jac = []
    for a in range(len(pid_sets)):
        for b in range(a + 1, len(pid_sets)):
            u = len(pid_sets[a] | pid_sets[b])
            if u:
                jac.append(len(pid_sets[a] & pid_sets[b]) / u)
    summary = {
        "n_clients": len(rows),
        "images_min": int(counts.min()),
        "images_max": int(counts.max()),
        "images_mean": float(counts.mean()),
        "cv_images": float(cv),
        "mean_pairwise_jaccard_pid": float(np.mean(jac)) if jac else 0.0,
    }
    return rows, summary


# ---------------------------------------------------------------- transforms
def build_transforms(img_h, img_w, is_train):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_h, img_w), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Pad(10),
            transforms.RandomCrop((img_h, img_w)),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), value=0.0),
        ])
    return transforms.Compose([
        transforms.Resize((img_h, img_w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


# ---------------------------------------------------------------- datasets
class TrainPairDataset(Dataset):
    """Each sample = (image, 1 caption, pid).

    RSTPReid has 2 captions per image, so #samples = 2 x #images for this client.
    Note: SDM only uses the relation pid_i == pid_j, so pids do NOT need to be
    remapped to a local index. Remapping would only matter for an ID-classification loss.
    """

    def __init__(self, items, indices, root, transform, tokenizer, text_len):
        self.root = root
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.samples = []
        for i in indices:
            it = items[i]
            for ci in range(len(it["captions"])):
                self.samples.append((i, ci))
        self.items = items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        i, ci = self.samples[k]
        it = self.items[i]
        img = Image.open(os.path.join(self.root, "imgs", it["img_path"])).convert("RGB")
        img = self.transform(img)
        tok = self.tokenizer(
            it["captions"][ci], padding="max_length", truncation=True,
            max_length=self.text_len, return_tensors="pt",
        )
        return {
            "pixel_values": img,
            "input_ids": tok["input_ids"][0],
            "attention_mask": tok["attention_mask"][0],
            "pid": it["pid"],
        }


class EvalImageDataset(Dataset):
    def __init__(self, items, root, transform):
        self.items, self.root, self.transform = items, root, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(os.path.join(self.root, "imgs", it["img_path"])).convert("RGB")
        return {"pixel_values": self.transform(img), "pid": it["pid"]}


class EvalTextDataset(Dataset):
    def __init__(self, items, tokenizer, text_len):
        self.rows = []
        for it in items:
            for c in it["captions"]:
                self.rows.append((c, it["pid"]))
        self.tokenizer, self.text_len = tokenizer, text_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        cap, pid = self.rows[i]
        tok = self.tokenizer(cap, padding="max_length", truncation=True,
                             max_length=self.text_len, return_tensors="pt")
        return {"input_ids": tok["input_ids"][0],
                "attention_mask": tok["attention_mask"][0],
                "pid": pid}
