"""Pure-Python smoke test (no torch/GPU needed) for the partitioning and metric logic."""
import sys, types, json, os, tempfile

# --- stub torch/torchvision/PIL so data.py can be imported without them ---
for name in ["torch", "torch.utils", "torch.utils.data", "torchvision",
             "torchvision.transforms", "PIL", "PIL.Image"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["torch.utils.data"].Dataset = object
class _IM: BICUBIC = "bicubic"
sys.modules["torchvision.transforms"].InterpolationMode = _IM
for fn in ["Compose", "Resize", "RandomHorizontalFlip", "Pad", "RandomCrop",
           "ToTensor", "Normalize", "RandomErasing"]:
    setattr(sys.modules["torchvision.transforms"], fn, lambda *a, **k: None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clip_lora_fedavg.data import parse_camera_id, build_partition, partition_stats, load_annotations

# ---------------- 1. camera id parsing ----------------
cases = {
    "0001_c1_0001.jpg": 1,
    "0042_c15_0007.jpg": 15,
    "1234_c3_0012.jpg": 3,
    "0001_001_02_0303morning_0015_0.jpg": 2,
    "weird_name.jpg": -1,
}
for name, want in cases.items():
    got = parse_camera_id(name)
    assert got == want, f"parse_camera_id({name}) = {got}, want {want}"
print("1. parse_camera_id: OK")

# ---------------- 2. simulate RSTPReid ----------------
# 4101 identities, 5 images each from 15 cameras, 2 captions each
import numpy as np
rng = np.random.RandomState(0)
raw = []
N_ID, N_CAM = 400, 15          # scaled down for a fast test
for pid in range(N_ID):
    cams = rng.choice(N_CAM, size=5, replace=False) + 1
    split = "train" if pid < N_ID - 40 else "test"
    for seq, cam in enumerate(cams):
        raw.append({
            "id": pid, "split": split,
            "img_path": f"{pid:04d}_c{cam}_{seq:04d}.jpg",
            "captions": [f"caption a {pid}", f"caption b {pid}"],
        })
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "data_captions.json"), "w") as f:
        json.dump(raw, f)
    splits = load_annotations(d)
train_items = splits["train"]
print(f"2. load_annotations: train {len(train_items)} images, test {len(splits['test'])} images")
assert len(train_items) == (N_ID - 40) * 5

# ---------------- 3. partitioning strategies ----------------
for strat, kw in [("camera", {}), ("iid", {}), ("dirichlet", {})]:
    part = build_partition(train_items, strat, 15, 0.5, 42)
    rows, summ = partition_stats(part, train_items)
    covered = sum(len(v) for v in part.values())
    assert covered == len(train_items), f"{strat}: data loss ({covered} vs {len(train_items)})"
    assert len(set().union(*[set(v) for v in part.values()])) == len(train_items), \
        f"{strat}: duplicate index across clients"
    print(f"3. {strat:10s} -> {summ['n_clients']:2d} clients | "
          f"images {summ['images_min']}-{summ['images_max']} | "
          f"CV {summ['cv_images']:.3f} | pid Jaccard {summ['mean_pairwise_jaccard_pid']:.3f}")

# ---------------- 4. metrics via numpy (mirrors evaluate.compute_metrics) ----------------
def metrics_np(sim, tpid, ipid):
    idx = np.argsort(-sim, axis=1)
    matches = (ipid[idx] == tpid[:, None]).astype(float)
    nq, ng = matches.shape
    cmc = np.clip(matches.cumsum(1), 0, 1)
    out = {f"R@{k}": cmc[:, k-1].mean()*100 for k in (1, 5, 10)}
    n_rel = matches.sum(1)
    tmp = matches.cumsum(1) / np.arange(1, ng+1)[None, :]
    out["mAP"] = ((tmp*matches).sum(1) / np.maximum(n_rel, 1)).mean()*100
    inp = [n_rel[i]/(np.nonzero(matches[i])[0][-1]+1) for i in range(nq) if n_rel[i] > 0]
    out["mINP"] = float(np.mean(inp))*100
    return out

# perfect case: every query ranks its positives first
ipid = np.repeat(np.arange(10), 5)          # 10 identities x 5 images
tpid = np.repeat(np.arange(10), 2)          # 10 identities x 2 captions
sim = (ipid[None, :] == tpid[:, None]).astype(float)
m = metrics_np(sim, tpid, ipid)
assert abs(m["R@1"]-100) < 1e-6 and abs(m["mAP"]-100) < 1e-6 and abs(m["mINP"]-100) < 1e-6
print(f"4. perfect-case metrics: R@1 {m['R@1']:.1f} mAP {m['mAP']:.1f} mINP {m['mINP']:.1f}")

sim_rand = np.random.RandomState(1).rand(20, 50)
m2 = metrics_np(sim_rand, tpid, ipid)
print(f"   random-case metrics: R@1 {m2['R@1']:.1f} R@10 {m2['R@10']:.1f} "
      f"mAP {m2['mAP']:.1f} mINP {m2['mINP']:.1f}   (expect R@1 ~ 10)")

# ---------------- 5. FedAvg via numpy ----------------
def fedavg_np(states, weights):
    tot = sum(weights)
    return {k: sum(s[k]*w for s, w in zip(states, weights))/tot for k in states[0]}
s1 = {"a": np.ones((2, 2))}
s2 = {"a": np.zeros((2, 2))}
r = fedavg_np([s1, s2], [3, 1])
assert np.allclose(r["a"], 0.75), r["a"]
print("5. weighted FedAvg: OK (3:1 -> 0.75)")

print("\nALL LOGIC TESTS PASSED")
