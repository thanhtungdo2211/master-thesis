# clip_lora_fedavg: FedAvg + LoRA + CLIP on RSTPReid

Minimal baseline for the thesis *Parameter-Efficient Fine-Tuning for Foundation Models in Federated Learning*, applied to Text-Based Person Search. This is the first pipeline in `experiments/` — later ones (different PEFT methods, different aggregation strategies, cross-dataset setups) will live alongside it as their own subfolders.

---

## Design decisions and rationale

| Decision | Reason |
|---|---|
| **CLIP ViT-B/16 + SDM loss only**, IRRA's IRR/MLM/ID heads dropped | The ID loss uses a classifier over the identity set → each client has a different ID set → different head size → **cannot FedAvg**. The MLM head is ~45M params → ~172 MB of wasted uplink per round. |
| **Manual FedAvg**, not Flower at this stage | Flower + Ray on Colab tends to OOM and crash workers. A manual loop is far easier to debug and iterates faster. Move to Flower once the baseline is stable (see bottom of file). |
| **Only 1 model kept on GPU** | 15 clients x CLIP ViT-B/16 doesn't fit Colab VRAM. Per client: load global state -> train -> pull the LoRA state back to CPU. |
| **No PID remapping to local indices** | SDM only uses the relation `pid_i == pid_j`, which is invariant under any bijective remapping. Remapping would only matter for an added ID-classification loss. |
| **384x128** with bicubic-interpolated position embeddings | Matches IRRA/UP-Person/DM-Adapter so numbers are comparable. Resizing to a square 224x224 would distort the aspect ratio. |
| **Global model evaluated on a centralized test set** | 200 identities held on the server, never evaluated per-client. |

---

## Setup

```bash
pip install "transformers==4.57.6" torch torchvision pillow
```

`transformers` is pinned to the exact version this pipeline was developed
against, not a floor (`>=`). `CLIPModel.get_image_features` / `get_text_features`
have changed implementation across releases — in some versions they return the
raw `BaseModelOutputWithPooling` from `vision_model(...)` instead of the
projected embedding tensor, which surfaces as `AttributeError: '...' object
has no attribute 'float'` deep inside `federated.train_one_client`. This bit
us specifically on Colab: its preinstalled `transformers` already satisfied
`>=4.40`, so `pip install "transformers>=4.40"` silently kept the old,
incompatible version instead of installing the tested one. Always pin exactly.

Dataset layout:
```
RSTPReid/
  imgs/0001_c1_0001.jpg ...
  data_captions.json
```

## Running

```bash
# B5: main focus — FL camera non-IID + LoRA r=4
python -m clip_lora_fedavg.main --root /path/to/RSTPReid \
  --out_dir runs/B5_camera_lora4 \
  --partition camera --tuning lora --lora_rank 4 \
  --rounds 50 --local_epochs 1 --batch_size 64 --lr 1e-4

# B3: IID control
python -m clip_lora_fedavg.main --root /path/to/RSTPReid \
  --out_dir runs/B3_iid_lora4 \
  --partition iid --num_clients 15 --tuning lora --lora_rank 4 \
  --rounds 50 --batch_size 64 --lr 1e-4

# B4: full fine-tuning (10x lower lr, smaller batch to fit VRAM)
python -m clip_lora_fedavg.main --root /path/to/RSTPReid \
  --out_dir runs/B4_camera_full \
  --partition camera --tuning full \
  --rounds 50 --batch_size 32 --lr 1e-5
```

Smoke test before a real run:
```bash
python -m clip_lora_fedavg.main --root /path/to/RSTPReid --out_dir runs/smoke \
  --rounds 2 --eval_every 2 --max_clients_per_round 3 --debug_steps 3 --batch_size 16
```

If you already have a partition file from your own `rstpreid_federated.py`, use it directly:
```bash
--partition_file /path/to/partition_camera.json   # {client_id: [image_index,...]}
```

### Cheaper pilot run: only 3 clients

`--select_clients` DROPS the data of every non-selected client entirely (unlike
`--max_clients_per_round`, which only rotates sampling per round while still
using all the data eventually). This is the way to genuinely reduce total
training volume per round — use it for a quick pilot on Colab before running
the full 15-client experiment.

```bash
python -m clip_lora_fedavg.main --root /path/to/RSTPReid \
  --out_dir runs/pilot_3cam \
  --partition camera --select_clients 1,2,3 \
  --tuning lora --lora_rank 4 \
  --rounds 50 --batch_size 64 --lr 1e-4
```

Note: any 3 cameras work, but check `partition_stats.json` first — some
cameras are very unbalanced in size (e.g. camera 1 alone holds ~3000/3440
images if you pick `1,2,3`), so pick cameras with similar `n_images` if you
want a more "fair" split across the 3 clients.

---

## Output

Each run produces, under `--out_dir`:

| File | Content |
|---|---|
| `log.csv` | round, loss, R@1/5/10, mAP, mINP, cumulative uplink (MB), elapsed time |
| `partition_stats.json` | images/IDs/captions per client, CV, mean pairwise pid Jaccard |
| `partition.json` | the partition actually used, for reproducibility |
| `args.json` | full config + trainable param count + MB/round/client |
| `best.pt` | best LoRA checkpoint by R@1 |

---

## Reference numbers

**Published results on RSTPReid** (source: `anosorae/IRRA` repo README):

| Method | R@1 | R@5 | R@10 | mAP | mINP |
|---|---|---|---|---|---|
| CFine (full FT) | 50.55 | 72.50 | 81.60 | – | – |
| **CLIP baseline** | **54.05** | 80.70 | 88.00 | 43.41 | 22.31 |
| IRRA (full, +IRR/MLM/ID) | 60.20 | 81.30 | 88.20 | 47.17 | 25.28 |
| UP-Person (PETL, centralized) | 63.15 | – | – | – | – |

**The "CLIP baseline" row (54.05) is the relevant reference point**, since its architecture matches this pipeline (CLIP dual-encoder, no IRR). The gap from 54.05 to 60.20 is exactly the contribution of IRR+MLM+ID, which we intentionally drop.

Expectation: **B5 (FL non-IID + LoRA) should land below 54.05.** That gap is the actual result to measure, not a bug.

**Communication cost** (15 clients x 50 rounds, fp32, uplink-only):

| Config | Trainable | MB/round/client | Total uplink |
|---|---|---|---|
| Full fine-tuning | ~150M | ~573 MB | ~420 GB |
| LoRA r=4 (Q,K,V,O) | ~0.49M | ~1.9 MB | ~1.4 GB |
| LoRA r=16 | ~1.97M | ~7.5 MB | ~5.5 GB |

The script prints the actually measured numbers at startup — use those instead of the estimates above.

---

## Three numbers to extract

- **FL gap** = B0 - B2 — the cost of going distributed
- **Non-IID gap** = B3 - B5 — the cost specifically from non-IID data
- **PEFT gap** = B4 - B5 — the accuracy traded for a ~300x reduction in communication

---

## Known limitations (worth writing into the thesis)

1. **LoRA aggregation bias.** FedAvg averages A and B separately, but `Avg(B) @ Avg(A) != Avg(B @ A)`. This is a known issue raised in FedIT / FFA-LoRA (arXiv 2403.12313) / FLoRA (arXiv 2409.05976) / LoRA-FAIR (arXiv 2411.14961). This baseline uses naive FedAvg to establish a reference point — this is exactly where a new method could be proposed.

2. **Camera-based != label-space disjoint.** Since every ID in RSTPReid appears in ~5/15 cameras, partitioning by camera creates **feature/domain skew**, not label-space disjointness. The script prints the mean pairwise pid Jaccard between clients so you can verify this. A truly disjoint label-space scenario requires cross-dataset partitioning (CUHK-PEDES / ICFG-PEDES / RSTPReid as 3 separate clients).

3. **Clients with few IDs can degenerate SDM.** If a camera has too few identities, a batch may not contain enough positive pairs. Check the `n_ids` column in `partition_stats.json`; if any client has under ~32 IDs, consider merging cameras or lowering that client's batch size.

4. **AMP fp16.** The loss is computed in fp32 to avoid softmax/KL underflow, but if you hit NaNs, rerun with `--no_amp`.

---

## Migrating to Flower later

Three functions in `federated.py` map directly onto the Flower client API:

| Current function | Flower |
|---|---|
| `trainable_state_dict(model)` | `NumPyClient.get_parameters()` |
| `load_trainable_state_dict(model, state)` + `train_one_client(...)` | `NumPyClient.fit()` |
| `evaluate(...)` | `NumPyClient.evaluate()` |
| `fedavg(states, weights)` | `flwr.server.strategy.FedAvg` |

Just needs wrapping — no need to rewrite the training logic.

---

## Code layout

```
clip_lora_fedavg/
  config.py       argparse, all hyperparameters
  data.py         RSTPReid loading, camera-id parsing, 3 partition strategies, Dataset classes
  lora.py         generic LoRA layer, injection, trainable-state helpers
  clip_model.py   CLIP wiring: position-embedding interpolation, build_model, encode_image/text
  loss.py         SDM loss
  evaluate.py     feature extraction + R@1/5/10, mAP, mINP
  federated.py    train_one_client + fedavg
  main.py         FedAvg loop, logging
notebooks/
  clip_lora_fedavg_colab.ipynb
tests/
  test_logic.py   pure-Python logic smoke test (no GPU needed)
```

Run the logic test (no GPU, no dataset needed):
```bash
python tests/test_logic.py
```
