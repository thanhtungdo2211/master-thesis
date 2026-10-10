#!/usr/bin/env python3
"""Generate every number-bearing table / plot-data file of the paper from results/*.csv.

Nothing in the .tex sources is typed by hand for experimental numbers: rerun this script
after changing anything in results/ and recompile.   (stdlib only; no pandas needed)

    cd paper && python3 scripts/gen_tables.py
"""
import csv
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "generated")
os.makedirs(OUT, exist_ok=True)

GROUP_A = {1, 3, 5, 7, 14, 15}
GROUP_B = {2, 6, 8, 9, 10, 11, 12, 13}
BRIDGE = {4}
BATCH = 64


def read(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def evals(rows):
    """{round: (R@1, R@5, R@10, mAP, mINP)} for rounds that were evaluated."""
    return {int(r["round"]): tuple(float(r[k]) for k in ("R@1", "R@5", "R@10", "mAP", "mINP"))
            for r in rows if r["R@1"]}


def write(name, text):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)


c1 = evals(read("C1true_log.csv"))
b1 = evals(read("B5_v1_log.csv"))
b2 = evals(read("B5_v2_log.csv"))
b1_rows, b2_rows, c1_rows = read("B5_v1_log.csv"), read("B5_v2_log.csv"), read("C1true_log.csv")

# ---------------------------------------------------------------- plateau of centralized run
plat = [c1[r] for r in range(15, 21)]
plat_r1 = st.mean(p[0] for p in plat)
plat_r1_sd = st.stdev(p[0] for p in plat)
plat_map = st.mean(p[3] for p in plat)
plat_map_sd = st.stdev(p[3] for p in plat)
best_r1_round = max(c1, key=lambda r: c1[r][0])
best_map_round = max(c1, key=lambda r: c1[r][3])
late = [c1[r] for r in range(10, 21)]
cv_r1 = st.stdev(p[0] for p in late) / st.mean(p[0] for p in late)
cv_map = st.stdev(p[3] for p in late) / st.mean(p[3] for p in late)

# ---------------------------------------------------------------- macros reused in the text
m = {
    "CPlatR": plat_r1, "CPlatRsd": plat_r1_sd, "CPlatM": plat_map, "CPlatMsd": plat_map_sd,
    "CBestR": c1[best_r1_round][0], "CBestRround": best_r1_round,
    "CBestM": c1[best_map_round][3], "CBestMround": best_map_round,
    "CBestRmAP": c1[best_r1_round][3],
    "CVR": cv_r1, "CVM": cv_map,
    "CFinalLoss": float(c1_rows[-1]["loss"]), "CFirstLoss": float(c1_rows[0]["loss"]),
    "CHours": float(c1_rows[-1]["elapsed_s"]) / 3600,
    "BoneLast": int(b1_rows[-1]["round"]), "BtwoLast": int(b2_rows[-1]["round"]),
    "BoneRThirty": b1[30][0], "BoneMThirty": b1[30][3],
    "BtwoRTwenty": b2[20][0], "BtwoMTwenty": b2[20][3],
    "UplinkTotalOne": float(b1_rows[-1]["cum_uplink_MB"]),
}
with open(os.path.join(OUT, "macros.tex"), "w") as f:
    for k, v in m.items():
        if isinstance(v, int):
            f.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
        elif k in ("CVR", "CVM"):
            f.write(f"\\newcommand{{\\num{k}}}{{{v:.4f}}}\n")
        else:
            f.write(f"\\newcommand{{\\num{k}}}{{{v:.2f}}}\n")

# ---------------------------------------------------------------- FL-gap table
def cell(x):
    return "--" if x is None else f"{x:.2f}"


def panel(idx, title, f):
    """One metric panel of the centralized-vs-federated table. Only the federated run that
    reaches round 30 is reported; the second federated run is used solely for the per-client
    analysis, so it does not appear here."""
    f.write(f"\\midrule\n\\multicolumn{{4}}{{l}}{{\\textit{{{title}}}}}\\\\\n")
    for r in (5, 10, 15, 20, 25, 30):
        cen = c1[r][idx] if r in c1 else (plat_r1 if idx == 0 else plat_map)
        dag = "" if r in c1 else r"$^\dagger$"
        x1 = b1[r][idx] if r in b1 else None
        g1 = None if x1 is None else cen - x1
        f.write(f"{r} & {cen:.2f}{dag} & {cell(x1)} & {cell(g1)}\\\\\n")


with open(os.path.join(OUT, "tab_flgap.tex"), "w") as f:
    f.write("\\begin{tabular}{r cc c}\n\\toprule\n"
            "Round & Centralized & Federated & $\\Delta$\\\\\n")
    panel(0, "R@1 (\\%)", f)
    panel(3, "mAP (\\%)", f)
    f.write("\\bottomrule\n\\end{tabular}\n")

# Spread between the two federated runs. They share a seed, so this reflects execution
# nondeterminism and is a lower bound on the variance over seeds. Reported in the text only.
noise_r1 = max(abs(b1[r][0] - b2[r][0]) for r in (5, 10, 15, 20))
noise_map = max(abs(b1[r][3] - b2[r][3]) for r in (5, 10, 15, 20))
with open(os.path.join(OUT, "macros.tex"), "a") as f:
    f.write(f"\\newcommand{{\\numNoiseR}}{{{noise_r1:.2f}}}\n\\newcommand{{\\numNoiseM}}{{{noise_map:.2f}}}\n")

# ---------------------------------------------------------------- per-client heterogeneity
cl = read("B5_v2_client_loss.csv")
by = defaultdict(dict)
size = {}
for r in cl:
    by[int(r["round"])][int(r["client_id"])] = float(r["loss"])
    size[int(r["client_id"])] = int(r["n_samples"])
rounds = sorted(by)


def cv(vals):
    return st.pstdev(vals) / st.mean(vals)


grp = {}
for rd in rounds:
    d = by[rd]
    a = st.mean(d[c] for c in GROUP_A)
    b = st.mean(d[c] for c in GROUP_B)
    br = st.mean(d[c] for c in BRIDGE)
    grp[rd] = (a, b, br, b - a, cv(list(d.values())), st.mean(d.values()))
with open(os.path.join(OUT, "group_loss.csv"), "w") as f:
    f.write("round,groupA,groupB,bridge,gap,cv,mean\n")
    for rd in rounds:
        f.write(f"{rd}," + ",".join(f"{x:.4f}" for x in grp[rd]) + "\n")

last = rounds[-1]
with open(os.path.join(OUT, "tab_group.tex"), "w") as f:
    f.write("\\begin{tabular}{r ccc c c}\n\\toprule\n"
            "Round & Group A & Group B & Bridge (c4) & Gap $B{-}A$ & CV across clients\\\\\n\\midrule\n")
    for rd in (1, 5, 10, 15, last):
        a, b, br, gap, c, _ = grp[rd]
        f.write(f"{rd} & {a:.2f} & {b:.2f} & {br:.2f} & {gap:.2f} & {c:.3f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(OUT, "macros.tex"), "a") as f:
    f.write(f"\\newcommand{{\\numLastRound}}{{{last}}}\n")
    f.write(f"\\newcommand{{\\numGapFirst}}{{{grp[1][3]:.2f}}}\n\\newcommand{{\\numGapLast}}{{{grp[last][3]:.2f}}}\n")
    f.write(f"\\newcommand{{\\numCVFirst}}{{{grp[1][4]:.3f}}}\n\\newcommand{{\\numCVLast}}{{{grp[last][4]:.3f}}}\n")
    f.write(f"\\newcommand{{\\numLossLo}}{{{min(min(d.values()) for d in by.values()):.2f}}}\n")
    f.write(f"\\newcommand{{\\numLossHi}}{{{max(max(d.values()) for d in by.values()):.2f}}}\n")

# Pearson r between client size and final loss (all clients / without client 2)
def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den


cs = sorted(size)
r_all = pearson([size[c] for c in cs], [by[last][c] for c in cs])
cs2 = [c for c in cs if c != 2]
r_no2 = pearson([size[c] for c in cs2], [by[last][c] for c in cs2])
with open(os.path.join(OUT, "macros.tex"), "a") as f:
    f.write(f"\\newcommand{{\\numPearsonAll}}{{{r_all:.2f}}}\n\\newcommand{{\\numPearsonNoTwo}}{{{r_no2:.2f}}}\n")

# per-client table (sorted by size)
def grp_name(c):
    return "A" if c in GROUP_A else "B" if c in GROUP_B else "bridge"


with open(os.path.join(OUT, "tab_client_loss.tex"), "w") as f:
    f.write("\\begin{tabular}{r r r c c c c}\n\\toprule\n"
            "Client & Samples & Steps/round & Group & Loss r.1 & Loss r.%d & Drop (\\%%)\\\\\n\\midrule\n" % last)
    for c in sorted(size, key=lambda k: -size[k]):
        l1, ll = by[1][c], by[last][c]
        f.write(f"{c} & {size[c]:,} & {size[c] // BATCH} & {grp_name(c)} & {l1:.2f} & {ll:.2f} & "
                f"{100 * (l1 - ll) / l1:.1f}\\\\\n".replace(",", "\\,"))
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- partition table + effective weights
ps = json.load(open(os.path.join(RES, "partition_stats.json")))
rows = []
for c in ps["per_client"]:
    cid = int(c["client"].split("_")[1])
    rows.append((cid, c["num_images"], c["num_identities"], c["images_per_identity"],
                 c["num_captions"]))
tot_img = sum(r[1] for r in rows)
eff = {r[0]: r[4] * (r[4] // BATCH) for r in rows}
eff_tot = sum(eff.values())
top = sorted(rows, key=lambda r: -r[1])
with open(os.path.join(OUT, "tab_partition.tex"), "w") as f:
    f.write("\\begin{tabular}{r c r r c r r r}\n\\toprule\n"
            "Client (camera) & Group & Images & \\% total & IDs & Img/ID & Steps/round & \\% effective\\\\\n\\midrule\n")
    for cid, ni, nid, ipi, ncap in top:
        f.write(f"{cid} & {grp_name(cid)} & {ni:,} & {100 * ni / tot_img:.1f} & {nid:,} & {ipi:.2f} & "
                f"{ncap // BATCH} & {100 * eff[cid] / eff_tot:.3f}\\\\\n".replace(",", "\\,"))
    f.write("\\bottomrule\n\\end{tabular}\n")

imgs = [r[1] for r in rows]
top2 = sorted(rows, key=lambda r: -r[1])[:2]
top5 = sorted(rows, key=lambda r: -r[1])[:5]
sh = lambda rs: 100 * sum(r[1] for r in rs) / tot_img
eh = lambda rs: 100 * sum(eff[r[0]] for r in rs) / eff_tot
gA = [r for r in rows if r[0] in GROUP_A]
gB = [r for r in rows if r[0] in GROUP_B]
gC = [r for r in rows if r[0] in BRIDGE]
with open(os.path.join(OUT, "macros.tex"), "a") as f:
    f.write(f"\\newcommand{{\\numImgMin}}{{{min(imgs)}}}\n\\newcommand{{\\numImgMax}}{{{max(imgs):,}}}\n".replace(",", "\\,"))
    f.write(f"\\newcommand{{\\numImgRatio}}{{{max(imgs) / min(imgs):.1f}}}\n")
    f.write(f"\\newcommand{{\\numTotImg}}{{{tot_img:,}}}\n".replace(",", "\\,"))
    f.write(f"\\newcommand{{\\numTotID}}{{{ps['total_train_identities']:,}}}\n".replace(",", "\\,"))
    f.write(f"\\newcommand{{\\numCVSize}}{{{ps['load_imbalance_cv']:.3f}}}\n")
    f.write(f"\\newcommand{{\\numTwoShareNom}}{{{sh(top2):.1f}}}\n\\newcommand{{\\numTwoShareEff}}{{{eh(top2):.1f}}}\n")
    f.write(f"\\newcommand{{\\numFiveShareNom}}{{{sh(top5):.1f}}}\n\\newcommand{{\\numFiveShareEff}}{{{eh(top5):.1f}}}\n")
    f.write(f"\\newcommand{{\\numShareA}}{{{sh(gA):.1f}}}\n\\newcommand{{\\numShareB}}{{{sh(gB):.1f}}}\n\\newcommand{{\\numShareC}}{{{sh(gC):.1f}}}\n")
    f.write(f"\\newcommand{{\\numImgA}}{{{sum(r[1] for r in gA):,}}}\n\\newcommand{{\\numImgB}}{{{sum(r[1] for r in gB):,}}}\n".replace(",", "\\,"))
    smallest = min(rows, key=lambda r: r[1])
    f.write(f"\\newcommand{{\\numEffTwo}}{{{100 * eff[smallest[0]] / eff_tot:.4f}}}\n")
    f.write(f"\\newcommand{{\\numNomTwo}}{{{100 * smallest[1] / tot_img:.2f}}}\n")
    f.write(f"\\newcommand{{\\numSqueeze}}{{{(smallest[1] / tot_img) / (eff[smallest[0]] / eff_tot):.0f}}}\n")

# ---------------------------------------------------------------- communication
with open(os.path.join(RES, "..", "..", "data", "runs", "B5_camera_lora4_v2_20260709", "args.json")) as f:
    args = json.load(f)
tr, tot = args["trainable_params"], args["total_params"]
with open(os.path.join(OUT, "macros.tex"), "a") as f:
    f.write(f"\\newcommand{{\\numTrainable}}{{{tr:,}}}\n\\newcommand{{\\numTotalParams}}{{{tot:,}}}\n".replace(",", "\\,"))
    f.write(f"\\newcommand{{\\numTrainPct}}{{{100 * tr / tot:.2f}}}\n")
    f.write(f"\\newcommand{{\\numUplinkLoRA}}{{{tr * 4 / 2**20:.2f}}}\n")
    f.write(f"\\newcommand{{\\numUplinkFull}}{{{tot * 4 / 2**20:.0f}}}\n")
    f.write(f"\\newcommand{{\\numUplinkRatio}}{{{tot / tr:.0f}}}\n")
    f.write(f"\\newcommand{{\\numUplinkRoundLoRA}}{{{15 * tr * 4 / 2**20:.1f}}}\n")
    f.write(f"\\newcommand{{\\numUplinkRoundFull}}{{{15 * tot * 4 / 2**30:.1f}}}\n")
print("generated ->", OUT, sorted(os.listdir(OUT)))

# ---------------------------------------------------------------- IRRA log (loss_names: sdm+id+mlm)
import re

irra, ep = [], None
for line in open(os.path.join(RES, "irra_rstp_train_log.txt")):
    mm = re.search(r"Validation Results - Epoch: (\d+)", line)
    if mm:
        ep = int(mm.group(1))
    mm = re.match(r"\|\s*t2i\s*\|" + r"\s*([\d.]+)\s*\|" * 5, line)
    if mm and ep is not None:
        irra.append((ep,) + tuple(float(x) for x in mm.groups()))
ib = max(irra, key=lambda r: r[1])
iw = [r for r in irra if r[0] >= 51]
i_r1, i_r1sd = st.mean(r[1] for r in iw), st.stdev(r[1] for r in iw)
i_map, i_mapsd = st.mean(r[4] for r in iw), st.stdev(r[4] for r in iw)
c_best_map = c1[best_map_round][3]
with open(os.path.join(OUT, "macros.tex"), "a") as f:
    f.write(f"\\newcommand{{\\numIrraBestR}}{{{ib[1]:.2f}}}\n\\newcommand{{\\numIrraBestM}}{{{ib[4]:.2f}}}\n"
            f"\\newcommand{{\\numIrraBestEp}}{{{ib[0]}}}\n\\newcommand{{\\numIrraNEval}}{{{len(irra)}}}\n"
            f"\\newcommand{{\\numIrraConvR}}{{{i_r1:.2f}}}\n\\newcommand{{\\numIrraConvRsd}}{{{i_r1sd:.2f}}}\n"
            f"\\newcommand{{\\numIrraConvM}}{{{i_map:.2f}}}\n\\newcommand{{\\numIrraConvMsd}}{{{i_mapsd:.2f}}}\n"
            f"\\newcommand{{\\numIrraOverR}}{{{ib[1] - i_r1:.2f}}}\n"
            f"\\newcommand{{\\numGapIrraConvR}}{{{i_r1 - plat_r1:.2f}}}\n"
            f"\\newcommand{{\\numGapIrraConvM}}{{{i_map - plat_map:.2f}}}\n"
            f"\\newcommand{{\\numGapIrraBestR}}{{{ib[1] - c1[best_r1_round][0]:.2f}}}\n"
            f"\\newcommand{{\\numCOverR}}{{{c1[best_r1_round][0] - plat_r1:.2f}}}\n")
    f.write(f"\\newcommand{{\\numGapIrraLastR}}{{{i_r1 - b1[30][0]:.2f}}}\n")

# Best checkpoint of each setting: the evaluation with the highest R@1, mAP breaking a tie.
# Only the longer federated run is reported here; the second one is used in the per-client
# analysis alone.
c_best = c1[max(c1, key=lambda r: (c1[r][0], c1[r][3]))]
b_best_round = max(b1, key=lambda r: (b1[r][0], b1[r][3]))
b_best = b1[b_best_round]
with open(os.path.join(OUT, "tab_vs_irra.tex"), "w") as f:
    f.write("\\begin{tabular}{l r l ccccc}\n\\toprule\n"
            "Setting & Params & Loss & R@1 & R@5 & R@10 & mAP & mINP\\\\\n\\midrule\n")

    def row(name, params, loss, v):
        f.write(f"{name} & {params} & {loss} & " + " & ".join(f"{x:.2f}" for x in v) + "\\\\\n")

    # Published baseline row of the IRRA paper: the CLIP dual encoder alone, without
    # IRRA's cross-attention and identity modules. Transcribed, not measured here.
    row("CLIP ViT-B/16 baseline", "195M", "InfoNCE", (54.05, 80.70, 88.00, 43.41, 22.31))
    row("IRRA, full fine-tuning", "195M", "SDM+ID+MLM", ib[1:6])
    row("Centralized LoRA ($r{=}4$)", "0.49M", "SDM", c_best)
    row("Federated LoRA ($r{=}4$)", "0.49M", "SDM", b_best)
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(OUT, "macros.tex"), "a") as f:
    f.write(f"\\newcommand{{\\numFBestRound}}{{{b_best_round}}}\n"
            f"\\newcommand{{\\numFBestR}}{{{b_best[0]:.2f}}}\n"
            f"\\newcommand{{\\numFBestM}}{{{b_best[3]:.2f}}}\n")

# ---------------------------------------------------------------- uplink table (params x 4 bytes)
PUB = [("CFine", 205e6, "full"), ("IRRA", 195e6, "full"), ("CSKT", 12e6, "PETL"),
       ("DM-Adapter", 16e6, "PETL (MoE-Adapter)"), ("UP-Person", 7.4e6, "PETL (LoRA+Prefix+Adapter)")]
with open(os.path.join(OUT, "tab_uplink.tex"), "w") as f:
    f.write("\\begin{tabular}{l l r r}\n\\toprule\nMethod & Tuning & Trainable params & Est. uplink (MiB/client/round)\\\\\n\\midrule\n")
    for n, p, t in PUB:
        f.write(f"{n} & {t} & {p / 1e6:g}M & {p * 4 / 2**20:.0f}\\\\\n")
    f.write(f"\\textbf{{This pipeline}} (LoRA $r{{=}}4$, Q,K,V,O) & LoRA & {tr / 1e6:.2f}M & {tr * 4 / 2**20:.2f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("irra:", ib, i_r1, i_map)

# ---------------------------------------------------------------- plot data (no empty cells, pgfplots-friendly)
def curve(name, ev):
    with open(os.path.join(OUT, name), "w") as f:
        f.write("round,r1,r5,r10,map,minp\n")
        for r in sorted(ev):
            f.write(f"{r}," + ",".join(f"{x:.4f}" for x in ev[r]) + "\n")


curve("curve_c1.csv", c1)
curve("curve_b5v1.csv", b1)
curve("curve_b5v2.csv", b2)
with open(os.path.join(OUT, "curve_irra.csv"), "w") as f:
    f.write("epoch,r1,map\n")
    for r in irra:
        f.write(f"{r[0]},{r[1]:.4f},{r[4]:.4f}\n")
with open(os.path.join(OUT, "curve_loss.csv"), "w") as f:
    f.write("round,c1,b5v1,b5v2\n")
    c1l = {int(r["round"]): float(r["loss"]) for r in c1_rows}
    b1l = {int(r["round"]): float(r["loss"]) for r in b1_rows}
    b2l = {int(r["round"]): float(r["loss"]) for r in b2_rows}
    for r in range(1, max(b1l) + 1):
        f.write(f"{r},{c1l.get(r, 'nan')},{b1l.get(r, 'nan')},{b2l.get(r, 'nan')}\n")

# ---------------------------------------------------------------- extra macros for the results text
gaps = [grp[r][3] for r in rounds]
cvs = [grp[r][4] for r in rounds]
n_gap_down = sum(1 for a, b in zip(gaps, gaps[1:]) if b < a)
n_cv_down = sum(1 for a, b in zip(cvs, cvs[1:]) if b < a)
low = defaultdict(int)
hi = defaultdict(int)
for r in rounds:
    low[min(by[r], key=by[r].get)] += 1
    hi[max(by[r], key=by[r].get)] += 1
b1_30 = next(r for r in b1_rows if r["round"] == "30")
with open(os.path.join(OUT, "macros.tex"), "a") as f:
    f.write(f"\\newcommand{{\\numGapDown}}{{{n_gap_down}}}\n\\newcommand{{\\numCVDown}}{{{n_cv_down}}}\n"
            f"\\newcommand{{\\numSteps}}{{{len(rounds) - 1}}}\n"
            f"\\newcommand{{\\numLowClient}}{{{max(low, key=low.get)}}}\n\\newcommand{{\\numLowCount}}{{{max(low.values())}}}\n"
            f"\\newcommand{{\\numHiOne}}{{{max(hi, key=hi.get)}}}\n\\newcommand{{\\numHiOneCount}}{{{max(hi.values())}}}\n"
            f"\\newcommand{{\\numUplinkRthirty}}{{{float(b1_30['cum_uplink_MB']):.1f}}}\n"
            f"\\newcommand{{\\numUplinkFullThirty}}{{{30 * 15 * tot * 4 / 2**30:.0f}}}\n"
            f"\\newcommand{{\\numCOverM}}{{{c1[best_r1_round][3] - plat_map:.2f}}}\n")
