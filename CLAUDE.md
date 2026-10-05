# Paper: FL + PEFT (LoRA) + CLIP for Text-Based Person Search

## Rules
- Never invent numbers. All results come from results/*.csv.
- Every \cite key must exist in paper/refs.bib. Add new entries only via
  latex-document-skill's fetch_bibtex.sh (DOI/arXiv), never from memory.
- When describing the method, verify against fl_tbps/ code: code is ground truth.
- Compile: cd paper && latexmk -pdf -outdir=build main.tex

## Core facts
- Claim: first work combining FL + PEFT + CLIP + TBPS.
- Setup: RSTPReid, camera-based non-IID split (feature/domain skew, not label skew).
- LoRA r=4 on Q,K,V,O in both CLIP ViT-B/16 encoders; ~0.49M trainable params,
  ~1.9 MB uplink/round/client.
- Loss: SDM only (ID loss incompatible with FedAvg; MLM head too expensive).
- Baseline: IRRA README "CLIP baseline" row, 54.05 R@1.
  TODO: loss config of that row NOT yet verified. Do not describe it until confirmed.