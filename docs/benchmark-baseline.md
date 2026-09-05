# **Báo cáo Baseline: Benchmark các phương pháp Text-Based Person Search**

## **Bảng kết quả của centralized settings**

### **CUHK-PEDES**

| Method | Backbone | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| CMPM/C | RN50 / LSTM | 49.37 | – | 79.27 | – | – |
| DSSL | RN50 / BERT | 59.98 | 80.41 | 87.56 | – | – |
| SSAN | RN50 / LSTM | 61.37 | 80.15 | 86.73 | – | – |
| Han et al. | RN101 / Xformer | 64.08 | 81.73 | 88.19 | 60.08 | – |
| LGUR | DeiT-Small / BERT | 65.25 | 83.12 | 89.00 | – | – |
| IVT | ViT-B/16 / BERT | 65.59 | 83.11 | 89.21 | – | – |
| CFine | ViT-B/16 / BERT | 69.57 | 85.93 | 91.15 | – | – |
| **CLIP baseline** | ViT-B/16 / Xformer | 68.19 | 86.47 | 91.47 | 61.12 | 44.86 |
| **IRRA** | ViT-B/16 / Xformer | **73.38** | **89.93** | **93.71** | **66.13** | **50.24** |

### **ICFG-PEDES**

| Method | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
| ----- | ----- | ----- | ----- | ----- | ----- |
| CMPM/C | 43.51 | 65.44 | 74.26 | – | – |
| SSAN | 54.23 | 72.63 | 79.53 | – | – |
| IVT | 56.04 | 73.60 | 80.22 | – | – |
| CFine | 60.83 | 76.55 | 82.42 | – | – |
| **CLIP baseline** | 56.74 | 75.72 | 82.26 | 31.84 | 5.03 |
| **IRRA** | **63.46** | **80.24** | **85.82** | **38.05** | **7.92** |

### **RSTPReid**

| Method | Rank-1 | Rank-5 | Rank-10 | mAP | mINP |
| ----- | ----- | ----- | ----- | ----- | ----- |
| DSSL | 39.05 | 62.60 | 73.95 | – | – |
| SSAN | 43.50 | 67.80 | 77.15 | – | – |
| IVT | 46.70 | 70.00 | 78.80 | – | – |
| CFine | 50.55 | 72.50 | 81.60 | – | – |
| **CLIP baseline** | 54.05 | 80.70 | 88.00 | 43.41 | 22.31 |
| **IRRA** | **60.20** | **81.30** | **88.20** | **47.17** | **25.28** |

## **Bảng so sánh Full fine-tuning vs PEFT**

### **Rank-1 trên 3 benchmark**

| \# | Method | Venue | Tuning | Trainable param | CUHK | ICFG | RSTP |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0 | Zero-shot CLIP | ICML'21 | None | 0 | — | — | — |
| 1 | CFine | TIP'23 | Full FT | 205M | 69.57 | 60.83 | 50.55 |
| 2 | **IRRA** | CVPR'23 | Full FT | 195M | 73.38 | 63.46 | 60.20 |
| 3 | CSKT | 2024 | PETL | 12M | 69.70 | 58.90 | 57.75 |
| 4 | **DM-Adapter** | AAAI'25 | PETL (MoE-Adapter) | 16M | 72.17 | 62.64 | 60.00 |
| 5 | **UP-Person** (ViT-B/16) | TCSVT'25 | PETL (LoRA+Prefix+Adapter) | 7.4M | **74.17** | **65.02** | **63.15** |
| 6 | UP-Person (ViT-L/14) | TCSVT'25 | PETL | – | 76.04 | 65.98 | 64.45 |
| 7 |  |  |  |  |  |  |  |

### **Khoảng cách PEFT vs Full FT nới rộng khi dữ liệu ít đi**

| Dataset | Ảnh train | UP-Person R@1 | IRRA R@1 | Chênh lệch |
| ----- | ----- | ----- | ----- | ----- |
| CUHK-PEDES | 34,054 | 74.17 | 73.38 | **\+0.79** |
| ICFG-PEDES | 34,674 | 65.02 | 63.46 | **\+1.56** |
| RSTPReid | 18,505 | 63.15 | 60.20 | **\+2.95** |

### **Trainable params**

| Method | Trainable param | % so với IRRA | Memory (MB) | R@1 RSTPReid | R@1 / M param |
| ----- | ----- | ----- | ----- | ----- | ----- |
| CFine | 205M | 105.1% | 13,570 | 50.55 | 0.25 |
| IRRA | 195M | 100.0% | 7,034 | 60.20 | 0.31 |
| CSKT | 12M | 6.2% | 2,338 | 57.75 | 4.81 |
| DM-Adapter | 16M | 8.2% | 2,952 | 60.00 | 3.75 |
| **UP-Person** | **7.4M** | **3.8%** | 3,262 | **63.15** | **8.53** |

## **References**

**IRRA — Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval (CVPR 2023\)** Jiang, D.; Ye, M. — CVPR 2023, pp. 2787–2797

* arXiv: https://arxiv.org/abs/2303.12501  
* CVF open access: https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang\_Cross-Modal\_Implicit\_Relation\_Reasoning\_and\_Aligning\_for\_Text-to-Image\_Person\_Retrieval\_CVPR\_2023\_paper.pdf  
* GitHub: https://github.com/anosorae/IRRA

**UP-Person — Unified Parameter-Efficient Transfer Learning for Text-based Person Retrieval (IEEE TCSVT 2025\)** Liu, Y.; Li, Y.; Lan, X.; Yang, W.; Liu, Z.; Liao, Q.

* arXiv: https://arxiv.org/abs/2504.10084  
* IEEE Xplore: https://ieeexplore.ieee.org/document/11079682/  
* GitHub: https://github.com/Liu-Yating/UP-Person

**DM-Adapter — Domain-Aware Mixture-of-Adapters for Text-Based Person Retrieval (AAAI 2025\)** Liu, Y.; Liu, Z.; Lan, X.; Yang, W.; Li, Y.; Liao, Q. — AAAI 2025, 39(6), pp. 5703–5711

* arXiv: https://arxiv.org/abs/2503.04144  
* AAAI proceedings: https://ojs.aaai.org/index.php/AAAI/article/view/32608  
* DOI: 10.1609/aaai.v39i6.32608  
* GitHub: https://github.com/Liu-Yating/DM-Adapter

**CFine — CLIP-Driven Fine-grained Text-Image Person Re-identification (IEEE TIP 2023\)** Yan, S.; Dong, N.; Zhang, L.; Tang, J.

* arXiv: https://arxiv.org/abs/2210.10276  
* DOI: 10.1109/TIP.2023.3327924  
* GitHub: https://github.com/shuanglinyan/CFine

**CSKT — CLIP-based Synergistic Knowledge Transfer for Text-based Person Retrieval** Liu et al.

* arXiv: https://arxiv.org/abs/2309.09496

**TBPS-CLIP — An Empirical Study of CLIP for Text-based Person Search (AAAI 2024\)** Cao, M.; Bai, Y.; Zeng, Z.; Ye, M.; Zhang, M. — AAAI 2024, pp. 465–473

* arXiv: https://arxiv.org/abs/2308.10045  
* GitHub: https://github.com/Flame-Chasers/TBPS-CLIP

**RaSa — Relation and Sensitivity Aware Representation Learning (IJCAI 2023\)**

* GitHub: https://github.com/Flame-Chasers/RaSa

**APTM — Towards Unified Text-based Person Retrieval: MALS benchmark (ACM MM 2023\)**

* GitHub: https://github.com/Shuyu-XJTU/APTM  
* ACM DL: https://dl.acm.org/doi/10.1145/3581783.3611709

| Dataset | Nguồn |
| ----- | ----- |
| CUHK-PEDES | https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description |
| ICFG-PEDES | https://github.com/zifyloo/SSAN |
| RSTPReid | https://github.com/NjtechCVLab/RSTPReid-Dataset |

