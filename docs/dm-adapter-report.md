# DM-Adapter: Domain-Aware Mixture-of-Adapters for Text-Based Person Retrieval

## Abstract

- **Tên paper:** DM-Adapter: Domain-Aware Mixture-of-Adapters for Text-Based Person Retrieval (AAAI 2025)
- **Foundation Model sử dụng:** CLIP (Contrastive Language-Image Pre-training), backbone ViT-B/16.
- **Bài toán:** Text-based Person Retrieval (TPR)
- **Phương pháp đề xuất:** DM-Adapter. **Kết hợp Mixture-of-Experts (MoE) với PETL** cho bài toán TPR. Ý tưởng cốt lõi: thay vì dùng một adapter đơn lẻ, tác giả gắn một **cụm nhiều adapter chuyên biệt (Sparse Mixture-of-Adapters — SMA)** song song với tầng MLP của cả hai nhánh ảnh/văn bản, mỗi expert học một khía cạnh riêng của "person knowledge" để trích đặc trưng fine-grained; đồng thời thiết kế **Domain-Aware Router (DR)** để đưa tri thức đặc thù của miền (domain) vào cơ chế định tuyến. Chỉ huấn luyện **16M tham số** nhưng đạt SOTA giữa các phương pháp PETL.

## Purpose

Mục tiêu là giải quyết đồng thời hai điểm yếu tồn tại trong TPR khi tailoring CLIP về person domain:

- **(i) Full fine-tuning (FFT)** như IRRA, CFine tuy có khả năng transfer fine-grained tốt nhưng **tốn tài nguyên khổng lồ và dễ overfitting** trên các tập TPR quy mô nhỏ.
- **(ii) PETL hiện có** (điển hình là CSKT) tuy tiết kiệm tham số nhưng **thiếu khả năng trích đặc trưng fine-grained và thiếu cân nhắc đặc thù cho miền person** → hiệu năng còn hạn chế.

DM-Adapter muốn đạt **trade-off tốt nhất giữa hiệu năng và độ hiệu quả tham số**: giữ CLIP đóng băng, chỉ thêm một lượng nhỏ tham số nhưng vẫn "đẩy PETL đến giới hạn" cho tác vụ TPR nhờ tăng dung lượng mô hình một cách có kiểm soát (MoE) và bơm tri thức miền (domain prompts).

## Hypothesis

- **Giả thuyết 1:** Một adapter đơn lẻ không đủ dung lượng để biểu diễn đặc trưng fine-grained của người. Nếu mở rộng thành **nhiều expert adapter được kích hoạt thưa (sparse MoE)**, mỗi expert đảm nhiệm một khía cạnh khác nhau (màu tóc, trang phục, phụ kiện...), thì mô hình sẽ trích được đặc trưng chi tiết hơn mà chi phí tính toán vẫn giữ ổn định (do Top-K cố định).
- **Giả thuyết 2:** Router thuần data-driven (chỉ phụ thuộc input token) dễ gây **routing imbalance** và **bỏ qua tri thức miền**. Nếu **ghép thêm thông tin domain qua learnable prompts** vào hàm gating, router sẽ chọn expert phù hợp hơn với đặc thù TPR.
- **Giả thuyết 3:** Freeze toàn bộ backbone CLIP + chỉ train các module PETL nhẹ sẽ **bảo toàn tri thức pre-train**, chống overfitting tốt hơn FFT — đặc biệt trên dataset nhỏ (RSTPReid), nơi các phương pháp full-tuning nhiều tham số như CFine bị suy giảm generalization.

## Dataset

- **CUHK-PEDES:** 40,206 ảnh và 80,412 mô tả văn bản cho 13,003 danh tính. Train: 11,003 danh tính (34,054 ảnh, 68,126 text); val/test mỗi tập 1,000 danh tính.
- **ICFG-PEDES:** 54,522 ảnh cho 4,102 danh tính, mỗi ảnh một mô tả. Train 3,102 danh tính, test 1,000 danh tính.
- **RSTPReid:** 20,505 ảnh của 4,101 danh tính, mỗi ảnh 2 mô tả. Train 3,701 danh tính (18,505 ảnh), val 200 danh tính, test 200 danh tính. Đây là dataset **nhỏ nhất** → dùng để kiểm chứng khả năng chống overfitting.

## Architecture

**Tổng quan:** DM-Adapter giữ nguyên backbone CLIP (Image Encoder ViT-B/16 + Text Encoder Transformer) và **đóng băng (frozen) toàn bộ**. Chỉ chèn khối **DM-Adapter song song với tầng MLP** trong mỗi transformer block của cả hai nhánh, và chỉ huấn luyện các khối này (~16M tham số). Đặc trưng global lấy tại `[CLS]` (ảnh) và `[EOS]` (văn bản), sau đó khớp cặp bằng SDM loss.

- → **Xem Figure 3** (The overall framework): sơ đồ tổng thể hai nhánh ảnh/văn bản, vị trí chèn Mixture-of-Adapters + Router song song MLP, hai domain-aware prompt (image/text), luồng SDM loss và LB loss. *(Ảnh cần cắt để giải thích tổng quan.)*
- → **Xem Figure 1** (Evolution of paradigms): so sánh trực quan (a) FFT unfreeze toàn bộ vs (b) PETL freeze CLIP + single adapter, và điểm mới của DM-Adapter là thay single adapter bằng mixture-of-adapters. *(Ảnh minh hoạ động lực thiết kế.)*

**Backbone (frozen):**

- **Image Encoder:** CLIP ViT-B/16. Ảnh chia thành N patch không chồng lấn → linear projection + positional embedding, thêm `[CLS]` token, chuỗi `N + 1` token qua 12 transformer block; `v_cls` là biểu diễn global của ảnh.
- **Text Encoder:** CLIP Text Transformer 12 lớp. Tokenizer vocab 49,152, thêm `[BOS]`/`[EOS]`; lấy output tại `[EOS]` (`f_eos`) làm biểu diễn global của câu.

**DM-Adapter block — gồm 2 thành phần chính:**

| Thành phần | Vị trí gắn | Vai trò | Ghi chú |
| --- | --- | --- | --- |
| **SMA (Sparse Mixture-of-Adapters)** | Song song với tầng MLP của mỗi transformer block (cả 2 nhánh) | `n` expert adapter + router Top-K; mỗi expert chuyên một khía cạnh → trích đặc trưng **fine-grained** | Tăng dung lượng mô hình nhưng chi phí tính toán ổn định vì chỉ kích hoạt Top-K expert |
| **DR (Domain-Aware Router)** | Router bên trong SMA | Bơm **domain information** qua learnable prompts vào hàm gating → chọn expert hợp với miền person, giảm routing imbalance | Kèm **Load-Balancing (LB) loss** để cân bằng tải giữa các expert |

- → **Xem Figure 4** (Architecture of DM-Adapter): chi tiết cấu trúc SMA (các Expert 1…n, weighted sum theo gate) và DR (Input Gate + Domain Gate, Top-K Selection, Vision/Language Domain-Aware Prompt). *(Ảnh cần cắt để giải thích chi tiết module.)*

**Phân chia chức năng (điểm cốt lõi cần highlight):**

- **Mỗi adapter expert** = bottleneck đơn giản: down-projection `W_down` (d → d/reduction) → phi tuyến ReLU → up-projection `W_up` (d/reduction → d).
- **SMA thay 1 adapter bằng nhiều adapter** để mỗi expert xử lý input token dưới một "góc nhìn chuyên biệt" khác nhau về đặc trưng người → chính là cơ chế trích fine-grained ngầm định (implicit), không cần thêm module tương tác phức tạp như FFT.
- **DR gắn domain vào router:** thay vì router chỉ nhìn input token (`x·W`), DR bổ sung nhánh domain (`p·W_d`) từ learnable prompt → tri thức miền được "ghép cặp" với quá trình định tuyến.
- → **Xem Figure 6** (Visualization of Expert Weight): heatmap trọng số của 6 expert cho các token trong một mô tả (phân tích ở lớp 12 của CLIP) — minh chứng các expert thực sự **chuyên biệt hoá** vào các khía cạnh khác nhau. *(Ảnh cần cắt để minh hoạ tính chuyên biệt của expert.)*

**Cơ sở lý thuyết (các công thức chính):**

- **Eq. (1):** Adapter chuẩn với residual: `h' ← h + f(x·W_down)·W_up` — nền tảng của mọi expert.
- **Eq. (2)–(3):** Output SMA = tổng có trọng số của các adapter được chọn: `Σ G(x)_i · Adapter_i(x)`, với gating thưa `G(x) = Softmax(TopK(x·W))`. Cố định K ⇒ tăng `n` chỉ tăng dung lượng chứ không tăng chi phí tính toán.
- **Eq. (4):** Output cuối sau MLP + expert: `y = h_o + Σ Softmax(TopK(x·W))_i · Adapter_i(x)`, trong đó `h_o = x + MLP(LN(x))` là nhánh MLP gốc (frozen) → **SMA cộng song song với MLP**.
- **Eq. (5):** Domain-Aware Router — thay `x·W` bằng `x·W + p·W_d`: `y = h_o + Σ Softmax(TopK(x·W + p·W_d))_i · Adapter_i(x)`. Đây là điểm mấu chốt đưa domain prompt `p` vào gating.
- **Eq. (6):** Load-Balancing loss: `L_aux = α · Σ_i f_i · p_i`, với `f_i` = tỉ lệ token gán cho expert i (theo Top-K), `p_i` = trọng số routing trung bình cho expert i → khuyến khích các expert nhận số lượng token tương đương, tránh dồn tải vào một expert.

**Chi tiết từ source code (bổ sung ngoài paper — để implement chính xác):**

- **Adapter khởi tạo đúng cách:** `W_down` init Kaiming, `W_up` init **zeros** → lúc đầu adapter đóng góp ≈ 0, không phá tri thức CLIP gốc (proper initialization theo Gao et al. 2023). Không có residual bên trong expert; residual nằm ở nhánh MLP gốc.
- **DM-Adapter dùng LayerNorm riêng (`ln_3`) cho nhánh adapter**, tách khỏi `ln_2` của MLP: `x = x + MLP(ln_2(x)) + scale · SMA(ln_3(x))`.
- **Hệ số `scale` bất đối xứng giữa 2 nhánh:** nhánh ảnh `scale = 0.1`, nhánh văn bản `scale = 4` (một chi tiết engineering quan trọng, không nêu rõ trong paper).
- **Domain-Aware Router trong code = tổ hợp lồi có trọng số học được:** `gate_logits = (1−α)·input_gate(x) + α·task_gate(task_param)`, với `α` là `nn.Parameter` khởi tạo `0.5`, và `task_param = nn.Parameter(randn(d_model))` chính là domain-aware prompt. (Tương đương ý tưởng `x·W + p·W_d` trong Eq. 5 nhưng cài đặt dưới dạng blend học được.)
- Cả 12 lớp của cả 2 encoder đều được chèn DM-Adapter.

## Loss function

Dùng tổ hợp: **SDM (parameter-free) + LB auxiliary loss**. Cấu hình loss trong code: `loss_names = 'sdm+aux'`.

- **SDM (Similarity Distribution Matching)**, kế thừa từ IRRA:
  - Ý tưởng: đưa phân phối cosine similarity của ma trận `N×N` cặp image–text vào **KL divergence** để khớp với phân phối ground-truth.
  - `L_i2t = KL(p_i ‖ q_i)` — **Eq. (7)**; tổng hai chiều `L_sdm = L_i2t + L_t2i` — **Eq. (8)**.
  - Temperature điều khiển qua `logit_scale` (temperature = 0.02 mặc định trong code).
- **Load-Balancing (LB) loss** — **Eq. (6)**: cân bằng tải giữa các expert (xem phần architecture).
- **Tổng optimization objective — Eq. (9):** `L = L_sdm + α · (L_aux^I + L_aux^T)`, cộng LB loss từ cả nhánh ảnh và văn bản; hệ số `α = 0.5`.

**Lưu ý:** SDM là parameter-free (không thêm tham số ⇒ không tốn communication khi mở rộng sang Federated Learning). MLM được bật ở script train nhưng chỉ đóng vai trò **data augmentation trên văn bản** (mask token đầu vào kiểu BERT), không tính vào loss cuối (loss_names chỉ có `sdm+aux`).

## Dataloader

| Key | Value |
| --- | --- |
| Kích thước ảnh đầu vào | **384 × 128** |
| Stride size (patch) | 16 |
| Độ dài chuỗi token văn bản | **77** |
| Tokenizer | Simple tokenizer, vocab **49,152** |
| Data augmentation (image) | `img_aug` bật (random crop/flip...) |
| Data augmentation (text) | Random mask token kiểu BERT (`--MLM`, dùng như augmentation) |
| Optimizer | Adam |
| Epochs | 60 |
| Batch size | 128 |
| Learning rate | `3e-4` (initial), lr scheduler cosine, warmup 5 epoch |
| Phần cứng | 1× NVIDIA RTX 4090 24GB |

## Hyperparameters

| Tham số | Giá trị | Ghi chú |
| --- | --- | --- |
| Backbone | CLIP ViT-B/16 | frozen |
| Số expert `n` | **6** | quét 2/4/6/8/10, tốt nhất tại 6 (>6 giảm do vượt quy mô data) |
| Top-K | **2** | K=1 suy biến về single adapter; K>2 chững lại nhưng chậm hơn |
| Reduction `b` (bottleneck adapter) | **8** | theo CSKT |
| Trọng số aux loss `α` | **0.5** | |
| Temperature (SDM) | **0.02** | qua logit_scale |
| `scale` nhánh ảnh / văn bản | 0.1 / 4 | từ code |

**Hyper-parameter analysis (Figure 5):** (trên) số expert `n` — R@1 tăng dần đến `n=6` rồi giảm ⇒ dung lượng mô hình phải khớp quy mô data; (dưới) Top-K — K=1 kém, K≥2 bão hoà, tốc độ xử lý giảm khi K tăng ⇒ chọn K=2 để cân bằng. *(Có thể cắt Figure 5 nếu cần minh hoạ.)*

## Benchmark

**So với CSKT (PETL baseline) và IRRA/CFine (full fine-tuning):**

| Method | Tuning | Trainable param | R@1 CUHK | R@1 ICFG | R@1 RSTP |
| --- | --- | --- | --- | --- | --- |
| CFine | Full | 205M | 69.57 | 60.83 | 50.55 |
| IRRA | Full | 195M | 73.38 | 63.46 | 60.20 |
| CSKT | PETL | 12M | 69.70 | 58.90 | 57.75 |
| **DM-Adapter (Ours)** | **PETL** | **16M** | **72.17** | **62.64** | **60.00** |

- **Vượt CSKT rõ rệt** trên CUHK-PEDES: `+2.47%` R@1, `+1.82%` R@5, `+1.05%` R@10, `+2.16%` mAP; trên ICFG-PEDES: `+2.25%` R@1 — chứng minh MoE + domain prompts trích fine-grained tốt hơn PETL cơ bản.
- **Ngang ngửa IRRA** (full fine-tuning, 195M tham số + module reasoning phức tạp) nhưng chỉ dùng **16M tham số** ⇒ đạt trade-off hiệu năng/chi phí.
- **Chống overfitting trên dataset nhỏ:** trên RSTPReid, DM-Adapter vượt **CFine `+9.45%` R@1** — do CFine (full-tuning, 205M) overfit trên tập nhỏ nhất, còn PETL giữ được generalization.

**Hiệu quả bộ nhớ (Table 2, batch size = 32):**

| Method | R@1 | Memory (M) | Trainable param |
| --- | --- | --- | --- |
| IRRA | 73.38 | 7034 (28.64%) | 195M |
| CFine | 69.57 | 13570 (55.24%) | 205M |
| CSKT | 69.70 | 2338 (9.52%) | 12M |
| **DM-Adapter** | **72.17** | **2952 (12.02%)** | **16M** |

- Bộ nhớ chỉ **~2952 MB** (bằng ~42% của IRRA, ~22% của CFine) → phù hợp phần cứng hạn chế.

**Ablation study (Table 5, R@1 trung bình 3 dataset):**

| # | Cấu hình | Avg. R@1 |
| --- | --- | --- |
| 0 | Zero-shot CLIP | 10.96 |
| 1 | + MLP-Adapter (single) | 63.89 |
| 2 | + SMA (w/o LB) | 64.38 |
| 3 | + SMA (w/ LB) | 64.65 |
| 4 | + Domain-Aware Router (**DM-Adapter đầy đủ**) | **64.94** |

- SMA vs single adapter: `+1.00%` (CUHK), `+0.5%` (RSTP) → MoE tăng năng lực trích fine-grained.
- LB loss giúp cân bằng tải, hiệu quả rõ trên dataset nhỏ (RSTP).
- DR bổ sung domain info → cải thiện thêm; tổng cộng DM-Adapter (No.4) vượt single MLP-Adapter (No.1) `+1.05%` avg R@1.
- **Figure 6** trực quan hoá trọng số 6 expert → xác nhận các expert chuyên biệt vào các khía cạnh khác nhau.

## Ghi chú kết nối với đề tài (PEFT trong Federated Learning cho TPR)

- DM-Adapter thuộc nhánh **PETL trên CLIP** giống UP-Person, nhưng thay vì "hợp nhất Prefix/LoRA/Adapter", nó đi theo hướng **MoE hoá adapter (nhiều expert thưa) + bơm tri thức miền vào router**.
- Ưu điểm với FL: chỉ ~16M tham số trainable ⇒ **chi phí communication thấp** khi trao đổi cập nhật giữa client–server; backbone CLIP frozen dùng chung. SDM parameter-free cũng không phát sinh tham số truyền.
- Điểm cần lưu ý khi đưa vào FL: cơ chế **routing/Top-K và Load-Balancing** vốn nhạy với phân phối dữ liệu — trong môi trường **non-IID giữa các client**, cân bằng tải expert và tính ổn định của domain prompt là hai điểm rủi ro/đáng nghiên cứu (ví dụ: mỗi client có thể "kéo" expert theo domain riêng, gây drift). Đây có thể là hướng đóng góp: thiết kế router/expert ổn định dưới non-IID trong FL.
