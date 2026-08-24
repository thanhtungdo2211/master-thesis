# UP-Person: Unified Parameter-Efficient Transfer Learning for Text-based Person Retrieval

## Abstract

- **Tên paper:** UP-Person: Unified Parameter-Efficient Transfer Learning for Text-based Person Retrieval
- **Foundation Model sử dụng:** CLIP (Contrastive Language-Image Pre-training)
- **Bài toán:** Text-based Person Retrieval (TPR) - Tìm kiếm người bằng hình ảnh dựa trên câu truy vấn văn bản.
- **Phương pháp đề xuất:** UP-Person. Phương pháp này tích hợp 3 kỹ thuật PETL (Parameter-Efficient Transfer Learning) gồm Prefix, LoRA và Adapter vào một khối thống nhất để chuyển giao kiến thức từ mô hình CLIP, qua đó giảm thiểu tối đa tham số cần huấn luyện mà vẫn đạt hiệu suất SOTA.

## Purpose

Mục tiêu chính của bài báo là giải quyết rủi ro *overfitting* (học vẹt) và sự sụt giảm khả năng *generalization* (tổng quát hóa) khi fine-tune toàn bộ (full fine-tuning) một mô hình khổng lồ như CLIP cho bài toán tìm kiếm người. Việc cập nhật 100% trọng số của mô hình lớn không chỉ gây tốn kém tài nguyên (lưu trữ, giao tiếp) mà còn làm mất đi lượng kiến thức tổng quát vô giá mà CLIP đã học được trước đó. Do đó, bài báo thiết kế một giải pháp PETL thống nhất, chỉ fine-tune một lượng cực nhỏ tham số nhưng vẫn đạt hiệu năng mạnh mẽ.

## Hypothesis

- **Giả thuyết 1:** Fine-tuning toàn bộ mô hình (Full-tuning) sẽ phá vỡ không gian đặc trưng chung đã được pre-train của CLIP, khiến mô hình dễ bị overfitting trên các tập dữ liệu TPR có quy mô nhỏ/trung bình.
- **Giả thuyết 2:** Freeze backbone của CLIP và chỉ huấn luyện một số ít tham số bổ sung sẽ giúp bảo toàn kiến thức nền tảng, trong khi vẫn đủ linh hoạt để học thêm kiến thức đặc thù của tác vụ TPR.
- **Giả thuyết 3:** Các phương pháp PETL đứng riêng lẻ (chỉ dùng LoRA hoặc chỉ dùng Prefix) có thể bị giới hạn hoặc xung đột. Nếu kết hợp khéo léo Prefix, LoRA (để học thông tin cục bộ) và Adapter (để chỉnh đặc trưng toàn cục) vào một thiết kế hợp nhất (Unified) thì sẽ đạt được kết quả cao nhất.

## Dataset

- **CUHK-PEDES:** Bộ dữ liệu phổ biến nhất với 40,206 hình ảnh và 80,412 mô tả văn bản cho 13,003 danh tính.
- **ICFG-PEDES:** Chứa 54,522 hình ảnh cho 4,102 danh tính, mỗi hình ảnh có một mô tả văn bản tương ứng.
- **RSTPReid:** Chứa 20,505 hình ảnh của 4,101 danh tính, mỗi hình ảnh có 2 mô tả văn bản.

## Architecture

**Tổng quan:** UP-Person giữ nguyên backbone của CLIP (Image Encoder + Text Encoder) và **đóng băng (frozen) toàn bộ**. Chỉ chèn các khối PETL nhẹ vào trong mỗi transformer block và chỉ huấn luyện các khối này (~4.7% tham số, tức 7.4M / 157M).

![Overall framework](image.png)

**Backbone (frozen):**

- **Image Encoder:** CLIP ViT-B/16. Ảnh được chia thành N patch, thêm `[CLS]` token, cộng positional embedding, qua 12 transformer block.
- **Text Encoder:** CLIP Text Transformer, cũng 12 lớp. Token hóa bằng tokenizer vocab 49,152, thêm `[BOS]`/`[EOS]`, lấy output tại `[EOS]` làm biểu diễn toàn cục của câu.

**Unified PETL block** — chèn 3 submodule vào mỗi block của cả 2 encoder, mỗi cái đảm nhận một vai trò khác nhau để tránh xung đột:

| Submodule | Vị trí gắn | Vai trò | Loại thông tin |
| --- | --- | --- | --- |
| **LoRA** | Trên ma trận **Key (W_k) và Value (W_v)** của MHA | Cập nhật trọng số attention rank-thấp để bắt đặc trưng tinh vi | **Local** |
| **S-Prefix** | Prefix token `P_k`, `P_v` ghép vào **K, V** của MHA | Bơm thông tin task-specific, hướng attention vào nội dung TPR | **Local + task-specific** |
| **L-Adapter** | **Song song với LayerNorm** (cả LN của MHA lẫn LN của MLP) | Điều chỉnh phân phối đặc trưng tổng thể theo cách phi tuyến | **Global** |

**Phân chia chức năng (điểm cốt lõi cần highlight):**

L-Adapter lo phần **đặc trưng toàn cục (global)**; còn **LoRA + S-Prefix phối hợp trong Multi-Head Attention** để tập trung vào **đặc trưng cục bộ (local)** — ví dụ "balo đỏ", "giày New Balance nhiều màu". Ba thành phần này được chứng minh là **không xung đột về không gian lẫn chức năng**.

**Cơ sở lý thuyết:**

- **Eq. (9):** Phép phân tích attention với LoRA → sinh ra số hạng phụ `QKᵀΔV + QΔKᵀVᵀ + QΔKᵀΔVᵀ` = phần **local information of TPR**.
- **Eq. (10):** Thêm Prefix → có thêm số hạng `QP_kᵀP_v` = phần **task-specific information of TPR**.
- **Eq. (11):** Adapter điều chỉnh phân phối qua LayerNorm = phần **global information of TPR**.

**Hai submodule cải tiến:**

- **S-Prefix (Scalable Prefix) — Figure 3**
  - Vấn đề: prefix gốc hội tụ rất chậm trong TPR vì gradient của `P_k`, `P_v` quá nhỏ (do `λ(x) → 0`, xem Eq. 13–14).
  - Giải pháp: thêm một **hệ số learnable `Sp`** nhân vào phần attention của prefix để khuếch đại gradient (Eq. 15–16).
  - `Sp` khởi tạo = 10. (Nếu quá lớn, ví dụ `Sp=500`, sẽ gây gradient exploding → không hội tụ.)
- **L-Adapter (Layernorm Adapter) — Figure 4(d)**
  - Đặt **song song với LayerNorm** + residual connection (Eq. 17): `h ← LayerNorm(x) + s·Adapter(x)`.
  - Mục đích: điều chỉnh shift/scale theo cách **phi tuyến**, tránh phá hỏng kiến thức gốc của CLIP (khác với LN-tuning tuyến tính).
  - Tách biệt khỏi MHA/MLP nên **không chồng chéo (conflict)** với LoRA và S-Prefix — khác với parallel adapter thường (vốn span qua MHA/MLP gây xung đột).

## Loss function

Dùng **một hàm loss duy nhất, parameter-free**: **SDM (Similarity Distribution Matching)**, kế thừa từ IRRA (IRRA có thêm ID và MLM). Không dùng loss InfoNCE vì purpose là 1 ảnh có thể tương ứng với nhiều query.

**Ý tưởng:**

- Đưa phân phối cosine similarity của ma trận `N×N` cặp image–text vào **KL divergence** để khớp với phân phối ground-truth.
- Xác suất match `p_{i,j}` tính bằng softmax trên cosine similarity (có temperature `τ = 0.02`) — Eq. (18).
- Loss image→text: `L_{i2t} = KL(p_i ‖ q_i)` — Eq. (19).
- Tổng loss hai chiều: **`L_sdm = L_{i2t} + L_{t2i}`** — Eq. (20), trọng số 2 chiều đặt = 1.

![SDM loss comparison](image.png)

SDM cho kết quả tốt nhất, hơn ITC `+2.13%` R@1. Thêm ID hoặc MLM gần như không cải thiện nhưng làm **tăng vọt số tham số** (SDM: 7.4M → SDM+ID+MLM: 52.3M). Vì thế tác giả chỉ dùng SDM → đơn giản, hiệu quả, dễ mở rộng. (loss không thêm tham số ⇒ không tốn communication) → phù hợp với FL.

## Dataloader

| Key | Value |
| --- | --- |
| Kích thước ảnh đầu vào | **384 × 128** |
| Độ dài chuỗi token văn bản | **77** |
| Tokenizer | Simple tokenizer, vocab **49,152** |
| Data augmentation (text) | Random mask **15%** token, thay bằng `[MASK]` (kiểu BERT) |
| Optimizer | Adam |
| Epochs | 60 |
| Batch size | 128 |
| Learning rate | `1e-3` (CUHK, ICFG); `1e-4` (RSTPReid — vì data nhỏ, PETL nhạy với LR) |
| Phần cứng | 1× NVIDIA RTX 4090 24GB |

## Hyperparameters

| Tham số | CUHK-PEDES | ICFG-PEDES | RSTPReid |
| --- | --- | --- | --- |
| LoRA rank `r` | 32 | 32 | 16 |
| Prefix length `l` | 10 | 14 | 2 |
| Bottleneck `b` (L-Adapter) | 8 | 8 | 8 |
| `Sp` init | 10 | 10 | 10 |

## Benchmark

**So với IRRA và CFine (cả hai đều full fine-tuning):**

| Method | Tuning | Trainable param | R@1 CUHK | R@1 ICFG | R@1 RSTP |
| --- | --- | --- | --- | --- | --- |
| CFine | Full | 205M | 69.57 | 60.83 | 50.55 |
| IRRA | Full | 195M | 73.38 | 63.46 | 60.20 |
| DM-Adapter | PETL | 16M | 72.17 | 62.64 | 60.00 |
| **UP-Person (ViT-B/16)** | **PETL** | **7.4M** | **74.17** | **65.02** | **63.15** |
| UP-Person (ViT-L/14) | PETL | – | 76.04 | 65.98 | 64.45 |

- **Giảm 95.1% tham số huấn luyện** (7.4M vs 150M+) nhưng **vẫn vượt SOTA** trên cả 3 dataset.
- **Chống overfitting tốt hơn full-tuning:** trên RSTPReid (dataset nhỏ nhất), UP-Person vượt IRRA `+2.95%` R@1 — đúng như giả thuyết về generalization.
- **Domain generalization (Table XI):** khi train trên RSTP rồi test sang CUHK/RSTP, UP-Person vượt IRRA tới `+4.21%` và `+6.2%` R@1 → PETL giữ được kiến thức gốc của CLIP.
- **Hiệu quả tài nguyên (Table IV):** chỉ 3262 MB memory (chưa bằng nửa IRRA), FLOPs thấp nhất, LoRA merge được vào trọng số gốc nên **không thêm overhead lúc inference**.
