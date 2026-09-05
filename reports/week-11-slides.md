# Tuần 11 — Bản trình bày

## 1. Mô hình & Loss

- **CLIP ViT-B/16, 2 nhánh độc lập**: nhánh ảnh (384×128 → 192 patch → 12 block) và nhánh text (77 token → 12 block), chỉ gặp nhau ở bước cuối để tính cosine similarity.
- **Chỉnh cho ảnh người**: ảnh dọc 384×128 khác input vuông 224×224 của CLIP gốc → nội suy lại position embedding (14×14 → 24×8).
- **Loss = SDM**, chỉ dựa vào quan hệ "cùng người hay không" trong batch, **không có tham số học được**.
- **Vì sao bắt buộc là SDM**: loss ID-classification cần classifier riêng theo số người của từng client → mỗi client một kích thước khác nhau → **không FedAvg được**. Vì lý do này cũng bỏ luôn head IRR/MLM của IRRA.

## 2. LoRA

- **Cắm vào 4 lớp Linear trong attention** (`q/k/v/out_proj`) của mỗi block, cả 2 nhánh → **96 module**. Mọi thứ còn lại (embedding, LayerNorm, MLP, projection) đóng băng.
- **Công thức**: `out = Wx + (BA)x·(α/r)` — `W` giữ nguyên, chỉ `A`,`B` được train; `B` init = 0 nên lúc đầu model đúng bằng CLIP gốc.
- **Con số**: r=4 → **0.49M tham số train** thay vì **150M** (ít hơn ~300 lần).
- **Lợi ích chính trong FL không phải tiết kiệm VRAM, mà là communication**: mỗi client gửi ~**1.9 MB/round** thay vì ~**573 MB** → đây là "PEFT gap" cần đo.

## 3. FedAvg

- **1 round = 4 bước**: server gửi `global_state` (chỉ A/B) → client train local trên data riêng → gửi A/B + số sample về → server trung bình có trọng số theo số sample.
- **Công thức**: `θ_global = Σ (n_k / N) · θ_k`, tính riêng cho từng tensor. Client nhiều data thì đóng góp nhiều hơn.
- **Vì sao hoạt động được**: trung bình trọng số ≈ đi một bước gradient descent trên toàn bộ data gộp lại — mà không cần gửi data đi đâu.
- **Nhưng áp lên LoRA thì lệch**: thứ gửi đi không phải trọng số, mà là **2 mảnh phân rã** của nó → `Avg(B)·Avg(A) ≠ Avg(BA)`. Ba hệ quả: (a) ghép nhầm mảnh giữa các client, (b) kết quả bị kẹt ở rank ≤ r nên mất thông tin khi client học khác hướng, (c) phân rã không duy nhất nên A/B của 2 client có thể không cùng hệ quy chiếu.
- **Ý nghĩa**: pipeline hiện tại cố ý chấp nhận sai lệch này để lấy baseline → đây chính là khoảng trống đề xuất cải tiến (FFA-LoRA, FLoRA, LoRA-FAIR).
