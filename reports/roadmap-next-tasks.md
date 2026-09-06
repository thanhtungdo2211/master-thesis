# Roadmap: các nhiệm vụ tiếp theo cho pipeline `clip_lora_fedavg`

> Ngày viết: 2026-09-05. Phạm vi: 3 nhiệm vụ sắp tới cho pipeline FedAvg + LoRA + CLIP trên RSTPReid (`experiments/clip_lora_fedavg/`).

## Tổng quan

| # | Nhiệm vụ | Ưu tiên | Effort ước lượng | Phụ thuộc |
|---|---|---|---|---|
| 1 | Trực quan hóa loss tại client | Cao (làm trước) | Nhỏ (1 file sửa + 1 script mới) | Không |
| 2 | Thử nghiệm rank LoRA khác nhau | Trung bình | Vừa (viết script sweep + chạy nhiều lần) | Có thể tận dụng #1 để debug |
| 3 | Training LoRA trên centralized | Trung bình | Vừa (thêm 1 partition strategy mới) | Không phụ thuộc #1/#2, có thể làm song song |

Đề xuất thứ tự: làm **#1 trước** vì nhẹ, không tốn compute, và hữu ích ngay để debug/quan sát khi chạy các thử nghiệm ở #2 và #3. #2 và #3 có thể làm song song sau đó.

---

## 1. Trực quan hóa loss tại client

**Mục tiêu:** Xem loss hội tụ ra sao ở từng client riêng lẻ qua các round, để đánh giá mức độ heterogeneity/straggler giữa các client — đặc biệt quan trọng với các partition `camera`/`dirichlet` vốn non-IID theo thiết kế.

**Hiện trạng:** `train_one_client()` (`experiments/clip_lora_fedavg/federated.py:34-83`) đã trả về `avg_loss` cho từng client mỗi round, nhưng vòng lặp FedAvg trong `main.py:126-165` chỉ gộp thành `mean_loss = np.mean(losses)` rồi bỏ đi phần breakdown theo từng client.

**Việc cần làm:**
- Ghi thêm 1 file `client_loss.csv` (long-format: `round,client_id,n_samples,loss`) song song với `log.csv` hiện có.
- Viết script `plot_client_loss.py` đọc file này và vẽ: (a) line loss theo round cho từng client, (b) boxplot/spread độ lệch loss giữa các client tại mỗi round.
- Thêm 1 cell tương ứng vào notebook Colab để dùng ngay trong môi trường train chính.

**Output:** `client_loss.csv` trong mỗi run dir + `client_loss.png` (hoặc hiển thị trực tiếp trong notebook).

**Spec chi tiết:** xem file plan đã lưu tại `/home/tung/.claude/plans/y-l-c-c-nhi-m-snug-toucan.md` (Phần B) — đã triển khai trong cùng đợt code này.

---

## 2. Thử nghiệm rank LoRA khác nhau

**Mục tiêu:** Đo trade-off giữa số tham số trainable (uplink mỗi round) và accuracy (R@1/R@5/R@10/mAP/mINP) khi thay đổi `--lora_rank` (ví dụ: 2, 4, 8, 16, 32), giữ các hyperparameter khác cố định.

**Việc cần làm:**
- Viết 1 script sweep (subprocess hoặc gọi `clip_lora_fedavg.main.main(argv)` trực tiếp trong loop) chạy lần lượt từng rank, mỗi lần dùng `--out_dir runs/rank_sweep/r{R}` riêng, giữ nguyên `--partition`/`--seed`/các hyperparameter khác để so sánh công bằng.
- Tổng hợp kết quả cuối (best R@1 tại mỗi rank) thành bảng + biểu đồ accuracy-vs-rank và accuracy-vs-uplink — mở rộng pattern đã có ở cell 18 của notebook (đọc nhiều `log.csv` từ nhiều run, vẽ so sánh).
- Cân nhắc so sánh thêm `client_loss.csv` giữa các rank (từ mục 1) để xem rank có ảnh hưởng đến độ lệch loss giữa các client hay không (ví dụ rank thấp có khiến một số client khó fit hơn không).

**Output:** `runs/rank_sweep/r{2,4,8,16,32}/` (mỗi thư mục là 1 run đầy đủ) + bảng so sánh + 2 biểu đồ tổng hợp.

**Lưu ý:** Chưa có script sweep nào trong repo hiện tại — cần viết mới. Không có `peft` dependency nên không có tiện ích tự động set rank; chỉ cần truyền `--lora_rank` qua CLI (`config.py:36`).

---

## 3. Training LoRA trên centralized (non-FL)

**Mục tiêu:** Có một baseline "centralized LoRA" để tách bạch hai khoảng cách:
- **FL gap**: centralized LoRA so với FL LoRA (ảnh hưởng của việc phân tán/aggregation).
- **PEFT gap**: centralized LoRA so với centralized full fine-tune (đã có sẵn ở `data/irra_rtsp`, baseline IRRA full fine-tune).

Đây là điều README hiện tại (`experiments/README.md`) đã đặt vấn đề ("ba con số cần trích xuất") nhưng chưa có code để đo trực tiếp con số centralized-LoRA.

**Việc cần làm:**
- Thêm 1 partition strategy mới `"centralized"` vào `build_partition()` (`experiments/clip_lora_fedavg/data.py:63`), trả về đúng 1 client chứa toàn bộ `train_items` (không chia nhỏ).
- Chạy pipeline hiện tại (`main.py`) với partition này — vì chỉ có 1 client nên mỗi "round" tương đương 1 lần train tiếp trên toàn bộ dữ liệu, không có bước aggregate nào tạo khác biệt so với train tuần tự bình thường. `--rounds` × `--local_epochs` đóng vai trò tổng số epoch training thật.
- Không cần eval per-round quá thường xuyên (có thể tăng `--eval_every`) vì không có yếu tố FL để theo dõi hội tụ theo round riêng.

**Output:** 1 run dir centralized-LoRA (ví dụ `runs/centralized_lora4/`), dùng để so sánh trực tiếp với các run FL hiện có (B3/B4/B5) trong bảng tổng kết kết quả.

**Lưu ý:** Không cần chạy centralized full fine-tune riêng cho pipeline này vì đã có `data/irra_rtsp` (checkpoint + log) làm baseline full fine-tune tham chiếu.
