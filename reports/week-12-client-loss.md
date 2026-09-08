# Báo cáo Tuần 12 — Phân tích loss theo từng client (FedAvg + LoRA + CLIP)

> Dữ liệu: run thật `data/runs/B5_camera_lora4_v2_20260709/` (partition `camera`, `lora_rank=4`, `client_fraction=1.0` — cả 15 client tham gia mọi round, `local_epochs=1`, `batch_size=64`). Run đã hoàn thành **21/25 round** (dừng giữa chừng, có thể do Colab disconnect — `checkpoint.pt` cho phép resume tiếp).
> Nguồn số liệu: `client_loss.csv` (315 dòng = 21 round × 15 client) và `log.csv`, đọc trực tiếp bằng script, không suy diễn.
> Đối chiếu với hai báo cáo tuần trước: cấu trúc partition (`reports/week-11-partition.md`) và cơ chế FedAvg/LoRA (`reports/week-11.md`).

---

## 1. Tổng quan tiến độ (từ `log.csv`)

| Round | Mean loss (toàn cục) | R@1 |
|---|---|---|
| 1 | 25.84 | – |
| 5 | 18.84 | 47.50 |
| 10 | 17.24 | 53.55 |
| 15 | 16.49 | 55.85 |
| 20 | 15.70 | **56.20** |
| 21 | 15.55 | – |

Mean loss giảm đều, R@1 vượt CLIP baseline (54.05) từ round 10, đạt 56.20 ở round 20 — **theo con số tổng, mọi thứ đang hội tụ tốt**. Nhưng đây là loss **trung bình cộng gộp tất cả client mỗi round** (`mean_loss = np.mean(losses)` trong `main.py`) — nó che mất hoàn toàn sự khác biệt giữa các client. Phần dưới bóc tách con số này ra.

---

## 2. Độ lệch (heterogeneity) giữa các client **tăng dần theo thời gian**, không giảm

Tính mean/std/CV của loss **giữa 15 client** tại mỗi round (không phải theo thời gian của 1 client):

| Round | Mean | Std | **CV** | Client thấp nhất | Client cao nhất |
|---|---|---|---|---|---|
| 1 | 25.84 | 1.85 | 0.072 | 14 (21.18) | 9 (27.57) |
| 5 | 18.84 | 2.47 | 0.131 | 14 (14.49) | 13 (22.87) |
| 10 | 17.24 | 2.80 | 0.163 | 14 (12.35) | 10 (21.29) |
| 15 | 16.49 | 2.95 | 0.179 | 14 (11.38) | 10 (20.80) |
| 21 | 15.55 | 3.10 | **0.200** | 14 (10.49) | 10 (20.27) |

**Phát hiện quan trọng nhất:** CV của loss giữa các client đi từ **0.072 → 0.200**, tăng gần gấp 3 lần trong 21 round, và tăng gần như đơn điệu (xem `client_loss_by_group.png`, panel phải cho phiên bản theo cụm). Nói cách khác: **mean loss tổng thể giảm, nhưng các client không hội tụ đồng đều — khoảng cách giữa client "dễ" và client "khó" ngày càng doãng ra.** Đây chính là điều `log.csv`/R@1 không thể hiện được vì chỉ có 1 con số tổng mỗi round.

Client **14** (camera lớn nhất, 6,530 sample) luôn là client có loss thấp nhất ở **20/21 round**. Client **13** và **10** (2 trong số các client thuộc "Nhóm B" ở tuần 11) luân phiên là client có loss cao nhất ở toàn bộ 21 round.

![per-client loss](../data/runs/B5_camera_lora4_v2_20260709/client_loss.png)

---

## 3. Bảng chi tiết theo client (sắp theo quy mô)

| Client | Sample | Step/round | Nhóm (tuần 11) | Loss round 1 | Loss round 21 | Giảm |
|---|---|---|---|---|---|---|
| 14 | 6,530 | 102 | A | 21.18 | **10.49** | 50.5% |
| 1 | 6,142 | 95 | A | 23.40 | 12.44 | 46.8% |
| 5 | 5,008 | 78 | A | 23.83 | 12.94 | 45.7% |
| 7 | 4,090 | 63 | A | 23.72 | 11.07 | 53.3% |
| 11 | 3,640 | 56 | B | 26.34 | 17.62 | 33.1% |
| 13 | 3,042 | 47 | B | 27.18 | **20.00** | 26.4% |
| 12 | 2,374 | 37 | B | 26.22 | 16.17 | 38.4% |
| 15 | 1,998 | 31 | A | 26.34 | 14.68 | 44.3% |
| 4 | 1,028 | 16 | bridge | 27.32 | 17.86 | 34.6% |
| 10 | 820 | 12 | B | 27.51 | **20.27** | 26.3% |
| 6 | 776 | 12 | B | 26.54 | 14.72 | 44.5% |
| 3 | 626 | 9 | A | 27.06 | 15.10 | 44.2% |
| 8 | 594 | 9 | B | 27.56 | 19.16 | 30.5% |
| 9 | 230 | 3 | B | 27.57 | 18.15 | 34.2% |
| 2 | 112 | 1 | B | 25.77 | 12.60 | 51.1% |

Tương quan tuyến tính giữa **số sample của client** và **loss cuối cùng (round 21)**: **Pearson r = −0.563** (client nhiều data hơn → loss thấp hơn, đúng như dự đoán từ phân tích trọng số hiệu dụng ở tuần 11). Bỏ client 2 (trường hợp ngoại lệ, xem mục 5) thì **r = −0.693** — tương quan khá rõ.

---

## 4. Phát hiện chính: loss tách thành **2 dải** đúng theo cấu trúc cụm đã tìm thấy ở tuần 11

`week-11-partition.md` đã chỉ ra 15 camera-client tách thành 2 cụm gần như rời rạc về identity:
- **Nhóm A** (client 1, 3, 5, 7, 14, 15 — 65.9% dữ liệu)
- **Nhóm B** (client 2, 6, 8, 9, 10, 11, 12, 13 — 31.3% dữ liệu)
- **Cầu nối**: client 4 (2.8%)

Tính mean loss riêng cho từng nhóm ở mỗi round:

| Round | Nhóm A (mean) | Nhóm B (mean) | Cầu nối (c4) | **Gap B − A** |
|---|---|---|---|---|
| 1 | 24.26 | 26.84 | 27.32 | 2.58 |
| 5 | 16.73 | 20.17 | 20.89 | 3.44 |
| 10 | 14.81 | 18.76 | 19.68 | 3.96 |
| 15 | 13.90 | 18.13 | 18.83 | 4.23 |
| 21 | 12.79 | 17.34 | 17.86 | **4.55** |

![loss theo cụm](../data/runs/B5_camera_lora4_v2_20260709/client_loss_by_group.png)

**Gap giữa 2 nhóm tăng gần như đơn điệu suốt 21 round (2.58 → 4.55), không có dấu hiệu thu hẹp.** Đây là bằng chứng thực nghiệm trực tiếp — không còn là suy luận lý thuyết — cho vấn đề đã cảnh báo ở tuần 11:

> *"Non-IID của partition này không đồng nhất — trong nhóm là domain skew, giữa nhóm là label skew gần như tách biệt hoàn toàn."* (week-11-partition.md, mục 3)

Global model đang **học tốt cho Nhóm A và ngày càng bỏ lại Nhóm B phía sau**, dù Nhóm B vẫn đóng góp 31.3% dữ liệu và tham gia đầy đủ mọi round (`client_fraction=1.0`).

---

## 5. Vì sao lại như vậy — nối với 2 cơ chế đã phân tích ở tuần 11

### 5.1 Trọng số hiệu dụng lệch bậc 2 (mục 5, `week-11-partition.md`)

Với `local_epochs=1`, mỗi client chạy $S_k = \lfloor n_k/64\rfloor$ bước; kết hợp hệ số gộp $n_k/N$, trọng số hiệu dụng $\propto n_k^2$. Nhóm A (client 14, 1, 5, 7 — 4 client lớn nhất hệ thống) chiếm phần lớn hơn hẳn trong tổng gradient trung bình so với tỉ lệ dữ liệu danh nghĩa. Kết quả: `global_state` mỗi round bị kéo mạnh về phía cực tiểu hoá loss của Nhóm A, còn Nhóm B chỉ đóng góp một phần nhỏ tương ứng — khớp chính xác với xu hướng loss quan sát được ở đây.

### 5.2 LoRA aggregation bias có thể đang khuếch đại thêm (mục 3.4b, `week-11.md`)

$\overline{B}\,\overline{A}$ (kết quả FedAvg trên factor LoRA) luôn có **rank ≤ r = 4**, trong khi tổng "đúng" của các update cục bộ có thể cần rank tới $K\cdot r$ nếu các client học theo hướng khác nhau. Nhóm A và Nhóm B gần như là 2 domain riêng biệt (ảnh người ở 2 khu vực camera khác nhau, gần như không chung identity) — rất có khả năng chúng cần 2 "hướng thích nghi" khác nhau trong không gian LoRA. Với rank bị nén xuống 4, global $\Delta W$ **không đủ chỗ để đồng thời biểu diễn tốt cả 2 domain** → phải đánh đổi, và vì Nhóm A áp đảo về trọng số hiệu dụng (mục 5.1), phần rank hiệu dụng bị chiếm ưu tiên cho Nhóm A.

**Đây là giả thuyết có thể kiểm chứng trực tiếp bằng nhiệm vụ #2 trong roadmap (rank sweep):** nếu tăng `--lora_rank` (vd. 8, 16) làm gap Nhóm A/B thu hẹp lại (không chỉ R@1 tổng tăng), giả thuyết "rank nghẽn cổ chai" được củng cố. Nếu gap không đổi khi tăng rank, nguyên nhân chính nằm ở trọng số hiệu dụng ($n_k^2$), không phải giới hạn rank.

### 5.3 Ngoại lệ đáng chú ý: client 2 và client 6

Client 2 (112 sample, **chỉ 1 step/round** do `drop_last=True` cắt mất 42.9% dữ liệu — xem `week-11-partition.md` mục 5) và client 6 (776 sample, 12 step/round) đều thuộc Nhóm B nhưng có loss cuối cùng thấp bất thường so với các thành viên khác trong nhóm (12.60 và 14.72, gần mức Nhóm A). Tra lại ma trận containment ở tuần 11: cả hai đều có **containment rất cao với client 11** (0.94–0.95) — client lớn nhất trong Nhóm B. Suy đoán hợp lý: các client nhỏ này "ăn theo" (free-ride) chất lượng biểu diễn mà client 11 đóng góp cho các identity chung, chứ bản thân đóng góp trọng số của chúng gần như không đáng kể.

**Lưu ý về nhiễu:** với chỉ 1–12 step/round, giá trị `avg_loss` của các client này là ước lượng trên rất ít batch → nhiễu cao, không nên coi là tín hiệu hội tụ đáng tin cậy tương đương các client lớn (đây cũng là lý do khuyến nghị nhìn vào **gap theo nhóm** ở mục 4 làm kết luận chính, vì trung bình theo nhóm san bằng phần lớn nhiễu cấp-client này) — client 13 và 10, dù cũng nhỏ/vừa, lại luôn là 2 client tệ nhất bất chấp có containment cao với hầu hết Nhóm B, cho thấy bản thân việc "kết nối tốt trong nhóm" không đủ để đảm bảo loss thấp nếu cả nhóm đang bị model global đối xử bất lợi.

---

## 6. Tác động đến kết quả cuối cùng

1. **R@1 tổng (56.20 ở round 20) là con số trung bình trên 200 identity test, không tách theo domain** — `evaluate.py` hiện chỉ đánh giá trên tập test tập trung, không biết identity nào "thuộc" Nhóm A hay Nhóm B. Vì vậy **không thể khẳng định trực tiếp từ R@1 rằng model đang retrieve kém hơn với người thuộc Nhóm B** — nhưng gap loss huấn luyện ngày càng doãng ra là dấu hiệu rủi ro rất cụ thể cho việc này, và có thể kiểm chứng bằng cách gắn nhãn cụm (A/B/bridge) cho identity trong tập test rồi tách R@1 theo cụm (việc nhỏ, có thể làm ngay).
2. **Đóng góp 31.3% dữ liệu của Nhóm B đang bị "lãng phí" một phần** — họ tham gia đầy đủ (uplink, compute) mỗi round nhưng ảnh hưởng thực tế lên global model ngày càng giảm tương đối so với Nhóm A.
3. Đây là bằng chứng thực nghiệm ủng hộ trực tiếp 2 hướng cải tiến đã đề xuất ở tuần 11 (`week-11-partition.md`, mục "Hướng khai thác"): **Clustered FL** (huấn luyện riêng cho Nhóm A/B, hoặc gộp theo cụm phát hiện được) và **FedNova normalization** (sửa trọng số hiệu dụng $n_k^2$) — nay có số liệu per-client cụ thể để dùng làm baseline so sánh "trước/sau" khi thử các hướng này.

---

## 7. Tóm tắt & việc tiếp theo

1. **CV loss giữa client tăng 0.072 → 0.200** qua 21 round — heterogeneity không giảm dù mean loss tổng vẫn giảm đều.
2. **Gap Nhóm A/B tăng đơn điệu 2.58 → 4.55** — khớp chính xác với cấu trúc 2 cụm identity phát hiện ở tuần 11, là bằng chứng thực nghiệm (không còn là suy luận) cho hiện tượng non-IID "hai chế độ".
3. Tương quan **n_samples ↔ loss cuối = −0.69** (bỏ ngoại lệ) — cỡ dữ liệu vẫn là yếu tố chính, nhưng **containment/identity-sharing với client lớn cùng cụm** (client 2, 6) có thể giúp "ăn theo", còn **client hub của cụm yếu (13, 10)** vẫn chịu thiệt dù kết nối tốt trong cụm.
4. **Việc tiếp theo (đã đưa vào `reports/roadmap-next-tasks.md`):**
   - Rank sweep (nhiệm vụ #2) — kiểm chứng giả thuyết rank compression bằng cách xem gap A/B có thu hẹp khi tăng `lora_rank` hay không.
   - Centralized-LoRA baseline (nhiệm vụ #3) — nếu gap A/B biến mất hoàn toàn ở centralized, xác nhận nguyên nhân nằm ở cơ chế FedAvg (trọng số $n_k^2$ + rank compression), không phải bản thân dữ liệu/kiến trúc.
   - Đề xuất bổ sung: gắn nhãn cụm A/B/bridge cho test set, tách R@1/mAP theo cụm ở `evaluate.py` để xác nhận gap loss huấn luyện có phản ánh ra gap accuracy hay không.
