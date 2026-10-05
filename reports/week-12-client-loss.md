# Báo cáo Tuần 12 — Phân tích loss theo từng client (FedAvg + LoRA + CLIP)

> Dữ liệu: run thật `data/runs/B5_camera_lora4_v2_20260709/` (partition `camera`, `lora_rank=4`, `client_fraction=1.0` — cả 15 client tham gia mọi round, `local_epochs=1`, `batch_size=64`). Run đã hoàn thành **21/25 round** (dừng giữa chừng, có thể do Colab disconnect — `checkpoint.pt` cho phép resume tiếp).
> Nguồn số liệu: `client_loss.csv` (315 dòng = 21 round × 15 client) và `log.csv`, đọc trực tiếp bằng script, không suy diễn.
> Đối chiếu với hai báo cáo tuần trước: cấu trúc partition (`reports/week-11-partition.md`) và cơ chế FedAvg/LoRA (`reports/week-11.md`).
> **Bổ sung mục 7:** baseline centralized `data/runs/C1_true_centralized_lora4_20261309/` (20 round = 20 epoch, `--persistent_optimizer`, `--lr_schedule cosine`, `--eval_every 1`) — đã chạy xong, dùng để trả lời nhiệm vụ #3 của mục 8.

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

### 2.0 Đơn vị và range của loss

Loss ở đây là **SDM loss** (Similarity Distribution Matching) — về bản chất là **KL divergence** giữa 2 phân phối: (1) phân phối similarity mà model dự đoán (softmax của cosine similarity giữa ảnh và caption trong batch) và (2) phân phối "đúng" dựa trên nhãn (ảnh/caption nào cùng 1 người thì phải giống nhau). Vì là KL divergence nên:

- **Không có thang đo %** như accuracy — không thể nói "loss 15 nghĩa là đúng 85%".
- **Không âm, không có giới hạn trên cố định** — càng gần 0 thì model dự đoán càng khớp hoàn hảo với nhãn thật; càng lớn thì càng lệch. Về lý thuyết có thể lớn tùy ý nếu model dự đoán sai hoàn toàn.
- **Trong run này, giá trị quan sát được dao động khoảng 10.5 → 27.6** trên toàn bộ 21 round × 15 client (thấp nhất: client 14 ở round 21 = 10.49; cao nhất: client 9 ở round 1 = 27.57). Vì tất cả client dùng chung 1 công thức và cùng hệ số nhiệt độ (`temperature=0.02`), **so sánh loss giữa các client với nhau là hợp lệ** dù bản thân con số không có ý nghĩa tuyệt đối kiểu %.

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

**Panel bên phải — "Client loss heterogeneity per round" — đang thể hiện điều gì:**

Đây là 1 boxplot cho **mỗi round** (không phải cho từng client). Tại mỗi round, lấy 15 giá trị loss của 15 client, vẽ thành 1 hộp:

- Đường giữa hộp (màu cam) = **median** (giá trị loss ở giữa) của 15 client tại round đó.
- Thân hộp = khoảng **25%–75%** (phần lớn client nằm trong khoảng này).
- 2 râu (whisker) trên/dưới = **min–max**, tức client thấp nhất và cao nhất tại round đó.

→ Nó không nói "1 client cụ thể học nhanh hay chậm" (đó là việc của panel trái), mà nói **"tại mỗi thời điểm, 15 client đang tản mát ra sao so với nhau."** Đọc chuỗi 21 hộp từ trái sang phải: cả hộp dịch chuyển xuống dưới đều đặn (đúng — training vẫn đang tiến triển, ai cũng giảm loss), nhưng **chiều cao của hộp và độ dài của râu không co lại mà có xu hướng rộng ra** — nghĩa là độ tản mát (chênh lệch giữa client dễ nhất và khó nhất) **không thu hẹp theo thời gian, mà ngày càng lớn**. Đây là cách trực quan để thấy đúng điều số liệu CV ở trên đang chỉ ra, mà không cần đọc bảng số.

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

## 7. Baseline centralized LoRA (C1-true) — trần trên của pipeline

> Run: `data/runs/C1_true_centralized_lora4_20261309/`. Toàn bộ 18,505 ảnh trong **1 client**
> (FedAvg là phép đồng nhất → 1 round = 1 epoch), `--persistent_optimizer` (một `AdamW` duy nhất
> sống suốt run — training centralized đúng nghĩa, không bị reset moment mỗi epoch như đường code FL),
> `--lr_schedule cosine --warmup_rounds 3`, LoRA r=4 α=8, 20 epoch, batch 64, seed 42.
> 491,520 / 150,109,185 tham số trainable (**0.33%**). Tổng 8,573 s ≈ 2.4 h, `--eval_every 1`.

### 7.1 Kết quả

| | R@1 | R@5 | R@10 | mAP | mINP |
|---|---|---|---|---|---|
| **Best-of-20** (round 9 / 14) | **58.85** | 80.30 | 89.15 | **45.56** | **24.14** |
| **Mức hội tụ** (mean round 15–20) | **57.18** ± 0.28 | 80.92 | 88.57 | **45.42** ± 0.13 | 23.84 ± 0.13 |
| CLIP zero-shot (mốc IRRA repo) | 54.05 | – | – | – | – |
| IRRA full FT + IRR + MLM + ID | 60.20 | – | – | – | – |

Train loss giảm đơn điệu 21.16 → 8.07 và **vẫn đang giảm** ở epoch 20.

![centralized vs FL](../data/runs/C1_true_centralized_lora4_20261309/centralized_vs_fl.png)

### 7.2 R@1 là tiêu chí chọn model tệ — mAP ổn định hơn 2.6 lần

Trên các round đã hội tụ (10–20), độ biến thiên tương đối của hai chỉ số chênh nhau rõ rệt:

| Chỉ số | Mean (r10–20) | Std | **CV** |
|---|---|---|---|
| R@1 | 57.04 | 0.43 | **0.0076** |
| mAP | 45.42 | 0.13 | **0.0029** |

R@1 chỉ đếm **hit ở vị trí 1** trên 2,000 caption query → là một ước lượng nhị thức, sai số chuẩn
lý thuyết $\sqrt{p(1-p)/n} \approx 1.1$ pp so với giá trị thật. mAP trung bình trên **toàn bộ thứ hạng**
của mọi query nên nhiễu bị san phẳng.

**Hệ quả trực tiếp lên code:** `main.py` lưu `best.pt` theo `metrics["R@1"] > best_r1`, tức là lấy
**max của 20 lần đo trên chỉ số nhiễu nhất**. Kết quả là nó chọn round 9 (58.85) — cao hơn mức hội tụ
thật **+1.67 pp**. Nhìn sang mAP thì round 9 (45.455) còn *thua* round 14 (45.561), và đường mAP đã
phẳng từ round 8, tức là mô hình ở round 9 **không hề tốt hơn** mô hình ở round 20 — nó chỉ gặp may
ở top-1 của lần đo đó.

> **Khuyến nghị cho luận văn:** báo cáo **mức hội tụ** (mean của k round cuối) làm số chính, kèm
> best-of trong ngoặc và nói rõ nó là max-of-N. Nếu vẫn muốn chọn checkpoint tự động thì đổi tiêu chí
> sang mAP. Bảng nào chỉ ghi "best R@1" của các run có **số lần eval khác nhau** là bảng so sánh sai:
> run nào eval nhiều lần hơn thì được rút nhiều lá bài hơn, và sẽ thắng chỉ vì thế. Đây chính là
> trường hợp so C1-true (20 lần eval) với B5 (4 lần eval) trong mục 7.4 — phải so ở cùng round.

### 7.3 LR schedule + persistent optimizer **không nâng trần** — kết quả âm, nhưng hữu ích

Tuần trước có một run centralized theo cấu hình cũ (lr hằng số 1e-4, optimizer bị tạo lại mỗi epoch —
đường code FL). Run đó chỉ tồn tại trong console log, chưa tải về `data/runs/`, nên đây là đối chiếu
tham khảo chứ không phải số từ file:

| Round | Cũ: lr hằng số + reset opt | Mới (C1-true): cosine + persistent |
|---|---|---|
| 5 | loss 12.14 · R@1 57.45 · mAP 44.12 | loss 12.92 · R@1 54.15 · mAP 42.64 |
| 10 | loss 9.58 · R@1 56.35 · mAP 45.03 | loss 9.98 · R@1 57.00 · mAP 45.19 |
| 15 | loss 7.77 · R@1 57.20 · mAP **45.51** | loss 8.45 · R@1 57.55 · mAP 45.35 |
| 20 | loss **6.42** · R@1 55.90 · mAP 44.85 | loss **8.07** · R@1 56.95 · mAP 45.42 |

Hai điều đọc được:

1. **Trần không đổi.** mAP tốt nhất: cũ 45.51 (r15) vs mới 45.56 (r14) — chênh 0.05, tức là bằng nhau.
   Giả thuyết đặt ra khi sửa code ("lr hằng số khiến Adam nhảy quanh cực tiểu, thêm cosine sẽ kéo R@1 lên")
   **không được số liệu ủng hộ**. Ghi nhận là kết quả âm.
2. **Nhưng overfit giảm rõ.** Ở cùng epoch 20, run mới có train loss **cao hơn** (8.07 vs 6.42) mà test
   metric lại **bằng hoặc tốt hơn** (mAP 45.42 vs 44.85). Đúng dấu hiệu của regularization: model mới học
   thuộc train ít hơn mà tổng quát hóa không kém. Warmup cũng làm epoch 1–5 chậm hơn (R@1 54.15 vs 57.45
   ở r5) — cái giá phải trả, và nó hoàn lại từ round 10.

Kết luận thực chất: **57.2 / mAP 45.4 là trần của cấu hình hiện tại, và trần đó do *loss function* quyết định,
không phải do optimizer hay lr.** Bằng chứng: train loss vẫn đang giảm mạnh ở 8.07 khi test metric đã phẳng
12 round — model **không** bị giới hạn capacity, nó đang học thuộc. Thêm rank hay thêm epoch sẽ không giúp;
thứ còn thiếu là tín hiệu huấn luyện (IRRA dùng SDM **+ IRR + MLM + ID loss**, pipeline này chỉ có SDM).

### 7.4 FL gap: chi phí của federation là **số round**, không phải chất lượng cuối

So ở **cùng số round** (1 round FL ≈ 580 step ≈ 1 epoch centralized, nên đây là so compute-matched):

| Round | B5 FL camera (run 2) | C1-true centralized | Gap R@1 | Gap mAP |
|---|---|---|---|---|
| 5 | 47.50 / 36.51 | 54.15 / 42.64 | **+6.65** | +6.13 |
| 10 | 53.55 / 40.64 | 57.00 / 45.19 | **+3.45** | +4.54 |
| 15 | 55.85 / 42.33 | 57.55 / 45.35 | **+1.70** | +3.03 |
| 20 | 56.20 / 43.30 | 56.95 / 45.42 | **+0.75** | +2.12 |
| 30 (run 1) | 57.05 / 44.63 | *(đã phẳng: 57.18 / 45.42)* | +0.13 | +0.79 |

**Gap thu hẹp đơn điệu, không bão hòa ở một mức dương cố định.** Ở round 30, B5 run 1 đạt mAP 44.63 —
chỉ còn cách plateau centralized **0.79 mAP**, và vẫn đang leo. Nói cách khác, với partition camera này
**FedAvg không hạ trần accuracy, nó chỉ cần nhiều round hơn để tới nơi** (~1.5× số round).

Điều đó có ý nghĩa tốt cho luận đề: chi phí của federation ở đây quy về **truyền thông**, mà truyền thông
chính là thứ LoRA làm rẻ đi — 1.88 MB × 15 client = 28.2 MB/round, 30 round ≈ 0.83 GB tổng, so với
8.4 GB **mỗi round** nếu full fine-tune. Trả thêm 50% số round trên một kênh rẻ hơn 300 lần vẫn là lời.

**Ba cảnh báo phải ghi kèm khi trích bảng này:**

1. **Hai thế hệ run bị trộn.** C1-true dùng cosine + persistent optimizer; B5 dùng lr hằng số + reset
   optimizer mỗi round. Con số gap là **chỉ dấu, chưa phải phép đo sạch**. Muốn sạch thì chạy lại B5 bằng
   cell 12 của notebook (cùng cosine, 20 round).
2. **C1-true còn được lợi thế mà FL về mặt cấu trúc không thể có** — optimizer giữ state suốt run. Một phần
   của gap là "optimizer có trí nhớ", không phải "dữ liệu bị phân tán". Run **C1 (FL-symmetric)** — cùng
   cosine nhưng optimizer bị reset mỗi round như FL — mới tách được hai thành phần này. **Chưa chạy.**
3. B5 chỉ eval mỗi 5 round nên chỉ có 4–6 điểm; đường cong FL trong hình là nội suy thô giữa các điểm đó.

### 7.5 Run này **không** trả lời được câu hỏi gap Nhóm A/B

Nhiệm vụ #3 ở mục 8 đặt vấn đề: *"nếu gap A/B biến mất hoàn toàn ở centralized, xác nhận nguyên nhân nằm ở
cơ chế FedAvg"*. Run centralized **không thể** kiểm chứng điều đó: chỉ có 1 client nên `client_loss.csv` chỉ
có 1 dòng mỗi round, không còn cấu trúc nhóm nào để đo. Nó cho **trần trên**, không cho **phân rã theo nhóm**.

Muốn kiểm chứng giả thuyết A/B vẫn phải làm đúng việc đã đề xuất ở mục 8: **gắn nhãn cụm A/B/bridge cho
200 identity trong test set, rồi tách R@1/mAP theo cụm trong `evaluate.py`**. Khi đó mới so được
"centralized có gap A/B không" với "FL có gap A/B bao nhiêu" — và C1-true chính là mẫu đối chứng cần có
sẵn cho phép so đó. Theo nghĩa đó run này là **điều kiện cần đã hoàn thành**, không phải câu trả lời.

### 7.6 PEFT gap

Với **0.33% tham số** (491,520 / 150M), LoRA đạt 57.2 hội tụ / 58.85 best-of, so với:

- **CLIP zero-shot 54.05** → **+3.1 / +4.8 R@1**. LoRA r=4 chỉ chạm vào `q,k,v,out_proj` là đủ để vượt rõ.
- **IRRA full FT 60.20** → **−3.0 / −1.35 R@1**. Nhưng IRRA fine-tune **toàn bộ 150M** tham số *và* có thêm
  IRR + MLM + ID loss. Phần thiếu hụt này (xem 7.3) đến từ **loss function**, không từ PEFT.

Điều đáng viết vào luận văn: **không tách được "PEFT gap" ra khỏi "loss gap" bằng cặp số hiện có.** Muốn
đo PEFT gap sạch phải có baseline **full fine-tune + chỉ SDM, centralized** — tức là chính cell B4 nhưng ở
chế độ centralized. Đây là một ô trống trong bảng thí nghiệm, nên bổ sung.

---

## 8. Tóm tắt & việc tiếp theo

1. **CV loss giữa client tăng 0.072 → 0.200** qua 21 round — heterogeneity không giảm dù mean loss tổng vẫn giảm đều.
2. **Gap Nhóm A/B tăng đơn điệu 2.58 → 4.55** — khớp chính xác với cấu trúc 2 cụm identity phát hiện ở tuần 11, là bằng chứng thực nghiệm (không còn là suy luận) cho hiện tượng non-IID "hai chế độ".
3. Tương quan **n_samples ↔ loss cuối = −0.69** (bỏ ngoại lệ) — cỡ dữ liệu vẫn là yếu tố chính, nhưng **containment/identity-sharing với client lớn cùng cụm** (client 2, 6) có thể giúp "ăn theo", còn **client hub của cụm yếu (13, 10)** vẫn chịu thiệt dù kết nối tốt trong cụm.
4. **Baseline centralized (mục 7, nhiệm vụ #3 — đã chạy xong):** trần trên của pipeline là
   **57.18 R@1 hội tụ / 58.85 best-of**, mAP 45.42. Ba điều rút ra:
   - **FL không hạ trần, chỉ làm chậm** — gap so centralized thu hẹp đơn điệu 6.65 → 0.75 R@1 từ round 5
     đến round 20, và ở round 30 FL chỉ còn cách 0.79 mAP. Chi phí của federation ở partition này quy về
     **số round** (≈1.5×), tức về truyền thông — đúng thứ LoRA làm rẻ (28.2 MB/round cho cả 15 client).
   - **R@1 là tiêu chí chọn model tệ** — CV 0.0076 so với 0.0029 của mAP. `best.pt` chọn theo R@1 nên lấy
     đúng đỉnh nhiễu, cao hơn mức hội tụ thật 1.67 pp. Mọi bảng so sánh phải so **ở cùng round**, không so
     "best R@1" giữa các run có số lần eval khác nhau.
   - **Trần 57.2 do loss function quyết định, không do optimizer/lr** — cosine + persistent optimizer
     *không* nâng được trần (kết quả âm, mục 7.3), trong khi train loss vẫn đang giảm ở 8.07 lúc test metric
     đã phẳng 12 round. Thứ còn thiếu so với IRRA 60.20 là IRR + MLM + ID loss, không phải capacity của LoRA.
5. **Việc tiếp theo** (danh sách này chưa được đồng bộ vào `reports/roadmap-next-tasks.md` — cần cập nhật riêng)**:**
   - Rank sweep (nhiệm vụ #2) — kiểm chứng giả thuyết rank compression bằng cách xem gap A/B có thu hẹp khi tăng `lora_rank` hay không. **Lưu ý mới từ mục 7.3:** train loss centralized vẫn đang giảm khi test metric đã phẳng → bài toán là overfit chứ không phải thiếu capacity, nên nhiều khả năng đường cong rank sẽ **phẳng hoặc hình chữ U ngược**. Đó vẫn là kết quả đáng báo cáo ("r=4 đã bão hòa, trả thêm uplink không mua được gì"), nhưng đừng kỳ vọng nó thu hẹp gap tới IRRA.
   - **Chạy C1 (FL-symmetric)** — cùng cosine nhưng optimizer bị reset mỗi round như FL. Đây là mảnh còn
     thiếu để tách "chi phí phân tán dữ liệu" khỏi "chi phí optimizer mất trí nhớ" trong con số gap ở mục 7.4.
   - **Chạy lại B5 bằng cell 12** (cosine, 20 round) — bảng gap hiện tại đang trộn hai thế hệ run
     (C1-true cosine vs B5 lr hằng số), nên chỉ là chỉ dấu.
   - **Bổ sung ô trống: full fine-tune + chỉ SDM, centralized** — không có baseline này thì không tách được
     "PEFT gap" khỏi "loss gap" (mục 7.6).
   - Gắn nhãn cụm A/B/bridge cho test set, tách R@1/mAP theo cụm ở `evaluate.py` — **vẫn là việc bắt buộc**
     để kiểm chứng giả thuyết A/B; run centralized không thay thế được (mục 7.5).
