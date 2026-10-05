# Tuần 12 — Client loss - 20 round Federated Learning


## 1. Loss Chart

![per-client loss](../data/runs/B5_camera_lora4_v2_20260709/client_loss.png)

**Insight**

- Client **14** (đường thấp nhất): loss đi từ 21.18 (round 1) xuống **10.49** (round 21) — giảm hơn một nửa (50.5%).
- Client **10** (một trong 2 đường cao nhất): loss chỉ đi từ 27.51 xuống **20.27** — giảm chưa tới 1/3 (26.3%).
- Chênh lệch giữa client thấp nhất và cao nhất: round 1 chênh **6.4 điểm** (21.18 vs 27.57), round 21 chênh tới **9.8 điểm** (10.49 vs 20.27).
- Độ lệch chuẩn của loss giữa 15 client (đo bằng CV = độ lệch chuẩn / trung bình): **0.072 ở round 1 → 0.200 ở round 21** — tăng gần 3 lần.

**Insight 1:** Toàn bộ 15 client có **tốc độ học rất khác nhau, và khoảng cách giữa client ít dữ liệu và client nhiều không thu hẹp lại theo thời gian mà ngày càng rộng ra.**

---

## 2. Root Cause

Nguyên nhân nằm ở cách server gộp (trung bình) kết quả của 15 client lại thành 1 global model mỗi round.

**Bước 1 — client nhiều ảnh thì train được nhiều bước hơn trong cùng 1 round:**

| Client | Số data | Số step train/round (batch=64) |
|---|---|---|
| 14 | 6,530 | 102 bước |
| 13 | 3,042 | 47 bước |
| 2 | 112 | **1 bước duy nhất** |

Client 14 được tuning trên dữ liệu 102 lần mỗi round, trong khi client 2 chỉ được tuning đúng 1 lần..

**Bước 2 — server gộp có trọng số theo số ảnh, nên client nhiều ảnh vừa dịch chuyển xa hơn (bước 1), vừa được tính trọng số cao hơn khi gộp:**

Hai yếu tố này nhân dồn vào nhau. Theo tính toán ở tuần trước: 4 client lớn nhất (14, 1, 5, 7) nắm **58.8% dữ liệu** nhưng ảnh hưởng thực tế của họ lên global model mỗi round ước tính lên tới **hơn 80%**. 
**Bước 3 — hệ quả  qua các round:**

Vì global model bị nắn theo hướng khớp với client 14/1/5/7, nên khi 4 client này nhận lại model ở round sau, họ thấy nó *đã khớp sẵn* với dữ liệu của mình → loss thấp, giảm nhanh. Còn các client nhỏ (13, 10, 12, 9...) khi nhận lại model, vẫn thấy nó "xa lạ" với dữ liệu riêng của họ → loss giảm chậm, dừng ở mức cao. Vòng lặp này diễn ra liên tục suốt 21 round, nên khoảng cách không tự thu hẹp mà càng lúc càng xa.

Trong 15 camera, có 6 camera (client 1, 3, 5, 7, 14, 15 — gọi là **Nhóm A**, chiếm 65.9% ảnh) mà người xuất hiện ở đó **gần như không xuất hiện** ở 8 camera còn lại (client 2, 6, 8, 9, 10, 11, 12, 13 — gọi là **Nhóm B**, chiếm 31.3% ảnh)..


**Số liệu:** loss trung bình của Nhóm A và Nhóm B qua các round:

| Round | Nhóm A | Nhóm B | Chênh lệch |
|---|---|---|---|
| 1 | 24.26 | 26.84 | 2.58 điểm |
| 10 | 14.81 | 18.76 | 3.96 điểm |
| 21 | 12.79 | 17.34 | **4.55 điểm** |

**Insight 2:** Chênh lệch giữa 2 khu vực **tăng liên tục, không có dấu hiệu thu hẹp** dù cả 2 khu vực đều tham gia đóng góp dữ liệu đầy đủ mọi round. Model đang **ưu tiên học tốt cho khu vực A** (vì áp đảo cả số lượng ảnh lẫn mức độ ảnh hưởng lên global model, theo đúng cơ chế ở mục 2), còn khu vực B ngày càng bị bỏ lại phía sau dù không hề "vắng mặt" trong quá trình training.



**Client 2** chỉ có 112 ảnh, mỗi round chỉ train được đúng 1 batch (48 ảnh dư bị bỏ vì không đủ 1 batch) — tưởng sẽ là client học kém nhất vì gần như không đóng góp gì cho global model. Nhưng thực tế loss của client 2 lại **thấp bất ngờ, gần bằng mức Nhóm A** (12.60 ở round 21). Lý do: trong 53 người xuất hiện ở camera 2, có tới **50 người cũng xuất hiện ở camera 11** (client lớn nhất Nhóm B, 3,640 ảnh). Khi client 11 train và giúp model học tốt hơn cho những người này, client 2 **"hưởng ké" luôn kết quả đó** — dù bản thân gần như không tự đóng góp được gì. Client 6 (776 ảnh) cũng rơi vào tình huống tương tự nhờ trùng nhiều người với client 11.

→ **loss thấp không phải lúc nào cũng có nghĩa là client đó ít dữ liệu** — đôi khi chỉ vì nó may mắn trùng người với 1 client lớn khác đang gánh phần học chính.

Ngược lại, **client 13** (3,042 ảnh — không hề nhỏ, đứng thứ 6/15 về quy mô) lại luôn nằm trong nhóm 2 client tệ nhất suốt cả 21 round (loss cuối 20.00, chỉ giảm 26.4%). Vì client 13 thuộc Nhóm B (khu vực bị model global đối xử kém hơn), nên dù bản thân có kích thước kha khá, vẫn không đủ để thoát khỏi bất lợi chung của cả nhóm.

---

## 3. Baseline: gom hết dữ liệu về 1 chỗ thì LoRA làm được tới đâu?

Đã chạy thêm 1 run **không dùng Federated Learning** — toàn bộ 18,505 ảnh để chung 1 máy, train 20 epoch
(~2.4 giờ). Đây là mốc để biết: phần accuracy bị mất đi **do phải chia dữ liệu ra 15 camera** là bao nhiêu.

![centralized vs FL](../data/runs/C1_true_centralized_lora4_20261309/centralized_vs_fl.png)

**Con số**

| | R@1 |
|---|---|
| CLIP gốc, chưa train gì (mốc tham chiếu) | 54.05 |
| **LoRA gom chung 1 chỗ — mức ổn định** | **57.2** |
| **LoRA gom chung 1 chỗ — lần đo tốt nhất** | **58.85** |
| IRRA — train toàn bộ 150 triệu tham số + 3 loss phụ | 60.20 |

Chỉ train **0.33% số tham số** (0.49 triệu trên 150 triệu) mà hơn CLIP gốc **3 điểm**, và chỉ kém
bài IRRA **3 điểm** — trong khi IRRA phải train *toàn bộ* model.

---

**So sánh ở cùng số round** (1 round FL tốn đúng bằng 1 epoch gom chung, nên đây là so công bằng về
lượng tính toán):

| Round | FL (15 camera) | Gom chung 1 chỗ | Cách nhau |
|---|---|---|---|
| 5 | 47.50 | 54.15 | 6.65 điểm |
| 10 | 53.55 | 57.00 | 3.45 điểm |
| 20 | 56.20 | 56.95 | **0.75 điểm** |
| 30 | 57.05 | *(đã đứng yên ở ~57.2)* | **≈ 0** |

**Insight 3:** Chia dữ liệu ra 15 camera **không làm model kém đi, chỉ làm nó học chậm hơn**.
Khoảng cách thu hẹp đều và gần như biến mất ở round 30 — tức là FL chỉ cần **thêm khoảng 50% số round**
là đuổi kịp. Mà "thêm round" nghĩa là "thêm lượt gửi dữ liệu lên server", và đó đúng là thứ LoRA làm rẻ:
**28 MB mỗi round cho cả 15 client**, so với **8.4 GB mỗi round** nếu phải gửi nguyên model. Rẻ hơn 300 lần.
Đây là lập luận ủng hộ trực tiếp cho hướng đi của đề tài.

*Lưu ý: điều này nói về **accuracy tổng**. Nó **không** mâu thuẫn với mục 1–2: khoảng cách giữa các client
vẫn đang doãng ra, và Nhóm B vẫn bị bỏ lại. Con số tổng che mất điều đó — đúng như mục 1 đã chỉ ra.*

---

**Insight 4:** Trần hiện tại (~57 điểm) là do **thiếu loss**, không phải do LoRA quá nhỏ.

Bằng chứng: loss lúc train vẫn đang giảm mạnh (21.2 → 8.1) trong khi điểm test đã **đứng yên suốt
12 epoch cuối**. Dấu hiệu này nghĩa là model **đang học thuộc dữ liệu train** chứ không phải **thiếu sức chứa**.
Nên tăng kích thước LoRA hay train thêm epoch sẽ không giúp gì — thứ còn thiếu là 3 loss phụ mà IRRA có
còn pipeline này đã bỏ (vì chúng không gộp được trong Federated Learning, xem báo cáo tuần 11).

Đã thử thêm lịch giảm dần learning-rate và giữ optimizer liên tục — **không nâng được trần**, chỉ đỡ học thuộc hơn.
Đây là kết quả âm, nhưng nó loại trừ được một hướng nghi ngờ.

---

**Insight 5:** Không nên chọn model theo R@1.

R@1 chỉ đếm "đoán đúng ngay ở vị trí số 1", nên nó **nhiễu hơn mAP khoảng 2.6 lần**. Code hiện đang lưu
model tốt nhất theo R@1 → nó chọn trúng một lần đo may mắn ở round 9, **cao hơn mức thật 1.7 điểm**
(58.85 so với 57.2). Xét theo mAP thì round 9 còn thua round 14, và mAP đã đứng yên từ round 8 —
tức model ở round 9 **không hề tốt hơn**, chỉ gặp may ở lần đo đó.

→ Khi lập bảng so sánh phải so **ở cùng round**, không so "điểm cao nhất" giữa các run có số lần đánh giá
khác nhau: run nào được đo nhiều lần hơn sẽ thắng chỉ vì được rút nhiều lá bài hơn.

---

**Cần nói rõ khi trích bảng ở trên:** run gom chung dùng cấu hình mới (có lịch learning-rate), run FL dùng
cấu hình cũ — nên đây là **chỉ dấu, chưa phải phép đo sạch**. Ngoài ra run gom chung còn được giữ optimizer
liên tục, lợi thế mà FL về bản chất không thể có. Cần chạy thêm 2 run nữa để chốt con số.
