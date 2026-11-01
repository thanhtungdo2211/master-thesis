# Báo cáo Tuần 14: LoRA rank 8 so với rank 4 (FedAvg + LoRA + CLIP, camera non-IID)

> **Dữ liệu:** run mới `data/runs/r8/` (`lora_rank=8`, `lora_alpha=16`, partition `camera`, 15 client, `client_fraction=1.0`, `local_epochs=1`, `batch_size=64`, `lr=1e-4`, `--lr_schedule cosine --warmup_rounds 3`, `eval_every=1`, seed 42). Run đã chạy **đủ 35/35 round**.
> **Run đối chứng rank 4:** `data/runs/B5_camera_lora4_v1_20262308/` (chạy 34/50 round) và `data/runs/B5_camera_lora4_v2_20260709/` (chạy 21/25 round, đây là run đã phân tích ở báo cáo tuần 12).
> **Tham chiếu thêm:** baseline centralized `data/runs/C1_true_centralized_lora4_20261309/` (r=4, 20 epoch).
> **Nguồn số liệu:** `log.csv`, `client_loss.csv` và `args.json` của từng run, đọc trực tiếp bằng script, không có số nào suy diễn ra. `partition.json` của r8 giống hệt (byte-by-byte) với cả hai run r=4, nên **15 client và dữ liệu của từng client là như nhau**.
> **Hình:** `data/runs/r8/r4_vs_r8.png`

---

## TL;DR

1. **Tăng rank từ 4 lên 8 không cải thiện độ chính xác.** Ở mọi round đánh giá chung, R@1 của r=8 gần như trùng với r=4 (chênh nhau không quá 0.2 điểm so với ít nhất một trong hai run r=4). Cả hai cùng bão hoà quanh **R@1 ≈ 57**. Best R@1: r=8 đạt **57.40** (round 35), r=4 đạt **57.05** (round 25 và 30). Chênh +0.35 điểm, tương đương **7 trên 2000 query**, chỉ với 1 seed, nên không đủ để kết luận r=8 tốt hơn.
2. **Chi phí truyền thông tăng gấp đôi:** 3.75 MB so với 1.875 MB cho mỗi client mỗi round (fp32). Để đạt R@1 ≥ 57, r=8 cần **1350 MB** tổng uplink, còn r=4 cần **≤ 703 MB**. Xét trên trục accuracy theo communication, **r=4 trội hẳn**.
3. **Phần capacity tăng thêm được dùng để khớp dữ liệu train, không giúp generalize.** Train loss của r=8 thấp hơn ở 14/15 client (round 21: trung bình 14.94 so với 15.55). Mức chênh lệch loss giữa các client cũng tăng (CV 0.213 so với 0.200 ở round 21, lên 0.234 ở round 35). Nhưng R@1 trên test **không đổi**.
4. **Insight cho luận văn:** nút thắt của pipeline **không nằm ở capacity của adapter**. Kết quả này củng cố lựa chọn r=4 trong paper, và cho thấy hướng cải thiện tiếp theo nên nhắm vào non-IID/client drift chứ không phải tăng rank.
5. **So sánh này có confound, phải nêu rõ khi báo cáo:** r=4 chạy với lr **constant**, còn r=8 chạy với **cosine**. Ngoài ra run r=8 bị **resume với số round khác**, làm lr nhảy ở round 20 (xem mục 5). Cần một run r=4 cosine 35 round để đối chứng sạch.

---

## 1. Cấu hình hai bên

| | r=4 v1 | r=4 v2 | **r=8** |
|---|---|---|---|
| `lora_rank` / `lora_alpha` | 4 / 8 | 4 / 8 | **8 / 16** |
| alpha / r (hệ số scale của LoRA, `lora.py:20`) | 2.0 | 2.0 | 2.0 |
| Trainable params | 491,520 | 491,520 | **983,040** (đúng 2×) |
| Uplink / client / round (fp32) | 1.875 MB | 1.875 MB | **3.75 MB** |
| Uplink / round (15 client) | 28.1 MB | 28.1 MB | **56.2 MB** |
| LR schedule | constant 1e-4 (run cũ, chưa có flag `--lr_schedule`) | constant 1e-4 | **cosine, warmup 3, min 1e-6** |
| Round dự kiến / đã chạy | 50 / 34 | 25 / 21 | 35 / 35 |
| `eval_every` | 5 | 5 | 1 |
| Thời gian / round (`elapsed_s`) | ≈ 389 s | ≈ 440 s | ≈ 394 s |

Hai bên giữ nguyên alpha/r = 2, nên hệ số scale của update LoRA giống nhau và chỉ có **capacity (rank)** thay đổi. Đây đúng là thiết kế của cell rank sweep trong notebook. Tuy vậy, lr schedule khác nhau là một biến không kiểm soát được.

---

## 2. Độ chính xác trên test theo round

R@1 (%), các round mà cả hai bên đều có đánh giá:

| Round | r=4 v1 | r=4 v2 | **r=8** |
|---|---|---|---|
| 5 | 46.95 | 47.50 | 46.75 |
| 10 | 52.55 | 53.55 | 52.50 |
| 15 | 54.15 | 55.85 | 55.85 |
| 20 | 56.00 | 56.20 | 56.10 |
| 25 | 57.05 | – | 57.05 |
| 30 | 57.05 | – | 57.25 |
| 35 | – | – | **57.40** |

mAP (%):

| Round | r=4 v1 | r=4 v2 | **r=8** |
|---|---|---|---|
| 5 | 36.60 | 36.51 | 37.35 |
| 10 | 40.26 | 40.64 | 41.35 |
| 15 | 42.05 | 42.33 | 43.44 |
| 20 | 43.34 | 43.30 | 44.19 |
| 25 | 44.23 | – | 44.62 |
| 30 | 44.63 | – | 44.81 |
| 35 | – | – | **44.91** |

Ở round 30 (round cuối mà cả hai có số), các metric còn lại như sau. r=4 v1: R@5 79.30 / R@10 87.15 / mINP 22.65. r=8: R@5 79.85 / R@10 86.75 / mINP 22.73. Chênh nhau theo cả hai chiều và đều dưới 0.6 điểm.

**Cách đọc:**
- **R@1 của r=8 bám sát r=4 ở mọi mốc.** Ở round 20, r=8 nằm *giữa* hai run r=4. Ở round 15, r=8 bằng v2. Ở round 10, r=8 thấp hơn v1 0.05 điểm. Hai run r=4 với cùng config và cùng seed (v1, v2) cũng đã lệch nhau tới 1.7 điểm ở round 15 do non-determinism. Mức chênh giữa r=8 và r=4 nhỏ hơn mức dao động tự nhiên đó.
- **mAP của r=8 cao hơn r=4 một chút và đều đặn:** khoảng +0.9 đến +1.4 điểm ở round 15–20, nhưng thu hẹp còn +0.2 ở round 30. Đây là tín hiệu duy nhất nghiêng về r=8. Nó nhỏ, chỉ có 1 seed, và co lại khi train lâu hơn, nên chỉ nên ghi nhận chứ chưa thể kết luận.
- **Cả hai bão hoà ở cùng mức.** Từ round 24 đến 35, R@1 của r=8 chỉ dao động trong khoảng 57.05–57.40. r=4 v1 đứng ở 57.05 cả ở round 25 lẫn 30.
- Cả hai đều vượt CLIP baseline trong IRRA README (54.05): r=4 v2 từ round 15 (lần eval đầu tiên vượt ngưỡng), r=8 từ round 12.

![r4 vs r8](../data/runs/r8/r4_vs_r8.png)

*Panel trái: R@1 theo round. Panel giữa: R@1 theo tổng uplink tích luỹ. Panel phải: CV của loss giữa 15 client.*

---

## 3. Chi phí truyền thông: r=8 tốn gấp đôi để đạt cùng một mức

Tổng uplink tích luỹ (15 client, fp32, lấy từ cột `cum_uplink_MB`) tại round **đầu tiên** đạt ngưỡng:

| Ngưỡng R@1 | r=4 | r=8 | Tỉ lệ |
|---|---|---|---|
| ≥ 55.5 | 421.9 MB (v2, round 15) | 843.8 MB (round 15) | 2.0× |
| ≥ 57.0 | ≤ 703.1 MB (v1, round 25) | 1350.0 MB (round 24) | ≈ 1.9× |

(r=4 chỉ được đánh giá 5 round một lần, nên con số của r=4 là **cận trên**. Thực tế r=4 có thể đã chạm ngưỡng sớm hơn.)

Panel giữa của hình cho thấy rõ điều này: đường r=4 luôn nằm phía trên-bên trái đường r=8. **Ở bất kỳ ngân sách truyền thông nào, r=4 đều cho R@1 cao hơn hoặc bằng r=8.**

Thời gian mỗi round gần như không đổi (≈ 389 s ở r=4 v1, ≈ 394 s ở r=8). Như vậy, trong setup này, **cái giá của rank cao nằm gần như hoàn toàn ở communication, không phải compute**. Lưu ý: các run chạy trên Colab và không ghi lại loại GPU, nên chỉ nên xem đây là ước lượng thô.

---

## 4. Capacity tăng thêm đi đâu? Train loss giảm, test không đổi

Train loss theo từng client, đọc từ `client_loss.csv`. So r=8 với r=4 v2, vì v2 là run r=4 duy nhất có log loss theo client:

| Round | r=4 v2: mean / std / **CV** | r=8: mean / std / **CV** |
|---|---|---|
| 1 | 25.84 / 1.85 / 0.072 | 26.54 / 1.12 / 0.042 |
| 5 | 18.84 / 2.47 / 0.131 | 18.60 / 2.53 / 0.136 |
| 10 | 17.24 / 2.80 / 0.163 | 16.81 / 2.80 / 0.167 |
| 15 | 16.49 / 2.95 / 0.179 | 15.92 / 2.94 / 0.185 |
| 21 | 15.55 / 3.10 / **0.200** | 14.94 / 3.18 / **0.213** |
| 25 | – | 14.70 / 3.23 / 0.220 |
| 35 | – | 14.73 / 3.44 / **0.234** |

Mức giảm loss theo từng client ở round 21 (r=8 trừ r=4 v2), sắp xếp theo kích thước client:

| Client | n_samples | r=4 v2 | r=8 | Δ |
|---|---|---|---|---|
| 14 | 6530 | 10.49 | 9.50 | −0.99 |
| 1 | 6142 | 12.44 | 11.48 | −0.96 |
| 5 | 5008 | 12.94 | 11.79 | −1.15 |
| 7 | 4090 | 11.07 | 10.17 | −0.90 |
| 11 | 3640 | 17.62 | 17.09 | −0.54 |
| 13 | 3042 | 20.00 | 19.09 | −0.92 |
| 12 | 2374 | 16.17 | 15.26 | −0.91 |
| 15 | 1998 | 14.68 | 14.21 | −0.47 |
| 4 | 1028 | 17.86 | 17.86 | −0.00 |
| 10 | 820 | 20.27 | 19.56 | −0.71 |
| 6 | 776 | 14.72 | 14.62 | −0.10 |
| 3 | 626 | 15.10 | 14.52 | −0.59 |
| 8 | 594 | 19.16 | 18.74 | −0.43 |
| 9 | 230 | 18.15 | 17.36 | −0.78 |
| 2 | 112 | 12.60 | 12.89 | +0.29 |

**Phát hiện:**
- **r=8 khớp dữ liệu train tốt hơn:** loss thấp hơn ở 14/15 client. Bốn client lớn nhất (14, 1, 5, 7) giảm nhiều và đồng đều nhất, khoảng −1.0. Phần lớn các client nhỏ (< 1100 sample) giảm ít hơn.
- **Điều này không đến từ lr cao hơn.** Từ round 4 trở đi, lr của r=8 (cosine) luôn **≤** lr constant 1e-4 của r=4. Vì vậy loss thấp hơn chủ yếu là do capacity của adapter. Lưu ý thêm: ở round 1 loss của r=8 *cao hơn* (26.54 so với 25.84), đúng như kỳ vọng vì r=8 đang trong giai đoạn warmup với lr = 3.3e-5.
- **Train loss giảm nhưng R@1 test không tăng.** Đây là dấu hiệu điển hình cho thấy capacity thêm vào được dùng để **fit sâu hơn vào phân phối cục bộ của từng camera**, chứ không học thêm được đặc trưng tổng quát. Pattern này cũng khớp với C1 centralized: train loss xuống tới 8.07 nhưng R@1 chỉ khoảng 57, và đạt đỉnh **58.85 ở round 9** rồi đi xuống.
- **Mức chênh lệch giữa các client tiếp tục tăng, và tăng nhanh hơn ở r=8** (CV 0.213 so với 0.200 ở round 21, lên 0.234 ở round 35). Pattern "client lớn ngày càng dễ, client khó vẫn khó" từ tuần 12 vẫn giữ nguyên. Client 14 có loss thấp nhất ở **35/35 round**. Client 10 có loss cao nhất ở 25 round, client 13 ở 9 round. Rank cao hơn **không thu hẹp** khoảng cách này mà còn làm nó doãng ra thêm một chút.

> **Giả thuyết (chưa kiểm chứng):** với dữ liệu non-IID theo camera, LoRA rank cao hơn cho mỗi client nhiều "chỗ" hơn để học các hướng đặc thù cho camera của mình. FedAvg sau đó lấy trung bình các hướng khác nhau này, nên phần lợi ích bị triệt tiêu ở model toàn cục. Đây là cách giải thích *phù hợp* với dữ liệu, nhưng chưa có thí nghiệm nào tách riêng được nó (ví dụ đo cosine similarity giữa các update LoRA của từng client).

---

## 5. Cảnh báo về tính hợp lệ của phép so sánh

### 5.1 Run r=8 bị resume với `--rounds` khác, làm lr nhảy ở round 20

Cột `lr` trong `r8/log.csv` giảm đột ngột từ **7.43e-5 (round 19) xuống 4.56e-5 (round 20)**. Tính lại công thức cosine (warmup 3, min ratio 0.01) thì khớp chính xác:

| Round | lr nếu `--rounds 50` | lr nếu `--rounds 35` | lr trong log |
|---|---|---|---|
| 18 | 7.71e-5 | 5.54e-5 | 7.71e-5 |
| 19 | 7.43e-5 | 5.05e-5 | **7.43e-5** |
| 20 | 7.13e-5 | 4.56e-5 | **4.56e-5** |
| 21 | 6.83e-5 | 4.08e-5 | 4.08e-5 |

→ **Round 1–19 đã chạy với `--rounds 50`, rồi run được resume từ checkpoint với `--rounds 35` cho round 20–35.** `args.json` chỉ lưu lần chạy cuối (35), nên nhìn vào config sẽ không thấy chi tiết này. Hệ quả: schedule của r=8 không phải cosine 35 round "sạch", vì lr giảm đột ngột khoảng 39% ở round 20. Trên đường R@1 có một nhịp giảm nhẹ ở round 20 (56.60 xuống 56.10), rồi hồi lại 56.75 ở round 21, nên ảnh hưởng có vẻ nhỏ. Dù vậy vẫn phải nêu rõ nếu dùng run này trong paper.

### 5.2 Các confound còn lại

| Yếu tố | r=4 | r=8 | Ảnh hưởng |
|---|---|---|---|
| LR schedule | constant | cosine (bị nhảy ở round 20) | Lớn nhất. Không tách được tác động của rank với tác động của schedule. |
| Số round hoàn thành | 34 (v1), 21 (v2) | 35 | So sánh được tới round 30 (v1) |
| Độ dày của evaluation | 5 round một lần | mỗi round | Best R@1 của r=4 có thể bị bỏ sót đỉnh giữa hai lần eval |
| Số seed | 1 (v1 và v2 cùng seed 42 nhưng vẫn lệch tới 1.7 điểm) | 1 | Mức chênh dưới khoảng 1.5 điểm nên coi là noise |

Điểm đáng chú ý là: **dù schedule khác hẳn nhau (constant so với cosine), đường R@1 của hai bên vẫn gần như trùng nhau.** Điều này gợi ý rằng mức bão hoà khoảng 57 không nhạy với cả rank lẫn lr schedule, tức nút thắt nằm ở chỗ khác. Tuy vậy, đây chỉ là quan sát và chưa phải bằng chứng.

---

## 6. Insight rút ra

1. **Rank không phải nút thắt.** Tăng gấp đôi số tham số trainable (0.49M lên 0.98M) không làm R@1 tăng ở bất kỳ mốc nào. Với dữ liệu RSTPReid chia theo camera, **r=4 đã đủ capacity**.
2. **r=4 là điểm Pareto tốt hơn cho FL.** r=8 cho cùng accuracy với chi phí uplink gấp 2. Trong bối cảnh FL, nơi communication là chi phí chính, đây là lập luận trực tiếp ủng hộ lựa chọn r=4 trong paper (≈ 1.9 MB/client/round).
3. **Capacity thừa bị dùng để overfit cục bộ.** Train loss giảm, mức chênh lệch giữa các client tăng, nhưng test không đổi. Hướng cải thiện độ chính xác nên nhắm vào **non-IID và client drift** (ví dụ FedProx, SCAFFOLD, aggregation có trọng số khác, personalization), chứ không phải tăng rank.
4. **Khoảng cách FL so với centralized không đến từ capacity.** FL r=8 đạt best 57.40, C1 centralized r=4 đạt best 58.85. Khoảng cách khoảng 1.5 điểm còn lại không được rank cao hơn lấp vào.
5. **Bài học vận hành:** khi resume một run, **phải giữ nguyên `--rounds`**, nếu không lr schedule sẽ bị gãy mà `args.json` không ghi lại.

---

## 7. Việc cần làm tiếp

1. **[Quan trọng nhất] Chạy r=4 với đúng config của r=8:** cosine, 35 round, `eval_every 1`, từ đầu và không đổi `--rounds` giữa chừng. Đây là đối chứng duy nhất cho phép nói "rank không ảnh hưởng" mà không vướng confound về schedule. Nếu muốn đường r=8 thật sạch, chạy lại r=8 một mạch 35 round.
2. Nếu còn ngân sách GPU, hoàn thiện sweep `RANKS = [2, 4, 8, 16]` với cùng số round, để có đủ đường accuracy theo rank và accuracy theo uplink cho paper (cell 10/10b trong notebook đã có sẵn).
3. (Tuỳ chọn) Chạy thêm 1–2 seed cho r=4 và r=8. Mức chênh hiện tại (0.35 R@1) nhỏ hơn noise giữa hai run cùng seed (v1 so với v2).
4. Nếu đưa kết quả này vào paper: copy `log.csv` của r=8 vào `paper/results/` (theo quy tắc của repo: mọi số liệu trong paper phải lấy từ CSV ở đó), và ghi rõ confound về lr schedule.
