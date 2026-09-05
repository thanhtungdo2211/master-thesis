# Tuần 11 — Thống kê partition theo camera (RSTPReid, 15 client)

> Số liệu tính trực tiếp từ `data/RSTPReid/data_captions.json`, split `train` (18,505 ảnh / 3,701 identity / 37,010 caption).

---

## 1. Bảng thống kê từng camera-client

| Camera | Ảnh | % tổng | % lũy kế | ID | Ảnh/ID | Caption | Step/round (B=64) |
|---|---|---|---|---|---|---|---|
| 14 | 3,265 | 17.6% | 17.6% | 2,307 | 1.42 | 6,530 | 102 |
| 1 | 3,071 | 16.6% | 34.2% | 2,421 | 1.27 | 6,142 | 95 |
| 5 | 2,504 | 13.5% | 47.8% | 2,208 | 1.13 | 5,008 | 78 |
| 7 | 2,045 | 11.1% | 58.8% | 1,968 | 1.04 | 4,090 | 63 |
| 11 | 1,820 | 9.8% | 68.7% | 1,393 | 1.31 | 3,640 | 56 |
| 13 | 1,521 | 8.2% | 76.9% | 1,360 | 1.12 | 3,042 | 47 |
| 12 | 1,187 | 6.4% | 83.3% | 1,105 | 1.07 | 2,374 | 37 |
| 15 | 999 | 5.4% | 88.7% | 952 | 1.05 | 1,998 | 31 |
| 4 | 514 | 2.8% | 91.5% | 503 | 1.02 | 1,028 | 16 |
| 10 | 410 | 2.2% | 93.7% | 228 | **1.80** | 820 | 12 |
| 6 | 388 | 2.1% | 95.8% | 370 | 1.05 | 776 | 12 |
| 3 | 313 | 1.7% | 97.5% | 307 | 1.02 | 626 | 9 |
| 8 | 297 | 1.6% | 99.1% | 290 | 1.02 | 594 | 9 |
| 9 | 115 | 0.6% | 99.7% | 114 | 1.01 | 230 | 3 |
| **2** | **56** | **0.3%** | 100.0% | 53 | 1.06 | 112 | **1** |

**Tóm tắt phân tán quy mô:**

| Chỉ số | Giá trị |
|---|---|
| Ảnh min / max | 56 / 3,265 — **chênh 58.3 lần** |
| CV (std/mean) | **0.851** |
| Gini | **0.471** |
| Top-1 client | 17.6% tổng data |
| Top-3 client | 47.8% |
| Top-5 client | **68.7%** |

Ghi chú: hầu hết camera có **ảnh/ID ≈ 1.0** — tức mỗi người chỉ xuất hiện 1 lần ở camera đó. Ngoại lệ là camera 10 (1.80) và camera 14/1 (1.42/1.27). Hệ quả: trong batch train của phần lớn client, **positive pair chủ yếu đến từ 2 caption của cùng một ảnh**, chứ không phải 2 ảnh khác nhau của cùng một người — SDM có ít tín hiệu cross-view hơn kỳ vọng.

---

## 2. Identity trải trên bao nhiêu camera?

| Số camera 1 ID xuất hiện | Số ID | % |
|---|---|---|
| 1 camera | 6 | 0.2% |
| 2 camera | 94 | 2.5% |
| 3 camera | 653 | 17.6% |
| 4 camera | 1,314 | 35.5% |
| 5 camera | 1,634 | 44.2% |

**Trung bình 4.21 camera/ID, trung vị 4.** Chỉ 0.2% identity nằm gọn trong một camera duy nhất.

→ Đây **không phải label-space disjoint**. Các client chia sẻ identity rất rộng. Về nguyên tắc đây là **feature/domain skew**, không phải label skew.

---

## 3. Phát hiện quan trọng: cấu trúc **theo cụm (block-structured)**

Chỉ số `mean_pairwise_jaccard_pid = 0.124` trong `partition_stats.json` **gây hiểu nhầm** — nó là trung bình trên 105 cặp có kích thước rất chênh lệch. Bóc tách ra:

| Nhóm cặp | Mean Jaccard |
|---|---|
| Toàn bộ 105 cặp | 0.124 |
| Chỉ các cặp giữa 7 camera lớn (≥1,000 ID) | **0.344** |
| Các cặp có ít nhất 1 camera nhỏ (<400 ID) | 0.048 |

Nhìn vào ma trận **containment** $|A \cap B| / \min(|A|,|B|)$ (1.00 = tập nhỏ nằm gọn trong tập lớn):

```
     c1   c2   c3   c4   c5   c6   c7   c8   c9   c10  c11  c12  c13  c14  c15
c1   1.00 0.02 0.81 0.34 0.91 0.10 0.92 0.12 0.35 0.03 0.16 0.17 0.15 0.85 0.86
c3   0.81 0.00 1.00 0.16 0.66 0.03 0.57 0.00 0.00 0.00 0.30 0.26 0.30 0.67 0.01
c5   0.91 0.06 0.66 0.39 1.00 0.05 0.85 0.01 0.04 0.00 0.08 0.08 0.07 0.87 0.81
c7   0.92 0.04 0.57 0.17 0.85 0.25 1.00 0.03 0.43 0.00 0.11 0.06 0.08 0.83 0.72
c14  0.85 0.04 0.67 0.50 0.87 0.11 0.83 0.01 0.04 0.02 0.20 0.19 0.20 1.00 0.91
c15  0.86 0.00 0.01 0.25 0.81 0.04 0.72 0.00 0.01 0.03 0.13 0.09 0.12 0.91 1.00
------------------------------------------------------------------------------
c2   0.02 1.00 0.00 0.04 0.06 0.11 0.04 0.11 0.02 0.02 0.94 0.72 0.94 0.04 0.00
c6   0.10 0.11 0.03 0.18 0.05 1.00 0.25 0.21 0.34 0.07 0.95 0.62 0.96 0.11 0.04
c8   0.12 0.11 0.00 0.02 0.01 0.21 0.03 1.00 0.17 0.02 0.97 0.82 0.99 0.01 0.00
c9   0.35 0.02 0.00 0.05 0.04 0.34 0.43 0.17 1.00 0.07 0.88 0.67 0.72 0.04 0.01
c10  0.03 0.02 0.00 0.01 0.00 0.07 0.00 0.02 0.07 1.00 0.96 0.71 0.95 0.02 0.03
c11  0.16 0.94 0.30 0.71 0.08 0.95 0.11 0.97 0.88 0.96 1.00 0.98 0.97 0.20 0.13
c12  0.17 0.72 0.26 0.59 0.08 0.62 0.06 0.82 0.67 0.71 0.98 1.00 0.95 0.19 0.09
c13  0.15 0.94 0.30 0.69 0.07 0.96 0.08 0.99 0.72 0.95 0.97 0.95 1.00 0.20 0.12
```

Cấu trúc 2 khối hiện rõ:

| Nhóm | Camera | Ảnh | % tổng |
|---|---|---|---|
| **Nhóm A** | 1, 3, 5, 7, 14, 15 | 12,197 | 65.9% |
| **Nhóm B** | 2, 6, 8, 9, 10, 11, 12, 13 | 5,794 | 31.3% |
| **Cầu nối** | 4 | 514 | 2.8% |

- **Trong nhóm**: chia sẻ identity rất mạnh — c1↔c7 = 0.92, c1↔c5 = 0.91, c14↔c15 = 0.91; c11↔c12 = 0.98, c8↔c13 = 0.99, c2↔c11 = 0.94.
- **Giữa 2 nhóm**: gần như rời rạc — c1↔c2 = 0.02, c5↔c11 = 0.08, c14↔c8 = 0.01. Camera 14 và camera 2 chỉ **chung đúng 2 identity** trên tổng 53 của camera 2.
- Camera 4 nằm giữa (0.30 với A, 0.29 với B) — client duy nhất bắc cầu.

Kiểm chứng lại bằng cách khác: 53 identity của camera 2 trung bình xuất hiện ở **4.06 camera** — nhưng những camera đó là 11, 13, 12 (50/53, 50/53, 38/53), **không** phải 14, 1, 5.

### Ý nghĩa

Non-IID của partition này **không đồng nhất**, mà là **hai chế độ khác nhau tùy cặp client**:

| Cặp | Loại skew |
|---|---|
| Trong cùng nhóm | **Domain/feature skew** — cùng người, khác góc chụp/ánh sáng |
| Khác nhóm | **Label skew** — gần như không chung người nào |

Đây gần như chắc chắn phản ánh **bố cục vật lý thật** của hệ camera (RSTPReid kế thừa từ MSMT17): 2 khu vực/tòa nhà tách biệt, người đi qua khu A không được camera khu B ghi hình.

---

## 4. Về CV = 0.851 — nghiêm trọng tới mức nào?

Cần nói cho chính xác thay vì nói chung chung "cao hơn bình thường":

| Setup | CV xấp xỉ |
|---|---|
| IID split (chính là B3 — đối chứng của bạn) | ≈ 0 |
| CIFAR + Dirichlet, đa số paper FL (chia label, giữ size cân bằng) | thường < 0.3 |
| LEAF / FEMNIST (partition tự nhiên theo người viết) | ≈ 0.39 |
| **Partition camera của bạn** | **0.851** |
| LEAF / Shakespeare, Sent140 (partition tự nhiên) | 1.6 – 2.0 |

**Kết luận công bằng:** so với các setup **thực nghiệm có kiểm soát** (IID hoặc Dirichlet với size cân bằng — thứ mà đa số paper FL dùng để báo cáo) thì 0.851 là **lệch rất mạnh**. Nhưng so với các benchmark **partition tự nhiên** (LEAF) thì nó nằm ở mức trung bình — không phải trường hợp cực đoan.

Điều thực sự đáng lo không phải bản thân con số CV, mà là **hệ quả của nó khi kết hợp với FedAvg**.

---

## 5. Hệ quả: trọng số hiệu dụng bị lệch bậc 2

Với `local_epochs=1`, client $k$ chạy $S_k = \lfloor n_k/64 \rfloor$ bước, nên độ dịch chuyển $\|\theta_k - \theta_g\| \propto S_k$. Kết hợp với hệ số gộp $n_k/N$:

$$
\Delta\theta_{\text{global}} \approx -\eta \sum_k \frac{n_k S_k}{N}\bar{g}_k
\quad\Longrightarrow\quad
\text{trọng số hiệu dụng} \propto n_k^2
$$

| Client | Ảnh | Danh nghĩa $n_k/N$ | **Hiệu dụng** $\propto n_kS_k$ | Chênh |
|---|---|---|---|---|
| 14 | 3,265 | 17.6% | **27.3%** | ×1.55 |
| 1 | 3,071 | 16.6% | **23.9%** | ×1.44 |
| 5 | 2,504 | 13.5% | 16.0% | ×1.18 |
| 7 | 2,045 | 11.1% | 10.6% | ×0.95 |
| 11 | 1,820 | 9.8% | 8.3% | ×0.85 |
| 9 | 115 | 0.62% | 0.028% | ×0.045 |
| **2** | **56** | **0.30%** | **0.0046%** | **×0.015** |

- **Top-2 client**: 34.2% danh nghĩa → **51.2% thực tế** (chi phối quá bán global update).
- **Top-5 client**: 68.7% → **86.1%**.
- **Client 2**: bị bóp nhỏ **66 lần** — thực chất gần như không tồn tại trong model global.

Đây là **objective inconsistency**, phân tích trong FedNova (Wang et al., NeurIPS 2020). Không phải bug — FedAvg gốc của McMahan cũng có đúng vấn đề này — nhưng cần ghi vào mục "Giới hạn đã biết".

### Cộng thêm: drop_last ăn mất data của client nhỏ

`DataLoader(..., drop_last=True)` với `batch_size=64`:

| Client | Caption | Step | Sample bị bỏ |
|---|---|---|---|
| 2 | 112 | 1 | 48 (**42.9%**) |
| 9 | 230 | 3 | 38 (16.5%) |
| 3 | 626 | 9 | 50 (8.0%) |
| 10 | 820 | 12 | 52 (6.3%) |
| 14 | 6,530 | 102 | 2 (0.03%) |
| **Tổng** | 37,010 | **571** | 466 (1.3%) |

Tổng thể chỉ mất 1.3% nên không đáng lo, nhưng nó tập trung vào đúng những client vốn đã yếu thế.

---

## 6. Tóm tắt để đưa vào báo cáo

1. **Quy mô rất lệch**: 56 → 3,265 ảnh (58×), CV = 0.851, Gini = 0.471, top-5 client giữ 68.7% data.
2. **Identity chia sẻ rộng**: trung bình 1 người xuất hiện ở 4.21/15 camera, chỉ 0.2% người nằm gọn 1 camera → về bản chất là **domain skew**, không phải label skew.
3. **Nhưng có cấu trúc cụm**: 15 camera tách thành 2 nhóm gần như rời rạc về identity (A: 6 camera / 65.9% data; B: 8 camera / 31.3%; camera 4 bắc cầu). Trong nhóm Jaccard ~0.34–0.99, giữa nhóm ~0.01–0.20. → non-IID **hai chế độ**: domain skew trong nhóm, label skew giữa nhóm.
4. **Trọng số hiệu dụng lệch bậc 2**: top-2 client thực tế chiếm 51% global update thay vì 34%; client nhỏ nhất bị bóp 66 lần.
5. **Chỉ số `mean_pairwise_jaccard_pid` trong `partition_stats.json` (0.124) không nên trích dẫn đơn lẻ** — nó bị kéo xuống bởi các cặp camera lệch size và che mất cấu trúc cụm.

### Hướng khai thác

- **Clustered FL** (IFCA, CFL) là lựa chọn khớp tự nhiên với cấu trúc 2 cụm đã phát hiện — có thể là đóng góp riêng cho luận văn, ngoài phần LoRA aggregation bias.
- **FedNova normalization** — chỉ sửa hàm `fedavg` vài dòng, cho một ablation rẻ và sạch.
- Bổ sung vào `partition_stats()`: containment matrix và phân bố số camera/ID, thay vì chỉ mean Jaccard.
