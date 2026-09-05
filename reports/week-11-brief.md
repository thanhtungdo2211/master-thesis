# Tuần 11 — Bản tóm tắt: CLIP + LoRA + FedAvg

> Bản ngắn, dễ đọc. Bản chi tiết đầy đủ (kèm diagram) xem ở `week-11.md`.

---

## 1. Mô hình & Loss

### Mô hình: CLIP ViT-B/16 — 2 nhánh riêng biệt

CLIP có 2 "bộ mã hóa" chạy song song, **không dùng chung trọng số**:

| Nhánh | Đầu vào | Xử lý | Đầu ra |
|---|---|---|---|
| **Vision** | Ảnh 384×128 | Cắt thành patch 16×16 → grid 24×8 = 192 patch + [CLS] → 12 block transformer | `image_embeds` |
| **Text** | Caption | Tokenize (77 token) → 12 block transformer → lấy token [EOS] | `text_embeds` |

Hai nhánh **chỉ gặp nhau ở bước cuối**: chuẩn hóa L2 rồi tính cosine similarity giữa ảnh và câu mô tả. Càng giống nhau thì điểm càng cao.

**Một chỉnh sửa cần thiết:** CLIP gốc train trên ảnh vuông 224×224, nhưng ảnh người thì dọc (384×128). Nên phải nội suy lại position embedding từ grid 14×14 → 24×8, và sửa code HuggingFace để bỏ dòng kiểm tra "ảnh phải vuông".

### Loss: SDM — chọn vì bắt buộc, không phải vì thích

SDM chỉ cần biết trong batch **ảnh nào và caption nào cùng một người** (`pid_i == pid_j`), rồi ép model cho điểm similarity khớp với quan hệ đó. Điểm quan trọng: **SDM không có tham số học được nào**.

Vì sao điều đó quan trọng? Nếu dùng loss ID-classification (cách phổ biến trong ReID) thì mỗi client cần một lớp classifier có kích thước bằng **số người trong data của client đó**. Mà mỗi client có tập người khác nhau → 2 vấn đề:

- Kích thước classifier khác nhau → **không cộng trung bình được**.
- Kể cả trùng kích thước thì cột thứ 3 ở client A là người X, ở client B lại là người Y → cộng trung bình cũng vô nghĩa.

Vì lý do này pipeline cũng **bỏ luôn** các head phụ của IRRA (IRR, MLM). Nguyên tắc: mọi thứ trao đổi giữa client và server phải hoàn toàn giống nhau về cấu trúc.

---

## 2. LoRA cắm vào đâu

### Cắm ở đâu

Chỉ vào **4 lớp Linear trong khối attention** của mỗi transformer block: `q_proj`, `k_proj`, `v_proj`, `out_proj`. Áp dụng cho cả 2 nhánh:

> 12 block × 4 lớp × 2 nhánh = **96 module** được gắn LoRA

**Không đụng tới** (freeze hết): patch embedding, token/position embedding, LayerNorm, phần MLP (`fc1`/`fc2`), và 2 lớp projection cuối.

### Cách hoạt động

$$
\text{out} = \underbrace{W x}_{\text{đóng băng}} + \underbrace{(B A)\, x \cdot \frac{\alpha}{r}}_{\text{phần học được}}
$$

- `W` là trọng số CLIP gốc — giữ nguyên, không train.
- `A` và `B` là 2 ma trận nhỏ, chỉ 2 cái này được train.
- `B` khởi tạo bằng **0** → lúc bắt đầu delta = 0, model đúng bằng CLIP gốc. Nhờ vậy train không bị "phá" pretrained ngay từ đầu.

**Con số:** với r=4 → khoảng **0.49M tham số** cần train, so với **150M** nếu fine-tune toàn bộ. Ít hơn ~**300 lần**.

### Vì sao chọn attention?

Attention là chỗ model quyết định "nhìn vào đâu". Khi chuyển sang domain mới (ảnh người, mô tả người), cái cần thay đổi chủ yếu là **cách chú ý**, chứ không phải bộ trích đặc trưng cấp thấp. Paper LoRA gốc cũng thử nhiều tổ hợp và thấy Q/V (hoặc cả Q,K,V,O) cho hiệu quả tốt nhất trên mỗi đơn vị tham số.

Cắm cả 4 là chọn kiểu "phủ hết cho chắc" làm baseline. Code để mở qua flag `--lora_targets` nên muốn thử chỉ `k_proj,v_proj` (như UP-Person) thì đổi flag là được, không cần sửa code.

### LoRA giúp được gì — 3 điều, điều thứ 3 mới là chính

1. **Train nhẹ hơn** — chỉ backward qua phần adapter, không cần optimizer state cho 150M tham số.
2. **Không phá pretrained** — CLIP gốc giữ nguyên, delta nhỏ và có kiểm soát → đỡ overfit trên data ít của từng client.
3. **Giảm chi phí truyền dữ liệu — lý do chính trong FL** — mỗi round client chỉ gửi lên ~**1.9 MB** thay vì ~**573 MB**. Đây chính là "PEFT gap" cần đo trong luận văn.

---

## 3. FedAvg hoạt động thế nào

### 3.1 Một vòng (round) gồm 5 bước

1. **Server gửi xuống** — `global_state` (chỉ có `lora_A`, `lora_B` của 96 module).
2. **Client train** — nạp vào model, train vài epoch trên data riêng. Base CLIP vẫn đóng băng, chỉ A/B thay đổi.
3. **Client gửi lên** — trả về A/B đã train + số sample `n_k`.
4. **Server gộp** — tính trung bình có trọng số theo `n_k`.
5. **Lặp lại**. Cứ vài round thì đánh giá model global trên test set tập trung (200 người).

> **Ghi chú thực tế:** 1 GPU Colab không đủ VRAM để giữ 15 bản CLIP cùng lúc, nên code chỉ giữ **1 model**, nạp/rút trọng số LoRA lần lượt từng client. Về mặt thuật toán vẫn đúng FedAvg — mọi client đều bắt đầu từ cùng một `global_state`.

### 3.2 Công thức gộp

$$
\theta_{\text{global}} = \sum_k \frac{n_k}{N} \, \theta_k, \qquad N = \sum_k n_k
$$

Trọng số là **số sample** của mỗi client, không phải chia đều. Client nào có nhiều data hơn thì tiếng nói nặng hơn. Công thức áp dụng **riêng cho từng tensor** (mỗi `lora_A`/`lora_B` của mỗi layer là một tensor riêng).

### 3.3 Vì sao trung bình trọng số lại hoạt động được?

Trực giác đơn giản: nếu mỗi client chỉ train 1 epoch, thì client k làm đúng một bước gradient:

$$
\theta_k = \theta_g - \eta \nabla L_k(\theta_g)
$$

Thay vào công thức trung bình:

$$
\sum_k \frac{n_k}{N}\theta_k = \theta_g - \eta \underbrace{\sum_k \frac{n_k}{N}\nabla L_k(\theta_g)}_{\text{chính là gradient của loss toàn cục}}
$$

→ **Trung bình trọng số ≈ đi một bước gradient descent trên toàn bộ data gộp lại**, mà không cần gửi data đi đâu. Đó là lý do FedAvg hoạt động.

Sai lệch chỉ bắt đầu xuất hiện khi client train nhiều epoch (client drift) hoặc data giữa các client lệch nhau mạnh (non-IID).

### 3.4 Áp lên LoRA thì khác gì? (phần quan trọng nhất)

Nếu tune trực tiếp một block transformer cuối, thứ được gửi đi **chính là trọng số** $W$ → trung bình cộng là phép tuyến tính, không có vấn đề gì.

Nhưng với LoRA, thứ được gửi đi **không phải trọng số**, mà là **2 mảnh phân rã** $A$ và $B$ của nó ($\Delta W = BA$). Đây là nguồn gốc của 3 vấn đề:

| | Tune block cuối | LoRA |
|---|---|---|
| Gửi gì đi? | Chính là $W$ | $A$ và $B$ — 2 mảnh của $\Delta W = BA$ |
| Trung bình có tuyến tính? | **Có** | **Không** |
| Kết quả | Đúng nghĩa trung bình | $\overline{B}\,\overline{A} \ne \overline{BA}$ |

**(a) Trung bình sai chỗ.** Server gộp $A$ và $B$ **tách rời nhau**:

$$
\Delta W_{\text{global}} = \Big(\sum_k \tfrac{n_k}{N}B_k\Big)\Big(\sum_k \tfrac{n_k}{N}A_k\Big)
$$

Nhưng cái *đáng lẽ* phải có là:

$$
\overline{\Delta W} = \sum_k \tfrac{n_k}{N}\big(B_k A_k\big)
$$

Hai cái này **khác nhau**. Phần dư ra là các số hạng chéo $B_k A_{k'}$ — tức là **ghép nhầm mảnh của client này với mảnh của client kia**.

**(b) Mất thông tin do rank.** Trung bình đúng $\overline{\Delta W}$ là tổng của K ma trận rank-r → rank có thể lên tới $K \times r$. Nhưng $\overline{B}\,\overline{A}$ thì luôn bị kẹt ở rank $\le r$. Nghĩa là khi các client học theo hướng khác nhau (đúng kịch bản non-IID theo camera), FedAvg **bóp hết về một không gian hẹp** và mất thông tin.

**(c) Phân rã không duy nhất — vấn đề khó chịu nhất.** Với ma trận khả nghịch $T$ bất kỳ, $(BT)(T^{-1}A)$ cho ra **đúng cùng một** $\Delta W$. Nghĩa là 2 client hoàn toàn có thể học ra cùng kết quả nhưng $A$, $B$ lại khác nhau hoàn toàn. Cộng trung bình $A$ với $A$, $B$ với $B$ trong trường hợp đó là cộng những thứ **không cùng hệ quy chiếu**.

### 3.5 Ý nghĩa cho luận văn

Pipeline hiện tại **cố ý chấp nhận** vấn đề này để có một baseline tham chiếu ("LoRA thường + FedAvg thường thì được bao nhiêu"). Đây chính là khoảng trống để đề xuất cải tiến. Các hướng đã có trong literature:

| Hướng | Ý tưởng |
|---|---|
| **FFA-LoRA** | Đóng băng $A$, chỉ gộp $B$ → phép trung bình trở lại tuyến tính |
| **FLoRA** | Server dựng lại đúng $\overline{BA}$ rồi phân rã lại thành A, B mới |
| **LoRA-FAIR** | Thêm bước hiệu chỉnh trên server để bù phần sai lệch |
