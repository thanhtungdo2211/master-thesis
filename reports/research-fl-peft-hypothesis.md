# Research — Hypothesis của FL + PEFT cho TBPS, related work, và câu hỏi của cô

> Ngày: 2026-10-10. Nguồn: một lượt deep-research (5 hướng tìm kiếm, 20 nguồn đã đọc, 95 claim rút ra, 25 claim qua kiểm chứng đối kháng 3 phiếu: 19 đứng vững, 6 bị bác), đối chiếu với `paper/sections/*.tex` và `paper/refs.bib`.
>
> **Ký hiệu độ tin cậy**
> - ✅ **đã kiểm chứng**: claim đọc từ bản gốc, ≥2/3 phiếu xác nhận.
> - ⚠️ **chưa kiểm chứng**: mới thấy qua search snippet, phải đọc paper trước khi trích.
> - 💡 **suy luận**: suy luận của mình, chưa có nguồn.

---

## TL;DR

1. **Hypothesis về cost có precedent rõ** (FedCLIP, PromptFL, FFA-LoRA). Nhưng **PEFT không giải quyết non-IID**: FedPETuning cho thấy PEFT còn suy giảm mạnh hơn full fine-tuning khi heterogeneity tăng. Vì vậy hypothesis nên tách thành ba tầng:
   - privacy là động cơ;
   - cost là phần PEFT giải quyết;
   - camera/domain skew là câu hỏi nghiên cứu.
2. **Đã có FL cho TBPS**: FedSH (IEEE TMM 2024) tự nhận "first attempt", nhưng không dùng CLIP hay PEFT. Claim "first FL + PEFT + CLIP + TBPS" vẫn giữ được, với điều kiện trích FedSH và viết "to our knowledge".
3. **Về câu hỏi của cô**: federated ReID chuẩn (FedPav, FedReID) làm **ngược lại**: ID classifier ở lại client, server chỉ aggregate backbone. Hệ quả là lập luận bỏ ID loss trong báo cáo hiện tại chưa chặt. Muốn server học ID thì có ba hướng (chi tiết ở §3):
   - server có dữ liệu public để học ID;
   - server giữ embedding theo identity (FedAwS);
   - tách global LoRA và zone LoRA.

---

## 1. Làm rõ hypothesis: PEFT giải quyết vấn đề gì của FL?

### 1.1 Các paper PEFT-in-FL đặt vấn đề thế nào

| Paper | Vấn đề FL được nêu | Lời giải PEFT | Độ tin cậy |
|---|---|---|---|
| **FedCLIP** ([arXiv 2302.13485](https://arxiv.org/abs/2302.13485)) | "data distribution heterogeneity and high resource costs brought by large foundation models" | Attention-based adapter trên nhánh image; mọi thao tác federated chỉ làm trên adapter | ✅ |
| **PromptFL** ([arXiv 2208.11625](https://arxiv.org/abs/2208.11625)) | FL "requires adequate bandwidth for parameters communication and sufficient user data for local training" | Federated soft prompts (kiểu CoOp), CLIP frozen | ✅ |
| **FFA-LoRA** ([arXiv 2403.12313](https://arxiv.org/abs/2403.12313), ICLR 2024) | LoRA trong FL bất ổn dưới heterogeneity, multi-step local update và DP noise | Freeze A, chỉ train/gửi B → "further halve the communication cost" | ✅ |
| **FedPETuning** ([arXiv 2212.10025](https://arxiv.org/abs/2212.10025), ACL Findings 2023) | Benchmark PEFT trong FL (NLP) | So sánh adapter / LoRA / prefix / BitFit với full FT | ✅ (phần non-IID) |

Lưu ý khi trích:
- FedCLIP "+9% trên PACS" là khoảng 9 **điểm phần trăm** độ chính xác trung bình so với FedAvg/FedProx (cũng dùng CLIP).
- FedCLIP "283× faster" thực chất là **tỉ lệ số tham số trainable** (5.3E5 so với 1.5E8), không phải wall-clock (claim này chỉ đạt 2–1 phiếu).
- FedCLIP đăng ở IEEE Data Engineering Bulletin, venue khiêm tốn.
- FFA-LoRA giảm đúng ½ khi A và B cùng kích thước. Điều kiện này thỏa với Q/K/V/O của mình vì là ma trận vuông.

**Bằng chứng cost của chính pipeline** (lấy từ `paper/generated/macros.tex`):
- LoRA r=4: **491,520** tham số trainable (0.33% backbone), upload **1.88 MiB**/client/round.
- Full fine-tuning: **573 MiB**/client/round, tức gấp **×305**.

### 1.2 Phản biện: PEFT KHÔNG giải quyết non-IID ✅

- **FedPETuning, Sec. 4.5**: "FedPETuning is more susceptible to data heterogeneity than FedFT … degrades more dramatically" khi Dirichlet α giảm từ 1.0 xuống 0.1.
- **FedPEFT** ([arXiv 2210.01708](https://arxiv.org/abs/2210.01708), ViT-B / CIFAR-100): khi α = 0.1, Prompt giảm 2.10 điểm, Bias giảm 1.47, full FT giảm 1.41.
- **SLoRA** ([arXiv 2308.06522](https://arxiv.org/abs/2308.06522)): khoảng cách giữa full FT và PEFT tăng theo mức heterogeneity.

**Giới hạn của bằng chứng:**
- Phần lớn đến từ NLP với **label skew** (Dirichlet).
- Chưa có bằng chứng trực tiếp cho CLIP + **feature/domain skew** theo camera trong một bài toán retrieval.
- Đây chính là khoảng trống thesis lấp được.

### 1.3 Rủi ro kỹ thuật của pipeline hiện tại ✅

Vanilla FedAvg average riêng A và B, nên avg(B)·avg(A) ≠ avg(B·A) (FFA-LoRA, Eq. 3).
- FFA-LoRA coi sự "discordance" này là **một yếu tố then chốt** gây bất ổn (chứ không phải *nguyên nhân gốc duy nhất*).
- Báo cáo đã thừa nhận bias này ở `04_method.tex` §Aggregation bias, và code cũng ghi chú trong docstring của `experiments/clip_lora_fedavg/federated.py`.
- Các hướng sửa: FFA-LoRA, FLoRA, FedEx-LoRA, FedSA-LoRA.

### 1.4 Đề xuất viết lại hypothesis

> **Động cơ (privacy):** dữ liệu camera thuộc nhiều tổ chức khác nhau, là dữ liệu cá nhân, không gom về một chỗ được → FL. Giống FedSH và FedReID.
>
> **H1 (cost):** Freeze CLIP và chỉ federate LoRA thì FL-TBPS khả thi: upload giảm ×305 (1.88 so với 573 MiB/client/round), còn khoảng cách độ chính xác so với centralized có giới hạn và đo được.
>
> **H2 (heterogeneity):** Chia theo camera gây feature/domain skew mà PEFT không tự khử (bằng chứng từ FedPETuning). Thesis đo phần hiệu năng mất đi và thử khôi phục, bằng cách sửa aggregation bias của LoRA hoặc personalization theo zone.
>
> **(Tùy chọn) H3 (identity):** Thông tin ID mà SDM-only bỏ qua có thể đưa lại mà không phá aggregation, bằng ID head local hoặc cơ chế phía server (§3).

Cần sửa thêm trong báo cáo: intro và abstract đang viết PEFT "regularizes training on the limited data held by any one client" như một kết luận. Nên hạ thành giả thuyết cần kiểm chứng.

---

## 2. FL trong TBPS / person ReID: hypothesis của các nghiên cứu khác

| Công trình | Task | Hypothesis / động cơ | Backbone | CLIP / PEFT | Độ tin cậy |
|---|---|---|---|---|---|
| **FedSH** — Ma et al., *IEEE TMM* vol. 26, 2024, DOI `10.1109/TMM.2023.3330091` ([PDF](https://lab-ant.github.io/Papers/2024/FedSH_Towards_Privacy-Preserving_Text-Based_Person_Re-Identification.pdf)) | **TBPS** | Privacy giữa tổ chức (campus, government, company). 3 thách thức: (a) feature đồng nhất trong client vì ảnh cùng khu vực; (b) feature lệch giữa client làm mờ ranh giới identity; (c) communication bị nhân đôi vì two-tower | ResNet-50 + Bi-LSTM | Không. Ghi "CLIP and ALIGN" là future work | ✅ |
| **FedPav** — Zhuang et al., ACM MM 2020 ([arXiv 2008.11560](https://arxiv.org/abs/2008.11560)) | Image ReID | Privacy; statistical heterogeneity | ResNet-50 + ID loss | Không | ✅ |
| **FedReID** — Zhuang et al., TOMM 2022 ([arXiv 2205.12144](https://arxiv.org/abs/2205.12144)) | Image ReID | Privacy ("contains sensitive personal information"); kỹ thuật xoay quanh KD, client clustering, cosine-distance weights | ResNet-50 | Không | ✅ |
| **FedReID** — Wu & Gong, AAAI 2021 ([arXiv 2006.04150](https://arxiv.org/abs/2006.04150)) | Image ReID | ID space và context không trùng giữa các địa điểm | — | Không | ✅ |
| **FLoRA** — Nguyen, Munoz, Jannesari ([arXiv 2404.15182](https://arxiv.org/abs/2404.15182), preprint) | Image **classification** | FL + LoRA + CLIP; chỉ LoRA r=2 ở text encoder (ViT-B/32), image encoder frozen | CLIP | Có (LoRA) | ✅ (2–1) |
| **FedBPrompt** ([arXiv 2603.12912](https://arxiv.org/abs/2603.12912), 2026) | Image ReID, federated domain generalization | Visual prompt nhận biết phân bố cơ thể | ? | Có (prompt) | ⚠️ |

### Hệ quả cho claim novelty

- **Bỏ cách nói "first FL + TBPS".** FedSH đã có trước.
- **Nên viết:** "To our knowledge, this is the first work to combine FL with PEFT on a CLIP backbone for TBPS", và trích FedSH làm prior work trực tiếp.
- **FedSH chưa có trong `refs.bib`.** Thêm qua `fetch_bibtex.sh` với DOI ở trên.
- **Thách thức (a) và (b) của FedSH trùng với camera-level non-IID của mình**, nên dùng được để định vị thesis.
- **FLoRA trùng tên.** Key `wang2024florafederatedfinetuninglarge` trong `refs.bib` là FLoRA *cho LLM* (NeurIPS 2024). Còn arXiv 2404.15182 là FLoRA *cho CLIP*. Khi trích phải phân biệt.
- **FedBPrompt cần đọc kỹ** trước khi chốt claim, vì đây là PEFT + FL + ReID gần nhất.
- **Novelty mới dựa trên một lượt search không đầy đủ.** Cần thêm một lượt survey có hệ thống (arXiv, CVPR/ICCV/ECCV 2025–26, ACM MM, TMM, TIFS). Việc này đã có trong `08_next.tex`.

---

## 3. Câu hỏi của cô

> *Có mô hình nào mà weight ở client chỉ học thông tin ảnh–text như hiện tại, còn weight trên server học thông tin ID không? Áp dụng cho từng zone khác nhau khi search được không?*

### 3.1 Điểm cần làm rõ trước 💡

**SDM không thuần "ảnh–text".**
- Nhãn y_ij của SDM lấy từ identity: hai mẫu là positive nếu cùng người (`04_method.tex`: SDM "depends on identity labels solely through the relation y_i = y_j within a mini-batch").
- Vì vậy client **đã học thông tin ID ở mức "cùng người / khác người"** trong từng batch.
- Thứ bị bỏ là **ID classifier**, tức một ánh xạ cố định từ ảnh sang chỉ số identity. Không phải mọi thông tin ID đều bị bỏ.

### 3.2 Federated ReID chuẩn làm NGƯỢC lại với ý cô ✅

**FedPav:**
> "Models in clients share an identical backbone, varying the identity classifiers, so the clients only send model parameters of the backbone … Each client preserves the classifier layer."
>
> "Since ReID evaluation uses an image as a query …, we can omit the identity classifier in evaluation."

**Wu & Gong (AAAI 2021):** aggregate toàn bộ tham số làm giảm hiệu năng, vì "each local dataset is usually captured in different locations where the person ID space and context are non-overlapping".

**Hệ quả cho báo cáo:**
- Lập luận trong `04_method.tex` ("The identity loss cannot be used, because its shared classifier …") và trong `01_intro.tex` chỉ đúng nếu classifier là **shared**.
- Nên sửa thành: *"cannot be naively averaged; a local, non-aggregated head (FedPav) is the standard alternative"*.
- Nên thêm ablation **SDM + local ID head**. Upload vẫn chỉ là LoRA, nên chi phí communication không đổi.

### 3.3 Các thiết kế ứng viên

| # | Thiết kế | Cơ chế | Server học ID? | Khả thi | Rủi ro / điều kiện | Nguồn |
|---|---|---|---|---|---|---|
| A | **Local ID head** (FedPav-style) | Client: LoRA (aggregate) + ID head riêng (không gửi lên server). Bỏ head khi retrieval | Không (ngược ý cô) | **Cao** | Client nhỏ ít identity, nên head có thể overfit | ✅ FedPav |
| B | **Server học ID trên dữ liệu public** | Client train LoRA bằng SDM trên dữ liệu riêng. Server có một tập public có nhãn ID (ví dụ CUHK-PEDES ở server, RSTPReid ở client) và học ID head / LoRA riêng, hoặc distill (client là teacher, server là student) | **Có** | Trung bình | Server phải có dữ liệu; hiệu quả phụ thuộc domain gap (FedReID ghi nhận KD tốt hơn khi phân phối client giống public data) | ✅ FedReID TOMM 2022 (KD toàn model); 💡 phần tách client/server là suy luận |
| C | **Server giữ embedding theo identity** (FedAwS-style) | Mỗi client chỉ thấy positive của identity mình. Server giữ ma trận class-embedding và chạy thêm bước "spreadout" đẩy các identity xa nhau, nên negative được xử lý ở server | **Có**: gần nhất với ý cô | Thấp – trung bình | Cần ID label **thống nhất toàn cục** giữa các camera, tức là phải liên kết người qua các tổ chức. Server nắm vector đặc trưng của từng người → rủi ro privacy 💡 | ⚠️ FedAwS, ICML 2020 ([arXiv 2004.10342](https://arxiv.org/abs/2004.10342)) |
| D | **Global LoRA + zone LoRA** | Global LoRA được aggregate, học alignment chung. Zone/local LoRA ở lại, học đặc thù zone | Không trực tiếp; personalization theo zone | Trung bình – cao | Embedding giữa các zone có thể không còn so sánh được (§3.4) | ⚠️ FedDPA, pFedMMA, HiLoRA (bên dưới) |

**Chi tiết các mẫu cho thiết kế D** (⚠️ tất cả chưa kiểm chứng, phải đọc trước khi trích):

- **FedDPA**, NeurIPS 2024 ([arXiv 2403.19211](https://arxiv.org/abs/2403.19211)).
  - Mỗi client có một global adapter (chỉ phần này lên server) và một local adapter.
  - Khi inference, hai adapter được trộn theo trọng số động cho từng instance.
  - Mới thử trên NLP.
- **pFedMMA** ([arXiv 2507.05394](https://arxiv.org/abs/2507.05394)).
  - Adapter cho CLIP: phần down/up projection riêng từng modality ở lại local; **chỉ phần cross-modal projection dùng chung được aggregate**.
  - Đây là phiên bản "đối xứng" của ý cô: alignment ảnh–text lên server, đặc thù modality/domain ở client.
- **HiLoRA**, 2026 ([arXiv 2603.02785](https://arxiv.org/abs/2603.02785)).
  - LoRA ba tầng: root (global), cluster, leaf (client), cluster được suy ra tự động.
  - Khớp với phân cấp global / zone / camera.
- **pFedMoAP**, ICLR 2025 ([arXiv 2410.10114](https://arxiv.org/abs/2410.10114)).
  - Federated prompt cho CLIP: client tải nhiều prompt expert từ pool trên server.
  - Một gating network local kết hợp các expert.
- **SAGE-reID** ([arXiv 2508.06831](https://arxiv.org/html/2508.06831v2)).
  - Person ReID, centralized multi-source DA, không phải FL.
  - LoRA expert theo từng domain, gating dự đoán cách merge ngay khi inference.
  - Cho thấy LoRA expert theo domain chạy được cho retrieval mà overhead thấp.

### 3.4 Áp dụng theo zone khi search

**Bằng chứng: federated-by-camera là kịch bản kém nhất** ✅ (FedPav, Rank-1):

| Dataset | Centralized | Federated-by-identity | **Federated-by-camera** |
|---|---|---|---|
| Market-1501 | 88.93 | 85.69 | **61.13** |
| CUHK03-NP | 49.29 | 51.71 | **11.21** |

**Lý giải:**
- Tác giả FedPav: "Since each client only learns from one-camera-view data … the model is incapable to generalize to the multi-camera evaluation". Họ khuyến nghị kiến trúc client–edge–cloud.
- Kết quả này đến từ ResNet-50 + ID loss trên image ReID, không phải CLIP + LoRA + SDM trên TBPS. Vì vậy nó chỉ là luận cứ tương tự, nhưng khớp trực tiếp với split 15 camera của mình.
- Week-12 cũng thấy 2 nhóm camera (A: 1, 3, 5, 7, 14, 15; B: 2, 6, 8–13) gần như không chung identity. Đó là ứng viên tự nhiên cho "zone".

**Đề xuất thiết kế theo zone** 💡

1. **Phần global là bắt buộc.** Mô hình hoàn toàn riêng theo zone sẽ thiếu kiến thức cross-camera.
2. **Global LoRA + zone LoRA** (FedDPA/HiLoRA-style). Zone có thể là nhóm camera (A/B) hoặc cụm do clustered FL tìm ra.
3. **Search trong zone:** encode cả query và gallery của zone bằng global + adapter của zone đó.
4. **Search xuyên zone:** chỉ dùng global LoRA, hoặc mô hình merge.
5. **Ràng buộc:** nếu gallery mỗi zone được encode bằng adapter khác nhau thì embedding giữa các zone **không so sánh trực tiếp được**. Cross-zone search vì vậy cần một embedding space chung, hoặc phải encode lại gallery.

### 3.5 Thứ tự thử nghiệm gợi ý 💡

1. **A — SDM + local ID head.** Rẻ, trả lời ngay câu "bỏ ID loss có mất gì không" và quyết định cách sửa `04_method.tex`.
2. **D — global + zone LoRA với 2 zone A/B**, đánh giá R@1/mAP theo từng zone. Nối được với việc "report accuracy per cluster" đã có trong `08_next.tex`.
3. **B — cross-dataset:** server có CUHK-PEDES có nhãn, client là RSTPReid theo camera. Khớp với kế hoạch cross-dataset federation trong `08_next.tex`.
4. **C** chỉ thử nếu chấp nhận giả định có ID label toàn cục; nếu thử, phải thảo luận rủi ro privacy.

---

## 4. Việc cần sửa trong báo cáo (`paper/`)

| File | Sửa gì |
|---|---|
| `refs.bib` | Thêm FedSH (DOI `10.1109/TMM.2023.3330091`) qua `fetch_bibtex.sh`. Cân nhắc thêm FedPav, FedReID (Wu & Gong), FedPETuning, FedCLIP, PromptFL, FFA-LoRA |
| `01_intro.tex` | Viết lại hypothesis theo §1.4. Sửa câu "identity classifier … not all compatible with weight averaging" (§3.2). Hạ claim "regularizes" thành giả thuyết |
| `02_related.tex` | Thêm một subsection "Federated person ReID / TBPS" (FedSH, FedPav, FedReID). Sửa đoạn novelty: trích FedSH, viết "to our knowledge" |
| `04_method.tex` | Sửa lập luận bỏ ID loss: không average được ≠ không dùng được; nêu local head là phương án chuẩn |
| `08_next.tex` | Thêm ablation SDM + local ID head và thiết kế global + zone LoRA |

---

## 5. Không được trích (đã bị bác khi kiểm chứng)

| Claim | Nguồn | Phiếu |
|---|---|---|
| FedPETuning giảm ~97% communication / FedAP 60× / giữ >95% hiệu năng FedFT | 2212.10025 | 0–3 |
| PEFT giảm 40.7% rò rỉ DLG | 2212.10025 | 1–2 |
| FLoRA giảm communication 4766× (0.094 so với 577.078 MB) | 2404.15182 | 0–3 |
| FLoRA nhanh hơn 34.72×, tiết kiệm 2.47× memory | 2404.15182 | 0–3 |
| FedReID (Wu & Gong) để server không giữ ID classifier, "giống thesis" | 2006.04150 | 0–3 |
| FedReID benchmark lấy 9 dataset làm 9 client | 2205.12144 | 0–3 |

Có claim bị bác nghĩa là claim đó không khớp với bản gốc. Nếu thật sự cần con số liên quan, phải đọc lại paper.

## 6. Câu hỏi mở

1. Đã có paper 2025–2026 nào kết hợp FL + CLIP + PEFT cho TBPS chưa (có thể là hậu duệ của FedSH, hoặc FedBPrompt mở rộng)?
2. SDM + local ID head có cải thiện R@1/mAP so với SDM-only trên RSTPReid 15 camera không?
3. Server có học được ID mà **không cần** dữ liệu public không? Ba hướng cần xem: FedProto (prototype), FedAwS (spreadout), split learning với ID head ở server. Rủi ro privacy của việc gửi prototype theo identity là gì?
4. Chọn cơ chế personalization theo zone nào: FedRep/FedPer-style, clustered FL, hay mixture-of-LoRA? Cơ chế nào giữ được cross-zone retrieval trong một embedding space chung?

> Các paper sau có trong câu hỏi gốc nhưng **chưa có claim nào được kiểm chứng**: pFedPrompt, FedAPT, FedDAT, FedIT, HetLoRA, FedSA-LoRA, FedUReID, DACS, FedSTIL, FedProto, FedRep/FedPer/FedBABU, split learning, clustered FL. Hiện chỉ dùng làm danh sách đọc.
