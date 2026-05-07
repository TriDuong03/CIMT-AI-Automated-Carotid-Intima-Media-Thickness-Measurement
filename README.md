# CIMT-AI: Automated Carotid Intima-Media Thickness Measurement

> **Đo độ dày lớp nội trung mạc động mạch cảnh tự động bằng trí tuệ nhân tạo**

Dự án ứng dụng các mô hình học sâu (deep learning) để tự động phân đoạn và đo độ dày lớp IMT (Intima-Media Thickness) trên ảnh siêu âm động mạch cảnh — thay thế cho quy trình đo thủ công của bác sĩ.

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Kiến trúc mô hình](#kiến-trúc-mô-hình)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Quy trình xử lý dữ liệu](#quy-trình-xử-lý-dữ-liệu)
- [Huấn luyện](#huấn-luyện)
- [Đánh giá](#đánh-giá)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Kết quả](#kết-quả)

---

## Giới thiệu

**Carotid Intima-Media Thickness (CIMT/IMT)** là độ dày của lớp nội trung mạc (intima và media) của thành động mạch cảnh, đo được trên ảnh siêu âm B-mode. Chỉ số IMT là một marker lâm sàng quan trọng để đánh giá nguy cơ xơ vữa động mạch và các bệnh tim mạch.

Việc đo IMT thủ công tốn thời gian, phụ thuộc vào kinh nghiệm của bác sĩ và khó tái lập. Dự án này xây dựng pipeline tự động:

1. **Phân đoạn** vùng IMT trên ảnh siêu âm bằng mô hình deep learning.
2. **Trích xuất ranh giới** LI (Lumen-Intima) và MA (Media-Adventitia).
3. **Tính toán độ dày** IMT (theo mm) và so sánh với kết quả của chuyên gia.

---

## Dataset

Dự án sử dụng hai bộ dữ liệu công khai từ Mendeley Data:

| Bộ dữ liệu | Mô tả | Nguồn |
|---|---|---|
| **CUBS-Tech** | 500 ảnh siêu âm kỹ thuật (phantom/standardized), định dạng `.tiff`, kèm profile ranh giới thủ công | [Mendeley Data](https://data.mendeley.com/datasets/fpv535fss7/1) |
| **CUBS-2021 Clinical** | Ảnh siêu âm lâm sàng thực tế, kèm annotation của chuyên gia | [Mendeley Data](https://data.mendeley.com/datasets/m7ndn58sv6/1) |

**Định dạng annotation:**
- `{image_id}-LI.txt` — tọa độ (x, y) ranh giới Lumen-Intima
- `{image_id}-MA.txt` — tọa độ (x, y) ranh giới Media-Adventitia

**Phân chia tập dữ liệu (CUBS-Tech):**

| Tập | Số ảnh | Ghi chú |
|---|---|---|
| Train | 576 | 70% của 821 ảnh còn lại |
| Validation | 245 | 30% của 821 ảnh còn lại |
| Test | 100 | Cố định: tech_401 → tech_500 |

---

## Kiến trúc mô hình

Dự án triển khai và so sánh **4 kiến trúc phân đoạn**:

### 1. PlainConvUNet
U-Net thuần tích chập 6 tầng (encoder-decoder với skip connections).
- **Features:** 32 → 64 → 128 → 256 → 320 → 320
- **Normalization:** InstanceNorm2d
- **Activation:** LeakyReLU
- **Thư viện:** `dynamic-network-architectures` (nnUNet)

### 2. SwinUNETR
Kiến trúc lai Transformer–CNN, dùng Shifted Window Attention (Swin Transformer) làm encoder, CNN decoder.
- **Transformer depths:** (2, 2, 6, 2)
- **Attention heads:** (3, 6, 12, 24)
- **Thư viện:** MONAI

### 3. BasicUNetPlusPlus
UNet++ với dense skip connections giữa tất cả các tầng độ phân giải.
- **Features:** (32, 32, 64, 128, 256, 32)
- **Thư viện:** MONAI

### 4. UNetPlusPlus + ASPP (Dilated)
Mở rộng của BasicUNetPlusPlus, tích hợp module **ASPP (Atrous Spatial Pyramid Pooling)** tại bottleneck để tăng receptive field đa tỉ lệ.
- **Dilation rates:** (1, 2, 4, 8)
- Tích hợp qua hook — không xâm phạm kiến trúc gốc

**Input/Output cho tất cả mô hình:**
- Input: `(B, 1, H, W)` — ảnh grayscale
- Output: `(B, 2, H, W)` — logits phân đoạn nhị phân (background + IMT)

---

## Cấu trúc dự án

```
CIMT-AI-Automated-Carotid-Intima-Media-Thickness-Measurement/
├── README.md
├── configs/                                     # Cấu hình
│   └── dataset.yaml                             # Cấu hình đường dẫn dataset
│
├── data/                                        # Tiện ích xử lý dữ liệu
│   ├── __init__.py
│   ├── make_mask_from_profiles.py               # Chuyển profile → binary mask
│   ├── split_dataset.py                         # Phân chia train/val/test
│   └── visualize.py                             # Trực quan hóa ranh giới
│
├── models/                                      # Định nghĩa kiến trúc mô hình
│   ├── __init__.py
│   ├── plainconvunet.py
│   ├── swinunetr.py
│   ├── unetplusplus.py
│   └── unetplusplus_dilated.py
│
├── notebooks/                                   # Notebook huấn luyện (Kaggle)
│   ├── 01_plainconvunet.ipynb
│   ├── 02_swinunetr.ipynb
│   └── 03_unetplusplus.ipynb
│
├── inference/                                   # Đánh giá & đo IMT
│   └── evaluate_imt.py
│
└── assets/                                      # Ảnh minh họa
    └── tech_401_viz_gt.png
```

---

## Quy trình xử lý dữ liệu

### Bước 1 — Tạo mask từ profile thủ công

```bash
python cubs/make_mask_from_profiles.py --config cubs/config.yaml
```

Script đọc file tọa độ LI/MA và tạo binary mask PNG (0/255) tương ứng với vùng IMT giữa hai ranh giới.

### Bước 2 — Phân chia tập dữ liệu

```bash
python cubs/split_dataset.py
```

Tạo cấu trúc thư mục `data/train/`, `data/val/`, `data/test/` với ảnh và mask tương ứng.

### Bước 3 — Cấu hình dataset

Chỉnh sửa `cubs/config.yaml`:

```yaml
dataset_dir: "/path/to/CUBS_dataset"
image_subdir: "IMAGES"
profile_subdir: "SEGMENTATIONS/Manual-A1"
output_dir: null   # mặc định: <dataset_dir>/MASKS
ext: ".tiff"
```

---

## Huấn luyện

Toàn bộ quá trình huấn luyện được thực hiện trên **Kaggle Notebooks** với GPU NVIDIA Tesla T4.

Mở và chạy notebook tương ứng với mô hình muốn huấn luyện:

| Mô hình | Notebook |
|---|---|
| PlainConvUNet | `PlainConvUNet.ipynb` |
| SwinUNETR | `Swinunetr.ipynb` |
| UNet++ & UNet++ Dilated | `unet_plusplus_and_unetplusplusdilated.ipynb` |

**Pipeline trong mỗi notebook:**
1. Cài đặt thư viện (`nnunetv2`, `monai`, `albumentations`)
2. Tải và kiểm tra dữ liệu từ Kaggle Dataset
3. Định nghĩa augmentation (albumentations)
4. Vòng lặp huấn luyện với validation sau mỗi epoch
5. Lưu checkpoint tốt nhất theo Dice score
6. Trực quan hóa kết quả dự đoán

---

## Đánh giá

```bash
python inference/evaluate_imt.py
```

Cấu hình trong file (chỉnh trực tiếp các biến ở đầu script):

```python
DATASET_ROOT = "path/to/DATASET_CUBS_tech"
WEIGHT_PATH   = "weights/best_model_dilated.pt"
IMG_SIZE      = 512
```

**Các chỉ số đánh giá:**

| Nhóm | Chỉ số |
|---|---|
| Pixel-level | Dice, IoU, Hausdorff95 |
| Boundary-level | MAE/RMSE của ranh giới LI và MA (pixel) |
| IMT measurement | MAE/RMSE (mm, dùng calibration factor) |
| Statistical | Bland-Altman, Pearson correlation, bias, limits of agreement |

**Output:**
- `results_per_image.csv` — chỉ số từng ảnh
- `summary.txt` — thống kê tổng hợp, top/bottom performers

---

## Yêu cầu hệ thống

### Thư viện Python

```
torch
torchvision
monai
dynamic-network-architectures
albumentations
opencv-python
numpy
scipy
Pillow
matplotlib
PyYAML
```

### Phần cứng

- **Huấn luyện:** GPU NVIDIA (Tesla T4 trên Kaggle)
- **Inference:** GPU hoặc CPU (tự động nhận diện)

---

## Kết quả

Mô hình tốt nhất (**UNetPlusPlus + ASPP**) đạt được:

- Dice Score cao trên tập test (CUBS-Tech)
- Đo IMT tự động có tương quan cao với kết quả của chuyên gia (Pearson r)
- Phân tích Bland-Altman cho thấy bias nhỏ và limits of agreement chấp nhận được trong lâm sàng

> Chi tiết kết quả đầy đủ xem trong báo cáo đồ án.

---

## Tài liệu tham khảo

- **Dataset CUBS-Tech:** [https://data.mendeley.com/datasets/fpv535fss7/1](https://data.mendeley.com/datasets/fpv535fss7/1)
- **Dataset CUBS-2021 Clinical:** [https://data.mendeley.com/datasets/m7ndn58sv6/1](https://data.mendeley.com/datasets/m7ndn58sv6/1)
- **MONAI:** [https://monai.io](https://monai.io)
- **nnUNet / dynamic-network-architectures:** [https://github.com/MIC-DKFZ/dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)
- Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", *Nature Methods*, 2021
- Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images", *MICCAI*, 2021
- Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", *MICCAI*, 2018

---

*Đồ án tốt nghiệp — Đo độ dày lớp nội trung mạc động mạch cảnh tự động bằng AI*
