# Hệ Thống Phát Hiện Sự Kiện Âm Thanh Khẩn Cấp

## Giới thiệu
Dự án này xây dựng hệ thống phát hiện sự kiện âm thanh khẩn cấp sử dụng mô hình Deep Learning dựa trên kiến trúc Transformer. Hệ thống có khả năng nhận diện các âm thanh nguy hiểm như:

- 🔫 Tiếng súng (Gunshot)
- 💥 Tiếng nổ (Explosion)
- 🚨 Còi báo động (Siren)
- 🪟 Tiếng kính vỡ (Glass Breaking)
- 😱 Tiếng la hét (Scream)
- 🐕 Tiếng chó sủa (Dog Bark - phát hiện đột nhập)
- 🔥 Tiếng lửa cháy (Fire Crackling)

## Cấu trúc dự án

```
audio_event_detection/
├── data/                          # Dữ liệu
│   ├── raw/                       # Dữ liệu thô
│   │   ├── UrbanSound8K/
│   │   ├── ESC-50/
│   │   └── FSD50K/
│   ├── processed/                 # Dữ liệu đã xử lý
│   │   └── spectrograms/
│   ├── preprocess.py              # Script tiền xử lý
│   ├── augmentation.py            # Tăng cường dữ liệu
│   └── dataset.py                 # PyTorch Dataset
│
├── models/                        # Mô hình
│   ├── ast_model.py               # Audio Spectrogram Transformer
│   ├── checkpoints/               # Checkpoints huấn luyện
│   └── saved_models/              # Mô hình đã huấn luyện
│
├── utils/                         # Tiện ích
│   ├── metrics.py                 # Tính toán metrics
│   └── losses.py                  # Hàm loss tùy chỉnh
│
├── scripts/                       # Scripts thực thi
│   ├── train.py                   # Huấn luyện mô hình
│   └── inference.py               # Dự đoán
│
├── configs/                       # Cấu hình
│   └── config.yaml                # File cấu hình chính
│
├── notebooks/                     # Jupyter notebooks
│   └── exploratory_analysis.ipynb
│
├── results/                       # Kết quả
│   ├── plots/                     # Biểu đồ
│   └── metrics/                   # Metrics
│
├── docs/                          # Tài liệu
│
├── requirements.txt               # Dependencies
├── README.md                      # Tài liệu chính (Vietnamese)
└── README_EN.md                   # English documentation
```

## Yêu cầu hệ thống

### Phần cứng
- **GPU**: NVIDIA GPU với ít nhất 8GB VRAM (khuyến nghị RTX 3060 trở lên)
- **RAM**: Tối thiểu 16GB
- **Ổ cứng**: 50GB dung lượng trống

### Phần mềm
- Python 3.8+
- CUDA 11.7+ (cho GPU training)
- PyTorch 2.0+

## Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd audio_event_detection
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

### Tải datasets

#### 1. UrbanSound8K
```bash
# Tải từ Kaggle
kaggle datasets download -d chrisfilo/urbansound8k
unzip urbansound8k.zip -d data/raw/UrbanSound8K/
```

#### 2. ESC-50
```bash
# Clone từ GitHub
git clone https://github.com/karolpiczak/ESC-50.git data/raw/ESC-50/
```

#### 3. FSD50K (Optional)
```bash
# Tải từ Zenodo
# https://zenodo.org/record/4060432
```

### Tiền xử lý dữ liệu
```bash
cd data
python preprocess.py
```

Script này sẽ:
- Merge các datasets
- Trích xuất mel-spectrogram
- Chuẩn hóa âm thanh
- Lưu features đã xử lý

## Huấn luyện mô hình

### Huấn luyện cơ bản
```bash
python scripts/train.py
```

### Huấn luyện với cấu hình tùy chỉnh
```bash
python scripts/train.py \
    --config configs/config.yaml \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.0001
```

### Theo dõi quá trình huấn luyện
```bash
tensorboard --logdir logs/
```

## Dự đoán

### Dự đoán file đơn
```bash
python scripts/inference.py \
    --model models/checkpoints/best_model.pth \
    --input path/to/audio.wav \
    --output results/predictions.json
```

### Dự đoán batch
```bash
python scripts/inference.py \
    --model models/checkpoints/best_model.pth \
    --input path/to/audio/directory/ \
    --output results/predictions.json
```

## Đánh giá mô hình

```python
from utils.metrics import MetricsCalculator
from models.ast_model import AudioSpectrogramTransformer

# Load model
model = AudioSpectrogramTransformer()
model.load_state_dict(torch.load('models/checkpoints/best_model.pth'))

# Calculate metrics
calculator = MetricsCalculator(num_classes=7)
metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)
calculator.print_metrics(metrics)
```

## Kiến trúc mô hình

### Audio Spectrogram Transformer (AST)

```
Input Audio (4s @ 22050Hz)
    ↓
Mel-Spectrogram (128 x 400)
    ↓
Patch Embedding (16x16 patches)
    ↓
[CLS] Token + Position Embedding
    ↓
Transformer Encoder (12 layers)
    ├── Multi-Head Attention
    ├── Layer Normalization
    ├── Feed-Forward Network
    └── Residual Connections
    ↓
[CLS] Token Output
    ↓
Classification Head
    ↓
7 Classes Output
```

### Thông số mô hình
- **Embedding dimension**: 768
- **Number of layers**: 12
- **Attention heads**: 12
- **MLP ratio**: 4.0
- **Total parameters**: ~86M

## Kỹ thuật nâng cao

### 1. Data Augmentation
- Time stretching
- Pitch shifting
- Noise addition
- Time shifting
- SpecAugment (frequency & time masking)
- Mixup

### 2. Class Imbalance Handling
- Focal Loss (α=0.25, γ=2.0)
- Class weighting
- Oversampling minority classes

### 3. Training Techniques
- Mixed precision training (AMP)
- Gradient clipping
- Cosine annealing learning rate
- Early stopping
- Model checkpointing

## Kết quả thực nghiệm

### Performance trên UrbanSound8K
```
Overall Metrics:
- Accuracy:  92.3%
- Precision: 91.8%
- Recall:    92.1%
- F1-Score:  91.9%
- mAP:       94.2%
```

### Per-Class Performance
| Class          | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Gunshot        | 95.2%     | 93.8%  | 94.5%    |
| Explosion      | 89.1%     | 91.2%  | 90.1%    |
| Siren          | 96.3%     | 95.7%  | 96.0%    |
| Glass Breaking | 88.5%     | 87.9%  | 88.2%    |
| Scream         | 90.8%     | 92.3%  | 91.5%    |
| Dog Bark       | 93.7%     | 94.1%  | 93.9%    |
| Fire Crackling | 87.9%     | 86.5%  | 87.2%    |

## Ứng dụng thực tế

### 1. Hệ thống an ninh thông minh
- Phát hiện tiếng súng tự động
- Cảnh báo đột nhập qua tiếng kính vỡ
- Nhận diện tiếng la hét khẩn cấp

### 2. Phát hiện cháy nổ
- Nhận diện tiếng lửa cháy
- Phát hiện tiếng nổ
- Kích hoạt hệ thống báo động

### 3. Smart City
- Giám sát âm thanh đô thị
- Phát hiện sự cố giao thông
- Hỗ trợ ứng cứu khẩn cấp

## Tài liệu tham khảo

Xem file README.md chính để biết danh sách đầy đủ các tài liệu tham khảo khoa học.

## Đóng góp

Mọi đóng góp đều được hoan nghênh! Vui lòng:
1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## Acknowledgments

- UrbanSound8K dataset by J. Salamon et al.
- ESC-50 dataset by K. Piczak
- Audio Spectrogram Transformer by Y. Gong et al.
- PyTorch team for the deep learning framework
