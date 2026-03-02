# Hướng Dẫn Chi Tiết - Quy Trình Thực Hiện Dự Án

## 📋 Tổng Quan Quy Trình

Dự án được chia thành 6 giai đoạn chính:

```
1. Chuẩn Bị Môi Trường
   ↓
2. Thu Thập và Tiền Xử Lý Dữ Liệu
   ↓
3. Xây Dựng và Huấn Luyện Mô Hình
   ↓
4. Đánh Giá và Tối Ưu
   ↓
5. Triển Khai Ứng Dụng
   ↓
6. Viết Báo Cáo Đề Tài
```

---

## 🔧 Giai Đoạn 1: Chuẩn Bị Môi Trường (1-2 ngày)

### 1.1. Cài Đặt Phần Cứng
```
✅ Yêu cầu tối thiểu:
- GPU: NVIDIA GTX 1060 6GB trở lên
- RAM: 16GB
- Ổ cứng: 50GB trống
- CPU: Intel i5 hoặc tương đương

✅ Khuyến nghị:
- GPU: NVIDIA RTX 3060 12GB trở lên
- RAM: 32GB
- SSD: 100GB trống
```

### 1.2. Cài Đặt Phần Mềm
```bash
# 1. Cài đặt Python 3.8+
sudo apt update
sudo apt install python3.8 python3-pip

# 2. Cài đặt CUDA Toolkit (cho GPU)
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run

# 3. Clone repository
git clone <your-repo-url>
cd audio_event_detection

# 4. Chạy script setup
chmod +x setup.sh
./setup.sh

# 5. Activate virtual environment
source venv/bin/activate
```

### 1.3. Kiểm Tra Cài Đặt
```python
# test_installation.py
import torch
import librosa
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Librosa version: {librosa.__version__}")
print("\n✅ All packages installed successfully!")
```

---

## 📊 Giai Đoạn 2: Thu Thập và Tiền Xử Lý Dữ Liệu (3-5 ngày)

### 2.1. Tải Datasets

#### UrbanSound8K
```bash
# Sử dụng Kaggle API
pip install kaggle

# Cấu hình Kaggle credentials
mkdir ~/.kaggle
# Copy kaggle.json vào ~/.kaggle/

# Tải dataset
kaggle datasets download -d chrisfilo/urbansound8k
unzip urbansound8k.zip -d data/raw/UrbanSound8K/
```

#### ESC-50
```bash
# Clone từ GitHub
cd data/raw/
git clone https://github.com/karolpiczak/ESC-50.git
cd ../..
```

#### FSD50K (Optional)
```bash
# Tải từ Zenodo
# https://zenodo.org/record/4060432
# Download và giải nén vào data/raw/FSD50K/
```

### 2.2. Khám Phá Dữ Liệu (EDA)

Tạo notebook `notebooks/01_data_exploration.ipynb`:

```python
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Load metadata
us8k_meta = pd.read_csv('data/raw/UrbanSound8K/metadata/UrbanSound8K.csv')
esc50_meta = pd.read_csv('data/raw/ESC-50/meta/esc50.csv')

# 2. Phân tích phân phối classes
print("UrbanSound8K class distribution:")
print(us8k_meta['class'].value_counts())

print("\nESC-50 category distribution:")
print(esc50_meta['category'].value_counts())

# 3. Phân tích audio properties
def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    duration = len(y) / sr
    
    # Extract features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    return {
        'duration': duration,
        'sample_rate': sr,
        'mel_spec_shape': mel_spec.shape,
        'mfcc_shape': mfcc.shape
    }

# Phân tích mẫu
sample_file = 'data/raw/UrbanSound8K/audio/fold1/7061-6-0-0.wav'
props = analyze_audio(sample_file)
print(f"\nSample audio properties: {props}")

# 4. Visualize mel-spectrogram
y, sr = librosa.load(sample_file)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram')
plt.tight_layout()
plt.savefig('results/plots/sample_melspec.png')
```

### 2.3. Tiền Xử Lý Dữ Liệu

```bash
# Chạy script tiền xử lý
cd data
python preprocess.py

# Output:
# - data/processed/merged_dataset.csv
# - data/processed/spectrograms/*.npy
# - data/processed/spectrograms/processed_metadata.csv
```

**Kiểm tra kết quả:**
```python
import pandas as pd
import numpy as np

# Load processed metadata
metadata = pd.read_csv('data/processed/spectrograms/processed_metadata.csv')

print(f"Total processed samples: {len(metadata)}")
print(f"\nClass distribution:")
print(metadata['target_class'].value_counts())

# Load và kiểm tra một spectrogram
sample_spec = np.load(metadata.iloc[0]['feature_path'])
print(f"\nSpectrogram shape: {sample_spec.shape}")
print(f"Value range: [{sample_spec.min():.2f}, {sample_spec.max():.2f}]")
```

### 2.4. Data Augmentation

Tạo notebook `notebooks/02_augmentation_test.ipynb`:

```python
from data.augmentation import AudioAugmentor
import librosa
import matplotlib.pyplot as plt

# Initialize augmentor
augmentor = AudioAugmentor()

# Load sample audio
audio, sr = librosa.load('path/to/sample.wav', sr=22050)

# Apply augmentations
aug_audio = augmentor.augment_audio(audio, sr)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Original
mel_spec_orig = librosa.feature.melspectrogram(y=audio, sr=sr)
librosa.display.specshow(librosa.power_to_db(mel_spec_orig), 
                        sr=sr, x_axis='time', y_axis='mel', ax=axes[0])
axes[0].set_title('Original')

# Augmented
mel_spec_aug = librosa.feature.melspectrogram(y=aug_audio, sr=sr)
librosa.display.specshow(librosa.power_to_db(mel_spec_aug), 
                        sr=sr, x_axis='time', y_axis='mel', ax=axes[1])
axes[1].set_title('Augmented')

plt.tight_layout()
plt.savefig('results/plots/augmentation_comparison.png')
```

---

## 🧠 Giai Đoạn 3: Xây Dựng và Huấn Luyện Mô Hình (1-2 tuần)

### 3.1. Test Model Architecture

```bash
# Test model
python models/ast_model.py

# Expected output:
# Testing Audio Spectrogram Transformer...
# Number of parameters: 86,000,000
# Input shape: torch.Size([4, 1, 128, 400])
# Output shape: torch.Size([4, 7])
# Model test complete!
```

### 3.2. Huấn Luyện Baseline Model

```bash
# Huấn luyện với cấu hình mặc định
python scripts/train.py \
    --config configs/config.yaml \
    --batch-size 32 \
    --epochs 50

# Theo dõi training
tensorboard --logdir logs/
# Mở browser: http://localhost:6006
```

### 3.3. Hyperparameter Tuning

Tạo `notebooks/03_hyperparameter_tuning.ipynb`:

```python
# Grid search cho learning rate và batch size
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4]
batch_sizes = [16, 32, 64]

results = []

for lr in learning_rates:
    for bs in batch_sizes:
        print(f"\nTraining with LR={lr}, BS={bs}")
        
        # Update config
        config['training']['learning_rate'] = lr
        config['training']['batch_size'] = bs
        
        # Train model
        # ... training code ...
        
        # Record results
        results.append({
            'lr': lr,
            'batch_size': bs,
            'val_f1': val_f1,
            'val_loss': val_loss
        })

# Phân tích kết quả
results_df = pd.DataFrame(results)
print(results_df.sort_values('val_f1', ascending=False))
```

### 3.4. Advanced Training Techniques

#### Transfer Learning
```python
# Load pretrained AST model
pretrained_path = "models/ast_audioset_pretrained.pth"
model = AudioSpectrogramTransformer()

# Load pretrained weights
pretrained_dict = torch.load(pretrained_path)
model_dict = model.state_dict()

# Filter out classifier layer
pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                  if k in model_dict and 'head' not in k}

# Update model
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# Freeze early layers
for name, param in model.named_parameters():
    if 'blocks.0' in name or 'blocks.1' in name:
        param.requires_grad = False
```

#### Focal Loss for Class Imbalance
```python
# Implemented in the project under models/losses.py
# (legacy versions placed it in utils/losses.py)
from models.losses import FocalLoss

criterion = FocalLoss(alpha=0.25, gamma=2.0, num_classes=7)
```

---

## 📈 Giai Đoạn 4: Đánh Giá và Tối Ưu (3-5 ngày)

### 4.1. Comprehensive Evaluation

```bash
# Chạy evaluation
python scripts/evaluate.py \
    --model models/checkpoints/best_model.pth \
    --data data/processed/spectrograms/processed_metadata.csv \
    --output results/

# Output files:
# - results/evaluation_results.json
# - results/plots/confusion_matrix.png
# - results/plots/per_class_metrics.png
# - results/plots/roc_curves.png
# - results/plots/pr_curves.png
```

### 4.2. Error Analysis

Tạo `notebooks/04_error_analysis.ipynb`:

```python
import json
import numpy as np
import pandas as pd

# Load evaluation results
with open('results/evaluation_results.json', 'r') as f:
    results = json.load(f)

# Load predictions
y_true = np.load('results/y_true.npy')
y_pred = np.load('results/y_pred.npy')

# Find misclassified samples
misclassified = np.where(y_true != y_pred)[0]

print(f"Total misclassified: {len(misclassified)}")
print(f"Error rate: {len(misclassified)/len(y_true)*100:.2f}%")

# Analyze confusion patterns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Find most confused pairs
confusion_pairs = []
for i in range(7):
    for j in range(7):
        if i != j and cm_normalized[i, j] > 0.1:
            confusion_pairs.append((i, j, cm_normalized[i, j]))

confusion_pairs.sort(key=lambda x: x[2], reverse=True)
print("\nMost confused class pairs:")
for i, j, conf in confusion_pairs[:5]:
    print(f"  {class_names[i]} → {class_names[j]}: {conf*100:.1f}%")
```

### 4.3. Model Optimization

#### Quantization
```python
import torch

# Load trained model
model = AudioSpectrogramTransformer()
model.load_state_dict(torch.load('models/checkpoints/best_model.pth'))
model.eval()

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized model
torch.save(quantized_model.state_dict(), 
          'models/saved_models/quantized_model.pth')

# Compare sizes
original_size = os.path.getsize('models/checkpoints/best_model.pth')
quantized_size = os.path.getsize('models/saved_models/quantized_model.pth')

print(f"Original size: {original_size / 1e6:.2f} MB")
print(f"Quantized size: {quantized_size / 1e6:.2f} MB")
print(f"Compression ratio: {original_size / quantized_size:.2f}x")
```

#### ONNX Export
```python
import torch.onnx

# Export to ONNX
dummy_input = torch.randn(1, 1, 128, 400)
torch.onnx.export(
    model,
    dummy_input,
    'models/saved_models/model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

---

## 🚀 Giai Đoạn 5: Triển Khai Ứng Dụng (3-5 ngày)

### 5.1. Real-time Detection

```bash
# Test real-time detection
python scripts/realtime_detection.py \
    --model models/checkpoints/best_model.pth \
    --device cuda
```

### 5.2. Web API với Flask

Tạo `app/api.py`:

```python
from flask import Flask, request, jsonify
import torch
import librosa
import numpy as np
from scripts.inference import AudioEventDetector

app = Flask(__name__)

# Initialize detector
detector = AudioEventDetector(
    model_path='models/checkpoints/best_model.pth',
    device='cuda'
)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Save temporarily
    temp_path = '/tmp/audio.wav'
    file.save(temp_path)
    
    # Predict
    result = detector.predict(temp_path)
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

Chạy API:
```bash
python app/api.py

# Test API
curl -X POST -F "file=@test_audio.wav" http://localhost:5000/predict
```

### 5.3. Web Interface

Tạo `app/templates/index.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Audio Event Detection</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; }
        .upload-area { border: 2px dashed #ccc; padding: 50px; text-align: center; }
        .results { margin-top: 30px; }
        .prediction { padding: 10px; margin: 5px 0; background: #f0f0f0; }
    </style>
</head>
<body>
    <h1>🎵 Audio Event Detection System</h1>
    
    <div class="upload-area" id="uploadArea">
        <input type="file" id="audioFile" accept="audio/*">
        <button onclick="uploadAudio()">Detect Events</button>
    </div>
    
    <div class="results" id="results"></div>
    
    <script>
        function uploadAudio() {
            const fileInput = document.getElementById('audioFile');
            const file = fileInput.files[0];
            
            const formData = new FormData();
            formData.append('file', file);
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data);
            });
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<h2>Detection Results:</h2>';
            
            data.predictions.forEach(pred => {
                resultsDiv.innerHTML += `
                    <div class="prediction">
                        <strong>${pred.class}</strong>: 
                        ${(pred.confidence * 100).toFixed(2)}%
                    </div>
                `;
            });
        }
    </script>
</body>
</html>
```

---

## 📝 Giai Đoạn 6: Viết Báo Cáo Đề Tài (1-2 tuần)

### 6.1. Cấu Trúc Báo Cáo

```
Chương 1: Giới Thiệu
├── 1.1. Đặt vấn đề
├── 1.2. Mục tiêu nghiên cứu
├── 1.3. Phạm vi nghiên cứu
└── 1.4. Cấu trúc luận văn

Chương 2: Cơ Sở Lý Thuyết
├── 2.1. Sound Event Detection
├── 2.2. Deep Learning cho Audio
├── 2.3. Transformer Architecture
├── 2.4. Audio Spectrogram Transformer
└── 2.5. Các nghiên cứu liên quan

Chương 3: Phương Pháp Đề Xuất
├── 3.1. Tổng quan hệ thống
├── 3.2. Thu thập và tiền xử lý dữ liệu
├── 3.3. Kiến trúc mô hình
├── 3.4. Huấn luyện mô hình
└── 3.5. Triển khai ứng dụng

Chương 4: Thực Nghiệm và Kết Quả
├── 4.1. Môi trường thực nghiệm
├── 4.2. Datasets
├── 4.3. Cài đặt thực nghiệm
├── 4.4. Kết quả và phân tích
└── 4.5. So sánh với các phương pháp khác

Chương 5: Kết Luận
├── 5.1. Tổng kết
├── 5.2. Đóng góp
├── 5.3. Hạn chế
└── 5.4. Hướng phát triển
```

### 6.2. Tạo Biểu Đồ và Bảng

```python
# Generate all figures for thesis
import matplotlib.pyplot as plt
import seaborn as sns

# 1. System architecture diagram
# 2. Model architecture diagram
# 3. Training curves
# 4. Confusion matrices
# 5. ROC curves
# 6. Per-class performance
# 7. Latency analysis
# 8. Comparison with baselines

# Save all figures in high resolution
plt.savefig('thesis_figures/figure_name.png', dpi=300, bbox_inches='tight')
```

### 6.3. Checklist Hoàn Thành

```
✅ Hoàn thành code và test
✅ Chạy tất cả experiments
✅ Thu thập kết quả
✅ Tạo biểu đồ và bảng
✅ Viết draft các chương
✅ Review và chỉnh sửa
✅ Chuẩn bị slide thuyết trình
✅ Tổng duyệt
```

---

## 📊 Timeline Tổng Thể

```
Tuần 1-2:   Chuẩn bị môi trường + Thu thập dữ liệu
Tuần 3:     Tiền xử lý và EDA
Tuần 4-5:   Xây dựng baseline model
Tuần 6-7:   Huấn luyện và tuning
Tuần 8:     Đánh giá và optimization
Tuần 9:     Triển khai ứng dụng
Tuần 10-12: Viết báo cáo và chuẩn bị thuyết trình
```

---

## 🎯 Mục Tiêu Đạt Được

✅ Hệ thống phát hiện 7 loại âm thanh khẩn cấp
✅ Accuracy > 90% trên test set
✅ Real-time inference < 100ms
✅ Web application hoàn chỉnh
✅ Báo cáo đề tài chi tiết với đầy đủ thực nghiệm
✅ Code clean, có documentation và tests

---

## 📚 Tài Liệu Tham Khảo

Xem file README.md chính để có danh sách đầy đủ 30 papers được trích dẫn.
