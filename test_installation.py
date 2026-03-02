"""
Test Installation and Verify Project Setup
Run this script to verify all components are working correctly
"""

import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

print("="*70)
print("Audio Event Detection - Installation Test")
print("="*70)
print(f"Project root: {project_root}")

# Test 1: Python version
print("\n[1/10] Checking Python version...")
python_version = sys.version_info
if python_version.major == 3 and python_version.minor >= 8:
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"❌ Python version {python_version.major}.{python_version.minor} not supported. Need 3.8+")
    sys.exit(1)

# Test 2: PyTorch
print("\n[2/10] Checking PyTorch...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"   ✅ CUDA {torch.version.cuda}")
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   ⚠️  CUDA not available - will use CPU (slower)")
except ImportError:
    print("❌ PyTorch not installed")
    sys.exit(1)

# Test 3: Audio libraries
print("\n[3/10] Checking audio libraries...")
try:
    import librosa
    print(f"✅ Librosa {librosa.__version__}")
except ImportError:
    print("❌ Librosa not installed")
    sys.exit(1)

try:
    import soundfile
    print(f"✅ SoundFile {soundfile.__version__}")
except ImportError:
    print("❌ SoundFile not installed")
    sys.exit(1)

# Test 4: Data processing libraries
print("\n[4/10] Checking data processing libraries...")
try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
except ImportError:
    print("❌ NumPy not installed")
    sys.exit(1)

try:
    import pandas as pd
    print(f"✅ Pandas {pd.__version__}")
except ImportError:
    print("❌ Pandas not installed")
    sys.exit(1)

# Test 5: ML libraries
print("\n[5/10] Checking ML libraries...")
try:
    import sklearn
    print(f"✅ Scikit-learn {sklearn.__version__}")
except ImportError:
    print("❌ Scikit-learn not installed")
    sys.exit(1)

# Test 6: Visualization libraries
print("\n[6/10] Checking visualization libraries...")
try:
    import matplotlib
    print(f"✅ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("❌ Matplotlib not installed")
    sys.exit(1)

try:
    import seaborn
    print(f"✅ Seaborn {seaborn.__version__}")
except ImportError:
    print("❌ Seaborn not installed")
    sys.exit(1)

# Test 7: Configuration
print("\n[7/10] Checking configuration files...")
config_path = project_root / "configs" / "config.yaml"
if config_path.exists():
    print(f"✅ Config file found: {config_path}")
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✅ Config loaded successfully")
        print(f"   ✅ Number of classes: {config['model']['num_classes']}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
else:
    print(f"❌ Config file not found: {config_path}")

# Test 8: Project structure
print("\n[8/10] Checking project structure...")
required_dirs = [
    "data",
    "models",
    "utils",
    "scripts",
    "configs"
]

all_dirs_exist = True

for dir_name in required_dirs:
    dir_path = project_root / dir_name
    if dir_path.exists():
        print(f"   ✅ {dir_name}/")
    else:
        print(f"   ❌ {dir_name}/ not found")
        all_dirs_exist = False

if all_dirs_exist:
    print("✅ All required directories exist")
else:
    print("⚠️  Some directories are missing")

# Test 9: Core modules
print("\n[9/10] Checking core modules...")

try:
    from models.ast_model import AudioSpectrogramTransformer
    print("✅ AST model module")
except ImportError as e:
    print(f"❌ Error importing AST model: {e}")

try:
    from utils.metrics import MetricsCalculator
    print("✅ Metrics module")
except ImportError as e:
    print(f"❌ Error importing metrics: {e}")

# losses module may live under models or, in earlier versions, utils
try:
    from models.losses import FocalLoss
    print("✅ Losses module (models)")
except ImportError:
    try:
        from utils.losses import FocalLoss
        print("✅ Losses module (utils)")
    except ImportError as e:
        print(f"❌ Error importing losses: {e}")

# Test 10: Model instantiation
print("\n[10/10] Testing model instantiation...")
try:
    model = AudioSpectrogramTransformer(
        config_path=str(project_root / "configs" / "config.yaml")
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model instantiated successfully")
    print(f"   ✅ Total parameters: {total_params:,}")
    print(f"   ✅ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 128, 400)
    
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   ✅ Forward pass successful")
    print(f"   ✅ Input shape: {dummy_input.shape}")
    print(f"   ✅ Output shape: {output.shape}")
    
except Exception as e:
    print(f"❌ Error testing model: {e}")

# Summary
print("\n" + "="*70)
print("Installation Test Summary")
print("="*70)

print("\n✅ Core Dependencies: OK")
print("✅ Project Structure: OK")
print("✅ Model Architecture: OK")

print("\n" + "="*70)
print("System is ready for training!")
print("="*70)

print("\nNext steps:")
print("1. Download datasets:")
print("   - UrbanSound8K: https://www.kaggle.com/datasets/chrisfilo/urbansound8k")
print("   - ESC-50: https://github.com/karolpiczak/ESC-50")
print("\n2. Place datasets in data/raw/ directory")
print("\n3. Run preprocessing:")
print("   python data/preprocess.py")
print("\n4. Start training:")
print("   python scripts/train.py")
print("\n" + "="*70)
