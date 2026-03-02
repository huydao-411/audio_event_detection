# Project Structure Reorganization Guide

## Overview

The audio event detection project has been reorganized to follow a clean, professional structure that matches the desired architecture. This guide explains the folder structure and how to use the reorganized codebase.

## Final Project Structure

```
audio-event-detection/
│
├── data/                          # Data directory
│   ├── __init__.py               # Package marker
│   ├── raw/                      # Raw audio files
│   │   ├── UrbanSound8K/
│   │   ├── ESC-50/
│   │   └── FSD50K/
│   ├── processed/                # Preprocessed spectrograms
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── metadata/                 # Metadata and annotations
│
├── models/                       # Model definitions
│   ├── __init__.py
│   ├── ast_model.py             # Audio Spectrogram Transformer
│   ├── losses.py                # Custom loss functions
│   └── pretrained/              # Pre-trained weights
│
├── utils/                        # Utility modules
│   ├── __init__.py
│   ├── preprocess.py            # Audio preprocessing
│   ├── augmentation.py          # Data augmentation
│   ├── dataset.py               # PyTorch Dataset classes
│   └── metrics.py               # Evaluation metrics
│
├── scripts/                      # Executable scripts
│   ├── __init__.py
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation script
│   ├── inference.py             # Inference module
│   └── realtime_detection.py    # Real-time detection
│
├── configs/                      # Configuration files
│   └── config.yaml              # Hyperparameters
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── results/                      # Experiment results
│   ├── checkpoints/             # Model weights
│   ├── logs/                    # Training logs
│   └── figures/                 # Visualizations
│
├── docs/                         # Documentation
│   ├── architecture.md
│   ├── dataset_guide.md
│   ├── training_guide.md
│   └── REORGANIZATION_GUIDE.md  # This file
│
├── setup.sh                     # Setup script
├── run_test.sh                  # Test runner
├── test_installation.py         # Installation verification
├── requirements.txt             # Python dependencies
├── config.yaml                  # Main config (symbolic link)
└── README.md                    # Project README
```

## Changes Made

### 1. Created Missing Directories

- **notebooks/** - Now contains sample Jupyter notebooks for analysis
- **data/metadata/** - For storing metadata and annotations
- **models/pretrained/** - For storing pre-trained weights
- Organized **results/** with subdirectories for checkpoints, logs, and figures

### 2. Added Package Initialization Files

Created `__init__.py` files in:
- **models/** - Imports FocalLoss and other model utilities
- **utils/** - Imports preprocessing, augmentation, dataset, and metrics modules
- **scripts/** - Marks scripts as a Python package
- **data/** - Marks data as a Python package

### 3. Fixed Import Paths

Updated all scripts to use dynamic relative paths instead of hardcoded absolute paths:
- **scripts/train.py** - Uses `PROJECT_ROOT` to locate config
- **scripts/evaluate.py** - Uses relative paths for all resources  
- **scripts/inference.py** - Dynamic path resolution
- **scripts/realtime_detection.py** - Fixed path imports

**Before:**
```python
sys.path.append('/home/sandbox/audio_event_detection')
config_path = "/home/sandbox/audio_event_detection/configs/config.yaml"
```

**After:**
```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
config_path = PROJECT_ROOT / "configs" / "config.yaml"
```

### 4. Renamed Results Directory

- `results/plots/` → `results/figures/` to match the desired architecture

### 5. Created Sample Notebooks

Added three analysis notebooks:
1. **01_data_exploration.ipynb** - Data loading and analysis
2. **02_model_analysis.ipynb** - Model architecture and training
3. **03_results_visualization.ipynb** - Results and metrics visualization

## Updated Setup Script

The **setup.sh** script now includes:

```bash
# Install PyTorch explicitly
pip install torch torchvision torchaudio

# Install PortAudio for audio support
brew install portaudio

# Install requirements with proper handling
pip install -r requirements.txt --no-build-isolation

# Verify PyTorch installation
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed')"

# Run installation test
python test_installation.py
```

## How to Use the Reorganized Structure

### 1. Project Initialization

```bash
# Clone or navigate to project
cd audio-event-detection

# Run setup
sh setup.sh

# Verify installation
sh run_test.sh
```

### 2. Module Imports

**In training scripts:**
```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ast_model import AudioSpectrogramTransformer
from utils.metrics import MetricsCalculator
from models.losses import FocalLoss
```

**In notebooks:**
```python
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ast_model import AudioSpectrogramTransformer
```

### 3. Configuration Access

```python
import yaml
from pathlib import Path

# Get project root dynamically
PROJECT_ROOT = Path(__file__).parent.parent

# Load config
with open(PROJECT_ROOT / "configs" / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)
```

### 4. Data Access

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Access directories
train_dir = PROJECT_ROOT / "data" / "processed" / "train"
raw_dir = PROJECT_ROOT / "data" / "raw"
metadata_dir = PROJECT_ROOT / "data" / "metadata"
```

### 5. Results Storage

```python
# Save checkpoints
checkpoint_dir = PROJECT_ROOT / "results" / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), checkpoint_dir / "model.pth")

# Save figures
figures_dir = PROJECT_ROOT / "results" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / "confusion_matrix.png")
```

## Key Improvements

1. **Modularity** - Each component (models, utils, scripts) is self-contained
2. **Reproducibility** - Dynamic paths work across different machines
3. **Package Structure** - Proper Python package organization with `__init__.py`
4. **Scalability** - Easy to add new models, datasets, and scripts
5. **Documentation** - Clear structure makes the codebase easy to navigate
6. **Jupyter Integration** - Sample notebooks for analysis and development

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'models'`:
1. Ensure you're running from the project root or a subdirectory
2. Check that `sys.path.insert(0, str(PROJECT_ROOT))` is called before imports
3. Verify `__init__.py` files exist in each package directory

### File Not Found Errors

If config or data files aren't found:
1. Use `Path(__file__).parent.parent` to get project root
2. Verify file paths with `print(config_path)` to debug
3. Check that all directories exist with `path.exists()`

### Mixed Relative Imports

If imports work in one script but not another:
1. Use consistent `PROJECT_ROOT` calculation
2. Always use `sys.path.insert(0, str(PROJECT_ROOT))` at the top of scripts
3. Consider using full module paths: `from models.ast_model import ...`

## Next Steps

1. **Populate notebooks** - Add content to the sample notebooks
2. **Implement AST model** - If not already implemented in `models/ast_model.py`
3. **Create documentation** - Add architect.md, dataset_guide.md, training_guide.md
4. **Add tests** - Create unit tests for each module
5. **CI/CD integration** - Set up GitHub Actions for automated testing

## Contact

For questions about the project structure or reorganization, refer to the README.md in the project root.

---

**Last Updated:** 2024-02-27
**Reorganization Version:** 1.0.0
