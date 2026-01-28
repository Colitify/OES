# OES/LIBS Spectral Analysis Toolkit

A Python-based toolkit for analyzing optical emission spectroscopy (OES) and laser-induced breakdown spectroscopy (LIBS) data using machine learning.

## Features

- **Data Loading**: Support for LIBS contest format CSV files with automatic wavelength and target extraction
- **Preprocessing**: Baseline correction (ALS), smoothing (Savitzky-Golay), normalization (SNV, MinMax, L2)
- **Feature Engineering**: PCA, PLS, wavelength selection, peak detection
- **Traditional ML Models**: PLS, Ridge, Lasso, ElasticNet, Random Forest, SVR
- **Deep Learning Models**: 1D-CNN, LSTM, Transformer
- **Hyperparameter Optimization**: Optuna and GridSearchCV support
- **Evaluation**: Cross-validation, metrics comparison, visualization

## Project Structure

```
libs-spectral-analysis/
├── .claude/
│   └── skills/
│       └── oes-libs-spectral-analysis/    # Claude Code skill
├── data/
│   ├── train_dataset_RAW.csv              # Training data
│   └── test_dataset_RAW.csv               # Test data
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Data loading module
│   ├── preprocessing.py                   # Preprocessing module
│   ├── features.py                        # Feature engineering module
│   ├── models/
│   │   ├── __init__.py
│   │   ├── traditional.py                 # Traditional ML models
│   │   └── deep_learning.py               # Deep learning models
│   ├── optimization.py                    # Hyperparameter optimization
│   └── evaluation.py                      # Evaluation and visualization
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Data exploration
│   ├── 02_preprocessing.ipynb             # Preprocessing experiments
│   ├── 03_traditional_ml.ipynb            # Traditional ML experiments
│   ├── 04_deep_learning.ipynb             # Deep learning experiments
│   └── 05_final_submission.ipynb          # Final submission
├── outputs/
│   ├── models/                            # Saved models
│   ├── predictions/                       # Prediction results
│   └── figures/                           # Figures and plots
├── requirements.txt
├── README.md
└── main.py                                # Main pipeline script
```

## Installation

1. Create and activate a virtual environment:
```bash
conda activate pytorch_env
# or
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Command Line

```bash
# Basic training
python main.py --train data/train_dataset_RAW.csv --model ridge

# With hyperparameter optimization
python main.py --train data/train_dataset_RAW.csv --model ridge --optimize

# Compare all models
python main.py --train data/train_dataset_RAW.csv --model all

# Generate test predictions
python main.py --train data/train_dataset_RAW.csv --test data/test_dataset_RAW.csv --model ridge
```

### Using Python API

```python
from src.data_loader import SpectralDataset
from src.preprocessing import Preprocessor
from src.features import FeatureExtractor
from src.models.traditional import get_traditional_models, train_traditional_model
from src.evaluation import evaluate_model

# 1. Load data
data = SpectralDataset.from_csv("data/train_dataset_RAW.csv", n_wavelengths=40002)

# 2. Preprocess
preprocessor = Preprocessor(baseline="als", normalize="snv", denoise="savgol")
X = preprocessor.fit_transform(data.spectra)

# 3. Feature extraction
extractor = FeatureExtractor(method="pca", n_components=50)
X_feat = extractor.fit_transform(X, data.targets)

# 4. Train model
models = get_traditional_models()
model = train_traditional_model(models["ridge"], X_feat, data.targets)

# 5. Evaluate
metrics, predictions = evaluate_model(model, X_feat, data.targets, cv=5)
```

### Using Jupyter Notebooks

Run the notebooks in order:
1. `01_data_exploration.ipynb` - Explore and understand the data
2. `02_preprocessing.ipynb` - Experiment with preprocessing methods
3. `03_traditional_ml.ipynb` - Train and compare traditional ML models
4. `04_deep_learning.ipynb` - Train deep learning models
5. `05_final_submission.ipynb` - Generate final predictions

## Preprocessing Pipeline

```
Raw Spectrum → Baseline Correction → Smoothing → Normalization
                    (ALS)           (Savgol)      (SNV)
```

### Baseline Correction (ALS)
- Removes fluorescence background
- Parameters: `lam` (smoothness), `p` (asymmetry)

### Smoothing (Savitzky-Golay)
- Reduces noise while preserving peaks
- Parameters: `window_length`, `polyorder`

### Normalization
- **SNV**: Standard Normal Variate (mean=0, std=1)
- **MinMax**: Scale to [0, 1]
- **L2**: Unit vector normalization

## Models

### Traditional ML
| Model | Best For |
|-------|----------|
| PLS | High-dimensional data, correlated features |
| Ridge | Linear relationships, regularization needed |
| Lasso | Feature selection, sparse solutions |
| Random Forest | Non-linear relationships, feature importance |

### Deep Learning
| Model | Architecture |
|-------|--------------|
| Conv1D | 1D CNN with multiple conv layers |
| LSTM | Bidirectional LSTM for sequential patterns |
| Transformer | Self-attention for global dependencies |

## Hyperparameter Optimization

```python
from src.optimization import optimize_ridge, optimize_pls

# Optimize Ridge with Optuna
best_params, best_score = optimize_ridge(X, y, n_trials=100)

# Optimize PLS
best_params, best_score = optimize_pls(X, y, n_trials=50)
```

## Data Format

Expected CSV format (LIBS contest style):
- First N columns: Wavelengths (e.g., 200.0, 200.1, ..., 1000.0)
- Remaining columns: Target values (e.g., Cr, Mn, Mo, Ni)

```csv
200.0,200.1,...,1000.0,Cr,Mn,Mo,Ni
0.123,0.456,...,0.789,1.23,0.45,0.67,0.89
...
```

## Dependencies

- numpy>=1.21
- pandas>=1.3
- scipy>=1.7
- scikit-learn>=1.0
- torch>=1.10
- optuna>=3.0
- matplotlib>=3.5
- seaborn>=0.11
- jupyter>=1.0
- tqdm>=4.62

## License

MIT License
