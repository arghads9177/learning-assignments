# Installation & Setup Guide

## Environment Setup

All required packages have been installed using `uv` (ultra-fast Python package installer).

### Installed Packages

#### Core Data Science & ML
- **numpy** 2.5.1 — Numerical computing
- **pandas** 3.0.3 — Data manipulation and analysis
- **scikit-learn** 1.9.0 — Machine learning algorithms (metrics, preprocessing)

#### Visualization
- **matplotlib** 3.11.0 — Static plotting library
- **seaborn** 0.13.2 — Statistical data visualization

#### Computer Vision
- **pillow** 12.3.0 — Image processing
- **opencv-python** 5.0.0 — Advanced computer vision

#### Jupyter & Interactive Computing
- **jupyter** 1.1.1 — Jupyter notebook server & utilities
- **ipykernel** 7.3.0 — IPython kernel for Jupyter
- **ipywidgets** 8.1.8 — Interactive widgets for notebooks
- **notebook** 7.6.0 — Jupyter Notebook application

#### Utilities
- **tqdm** 4.68.4 — Progress bars for loops

#### Development Tools
- **jedi** 0.20.0 — Autocompletion and code analysis
- **debugpy** 1.8.21 — Python debugger

### Installation Command Used

```bash
uv add pandas seaborn tqdm jupyter ipykernel scikit-learn
```

This automatically resolved and installed 102 packages total (including all dependencies).

## Virtual Environment

The environment is located at:
```
/home/argha-ds/datascience/computer vision/learning-assignments/.venv/
```

### Activate Virtual Environment

```bash
cd "/home/argha-ds/datascience/computer vision/learning-assignments"
source .venv/bin/activate
```

### Verify Installation

```bash
python -c "import pandas, seaborn, tqdm, sklearn; print('✓ All packages installed')"
```

## Running Notebooks

### For Notebook 1 (Local - Dataset Preparation)

```bash
# Activate environment
source .venv/bin/activate

# Start Jupyter
jupyter notebook

# Navigate to: Chapter2/module3/01_dataset_preparation.ipynb
```

### For Notebook 2 (Google Colab - Model Training)

**Note**: TensorFlow is **pre-installed** on Google Colab, so no additional setup needed.

1. Upload notebook to Google Colab
2. Enable GPU: Runtime → Change runtime type → T4
3. Run all cells

## Package Details

### Why These Packages?

| Package | Purpose | Used In |
|---------|---------|---------|
| **numpy** | Array operations, random numbers | Both notebooks |
| **pandas** | Dataset statistics, DataFrame operations | Notebook 1 |
| **matplotlib** | Plotting loss/accuracy curves, confusion matrix | Notebook 1 & 2 |
| **seaborn** | Statistical visualizations, heatmaps | Notebook 1 & 2 |
| **pillow** | Load, verify, resize images | Notebook 1 & 2 |
| **tqdm** | Progress bars for file processing | Notebook 1 |
| **scikit-learn** | Classification metrics, confusion matrix | Notebook 2 |
| **opencv-python** | Computer vision operations | Existing project |
| **jupyter** | Run .ipynb notebooks locally | Notebook 1 |
| **ipykernel** | Jupyter kernel support | Notebook 1 |

### Version Info

Generated on: 2026-07-10

All packages pinned to compatible, stable versions.

## Updating Packages

To update all packages to latest versions:

```bash
uv sync --upgrade
```

To update a specific package:

```bash
uv add package-name --upgrade
```

## Files

- **pyproject.toml** — Project metadata and dependencies (managed by uv)
- **requirements.txt** — Human-readable list of dependencies
- **uv.lock** — Lock file with exact versions (for reproducibility)

## Troubleshooting

### "Module not found" error

Ensure virtual environment is activated:
```bash
source .venv/bin/activate
```

Verify the module is installed:
```bash
uv pip list | grep module-name
```

### Import errors in notebooks

1. Restart kernel: Kernel → Restart
2. Verify environment is activated
3. Check cell imports: `import pandas as pd` (not `from pandas import ...`)

### Jupyter notebook not found

Make sure you're in the correct directory:
```bash
cd "/home/argha-ds/datascience/computer vision/learning-assignments"
jupyter notebook Chapter2/module3/01_dataset_preparation.ipynb
```

## Next Steps

1. Review **QUICK_START.md** in Chapter2/module3/ for workflow
2. Run Notebook 1 locally: `01_dataset_preparation.ipynb`
3. Follow instructions in README_WORKFLOW.md for Notebook 2

---

**Command Summary**

```bash
# Activate environment
cd "/home/argha-ds/datascience/computer vision/learning-assignments"
source .venv/bin/activate

# Run notebook
jupyter notebook Chapter2/module3/01_dataset_preparation.ipynb

# Install additional package (if needed)
uv add package-name

# View installed packages
uv pip list
```
