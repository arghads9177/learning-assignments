# Quick Start Guide

## 3-Step Setup

### Step 1: Prepare Dataset (Local - ~5 minutes)
```bash
cd "Chapter2/module3"
source .venv/bin/activate
pip install pillow numpy pandas matplotlib seaborn tqdm
jupyter notebook 01_dataset_preparation.ipynb
# Run all cells → Done!
```

**What happens:**
- ✓ Cleans corrupted & duplicate images
- ✓ Splits data into train/val/test (70/15/15)
- ✓ Creates `dataset/splits/` folder
- ✓ Generates statistics & visualizations

---

### Step 2: Train Model (Google Colab - ~30-60 minutes)

**Setup (once):**
1. Upload `02_model_training_colab.ipynb` to Google Drive
2. Open with Google Colaboratory
3. Enable GPU: Runtime → Change runtime type → T4

**Configure:**
```python
# Update this path to match your Google Drive location
DATASET_BASE = Path('/content/drive/MyDrive/[your_path]/flowers')
```

**Run:**
- Click Runtime → Run all
- Monitor training progress (watch loss decrease)
- Model saves automatically ✓

**What happens:**
- ✓ Builds 4-block CNN
- ✓ Trains with early stopping
- ✓ Evaluates on test set
- ✓ Generates metrics & visualizations
- ✓ Saves trained model

---

### Step 3: Test Model (Colab)

```python
# Test on custom image
test_image_interactive('/path/to/image.jpg')
```

Shows:
- Predicted class + confidence
- All class probabilities
- Visual result

---

## Files Created

### Notebook 1 (Local)
```
01_dataset_preparation.ipynb
├── Input: dataset/raw/
└── Output: 
    ├── dataset/splits/ (organized train/val/test)
    ├── class_distribution_before_cleaning.png
    └── train_val_test_split.png
```

### Notebook 2 (Colab)
```
02_model_training_colab.ipynb
├── Input: dataset/splits/
└── Output:
    ├── flower_classifier_model.h5
    ├── flower_classifier_savedmodel/
    ├── training_history.png
    ├── confusion_matrix.png
    ├── test_predictions.png
    ├── sample_images.png
    └── training_summary.txt
```

---

## Expected Results

**Data Cleaning:**
```
Original: 166 images
Corrupted removed: [X]
Duplicates removed: [Y]
Final clean: [166 - X - Y] images
```

**Train/Val/Test Split:**
```
Train: ~116 images (70%)
Val:   ~25 images (15%)
Test:  ~25 images (15%)
```

**Model Performance:**
```
Test Accuracy: ~90-95%
Per-class precision: ~90-95%
Training time: ~30-60 min on T4 GPU
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| "No GPU" | Runtime → Change runtime type → T4 |
| Path error | Update `DATASET_BASE` in notebook |
| Out of memory | Reduce batch size: 32 → 16 |
| Slow training | Use GPU (not CPU) |
| Poor accuracy | Verify data was cleaned in step 1 |

---

## Next Steps

After training:
1. Download `flower_classifier_model.h5` from Colab
2. Test on your own flower images
3. Deploy model to production (optional)
4. Try transfer learning for better accuracy

---

**Total Time: ~1.5 hours** (mostly GPU training)  
**GPU Required: Yes** (Colab T4 recommended)  
**Local Compute: Python 3.8+**
