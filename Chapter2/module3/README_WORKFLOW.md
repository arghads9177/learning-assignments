# Flower Classification - Complete Workflow

This directory contains a complete pipeline for flower image classification with 3 classes: **Hibiscus, Rose, and Sunflower**.

## Dataset Overview

```
dataset/raw/
├── hibiscus/  (57 images)
├── rose/      (47 images)
└── sunflower/ (62 images)
Total: 166 images
```

## Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: LOCAL EXECUTION - Dataset Preparation               │
├─────────────────────────────────────────────────────────────┤
│ Notebook: 01_dataset_preparation.ipynb                      │
│                                                              │
│ Tasks:                                                       │
│  1. Explore dataset & generate statistics                   │
│  2. Detect & remove corrupted files                        │
│  3. Detect & remove duplicate images                       │
│  4. Split data into train/val/test (70/15/15)             │
│  5. Save organized dataset in splits/ folder               │
│                                                              │
│ Output: dataset/splits/ directory ready for training       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: CLOUD EXECUTION - Model Training & Evaluation       │
├─────────────────────────────────────────────────────────────┤
│ Notebook: 02_model_training_colab.ipynb (Google Colab)    │
│ GPU: T4 (recommended)                                       │
│                                                              │
│ Tasks:                                                       │
│  1. Load prepared dataset                                  │
│  2. Build CNN architecture                                │
│  3. Train with:                                           │
│     • Early stopping (patience=15)                        │
│     • Model checkpoints (best weights)                    │
│     • Learning rate reduction                            │
│     • Data augmentation                                  │
│  4. Evaluate on test set                                 │
│  5. Plot training curves                                 │
│  6. Generate confusion matrix                            │
│  7. Test predictions on sample images                    │
│                                                              │
│ Outputs: Models, metrics, visualizations                  │
└─────────────────────────────────────────────────────────────┘
```

---

## STEP 1: Local Dataset Preparation

### Prerequisites
```bash
python -m pip install pillow numpy pandas matplotlib seaborn tqdm
```

### Run Locally
```bash
# Navigate to project directory
cd "/path/to/datascience/computer vision/learning-assignments/Chapter2/module3"

# Activate virtual environment
source .venv/bin/activate

# Open and run notebook
jupyter notebook 01_dataset_preparation.ipynb
```

### What This Notebook Does

#### 1. **Dataset Exploration**
   - Lists all classes and their image counts
   - Generates class distribution visualization
   - Prints dataset statistics

#### 2. **Data Cleaning**
   
   **Corrupted Files Detection:**
   - Validates each image file using PIL
   - Identifies files that can't be opened
   - Lists corrupted files with error details
   
   **Duplicate Detection:**
   - Calculates SHA256 hash of each image
   - Identifies identical images
   - Groups duplicate sets
   
   **Removal:**
   - Removes corrupted files
   - Removes duplicate copies (keeps first)
   - Reports total removed

#### 3. **Train/Val/Test Split**
   - Creates folder structure:
     ```
     dataset/splits/
     ├── train/
     │   ├── hibiscus/
     │   ├── rose/
     │   └── sunflower/
     ├── val/
     │   ├── hibiscus/
     │   ├── rose/
     │   └── sunflower/
     └── test/
         ├── hibiscus/
         ├── rose/
         └── sunflower/
     ```
   - Split ratio: **70% train, 15% val, 15% test**
   - Maintains random seed (42) for reproducibility
   - Generates visualization comparing splits

#### 4. **Outputs**
   - `dataset/splits/` - Organized dataset ready for training
   - `dataset/processed/class_distribution_before_cleaning.png` - Visual statistics
   - `dataset/processed/train_val_test_split.png` - Split distribution

### Expected Output
```
DATASET PREPARATION COMPLETE
=========================================
📊 Summary:
  • Original images: 166
  • Corrupted files removed: [count]
  • Duplicate files removed: [count]
  • Final clean images: [final_count]

📁 Dataset location: dataset/splits/

✓ Ready for model training on Google Colab!
  Use notebook: 02_model_training_colab.ipynb
```

---

## STEP 2: Google Colab Model Training

### Prerequisites
- Google Drive account
- Google Colab access
- GPU quota (T4 GPU recommended)

### Setup Instructions

#### 1. **Prepare Dataset for Colab**
   ```bash
   # Option A: Upload to Google Drive manually
   # Create folder: Google Drive > Colab > datasets > flowers
   # Upload dataset/splits/ folder there
   
   # Option B: Use Colab upload
   # You can upload from Colab interface during notebook execution
   ```

#### 2. **Open Notebook in Colab**
   - Upload `02_model_training_colab.ipynb` to Google Drive
   - Right-click → Open with → Google Colaboratory
   - Or visit: https://colab.research.google.com/

#### 3. **Enable GPU**
   ```
   Runtime → Change runtime type → GPU → Select "T4"
   ```

#### 4. **Configure Paths (IMPORTANT)**
   
   In the first code cell, update the dataset path to match your Google Drive:
   
   ```python
   # Example: if you uploaded to: My Drive/Colab/datasets/flowers
   DATASET_BASE = Path('/content/drive/MyDrive/Colab/datasets/flowers')
   
   # Or your custom path
   DATASET_BASE = Path('/content/drive/MyDrive/[your_folder]/flowers')
   ```

#### 5. **Run All Cells**
   ```
   Runtime → Run all
   ```

### What This Notebook Does

#### 1. **Data Loading**
   - Loads images from train/val/test directories
   - Applies data augmentation to training data:
     - Rotation, shifting, shearing, zooming
     - Horizontal flips
   - Creates data generators with batch size 32
   - Displays sample images from dataset

#### 2. **Build CNN Model**
   
   **Architecture:**
   ```
   Input (224x224x3)
   ↓
   Conv Block 1: 32 filters → MaxPool → Dropout(0.25)
   ↓
   Conv Block 2: 64 filters → MaxPool → Dropout(0.25)
   ↓
   Conv Block 3: 128 filters → MaxPool → Dropout(0.25)
   ↓
   Conv Block 4: 256 filters → MaxPool → Dropout(0.25)
   ↓
   Flatten
   ↓
   Dense 512 → BatchNorm → Dropout(0.5)
   ↓
   Dense 256 → BatchNorm → Dropout(0.5)
   ↓
   Dense 3 (softmax) → Output [hibiscus, rose, sunflower]
   ```
   
   **Features:**
   - Batch normalization for training stability
   - Dropout for regularization (prevent overfitting)
   - Increasing channel depth (32→64→128→256)

#### 3. **Training**
   
   **Configuration:**
   - Optimizer: Adam (learning rate = 1e-3)
   - Loss: Categorical Crossentropy
   - Epochs: Up to 100 (early stopping usually ~30-50)
   
   **Callbacks:**
   - **Early Stopping**: Stops if validation loss doesn't improve for 15 epochs
   - **Model Checkpoint**: Saves best weights
   - **Learning Rate Reduction**: Reduces LR by 50% if stuck
   - **TensorBoard**: Logs for visualization
   
   **Training runs on GPU** for faster computation (~1-2 minutes per epoch)

#### 4. **Evaluation**
   
   - **Test Accuracy**: Overall accuracy on test set
   - **Classification Report**: Per-class precision, recall, F1-score
   - **Confusion Matrix**: Shows misclassifications
   - **Confusion Matrix Heatmap**: Visual representation

#### 5. **Visualization**
   
   Generates:
   - `training_history.png` - Accuracy and loss curves
   - `confusion_matrix.png` - Confusion matrix heatmap
   - `test_predictions.png` - Sample predictions with true labels
   - `sample_images.png` - Dataset samples

#### 6. **Test with Real Data**
   
   **Interactive Function:**
   ```python
   test_image_interactive('/path/to/image.jpg')
   ```
   
   Shows:
   - Image and prediction
   - Confidence score
   - All class probabilities with bar chart

### Expected Training Results

```
TRAINING METRICS
=========================================

Best Val Accuracy: ~0.92-0.96 (typically Epoch 20-40)
Final Test Accuracy: ~0.90-0.94

Per-class performance (example):
  hibiscus:  Precision ~0.92, Recall ~0.90
  rose:      Precision ~0.94, Recall ~0.95
  sunflower: Precision ~0.91, Recall ~0.92
```

### Model Outputs

After training, you'll get:
- `flower_classifier_model.h5` - Trained model (HDF5 format)
- `flower_classifier_savedmodel/` - SavedModel format (recommended for production)
- `training_summary.txt` - Complete metrics report

---

## Complete Workflow Checklist

### Local Phase (Notebook 1)
- [ ] Install required packages
- [ ] Run `01_dataset_preparation.ipynb` locally
- [ ] Verify `dataset/splits/` folder created
- [ ] Check data cleaning report
- [ ] Confirm train/val/test split proportions

### Colab Phase (Notebook 2)
- [ ] Enable GPU in Colab (Runtime → Change runtime type)
- [ ] Mount Google Drive
- [ ] Update dataset path (if needed)
- [ ] Run all cells
- [ ] Monitor training loss decreasing
- [ ] Check test accuracy
- [ ] Review confusion matrix for errors
- [ ] Test predictions interactively

### Post-Training
- [ ] Download trained model from Colab
- [ ] Save training visualizations
- [ ] Document final metrics
- [ ] Test on custom flower images (optional)

---

## Key Files

```
Chapter2/module3/
├── 01_dataset_preparation.ipynb      ← Run locally first
├── 02_model_training_colab.ipynb     ← Run on Colab with GPU
├── README_WORKFLOW.md                ← This file
│
├── dataset/
│   ├── raw/
│   │   ├── hibiscus/
│   │   ├── rose/
│   │   └── sunflower/
│   ├── processed/               (generated by notebook 1)
│   └── splits/                  (generated by notebook 1)
│       ├── train/
│       ├── val/
│       └── test/
│
└── [Colab outputs saved to Google Drive]
```

---

## Troubleshooting

### "No GPU found" in Colab
- Solution: Runtime → Change runtime type → GPU (T4)

### "File not found" error
- Check dataset path in the configuration cell
- Verify files are uploaded to Google Drive

### Out of memory error
- Reduce batch size (32 → 16)
- Reduce image size (224 → 128)
- Run on Colab (has more GPU memory)

### Poor training results
- Check data was cleaned properly in step 1
- Verify class distribution is reasonable
- Try more epochs (increase patience)
- Increase data augmentation

### Corrupted files not detected
- Some files might be readable by PIL but still problematic
- Check manually in `dataset/raw/` if needed
- Verify file extensions are correct

---

## Performance Tips

1. **Local Execution**
   - Close other applications for faster processing
   - SSD is faster than HDD for file I/O

2. **Colab Execution**
   - Use T4 GPU (free tier) or better (V100, A100)
   - Disable auto-disconnection: Settings → Keep VM running
   - Use small batch size initially to test

3. **Model Improvement**
   - Add more training data (collect more images)
   - Use transfer learning (MobileNet, EfficientNet)
   - Increase model capacity (more layers/filters)
   - Train longer with different augmentations

---

## Questions & Support

For issues:
1. Check error messages carefully
2. Review the troubleshooting section
3. Verify all input paths are correct
4. Check notebook cell output for details

---

**Version**: 1.0  
**Created**: 2026-07-10  
**Last Updated**: 2026-07-10
