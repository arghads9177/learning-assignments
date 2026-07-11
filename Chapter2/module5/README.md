# Module 5: Transfer Learning for Flower Classification

## Overview

This module implements **Transfer Learning** - a fundamental technique in deep learning where pretrained models are leveraged to improve performance on new tasks, especially with limited data.

## What is Transfer Learning?

Transfer Learning utilizes knowledge learned from large-scale datasets (like ImageNet) to solve new classification tasks with smaller datasets. Instead of training from scratch, we:

1. **Use pretrained weights** from models trained on millions of images
2. **Leverage learned features** that can identify edges, textures, and patterns
3. **Train only the classifier head** or fine-tune the entire network

### Benefits
- ✅ **Better accuracy** with limited data (100-1000 images)
- ✅ **Faster training** compared to training from scratch
- ✅ **Reduced computational cost** (fewer epochs needed)
- ✅ **Generalization** from ImageNet to flower classification

## Notebook Structure

### 📝 Cell-by-Cell Breakdown

1. **Setup & GPU Check** - Verify T4 GPU is available
2. **Mount Google Drive** - Access datasets and save outputs
3. **Install Dependencies** - timm, transformers libraries
4. **Import Libraries** - PyTorch, torchvision, sklearn
5. **Configure Paths** - Dataset and output directories
6. **Load Data** - Apply augmentation from Module 4
7. **Define Model Wrapper** - TransferLearningModel class
8. **Training Functions** - train_epoch(), val_epoch(), train_model()
9. **Feature Extraction** - Train with frozen backbone
10. **Fine-tuning** - Train with unfrozen backbone
11. **Results Comparison** - Analyze performance
12. **Visualizations** - Plot training curves and metrics
13. **Best Model Analysis** - Confusion matrix and classification report
14. **Save Results** - Export model and summary

## Transfer Learning Strategies

### 1. Feature Extraction (Frozen Backbone) 🔒
```
Backbone (pretrained, frozen) → New Classifier Head (trained)
                    ↓
            Only classifier learns
```

**When to use:**
- Very small datasets (< 100 images)
- Limited computational resources
- Quick baseline results

**Characteristics:**
- Backbone parameters: `requires_grad = False`
- Only classifier weights are updated
- Fast training (typically 5-10 epochs)
- Generally good accuracy on small datasets

### 2. Fine-tuning (Unfrozen Backbone) 🔓
```
Backbone (pretrained, unfrozen) → Classifier Head (trained)
         ↓
All layers learn with lower LR
```

**When to use:**
- Medium datasets (1K-10K images)
- Have sufficient computational power
- Want maximum accuracy

**Characteristics:**
- All backbone parameters: `requires_grad = True`
- Lower learning rate (LR / 10) to prevent catastrophic forgetting
- Slower training (typically 20-50 epochs)
- Often achieves higher accuracy than feature extraction

**Why lower learning rate?**
- Pretrained weights are already good
- Large updates could destroy learned features
- Small updates allow fine adaptation to new task

## Models Evaluated

| Model | Parameters | Year | Best For |
|-------|-----------|------|----------|
| **ResNet50** | 25.5M | 2015 | Balanced accuracy/speed |
| **EfficientNet-B0** | 5.3M | 2019 | Efficiency + accuracy |
| **MobileNetV2** | 3.5M | 2018 | Mobile deployment |
| **ConvNeXt-Tiny** | 28.6M | 2022 | Modern architecture |

## Key Hyperparameters

```python
IMG_HEIGHT, IMG_WIDTH = 224, 224      # Input size
BATCH_SIZE = 32                        # Samples per batch
NUM_CLASSES = 3                        # Flowers: hibiscus, rose, sunflower
LEARNING_RATE = 1e-3                   # Initial LR
LR_FINETUNING = 1e-4                   # Fine-tuning LR (LR / 10)
EPOCHS = 50                            # Max training epochs
EARLY_STOPPING_PATIENCE = 10           # Stop if no improvement
```

## Data Augmentation (from Module 4)

The notebook uses augmentation strategies from Module 4:

```python
train_transform = [
    RandomHorizontalFlip(p=0.5)
    RandomVerticalFlip(p=0.3)
    RandomRotation(degrees=20)
    RandomAffine(translate=(0.1, 0.1))
    ColorJitter(brightness, contrast, saturation, hue)
    GaussianBlur(sigma=(0.1, 2.0))
    AutoAugment(policy=IMAGENET)
    RandomPerspective(distortion=0.2)
]
```

## Expected Results

Based on the flower dataset with transfer learning:

### Feature Extraction (Frozen)
```
Model              Test Accuracy    Val Accuracy    Epochs
MobileNetV2        ~85-90%          ~80-85%         8-12
EfficientNet-B0    ~80-85%          ~75-80%         10-15
ResNet50           ~85-92%          ~80-88%         10-15
ConvNeXt-Tiny      ~88-95%          ~85-92%         12-18
```

### Fine-tuning (Unfrozen)
```
Model              Test Accuracy    Val Accuracy    Epochs
MobileNetV2        ~90-95%          ~85-90%         15-25
EfficientNet-B0    ~88-93%          ~83-88%         18-28
ResNet50           ~90-96%          ~85-92%         20-30
ConvNeXt-Tiny      ~92-97%          ~88-94%         22-32
```

*Note: Exact results depend on dataset split, random seed, and training variance*

## How to Use

### Prerequisites
- Google Colab account with T4 GPU
- Flower dataset in Google Drive
- Module 4 data augmentation notebook (reference)

### Running on Colab

1. **Open notebook**: Upload to Google Colab
2. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
3. **Run cells**: Execute sequentially from top to bottom
4. **Monitor training**: Watch for validation accuracy improvements
5. **Save results**: Automatically saved to Google Drive

### Expected Runtime
- Per model (frozen): ~5-10 minutes
- Per model (fine-tuning): ~15-25 minutes
- Total (4 models × 2 strategies): ~60-90 minutes

## Output Files

```
flower_classifier_transfer_learning_output/
├── transfer_learning_results.csv          # Metrics for all models
├── transfer_learning_results.png          # Performance comparison charts
├── confusion_matrix_best_model.png        # Confusion matrix
├── checkpoints/
│   ├── frozen_ResNet50/best_model.pt
│   ├── unfrozen_ResNet50/best_model.pt
│   ├── frozen_EfficientNet/best_model.pt
│   └── ... (more checkpoints)
├── best_model_*.pt                        # Best performing model
└── summary_report.txt                     # Detailed analysis
```

## Interpreting Results

### Validation Accuracy Curves
- **Steep initial rise**: Model learning quickly
- **Plateau**: Model converging
- **Decline**: Overfitting (early stopping helps)

### Test Accuracy Metrics
- **High Test Acc**: Good generalization
- **Large Train-Test Gap**: Possible overfitting
- **Similar across strategies**: Dataset/augmentation working well

### Confusion Matrix
- **Diagonal values**: Correct predictions
- **Off-diagonal values**: Misclassified samples
- **Row-wise analysis**: Model's confidence by class

## Advanced Concepts

### Why Does Transfer Learning Work?

1. **Hierarchical Features**:
   - Early layers: Edges, corners, textures
   - Middle layers: Shapes, patterns
   - Deep layers: Complex object parts

2. **Domain Similarity**:
   - Flowers and ImageNet share visual patterns
   - Pretrained features generalize well

3. **Inductive Bias**:
   - CNN architecture expects visual data
   - Convolutional structure captures spatial relationships

### Catastrophic Forgetting

When fine-tuning with large learning rates:
- Model "forgets" pretrained knowledge
- Performance on validation drops
- Solution: Use lower learning rate (0.1× to 0.01×)

### When to Use Each Strategy

```
Dataset Size    Strategy                  Typical LR
< 100 images    Feature Extraction        N/A (backbone frozen)
100-1K images   Feature Extraction        1e-3
1K-10K images   Progressive Fine-tuning   1e-3 then 1e-4
> 10K images    Full Fine-tuning          1e-3 then 1e-4
```

## Troubleshooting

### GPU Memory Issues
```python
# Solution: Reduce batch size
BATCH_SIZE = 16  # or 8
```

### Poor Accuracy
```python
# Check: Data augmentation not too aggressive
# Check: Learning rate appropriate
# Try: Progressive fine-tuning strategy
```

### Slow Convergence
```python
# Use: Learning rate scheduler
# Use: Gradient accumulation
# Consider: Feature extraction first, then fine-tune
```

## Comparison with Module 4

| Aspect | Module 4 (CNN) | Module 5 (Transfer Learning) |
|--------|---|---|
| Training from | Scratch | Pretrained weights |
| Backbone | Custom CNN | ImageNet pretrained |
| Typical accuracy | 60-70% | 85-95% |
| Training time | 30-60 min | 10-30 min |
| Data requirement | 1000+ images | 100+ images |
| Overfitting risk | High | Low |

## Next Steps

1. **Experiment**: Try different learning rates
2. **Visualize**: Plot attention maps from models
3. **Ensemble**: Combine multiple models
4. **Deploy**: Convert to mobile format (ONNX, TFLite)
5. **Optimize**: Knowledge distillation for smaller models

## Key Takeaways

✅ Transfer learning dramatically improves performance with small datasets
✅ Feature extraction is fast but fine-tuning often yields better results
✅ Progressive training (frozen → unfrozen) balances stability and accuracy
✅ Lower learning rates essential for preventing catastrophic forgetting
✅ Model choice depends on accuracy vs efficiency requirements

## References

- Yosinski et al. (2014): "How transferable are features in deep neural networks?"
- He et al. (2015): "Deep Residual Learning for Image Recognition" (ResNet)
- Tan & Le (2019): "EfficientNet: Rethinking Model Scaling" (EfficientNet)
- Sandler et al. (2018): "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
- Liu et al. (2022): "A ConvNet for the 2020s" (ConvNeXt)

---

**Module Created**: July 11, 2026  
**Framework**: PyTorch + torchvision + timm  
**Suitable for**: Google Colab with T4 GPU  
**Estimated Duration**: 2-3 hours
