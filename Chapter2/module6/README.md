# Module 6: Fine-Tuning Techniques for Transfer Learning

## Overview
This module explores advanced fine-tuning techniques to optimize transfer learning models. You'll learn how to strategically configure layer freezing, learning rate scheduling, optimizers, and other hyperparameters to achieve better performance.

## Topics Covered

### 1. **Layer Freezing Strategies**
- Full Frozen Backbone (feature extraction only)
- Partial Freezing (freeze early layers, unfreeze later ones)
- Progressive Unfreezing (gradual layer release)
- Full Unfrozen (fine-tune entire model)

**Key Insight:** Progressive unfreezing balances stability and accuracy on small datasets.

### 2. **Learning Rate Scheduling**
Four essential schedulers implemented and compared:

| Scheduler | Best For | Characteristics |
|-----------|----------|-----------------|
| **ReduceLROnPlateau** | General purpose | Adaptive, reduces LR when validation loss plateaus |
| **StepLR** | Fixed schedules | Reduces LR every N epochs by fixed factor |
| **ExponentialLR** | Smooth decay | Exponentially decreases LR each epoch |
| **CosineAnnealingLR** | Stable convergence | Smooth cosine decay from max to min LR |

### 3. **Weight Decay (L2 Regularization)**
- Prevents overfitting on small datasets
- Recommended values: 1e-4 to 1e-3
- Crucial for full model fine-tuning
- Less important for frozen backbone scenarios

### 4. **Optimizers**
Five optimizer configurations compared:

| Optimizer | Characteristics | Recommendation |
|-----------|-----------------|-----------------|
| **Adam** | No regularization | Baseline comparison |
| **Adam + Weight Decay** | Manual weight decay | Decent generalization |
| **AdamW** | Built-in weight decay | **RECOMMENDED** |
| **SGD** | Momentum-based | Good generalization |
| **SGD + Nesterov** | Accelerated gradient | Alternative to Adam |

**Finding:** AdamW provides best balance of convergence speed and generalization.

### 5. **One-Cycle Learning Rate**
A modern scheduling technique that:
- Increases LR from minimum to maximum in first phase
- Decreases LR from maximum to minimum in second phase
- Often achieves faster convergence than traditional decay
- Requires knowing total training steps

**Variants Tested:**
- Cosine annealing strategy (smoother)
- Linear annealing strategy (simpler)
- Different pct_start values (phase 1 duration)

### 6. **Cosine Annealing with Warm Restarts**
- Enables model to escape sharp minima
- Periodic learning rate restarts
- T_mult parameter controls restart growth
- Better exploration of loss landscape

### 7. **Differential Learning Rates (Layer Groups)**
Assign different learning rates to different model layers:
- Early layers: 1e-5 (preserve general features)
- Middle layers: 1e-4 to 1e-3 (balanced tuning)
- Later layers: 1e-2 (adapt to task)
- Classifier: 1e-2 (learn task-specific features)

**Benefit:** 1-2% accuracy improvement on small datasets.

### 8. **Mixed Precision Training (Concept)**
Understanding automatic mixed precision (AMP):
- Uses float16 for faster computation
- Maintains float32 for numerical stability
- Requires loss scaling to prevent underflow
- ~40% faster training, minimal accuracy loss
- Modern GPUs required (RTX 2080+, A100, T4)

## Notebook Structure

```
01_fine_tuning.ipynb
├── Setup & GPU Check
├── Data Loading with Augmentation
├── Model Architecture Definition
├── Training Functions with Schedulers
├── Experiment 1: Layer Freezing Strategies
├── Experiment 2: Learning Rate Schedulers
├── Experiment 3: Optimizers Comparison
├── Experiment 4: One-Cycle Learning Rate
├── Experiment 5: Cosine Annealing with Warm Restarts
├── Experiment 6: Differential Learning Rates
├── Results Consolidation
├── Visualizations
├── Mixed Precision Training Guide
├── Best Model Analysis
├── Recommendations & Best Practices
└── Summary Report
```

## Key Results

The notebook systematically evaluates 18+ configurations across 6 categories:

- **Total Experiments:** 18+ configurations
- **Best Strategy:** [Determined by your run]
- **Dataset:** Flower Classification (Hibiscus, Rose, Sunflower)
- **Architecture:** ResNet50 + Custom Classifier Head

## Generated Outputs

### CSV Files
- `fine_tuning_results.csv` - Complete results table with all metrics

### Visualizations
- `fine_tuning_by_category.png` - Bar chart comparison by technique
- `learning_rate_schedules.png` - LR curves during training
- `validation_curves_comparison.png` - Validation accuracy across techniques

### Documentation
- `mixed_precision_training_guide.txt` - Comprehensive AMP tutorial
- `fine_tuning_recommendations.txt` - Best practices and guidelines
- `fine_tuning_module_summary_report.txt` - Complete analysis report
- `best_model_config.txt` - Configuration of best model

### Model Checkpoints
- `best_fine_tuned_model_*.pt` - Best model weights
- `checkpoints/*/best_model.pt` - Intermediate checkpoints

## Quick Start Guide

### For Small Datasets (<500 images)
```python
1. Freeze backbone (feature extraction)
2. Train classifier for 5-10 epochs
3. Unfreeze backbone with 10x lower LR
4. Use ReduceLROnPlateau scheduler
5. Apply weight decay (1e-4)
6. Use AdamW optimizer
```

### For Medium Datasets (500-5000 images)
```python
1. Train with frozen backbone first
2. Gradually unfreeze layer groups
3. Use differential learning rates
4. Apply OneCycleLR scheduler
5. Consider mixed precision if training is slow
```

### For Large Datasets (>5000 images)
```python
1. Full model fine-tuning from start
2. Use higher base learning rates
3. Apply aggressive data augmentation
4. Mixed precision training recommended
5. Consider ensemble methods
```

## Best Practices Summary

✅ **DO:**
- Start with frozen backbone for small datasets
- Use AdamW with weight decay (1e-4)
- Monitor validation loss (not just accuracy)
- Apply learning rate scheduling
- Save best model checkpoint
- Use different LRs for different layers
- Validate on held-out test set

❌ **DON'T:**
- Use same learning rate for all layers
- Forget weight decay during optimization
- Train without early stopping
- Change multiple hyperparameters at once
- Neglect data augmentation
- Unfreeze backbone immediately on small data
- Use old Adam without weight decay

## Hyperparameter Guidelines

### Learning Rates
- **Feature Extraction (Frozen):** 1e-3 to 1e-2
- **Fine-tuning (Unfrozen):** 1e-4 to 1e-5
- **Classifier Layer:** 1e-2 (10x higher)

### Weight Decay
- **Small Dataset:** 1e-4 to 1e-3
- **Large Dataset:** 1e-5 to 1e-4
- **AdamW:** Built-in, use 1e-4 as default

### Batch Size
- **Small Dataset:** 16-32
- **Large Dataset:** 32-64
- **Limited Memory:** 8-16

### Early Stopping Patience
- **Default:** 10 epochs
- **Large Dataset:** 20 epochs
- **Small Dataset:** 5 epochs

## Advanced Techniques

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for images, labels in train_loader:
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Layer-wise Learning Rates
```python
param_groups = [
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-3},
    {'params': model.layer4.parameters(), 'lr': 1e-2},
]
optimizer = optim.Adam(param_groups)
```

## Common Issues & Solutions

### Issue: NaN Loss Values
**Solutions:**
- Reduce learning rate
- Disable mixed precision training
- Check data preprocessing
- Verify loss scaling

### Issue: Slow Convergence
**Solutions:**
- Increase learning rate
- Use OneCycleLR scheduler
- Apply mixed precision training
- Increase batch size

### Issue: Overfitting
**Solutions:**
- Increase weight decay
- Apply more data augmentation
- Use earlier stopping
- Freeze more layers

### Issue: Low Final Accuracy
**Solutions:**
- Train longer (increase epochs)
- Adjust learning rate schedule
- Use differential learning rates
- Try different optimizer

## Resources & References

- [PyTorch Optim Documentation](https://pytorch.org/docs/stable/optim.html)
- [Learning Rate Scheduling Best Practices](https://arxiv.org/abs/1908.03265)
- [Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [Transfer Learning with PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## Running on Google Colab

1. Open the notebook: `01_fine_tuning.ipynb`
2. Enable GPU: Runtime → Change runtime type → GPU (T4)
3. Run all cells sequentially
4. Outputs will be saved to your Google Drive

**Estimated Runtime:** 15-20 minutes (all experiments)

## Next Steps

1. **Module 7:** Model Compression & Quantization
2. **Module 8:** Model Deployment & Inference
3. **Module 9:** Advanced Topics (Attention, Vision Transformers)

---

**Created:** July 2026  
**Framework:** PyTorch 2.11.0+  
**Dataset:** Flower Classification (3 classes)  
**GPU Requirement:** NVIDIA GPU with 4GB+ VRAM (T4 or better)
