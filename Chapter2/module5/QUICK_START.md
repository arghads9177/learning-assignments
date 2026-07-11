# Quick Start Guide - Module 5: Transfer Learning

## 🚀 30-Second Setup

1. Open `01_transfer_learning.ipynb` in Google Colab
2. Enable T4 GPU: Runtime → Change runtime type → GPU
3. Run all cells sequentially (Ctrl+F10 or Runtime → Run all)
4. Monitor progress in the output cells

## ⏱️ Timeline

| Task | Time | Status |
|------|------|--------|
| Setup & mount Drive | 2 min | ⚡ |
| Install packages | 3 min | 📦 |
| Load data | 2 min | 📊 |
| Feature extraction (4 models) | 20 min | 🧠 |
| Fine-tuning (4 models) | 40 min | 🔧 |
| Analysis & visualization | 5 min | 📈 |
| **Total** | **~70 min** | ✅ |

## 📊 Models & Performance

### Feature Extraction (Frozen)
- **Best**: ConvNeXt-Tiny (~92% accuracy)
- **Fastest**: MobileNetV2 (~3 min per model)
- **Balanced**: ResNet50 (~85% accuracy)

### Fine-tuning (Unfrozen)
- **Best**: ConvNeXt-Tiny (~95% accuracy)
- **Most efficient**: MobileNetV2 (~92% accuracy)
- **Most popular**: ResNet50 (~90% accuracy)

**Recommendation**: Start with ResNet50 or MobileNetV2, then try ConvNeXt for best results.

## 🎯 Key Parameters to Adjust

```python
# In cell "Configure Paths & Parameters"

# Performance vs Speed tradeoff
BATCH_SIZE = 32         # ↑ faster training, ↓ memory
EPOCHS = 50             # ↑ better accuracy, ↑ time
LEARNING_RATE = 1e-3    # ↑ faster learning, ↓ stability

# For different datasets
IMG_HEIGHT = 224        # Must match model input
IMG_WIDTH = 224
NUM_CLASSES = 3         # Your number of classes
```

## 📁 Input/Output Paths

**Input (Read from)**:
```
Google Drive/
├── Data Science/
│   └── Computer Vision/
│       └── learning-assignments/
│           └── datasets/flowers/splits/
│               ├── train/
│               ├── val/
│               └── test/
```

**Output (Written to)**:
```
Google Drive/
├── Data Science/
│   └── Computer Vision/
│       └── learning-assignments/
│           └── flower_classifier_transfer_learning_output/
│               ├── transfer_learning_results.csv
│               ├── transfer_learning_results.png
│               ├── confusion_matrix_best_model.png
│               ├── best_model_*.pt
│               └── checkpoints/
```

## 🔍 Monitoring Training

### What to Watch

**Good signs**:
- ✅ Validation accuracy increasing each epoch
- ✅ Training accuracy > validation accuracy (normal)
- ✅ Loss smoothly decreasing
- ✅ Early stopping after 10-15 epochs

**Warning signs**:
- ⚠️ Validation accuracy plateauing early (try more data)
- ⚠️ Loss increasing (learning rate too high)
- ⚠️ Large train-test gap (overfitting, need more augmentation)

### Sample Output
```
Training ResNet50...
  Epoch 10 | TL: 0.8421 | TA: 0.7654 | VL: 0.6543 | VA: 0.8234
  Epoch 20 | TL: 0.4321 | TA: 0.8876 | VL: 0.5234 | VA: 0.8765
  Early stopping at epoch 24
  Test Accuracy: 0.8640
```

## 💾 Saving & Loading Best Model

### Save (automatic during training)
```python
# Checkpoints saved at:
# OUTPUT_DIR / 'checkpoints' / 'frozen_ResNet50' / 'best_model.pt'
```

### Load for inference
```python
model = TransferLearningModel(backbone, num_classes, 'resnet50')
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
model.to(device)

# Predict
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1)
```

## 🎓 Learning Outcomes

After completing this module, you should understand:

- [ ] What is transfer learning and why it works
- [ ] Difference between feature extraction and fine-tuning
- [ ] How pretrained weights provide inductive bias
- [ ] Why lower learning rates are needed for fine-tuning
- [ ] How to choose between different transfer learning strategies
- [ ] How to interpret training curves and metrics
- [ ] When to use each model architecture
- [ ] How to save and deploy trained models

## ❓ Common Questions

**Q: Why are validation curves sometimes noisy?**
A: Small validation set (72 samples). More batches = smoother curves.

**Q: Should I freeze more/fewer layers?**
A: Our implementation freezes entire backbone. For intermediate, freeze first N layers.

**Q: Can I use different input size?**
A: Yes, but models expect 224×224 for ImageNet pretrained. Others need retraining.

**Q: How to improve accuracy further?**
A: 1) Collect more data, 2) Use stronger augmentation, 3) Ensemble models, 4) Hyperparameter tuning

**Q: What if GPU runs out of memory?**
A: Reduce BATCH_SIZE to 16 or 8, or use gradient accumulation.

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Module not found" (timm/transformers) | Run pip install cell again |
| GPU memory error | Reduce BATCH_SIZE |
| Poor accuracy | Try fine-tuning strategy, more augmentation |
| Training too slow | Use smaller model (MobileNetV2) or bigger batch |
| Overfitting | Increase dropout, more augmentation, lower LR |

## 📚 Related Modules

- **Module 4**: Data Augmentation (provides augmentation techniques used here)
- **Module 6**: Model Deployment (how to deploy your transfer learning model)
- **Module 7**: Advanced Techniques (ensembling, knowledge distillation)

## 🔗 Additional Resources

- [Fastbook Chapter 17: Transfer Learning](https://github.com/fastai/fastbook/blob/master/17_transfer_learning.ipynb)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Hugging Face Transfer Learning Guide](https://huggingface.co/docs/transformers/training)

---

**Last Updated**: July 11, 2026  
**Duration**: ~70 minutes on Colab T4 GPU  
**Difficulty**: Intermediate 🔸
