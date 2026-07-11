# Fine-Tuning Cheat Sheet

## Quick Reference for Module 6

### 1. Layer Freezing

```python
# Freeze entire backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Unfreeze backbone
for param in model.backbone.parameters():
    param.requires_grad = True

# Freeze early layers only (ResNet)
for param in model.backbone.layer1.parameters():
    param.requires_grad = False
for param in model.backbone.layer2.parameters():
    param.requires_grad = False
```

**Decision Tree:**
- Dataset < 100 images? → Freeze backbone completely
- Dataset 100-500 images? → Freeze early layers only
- Dataset 500-5000 images? → Partial freeze (2 layers)
- Dataset > 5000 images? → Unfreeze all or start unfrozen

---

### 2. Learning Rate Schedulers

#### ReduceLROnPlateau (Recommended)
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',           # Monitor val_loss
    factor=0.5,          # Multiply LR by 0.5
    patience=3,          # Wait 3 epochs before reducing
    verbose=True
)

for epoch in range(epochs):
    train()
    val_loss = validate()
    scheduler.step(val_loss)  # Pass metric to monitor
```

#### StepLR (Fixed Schedule)
```python
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,        # Reduce every 10 epochs
    gamma=0.5            # Multiply by 0.5
)

for epoch in range(epochs):
    train()
    scheduler.step()     # No metric needed
```

#### ExponentialLR (Smooth Decay)
```python
scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95           # Multiply by 0.95 each epoch
)

for epoch in range(epochs):
    train()
    scheduler.step()
```

#### CosineAnnealingLR (Smooth Cosine Decay)
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,           # Decay over 50 epochs
    eta_min=1e-6        # Minimum LR
)

for epoch in range(epochs):
    train()
    scheduler.step()
```

#### OneCycleLR (Modern, Fast)
```python
total_steps = len(train_loader) * epochs
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    total_steps=total_steps,
    pct_start=0.3,              # 30% increase phase
    anneal_strategy='cos'       # Cosine decay
)

for epoch in range(epochs):
    for batch in train_loader:
        train_batch()
        scheduler.step()        # Step per batch, not epoch!
```

#### CosineAnnealingWarmRestarts
```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,             # Cycle length 10 epochs
    T_mult=1,           # Keep cycle length same
    eta_min=1e-6
)

for epoch in range(epochs):
    train()
    scheduler.step()    # Periodic restarts
```

**Scheduler Comparison:**
| Scheduler | Dataset Size | Convergence | Stability | Best Case |
|-----------|---|---|---|---|
| ReduceLROnPlateau | Any | Medium | High | General purpose |
| StepLR | Any | Slow | High | Fixed schedules |
| ExponentialLR | Large | Fast | Medium | Smooth decay |
| CosineAnnealingLR | Any | Medium | High | Stable training |
| OneCycleLR | Large | Fast | Medium | Fast convergence |
| WarmRestarts | Large | Fast | High | Exploration |

---

### 3. Optimizers

#### Adam (Baseline)
```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0       # No L2 regularization
)
```

#### Adam + Weight Decay (Manual)
```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4    # L2 regularization applied
)
```

#### AdamW (Recommended)
```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4    # Decoupled weight decay
)
```

#### SGD with Momentum
```python
optimizer = optim.SGD(
    model.parameters(),
    lr=1e-2,            # Higher LR needed
    momentum=0.9,
    weight_decay=1e-4
)
```

#### SGD + Nesterov
```python
optimizer = optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
    nesterov=True,      # Accelerated gradient
    weight_decay=1e-4
)
```

**Optimizer Selection:**
- Default choice? → **AdamW**
- Complex models? → **AdamW**
- Generalization critical? → **SGD with momentum**
- Fast prototyping? → **Adam + Weight Decay**

---

### 4. Weight Decay (L2 Regularization)

```python
# With AdamW (Recommended)
optimizer = optim.AdamW(
    model.parameters(),
    weight_decay=1e-4
)

# Without explicit L2 penalty
optimizer = optim.Adam(
    model.parameters(),
    weight_decay=0
)

# Manual L2 penalty (Old way)
loss = criterion(outputs, labels)
l2_penalty = sum(p.pow(2).sum() for p in model.parameters())
total_loss = loss + 1e-4 * l2_penalty
```

**Weight Decay Values:**
- Small dataset (<500 images): 1e-3 to 1e-4
- Medium dataset: 1e-4
- Large dataset: 1e-5 to 1e-4
- No regularization: 0 (not recommended)

---

### 5. Differential Learning Rates (DLR)

```python
# Define layer groups with different LRs
param_groups = [
    {'params': model.backbone.layer1.parameters(), 'lr': 1e-5},
    {'params': model.backbone.layer2.parameters(), 'lr': 1e-4},
    {'params': model.backbone.layer3.parameters(), 'lr': 1e-3},
    {'params': model.backbone.layer4.parameters(), 'lr': 1e-2},
    {'params': model.classifier.parameters(), 'lr': 1e-2}
]

optimizer = optim.AdamW(param_groups)

# Training loop (no changes needed)
for epoch in range(epochs):
    for batch in train_loader:
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**LR Multiplier Strategy:**
- Layer 1 (edges, colors): 1x (smallest)
- Layer 2: 10x
- Layer 3: 100x
- Layer 4: 1000x
- Classifier: 1000x (largest)

---

### 6. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Create scaler for gradient scaling
scaler = GradScaler()

for epoch in range(epochs):
    for images, labels in train_loader:
        # Forward pass with automatic mixed precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        
        # Step and update scaler
        scaler.step(optimizer)
        scaler.update()
        
        optimizer.zero_grad()
```

**When to Use:**
- ✅ Model > 50M parameters
- ✅ Need 2-3x speedup
- ✅ Limited GPU memory
- ✅ Modern GPU (RTX 2080+, A100, T4)

**Benefits:**
- 40-50% faster training
- 30-50% less memory
- Minimal accuracy loss (<0.5%)

---

### 7. Complete Fine-Tuning Recipe

#### Small Dataset (<500 images)
```python
# Step 1: Load and freeze
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = CustomClassifier()

# Step 2: Train classifier only
optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

for epoch in range(10):
    train_epoch()
    scheduler.step(val_loss)

# Step 3: Unfreeze and fine-tune
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

for epoch in range(40):
    train_epoch()
    scheduler.step(val_loss)
```

#### Large Dataset (>5000 images)
```python
# Full model fine-tuning from start
model = models.resnet50(pretrained=True)
model.fc = CustomClassifier()

total_steps = len(train_loader) * 50
param_groups = create_layer_groups(model)
optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=1e-2, 
    total_steps=total_steps
)

for epoch in range(50):
    for batch in train_loader:
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step per batch
        optimizer.zero_grad()
```

---

### 8. Monitoring Training

```python
# Track key metrics
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

for epoch in range(epochs):
    # Training
    train_loss, train_acc = train_epoch()
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    
    # Validation
    val_loss, val_acc = validate()
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Learning rate
    current_lr = optimizer.param_groups[0]['lr']
    history['lr'].append(current_lr)
    
    # Scheduler step
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss > best_val_loss:
        patience_counter += 1
        if patience_counter >= patience:
            break
    else:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
```

---

### 9. Common Pitfalls & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| NaN loss | LR too high | Reduce LR by 10x |
| Slow convergence | LR too low | Increase LR by 10x |
| Overfitting | Too much training | Increase weight decay |
| Unstable loss | Wrong LR schedule | Use ReduceLROnPlateau |
| Low final acc | Early stopping too early | Increase patience |
| GPU OOM | Large batch/model | Reduce batch size or use mixed precision |

---

### 10. Default Configurations by Scenario

#### Quick Baseline (Guaranteed to work)
```python
backbone = models.resnet50(pretrained=True)
# Freeze backbone
for p in backbone.parameters():
    p.requires_grad = False

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
criterion = nn.CrossEntropyLoss()
```

#### Balanced Configuration (Good results)
```python
backbone = models.resnet50(pretrained=True)
# Unfreeze backbone with lower LR
param_groups = [
    {'params': backbone.parameters(), 'lr': 1e-4},
    {'params': classifier.parameters(), 'lr': 1e-3}
]

optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)
criterion = nn.CrossEntropyLoss()
```

#### Aggressive Configuration (Maximum performance)
```python
backbone = models.resnet50(pretrained=True)
# Full DLR
param_groups = [
    {'params': backbone.layer1.parameters(), 'lr': 1e-5},
    {'params': backbone.layer2.parameters(), 'lr': 1e-4},
    {'params': backbone.layer3.parameters(), 'lr': 1e-3},
    {'params': backbone.layer4.parameters(), 'lr': 1e-2},
    {'params': classifier.parameters(), 'lr': 1e-2}
]

optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
total_steps = len(train_loader) * 100
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-1, total_steps=total_steps, pct_start=0.3
)
criterion = nn.CrossEntropyLoss()
```

---

## Summary Decision Tree

```
START: What's your dataset size?

├─ Small (<500 images)
│  ├─ Freeze backbone completely
│  ├─ Use AdamW (lr=1e-3, weight_decay=1e-4)
│  ├─ Use ReduceLROnPlateau
│  └─ Early stopping: patience=10
│
├─ Medium (500-5000 images)
│  ├─ Freeze 2-3 early layers
│  ├─ Use AdamW with DLR
│  ├─ Use CosineAnnealingLR or ReduceLROnPlateau
│  └─ Early stopping: patience=15
│
└─ Large (>5000 images)
   ├─ Unfreeze all or partial freeze
   ├─ Use AdamW with differential LR
   ├─ Use OneCycleLR (faster convergence)
   └─ Consider mixed precision training
```

---

**Pro Tips:**
- 🎯 Start conservative (frozen backbone) and gradually unfreeze
- 📊 Always plot training curves (loss and accuracy)
- 💾 Save best model checkpoint, not last epoch
- 🔍 Monitor learning rate schedule during training
- 🧪 Try ReduceLROnPlateau first (safest choice)
- ⚡ Use OneCycleLR for maximum speed (requires tuning)
- 🛡️ Always apply weight decay (1e-4 is safe default)
- 📈 Track validation loss for early stopping, not accuracy

