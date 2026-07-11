# Module 9: Experiment Tracking

## Overview
This module tracks the transfer-learning experiment from [Module 5](../module5) (ResNet50 on the flower dataset, frozen-backbone vs. fine-tuned) using three different experiment-tracking tools, one per notebook. Each run records the same set of facts so the tools can be compared directly:

- Dataset version
- Model
- Epochs
- Learning rate
- Batch size
- Optimizer
- Scheduler
- Metrics (per-epoch train/val loss & accuracy, final test accuracy)
- Training time
- Hardware

## Notebooks

| Notebook | Tool | Scope |
|----------|------|-------|
| `01_tensorboard_tracking.ipynb` | **TensorBoard** | Full walkthrough — scalars per epoch + `add_hparams` summary, launched inline in Colab |
| `02_wandb_tracking.ipynb` | **Weights & Biases** | Introduction — cloud dashboard, live metric logging, anonymous login supported |
| `03_mlflow_tracking.ipynb` | **MLflow** | Preview — local file-based tracking store, params/metrics/model artifact logging, `mlflow.search_runs` query |

Each notebook is self-contained: mounts Drive, loads the Module 5 flower splits, trains ResNet50 with a frozen backbone and then fine-tuned, and logs everything to its respective tool. Run on Google Colab with a T4 GPU.

## Viewing results

- **TensorBoard**: run the `%tensorboard --logdir ...` cell inside the notebook.
- **W&B**: follow the run URL printed by `wandb.init()`.
- **MLflow**: run `mlflow ui --backend-store-uri file:<OUTPUT_DIR>/mlruns` from a terminal, or query runs in-notebook with `mlflow.search_runs()`.
