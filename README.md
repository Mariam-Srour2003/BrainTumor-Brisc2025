# Brain Tumor Analysis System — BRISC 2025

**Thesis Project · Université Saint-Joseph de Beyrouth (USJ)**
**Author:** Mariam Srour · `mariammsrourr2020@gmail.com`

A full-stack system for brain tumor detection, segmentation, and explainability built on the
[BRISC 2025](https://www.synapse.org/brisc2025) dataset (≈ 6,000 MRI slices, four tumor classes).
The system covers the complete ML pipeline — from raw MRI preprocessing through multi-model
training, evaluation, and explainable AI — exposed via a FastAPI service and a browser dashboard.

---

## Research Overview

### Problem Statement

Brain tumor diagnosis from MRI imaging is time-consuming and highly operator-dependent.
This thesis investigates whether a multi-model deep learning system can:

1. **Classify** MRI slices into four categories — *Glioma*, *Meningioma*, *Pituitary tumor*, *No tumor*
2. **Segment** tumor regions at pixel level
3. **Explain** predictions using visual attribution methods (Grad-CAM, SHAP, LIME)
4. **Handle anatomical variation** by training view-specific models for Axial (AX), Coronal (CO), and Sagittal (SA) planes

### Dataset — BRISC 2025

| Property | Value |
|----------|-------|
| Total images | ≈ 6,000 MRI slices |
| Classes | Glioma · Meningioma · Pituitary · No tumor |
| Anatomical views | Axial (AX) · Coronal (CO) · Sagittal (SA) |
| Tasks | Classification + Segmentation (paired image/mask) |
| Input size | Resized to **224 × 224** |

### Approach

The system trains and compares **14 model configurations** across three task categories:

- **Classification-only** — ResNet50, EfficientNet-B3, ViT, with both fine-tuned (ImageNet) and scratch variants
- **Segmentation-only** — U-Net, ResNet-UNet hybrid, Swin-HaFUNet
- **Joint / Multitask** — shared encoder with classification and segmentation heads (ADPT-Net, TGD-ADPT-Net, Multitask Joint)

Each model is also trained per-view (AX, CO, SA) to capture orientation-specific features.

---

## Milestones

| Milestone | Status |
|-----------|--------|
| Data preprocessing — normalize, resize, augment, split | ✅ Complete |
| Anatomical-view organisation (AX / CO / SA) | ✅ Complete |
| Baseline: U-Net segmentation + ResNet classification | ✅ Complete |
| Advanced: EfficientNet, ViT, ResNet-UNet hybrid, Swin-HaFUNet | ✅ Complete |
| Joint multitask pipeline + hyperparameter tuning | ✅ Complete |
| Medical metrics — Dice, IoU, Hausdorff, Accuracy, F1, AUC | ✅ Complete |
| Explainable AI — Grad-CAM, SHAP, LIME | ✅ Grad-CAM complete · SHAP/LIME require optional deps |
| FastAPI service layer — train, predict, evaluate, explain | ✅ Complete |
| Web dashboard — train, predict, evaluate, XAI visualisation | ✅ Complete |
| Ranked HTML evaluation report (per model, per view) | ✅ Complete |

---

## Project Layout

```
BrainTumerModels/
├── data/
│   ├── brisc2025/                      # Raw BRISC 2025 dataset
│   │   ├── classification_task/        # Images organised by class label
│   │   └── segmentation_task/          # train/test split with paired masks
│   └── brisc2025divided/               # Same structure, split by anatomical plane (AX/CO/SA)
├── models/
│   └── checkpoints/                    # Trained model weights (.pt)
├── outputs/
│   ├── processed/brisc2025/            # Materialised preprocessed dataset
│   ├── predictions/                    # Segmentation overlays & class predictions
│   ├── evaluations/                    # Per-checkpoint metrics.json, confusion matrix
│   ├── reports/                        # Ranked HTML evaluation reports
│   ├── explanations/                   # Grad-CAM / SHAP / LIME artefacts
│   └── uploads/                        # Temporary uploaded MRI images
├── src/ml_service/                     # Core ML service package
│   ├── api/                            # FastAPI application & route handlers
│   ├── core/                           # Config, constants, logging, device selection
│   ├── data/                           # Dataset discovery & preprocessing pipeline
│   ├── models/                         # Model definitions & registry
│   ├── training/                       # Trainer, tuning engine, callbacks
│   ├── evaluation/                     # Metrics, per-class analysis, report generator
│   ├── inference/                      # Prediction wrapper (TTA, calibration)
│   ├── explainability/                 # Grad-CAM, SHAP, LIME engine
│   └── experiments/                    # Top-level orchestration scripts
└── frontend/simple/                    # Browser dashboard (no build step)
    ├── index.html                      # Main UI — Dashboard / Train / Predict / Evaluate / XAI
    └── app.js                          # All client-side logic + live loss chart (SSE)
```

---

## Models

### Classification

| Model | Registry Name | Backbone | Init |
|-------|--------------|----------|------|
| ResNet | `classification.resnet.finetune` | ResNet50 + GeM pooling | ImageNet |
| ResNet | `classification.resnet.scratch` | ResNet50 + GeM pooling | Random |
| EfficientNet | `classification.efficientnet.finetune` | EfficientNet-B3 + catavgmax pooling | ImageNet |
| EfficientNet | `classification.efficientnet.scratch` | EfficientNet-B3 + catavgmax pooling | Random |
| ViT | `classification.vit.finetune` | ViT + stochastic depth | ImageNet |
| View Classifier | `classification.view_classifier` | Predicts AX / CO / SA plane | — |

### Segmentation

| Model | Registry Name | Notes |
|-------|--------------|-------|
| U-Net | `segmentation.unet.finetune` | Attention gates in decoder; pretrained encoder |
| U-Net | `segmentation.unet.scratch` | Same architecture, random init |
| ResNet-UNet Hybrid | `segmentation.resnet_hybrid.finetune` | ResNet50 encoder |
| ResNet-UNet Hybrid | `segmentation.resnet_hybrid.scratch` | ResNet50 encoder, random init |
| Swin-HaFUNet | `segmentation.swin_hafunet.finetune` | Swin Transformer encoder |

### Joint / Multitask

| Model | Registry Name | Notes |
|-------|--------------|-------|
| ADPT-Net | `hybrid.adpt_net` | Shared encoder, dual cls+seg heads |
| TGD-ADPT-Net | `hybrid.tgd_adpt_net` | ADPT-Net with task-guided decoding |
| Multitask Joint | `hybrid.multitask_joint` | Shared encoder, simultaneous cls+seg |

All models are trained per-view as well (suffix `.ax`, `.co`, `.sa`).

---

## Preprocessing Pipeline

The same pipeline is applied at dataset materialisation time and at inference time:

1. Load image — fix EXIF orientation, convert to RGB
2. Resize to **224 × 224** (bilinear for images, nearest-neighbour for masks)
3. CLAHE on YCbCr luminance channel for contrast enhancement
4. Normalize with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`
5. **Training-only augmentations:** random horizontal/vertical flip, ±10° rotation, brightness/contrast jitter, Gaussian blur, sharpening, Gaussian noise, random erasing, Mixup + CutMix — all geometric transforms applied to image and mask with a shared RNG seed

---

## Training Techniques

Key techniques applied across models (see `src/ml_service/training/trainer.py`):

| Technique | Detail |
|-----------|--------|
| Loss — classification | Focal loss + label smoothing |
| Loss — segmentation | Composite: 0.4 × Dice + 0.3 × BCE + 0.3 × FocalTversky |
| Regularisation | R-Drop (bidirectional KL on clean batches) |
| Weights | EMA (Exponential Moving Average) |
| Optimiser | AdamW + Lookahead wrapper (k=5, α=0.5) |
| LR schedule | Cosine warmup; backbone LR 10× lower than head |
| Gradient accumulation | Steps = 4 |
| Class imbalance | Balanced oversampling |
| Backbone unfreezing | Frozen for first 5 epochs, then gradual unfreeze |
| Regularisation | Gradient clipping |
| Early stopping | Patience = 10 on validation loss |
| Hyperparameter tuning | Random search via `POST /train/` `tune` action |

---

## Evaluation Metrics

### Classification

| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of correctly classified samples |
| F1 Score | Macro-averaged across all four classes |
| AUC | Macro-averaged one-vs-rest ROC AUC |

### Segmentation

| Metric | Description |
|--------|-------------|
| Dice Score | 2 × \|A∩B\| / (\|A\| + \|B\|) |
| IoU (Jaccard) | \|A∩B\| / \|A∪B\| |
| Hausdorff Distance | Maximum boundary point distance (pixels) |

### Ranking Score

The ranked evaluation report uses a single composite score per model:

| Task | Score |
|------|-------|
| Classification | F1 Score |
| Segmentation | Dice Score |
| Joint / Hybrid | (Classification F1 + Segmentation Dice) / 2 |

### Outputs per Evaluation

- `metrics.json` — all numeric metrics
- `confusion_matrix.png` — per-class confusion matrix image
- `analysis.json` — per-class precision, recall, F1 with support counts
- `summary.txt` — human-readable summary
- Ranked HTML report in `outputs/reports/`

---

## Explainable AI

| Method | Task | Requirement |
|--------|------|-------------|
| Grad-CAM | Classification, segmentation, joint | None — built-in |
| SHAP | Classification | `pip install -e ".[explainability]"` |
| LIME | Classification | `pip install -e ".[explainability]"` |

Outputs returned as `base64`-encoded data URIs so the browser can display them without file paths:

- **Overlay** — heatmap blended onto the original MRI
- **Heatmap** — raw attention/saliency map

---

## Anatomical View Organisation

The raw BRISC 2025 dataset mixes all three MRI planes. To split images by plane for view-specific training:

```powershell
python organize_brisc_data.py
```

This creates `data/brisc2025divided/` with the same structure, but with `AX/`, `CO/`, `SA/` subfolders under each class and split:

```
brisc2025divided/classification_task/train/
├── glioma/
│   ├── AX/    # *_ax_t1.* images
│   ├── CO/    # *_co_t1.* images
│   └── SA/    # *_sa_t1.* images
├── meningioma/ ...
├── no_tumor/   ...
└── pituitary/  ...
```

---

## Requirements

- Python 3.9+
- PyTorch 2.6 with CUDA 12.4 (CPU fallback supported)
- FastAPI + Uvicorn

### Install

```powershell
# Recommended — using uv
uv sync

# Or with pip
pip install -e .
pip install -e ".[explainability]"   # adds SHAP + LIME
```

> **Windows note:** if `.venv\Scripts\python.exe` is blocked by application-control policy,
> use the system Python at `C:\Users\maria\AppData\Local\Programs\Python\Python313\python.exe`.

---

## Running the Service

### 1. Start the API server

```powershell
$env:PYTHONPATH = "src"
uvicorn ml_service.api.main:create_app --factory --reload --host 127.0.0.1 --port 8000
```

API available at `http://127.0.0.1:8000` — interactive docs at `http://127.0.0.1:8000/docs`.

### 2. Open the dashboard

Open `frontend/simple/index.html` directly in a browser (no build step — connects to the API over localhost).

| Tab | Purpose |
|-----|---------|
| **Dashboard** | Health check, model list, quick dataset prepare |
| **Train** | Prepare dataset, train individual or all models, hyperparameter tuning |
| **Predict** | Upload MRI → class label or tumor mask + optional XAI overlay |
| **Evaluate** | Run test-set evaluation, view metrics cards, confusion matrix, ranked report |
| **XAI** | Generate and view Grad-CAM / SHAP / LIME heatmaps |

---

## API Reference

All endpoints served at `http://127.0.0.1:8000`.

### Health

```
GET /health  →  {"status": "ok"}
```

### Models

```
GET /models/list               →  {"models": [...]}
GET /models/info/{model_name}  →  model metadata
```

### Train

```
POST /train/
```

| `action` | Description |
|----------|-------------|
| `prepare` | Materialise cleaned dataset to `outputs/processed/` |
| `train` | One joint multitask train + eval cycle |
| `train_model` + `model_name` | Train a single model by registry name |
| `train_all` | Train every registered model |
| `tune` + `trials` + `metric_name` | Random-search hyperparameter tuning |

```
GET /train/stream/{model_name}   # SSE endpoint — streams live epoch progress to the dashboard
```

### Predict

```
POST /predict/upload
```

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | MRI image (JPEG or PNG) |
| `model_name` | string (optional) | Registry name; omit to use latest checkpoint |
| `explain` | bool | `true` to include XAI explanation |
| `method` | string | `gradcam` · `shap` · `lime` (default: `gradcam`) |
| `target_task` | string | `auto` · `classification` · `segmentation` · `both` |

Response includes `task`, class label + confidence (classification), `segmentation_mask_base64`, `segmentation_overlay_base64` (segmentation), and `explanation` with `overlay_base64` + `heatmap_base64` when requested.

### Evaluate

```
POST /evaluate/checkpoint
```

Body: `{"checkpoint_name": "classification.resnet.finetune", "split": "test"}`

Loads the checkpoint, runs evaluation, exports `metrics.json`, `analysis.json`,
`confusion_matrix.png`, `summary.txt`, and a ranked HTML report to `outputs/`.

```
POST /evaluate/report   # (re-)generate the ranked HTML report from existing checkpoint results
```

### Explain

```
POST /explain/upload   # single image
POST /explain/batch    # batch from a dataset split
```

---

## Integration With ASP.NET Backend

The ML service is consumed by `../BrainTumorBackend` (Angular + ASP.NET). Point the backend at:

```json
"MlService": { "BaseUrl": "http://127.0.0.1:8000" }
```

| Backend route | ML service route |
|--------------|-----------------|
| `POST /api/ml/predict` | `POST /predict/upload` |
| `POST /api/ml/train` | `POST /train/` |
| `GET  /api/ml/evaluate` | `POST /evaluate/` |
| `POST /api/ml/explain` | `POST /explain/upload` |

Always start the ML service **before** the ASP.NET backend.

---

## Development Notes

- `PYTHONPATH=src` is required for direct Python module runs.
- GPU is used automatically when PyTorch detects CUDA; otherwise falls back to CPU.
- Checkpoints are stored in both flat (latest) and versioned (timestamped) formats under `models/checkpoints/`.
- The API mounts `outputs/` at `/outputs` so segmentation overlays and explanation images are served as static files.
- CORS is enabled for all origins to support the local browser-based dashboard.
- Class label order: `0=glioma`, `1=meningioma`, `2=no_tumor`, `3=pituitary` (defined in `CLASS_NAMES` constant).
