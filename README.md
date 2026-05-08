# BrainTumerModels — BRISC 2025

Brain tumor segmentation and classification service built on the BRISC 2025 dataset.
Supports U-Net, ResNet hybrid, EfficientNet, ViT, and a joint multitask model that
simultaneously segments tumor regions and classifies tumor type.

---

## Project Status

| Milestone | Status |
|-----------|--------|
| Data preprocessing pipeline (normalize, resize, augment, split) | ✅ Complete |
| Baseline segmentation (U-Net) + classification (ResNet) | ✅ Complete — checkpoints in `models/checkpoints/` |
| Advanced segmentation (ResNet hybrid) + classification (EfficientNet, ViT) | ✅ Complete |
| Joint segmentation-classification pipeline + hyperparameter tuning | ✅ Complete |
| Medical metrics (Dice, IoU, Hausdorff, Accuracy, F1, AUC) + confusion matrix | ✅ Complete |
| Explainable AI — Grad-CAM, SHAP, LIME | ✅ Grad-CAM complete · SHAP/LIME require optional deps |
| FastAPI service layer (train, predict, evaluate, explain) | ✅ Complete |
| Web dashboard (train, predict, evaluate, XAI visualization) | ✅ Complete — `frontend/simple/` |

---

## Project Layout

```
BrainTumerModels/
├── data/
│   ├── brisc2025/                      # Raw BRISC 2025 dataset
│   │   ├── classification_task/        # Images organised by class
│   │   └── segmentation_task/          # train/test split with paired masks
│   └── brisc2025divided/               # Organized by anatomical orientation (AX/CO/SA)
├── models/checkpoints/                 # Trained model weights (.pt)
├── outputs/
│   ├── processed/brisc2025/            # Preprocessed dataset (materialised)
│   ├── predictions/                    # Segmentation overlays & masks
│   ├── explanations/                   # Grad-CAM / SHAP / LIME artefacts
│   └── uploads/                        # Temporary uploaded images
├── src/ml_service/                     # Core ML service package
│   ├── api/                            # FastAPI application & routes
│   ├── core/                           # Config, constants, logging, device
│   ├── data/                           # Dataset discovery & preprocessing
│   ├── models/                         # Model registry
│   ├── training/                       # Trainer, tuning engine, callbacks
│   ├── evaluation/                     # Metrics, analysis, reporting
│   ├── inference/                      # Prediction wrapper
│   ├── explainability/                 # Grad-CAM, SHAP, LIME engine
│   └── experiments/                    # Top-level orchestration
└── frontend/simple/                    # Web dashboard (HTML + JS)
    ├── index.html                      # Main UI — Dashboard/Train/Predict/Evaluate/XAI
    └── app.js                          # All client-side logic
```

---

## Data Organization by Anatomical Orientation

The BRISC 2025 dataset contains MRI scans in three anatomical planes: **Axial (AX)**, **Coronal (CO)**, and **Sagittal (SA)**.
The raw dataset stores these mixed together by default. To organize images by plane for plane-specific model training, use the organization script:

```powershell
python organize_brisc_data.py
```

This script creates a `brisc2025divided/` folder with the same structure as `brisc2025/`, but organizes images into `AX/`, `CO/`, and `SA/` subfolders:

**Classification task structure:**
```
brisc2025divided/classification_task/train/
├── glioma/
│   ├── AX/          # All axial plane glioma images
│   ├── CO/          # All coronal plane glioma images
│   └── SA/          # All sagittal plane glioma images
├── meningioma/
│   ├── AX/
│   ├── CO/
│   └── SA/
├── no_tumor/
└── pituitary/
```

**Segmentation task structure:**
```
brisc2025divided/segmentation_task/train/
├── images/
│   ├── AX/          # All axial plane images
│   ├── CO/          # All coronal plane images
│   └── SA/          # All sagittal plane images
└── masks/
    ├── AX/
    ├── CO/
    └── SA/
```

Images are organized based on their filename suffixes:
- `*_ax_t1.*` → `AX/` folder
- `*_co_t1.*` → `CO/` folder
- `*_sa_t1.*` → `SA/` folder

The script preserves all metadata files (README, manifest.csv, manifest.json) and uses file copying to maintain the original dataset integrity.

---

## Requirements

- Python 3.9+
- PyTorch 2.6 with CUDA 12.4 (or CPU fallback)
- FastAPI + Uvicorn

### Install

```powershell
# Install all dependencies via uv (recommended)
uv sync

# Or with pip
pip install -e .
pip install -e ".[explainability]"   # for SHAP + LIME
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

The API will be available at `http://127.0.0.1:8000`.
Interactive docs: `http://127.0.0.1:8000/docs`.

### 2. Open the dashboard

Open `frontend/simple/index.html` directly in your browser
(no build step needed — it connects to the API over localhost).

The dashboard has five tabs:

| Tab | What it does |
|-----|-------------|
| **Dashboard** | Health check, model list, quick dataset prepare |
| **Train** | Prepare dataset, train individual models, hyperparameter tuning |
| **Predict** | Upload MRI → get class label (classification) or tumor mask (segmentation) + optional XAI |
| **Evaluate** | Run full test-set evaluation, view metrics cards + confusion matrix |
| **XAI** | Generate Grad-CAM / SHAP / LIME explanations with heatmap overlay |

---

## API Reference

All endpoints are served at `http://127.0.0.1:8000`.

### Health

```
GET /health
→ {"status": "ok"}
```

### Models

```
GET  /models/list               → {"models": [...]}
GET  /models/info/{model_name}  → {"model_name": ..., "callable": ...}
```

### Train

```
POST /train/
```

Body (JSON) with `action` field:

| Action | Description |
|--------|-------------|
| `{"action": "prepare"}` | Materialise cleaned dataset to `outputs/processed/` |
| `{"action": "train"}` | One joint multitask train + eval cycle |
| `{"action": "train_model", "model_name": "classification.resnet.finetune"}` | Train one model by name |
| `{"action": "train_all"}` | Train every registered model |
| `{"action": "tune", "trials": 5, "metric_name": "val_loss"}` | Random-search hyperparameter tuning |

```
POST /train/model/{model_name}  # shorthand for train_model action
```

### Predict

```
POST /predict/upload
```

Form data fields:

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | MRI image (JPEG or PNG) |
| `model_name` | string (optional) | registry name, e.g. `classification.resnet.finetune`; omit for latest checkpoint |
| `explain` | bool | `true` to include XAI explanation |
| `method` | string | `gradcam` · `shap` · `lime` (default: `gradcam`) |
| `target_task` | string | `auto` · `classification` · `segmentation` · `both` |

Response includes:

- `task` — `"classification"`, `"segmentation"`, or `"joint"`
- For classification: `class_name`, `confidence`, `probabilities` (per-class)
- For segmentation: `segmentation_mask_base64`, `segmentation_overlay_base64`
- For joint: both of the above
- `explanation` (if requested): `overlay_base64`, `heatmap_base64`, metadata

### Evaluate

```
POST /evaluate/
```

Body: `{"predictions": [[...], ...], "targets": [...]}`
Returns: `accuracy`, `f1_score`, `auc`, `dice_score`, `iou`, `hausdorff_distance`,
`confusion_matrix`, `classwise`, `summary_table`.

```
POST /evaluate/checkpoint
```

Body: `{"checkpoint_name": "classification.resnet.finetune", "split": "test"}`
Loads the checkpoint, runs evaluation, exports `metrics.json`, `analysis.json`,
`confusion_matrix.png`, and `summary.txt` to `outputs/evaluations/`.

### Explain

```
POST /explain/upload          # single image
POST /explain/batch           # batch from dataset split
```

Form data same as `/predict/upload` plus optional `target_layer`.

---

## Models

### Classification

| Model | Registry Name |
|-------|--------------|
| ResNet (ImageNet pretrained) | `classification.resnet.finetune` |
| ResNet (scratch) | `classification.resnet.scratch` |
| EfficientNet (pretrained) | `classification.efficientnet.finetune` |
| EfficientNet (scratch) | `classification.efficientnet.scratch` |
| ViT (pretrained) | `classification.vit.finetune` |

### Segmentation

| Model | Registry Name |
|-------|--------------|
| U-Net (pretrained encoder) | `segmentation.unet.finetune` |
| U-Net (scratch) | `segmentation.unet.scratch` |
| ResNet U-Net hybrid (pretrained) | `segmentation.resnet_hybrid.finetune` |
| ResNet U-Net hybrid (scratch) | `segmentation.resnet_hybrid.scratch` |

### Joint Multitask

| Model | Registry Name |
|-------|--------------|
| Shared encoder + cls + seg heads | `hybrid.multitask_joint` |

---

## Preprocessing Pipeline

The same pipeline is used for dataset materialisation **and** prediction-time inference:

1. Load image — fix EXIF orientation, convert to RGB
2. Resize to **224 × 224** (bilinear for images, nearest-neighbour for masks)
3. Normalize with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`
4. Training only: random horizontal flip, optional vertical flip, ±10° rotation, brightness/contrast jitter

---

## Evaluation Metrics

### Classification

- **Accuracy** — fraction of correctly classified samples
- **F1 Score** — macro-averaged across all four classes
- **AUC** — macro-averaged one-vs-rest ROC AUC

### Segmentation

- **Dice Score** — 2 × |A∩B| / (|A| + |B|)
- **IoU** (Jaccard) — |A∩B| / |A∪B|
- **Hausdorff Distance** — max boundary point distance (pixels)

### Analysis Outputs

- Per-class confusion matrix (image + JSON)
- Per-class precision, recall, F1 with support counts
- Macro averages

---

## Explainable AI

| Method | Task | Notes |
|--------|------|-------|
| Grad-CAM | Classification, segmentation, joint | No extra deps; auto-selects last conv layer |
| SHAP | Classification | Requires `shap` package (`pip install -e ".[explainability]"`) |
| LIME | Classification | Requires `lime` package (same extra) |

Outputs include:
- Overlay PNG: heatmap blended onto original MRI
- Heatmap PNG: raw attention map
- Both returned as `base64`-encoded data URIs for direct use in the browser

---

## Integration With ASP.NET Backend

The ML service is consumed by `../BrainTumorBackend`.
Configure the backend to point at this service:

```json
"MlService": { "BaseUrl": "http://127.0.0.1:8000" }
```

Endpoint mapping:

| Backend route | ML service route |
|--------------|-----------------|
| `POST /api/ml/predict` | `POST /predict/upload` |
| `POST /api/ml/train` | `POST /train/` |
| `GET  /api/ml/evaluate` | `POST /evaluate/` |
| `POST /api/ml/explain` | `POST /explain/upload` |

Always start this ML service **before** the backend.

---

## Development Notes

- `PYTHONPATH=src` is required for direct Python module runs.
- GPU is used automatically when PyTorch detects CUDA; otherwise falls back to CPU.
- Trained checkpoints are stored at `models/checkpoints/` in both flat (latest) and versioned (timestamped) formats.
- The API mounts `outputs/` at `/outputs` — segmentation and explanation images are served as static files.
- CORS is enabled for all origins to support the local browser-based dashboard.
