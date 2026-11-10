# Facial Emotion Recognition — FastAPI Backend

Backend inference service for final-year project:  
**“Improved Facial Emotion Recognition Using Swin Transformer and YOLOv8 in Comparison with CNN”**

---

## 1. Overview & Architecture

This service acts as the inference engine for the React frontend, handling:
1. **YOLOv8 Face Detection (`POST /api/detect-face`)**: Detects human faces in webcam frames or static images, returning normalized bounding boxes `[x, y, width, height]`.
2. **Swin Transformer Emotion Classification (`POST /api/predict-emotion`)**: Preprocesses cropped face patches (224×224 RGB, ImageNet normalization) and outputs predictions across the 7 standardized classes:
   - *Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral*
3. **Model Comparison (`POST /api/compare-models`)**: Runs both the proposed Swin Transformer and the baseline CNN on the same face patch to provide a side-by-side comparative evaluation.
4. **Health & Status Check (`GET /health`)**: Inspects runtime environment (Python, PyTorch, CUDA, compute device) and the loading status of all 3 models.
5. **Multiple Face Analysis (`POST /api/analyze-faces`)**: Returns per-face YOLO detections and requested Swin, CNN, or comparison predictions.
6. **Sampled Video Jobs (`POST /api/video-jobs`, `GET /api/video-jobs/{job_id}`, `DELETE /api/video-jobs/{job_id}`)**: Samples uploaded videos and returns timestamped face predictions and an expression reliability report.

### Expression reliability interpretation

The video report is a consistency and image-quality summary, not a measure of model accuracy and not evidence of a person's actual emotional state. CNN–Swin agreement is calculated only when both models returned a label for the same tracked face and sample. Temporal consistency is the percentage of adjacent predictions within each temporary face track that kept the same label. Classifier confidence is not used in either calculation.

Quality measurements are calculated from each detected crop: pixel width/height, grayscale Laplacian variance (a blur/sharpness indicator), grayscale mean brightness on a 0–255 scale, and YOLO detection confidence. Occlusion/pose-based face visibility is reported as unavailable because no such estimator is present. Frames with no detected face count as insufficient-quality frames.

Status rules are configurable with environment variables (values below are defaults):

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `FER_RELIABILITY_MIN_USABLE_FRAMES` | `3` | Minimum sampled frames with a prediction and no configured quality flags. |
| `FER_RELIABILITY_MIN_AGREEMENT_PERCENT` | `80` | Required paired CNN/Swin label agreement for RELIABLE. |
| `FER_RELIABILITY_MIN_TEMPORAL_PERCENT` | `80` | Required unchanged adjacent same-track labels for RELIABLE. |
| `FER_RELIABILITY_MAX_LOW_QUALITY_PERCENT` | `30` | Maximum sampled frames with no face or all detected faces failing quality screens. |
| `FER_QUALITY_MIN_FACE_SIDE_PX` | `48` | Minimum short side of a measured face crop. |
| `FER_QUALITY_MIN_LAPLACIAN_VARIANCE` | `20` | Blur screening floor; lower values are flagged as possible blur. |
| `FER_QUALITY_MIN_BRIGHTNESS` / `FER_QUALITY_MAX_BRIGHTNESS` | `35` / `220` | Accepted mean grayscale brightness range. |
| `FER_QUALITY_MIN_DETECTION_CONFIDENCE` | `0.5` | YOLO score floor for the visual-quality screen (not emotion confidence). |

These are explicit screening heuristics, not validated scientific cutoffs. The server returns the effective thresholds and explanatory note with each completed report. When there are insufficient usable frames, the status is **INSUFFICIENT VISUAL EVIDENCE**. Otherwise, unavailable model pairs, rule failures, or instability produce **UNCERTAIN**. **RELIABLE** means only that the available outputs satisfy these configured consistency screens.

---

## 2. Windows Setup & Installation

### Step 1: Open PowerShell in the `backend` folder
```powershell
cd c:\Users\Admin\Desktop\face-feeling-atlas-main\backend
```

### Step 2: Create and activate a Python virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once).*

### Step 3: Upgrade pip and install base requirements
```powershell
python -m pip install --upgrade pip
pip install fastapi "uvicorn[standard]" pydantic python-multipart pillow numpy
```

### Step 4: Install PyTorch and Computer Vision packages
* **For CPU only:**
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install ultralytics timm opencv-python
  ```
* **For NVIDIA GPU (CUDA 12.1):**
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install ultralytics timm opencv-python
  ```

---

## 3. Running the Server

Start the FastAPI application with Uvicorn:
```powershell
uvicorn app:app --reload --port 8000
```

Once running:
* **Interactive API Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Alternative API Documentation (ReDoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Health Check & Model Inspection:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 4. Connecting the React Frontend

1. Ensure the frontend is running (default: `http://localhost:5173`).
2. Navigate to the **Settings** page in the web app: `http://localhost:5173/settings`.
3. Under **Model mode / inference backend**:
   - Set **Backend base URL** to: `http://localhost:8000`
   - Uncheck **Demo mode (simulated output)**.
4. The top header will immediately show:
   `MODEL MODE — INFERENCE BACKEND CONFIGURED`

---

## 5. Adding Trained Model Weights

Trained model weights should be placed in `backend/weights/`:

| Model | Expected Filename | Architecture / Format |
| :--- | :--- | :--- |
| **YOLOv8** | `backend/weights/yolov8n-face.pt` | Ultralytics YOLOv8 Face Detection |
| **Swin Transformer** | `backend/weights/swin_transformer_fer.pt` | PyTorch fine-tuned Swin Transformer (7 classes) |
| **CNN Baseline** | `backend/weights/cnn_baseline_fer.pt` | PyTorch baseline CNN (48×48 grayscale or RGB) |

> **Safety Notice:** The backend will never download unknown model weights automatically from the internet. When weights are placed in `backend/weights/` and the server is restarted, the endpoints automatically switch from pending mode to active inference mode.
