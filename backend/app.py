"""
FastAPI Backend for Facial Emotion Recognition Prototype.

Project Title:
"Improved Facial Emotion Recognition Using Swin Transformer and YOLOv8 in Comparison with CNN"

Provides API endpoints compatible with the React frontend's contracts:
- GET  /health              -> Health check & model status inspection
- GET  /                    -> Basic API information
- GET  /api/evaluation-metrics -> Actual evaluated benchmark metrics on FER2013 test set
- POST /api/detect-face     -> YOLOv8 Face Detection (contract for emotionService.ts)
- POST /api/predict-emotion  -> Swin Transformer Emotion Classification (contract for emotionService.ts)
- POST /api/compare-models  -> Side-by-side inference (Swin Transformer vs. CNN Baseline)
"""

import base64
import io
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .video_jobs import process_video_job, video_jobs
except ImportError:
    from video_jobs import process_video_job, video_jobs

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

# Base Directory & Weights Directory
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS_PATH = WEIGHTS_DIR / "yolov8n-face.pt"
SWIN_WEIGHTS_PATH = WEIGHTS_DIR / "swin_transformer_fer.pt"
CNN_WEIGHTS_PATH = WEIGHTS_DIR / "cnn_baseline_fer.pt"
SWIN_METRICS_PATH = WEIGHTS_DIR / "swin_evaluation_results.json"
CNN_METRICS_PATH = WEIGHTS_DIR / "cnn_evaluation_results.json"

# The 7 standardized FER emotion classes matching the frontend
# Exactly matching index order in FER2013 standard:
# 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

# Mapping from Swin Transformer checkpoint output classes (alphabetical)
# ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# to standardized FER classes ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
SWIN_ALPHA_TO_STD = {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 4, 6: 5}

# ---------------------------------------------------------------------------
# ML Framework Imports
# ---------------------------------------------------------------------------
HAS_PIL = False
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    pass

HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    pass

HAS_ULTRALYTICS = False
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    pass

HAS_TIMM = False
try:
    import timm
    HAS_TIMM = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# CNN Baseline PyTorch Definition (Exact architecture matching fer_saved_model)
# ---------------------------------------------------------------------------
if HAS_TORCH:
    class PyTorchCNNBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(2048, 128)
            self.fc2 = nn.Linear(128, 7)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            # Permute to channels_last to match the serialized TF dense kernel ordering
            x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x


# ---------------------------------------------------------------------------
# FastAPI Application & CORS Configuration
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Facial Emotion Recognition API",
    description=(
        "Backend inference service for 'Improved Facial Emotion Recognition "
        "Using Swin Transformer and YOLOv8 in Comparison with CNN'."
    ),
    version="1.0.0",
)

# CORS setup for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:8082",
        "http://127.0.0.1:8082",
        "http://localhost:8083",
        "http://127.0.0.1:8083",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas (Aligned with frontend src/lib/fer/types.ts & emotionService.ts)
# ---------------------------------------------------------------------------
class BoundingBox(BaseModel):
    x: float = Field(..., description="Normalized x coordinate (0.0 to 1.0)")
    y: float = Field(..., description="Normalized y coordinate (0.0 to 1.0)")
    width: float = Field(..., description="Normalized width (0.0 to 1.0)")
    height: float = Field(..., description="Normalized height (0.0 to 1.0)")


class DetectFaceRequest(BaseModel):
    image: Optional[str] = Field(
        None,
        description="Base64 encoded image string or data URL (e.g. data:image/jpeg;base64,...)",
    )


class DetectFaceResponse(BaseModel):
    detected: bool = Field(..., description="Whether at least one face was detected")
    faceCount: int = Field(..., description="Total number of faces found")
    confidence: Optional[float] = Field(
        None, description="Detector confidence score between 0.0 and 1.0"
    )
    box: Optional[BoundingBox] = Field(
        None, description="Primary bounding box (normalized 0-1)"
    )
    boxes: List[BoundingBox] = Field(
        default_factory=list, description="All detected bounding boxes"
    )
    source: str = Field(default="backend", description="Source indicator")


class PredictEmotionRequest(BaseModel):
    image: Optional[str] = Field(
        None,
        description="Base64 encoded cropped face patch or full image (data URL or raw base64)",
    )


class AnalyzeFacesRequest(BaseModel):
    image: str
    model: Literal["swin", "cnn", "compare"] = "swin"


class PredictEmotionResponse(BaseModel):
    emotion: str = Field(
        ...,
        description="Dominant predicted emotion (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)",
    )
    confidence: float = Field(
        ..., description="Prediction confidence score between 0.0 and 1.0"
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Percentage probability distribution across all 7 emotion classes (0-100)",
    )
    source: str = Field(default="backend", description="Source indicator")
    model_name: Optional[str] = Field(
        default="Swin Transformer", description="Model used for this prediction"
    )


class FaceAnalysisResponse(BaseModel):
    face_id: str
    box: BoundingBox
    detector_confidence: float
    predictions: Dict[str, PredictEmotionResponse]
    model_errors: Dict[str, str] = Field(default_factory=dict)


class AnalyzeFacesResponse(BaseModel):
    source: str = "backend"
    model: str
    face_count: int
    faces: List[FaceAnalysisResponse]
    model_errors: Dict[str, str] = Field(default_factory=dict)


class ModelComparisonResponse(BaseModel):
    source: str = "backend"
    swin_transformer: Optional[PredictEmotionResponse] = None
    cnn_baseline: Optional[PredictEmotionResponse] = None
    model_errors: Dict[str, str] = Field(default_factory=dict)
    comparison_summary: str = Field(
        ..., description="Summary comparing the predictions"
    )


# ---------------------------------------------------------------------------
# Model Registry & Lifecycle Management
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self):
        self.yolo_detector = None
        self.swin_classifier = None
        self.cnn_baseline = None
        self.load_errors = {"yolov8_face": None, "swin_transformer": None, "cnn_baseline": None}

    def status(self) -> Dict[str, dict]:
        return {
            "yolov8_face": {
                "name": "YOLOv8 Face Detector",
                "weights_path": str(YOLO_WEIGHTS_PATH),
                "weights_exist": YOLO_WEIGHTS_PATH.is_file(),
                "loaded": self.yolo_detector is not None,
                "framework_installed": HAS_ULTRALYTICS,
                "load_error": self.load_errors["yolov8_face"],
            },
            "swin_transformer": {
                "name": "Swin Transformer FER Classifier",
                "weights_path": str(SWIN_WEIGHTS_PATH),
                "weights_exist": SWIN_WEIGHTS_PATH.is_file(),
                "loaded": self.swin_classifier is not None,
                "framework_installed": HAS_TORCH and HAS_TIMM,
                "load_error": self.load_errors["swin_transformer"],
            },
            "cnn_baseline": {
                "name": "CNN Baseline FER Classifier",
                "weights_path": str(CNN_WEIGHTS_PATH),
                "weights_exist": CNN_WEIGHTS_PATH.is_file(),
                "loaded": self.cnn_baseline is not None,
                "framework_installed": HAS_TORCH,
                "load_error": self.load_errors["cnn_baseline"],
            },
        }


registry = ModelRegistry()


# ---------------------------------------------------------------------------
# Modular Model Loaders
# ---------------------------------------------------------------------------
def load_yolov8_detector():
    """Loads YOLOv8 Face Detection model from weights/yolov8n-face.pt."""
    if not HAS_ULTRALYTICS or not YOLO_WEIGHTS_PATH.is_file():
        return None

    try:
        model = YOLO(str(YOLO_WEIGHTS_PATH))
        return model
    except Exception as e:
        print(f"[Warning] Failed to load YOLOv8 weights: {e}", file=sys.stderr)
        return None


def load_swin_classifier():
    """Loads Swin Transformer model from weights/swin_transformer_fer.pt."""
    if not HAS_TORCH or not HAS_TIMM or not SWIN_WEIGHTS_PATH.is_file():
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=7)
        state = torch.load(str(SWIN_WEIGHTS_PATH), map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[Warning] Failed to load Swin Transformer weights: {e}", file=sys.stderr)
        return None


def load_cnn_baseline():
    """Loads CNN Baseline model from weights/cnn_baseline_fer.pt."""
    if not HAS_TORCH or not CNN_WEIGHTS_PATH.is_file():
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PyTorchCNNBaseline()
        state = torch.load(str(CNN_WEIGHTS_PATH), map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[Warning] Failed to load CNN Baseline weights: {e}", file=sys.stderr)
        return None


@app.on_event("startup")
def startup_event():
    """Load all models into memory upon backend startup."""
    registry.yolo_detector = load_yolov8_detector()
    registry.swin_classifier = load_swin_classifier()
    registry.cnn_baseline = load_cnn_baseline()
    for key, model, weights_path, frameworks in (
        ("yolov8_face", registry.yolo_detector, YOLO_WEIGHTS_PATH, HAS_ULTRALYTICS),
        ("swin_transformer", registry.swin_classifier, SWIN_WEIGHTS_PATH, HAS_TORCH and HAS_TIMM),
        ("cnn_baseline", registry.cnn_baseline, CNN_WEIGHTS_PATH, HAS_TORCH),
    ):
        if model is None:
            if not frameworks:
                registry.load_errors[key] = "Required inference framework is not installed."
            elif not weights_path.is_file():
                registry.load_errors[key] = f"Weights not found: {weights_path.name}"
            else:
                registry.load_errors[key] = "Model initialization failed; inspect backend logs for details."
        else:
            registry.load_errors[key] = None
    print("[INFO] Model loading complete.")
    for k, v in registry.status().items():
        print(f"  {k}: loaded={v['loaded']} (weights_exist={v['weights_exist']})")


# ---------------------------------------------------------------------------
# Image Utility Functions
# ---------------------------------------------------------------------------
def decode_base64_image(image_str: Optional[str]) -> "Image.Image":
    """Decodes a base64 string or data URL into a PIL Image."""
    if not image_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image payload was provided.",
        )

    if not HAS_PIL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pillow is not installed in the backend environment.",
        )

    if "," in image_str:
        _, base64_data = image_str.split(",", 1)
    else:
        base64_data = image_str

    try:
        image_bytes = base64.b64decode(base64_data, validate=True)
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        img = Image.open(io.BytesIO(image_bytes))
        return img.convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 image data: {str(e)}",
        )


def detect_faces_in_image(image: "Image.Image") -> List[dict]:
    """Return every YOLO face box, sorted by detector score, in normalized coordinates."""
    if registry.yolo_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLOv8 face detector model is not loaded.",
        )

    width, height = image.size
    results = registry.yolo_detector.predict(source=image, conf=0.25, verbose=False)
    faces = []
    for result in results:
        if result.boxes is None:
            continue
        for detected in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in detected.xyxy[0].tolist()]
            confidence = float(detected.conf[0])
            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(x1, min(float(width), x2))
            y2 = max(y1, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            box = {
                "x": round(x1 / width, 6),
                "y": round(y1 / height, 6),
                "width": round((x2 - x1) / width, 6),
                "height": round((y2 - y1) / height, 6),
            }
            faces.append({"box": box, "detector_confidence": confidence, "xyxy": (x1, y1, x2, y2)})
    return sorted(faces, key=lambda face: face["detector_confidence"], reverse=True)


def crop_face_for_classification(image: "Image.Image", xyxy: tuple) -> "Image.Image":
    """Center a modest square context around the detected face, clipped to the image."""
    x1, y1, x2, y2 = xyxy
    face_width = x2 - x1
    face_height = y2 - y1
    side = max(face_width, face_height) * 1.15
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    left = max(0, int(round(center_x - side / 2)))
    top = max(0, int(round(center_y - side / 2)))
    right = min(image.width, int(round(center_x + side / 2)))
    bottom = min(image.height, int(round(center_y + side / 2)))
    return image.crop((left, top, right, bottom)).convert("RGB")


def classify_face_image(face_image: "Image.Image", model_name: str) -> PredictEmotionResponse:
    """Run one existing trained classifier with its existing model-specific preprocessing."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed in the backend environment.")

    from torchvision import transforms

    if model_name == "swin":
        model = registry.swin_classifier
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=registry.load_errors["swin_transformer"] or "Swin Transformer is not loaded.",
            )
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        tensor = transform(face_image).unsqueeze(0).to(next(model.parameters()).device)
        with torch.inference_mode():
            probs_raw = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        probabilities = {
            EMOTIONS[standard_index]: round(float(probs_raw[raw_index]) * 100, 1)
            for raw_index, standard_index in SWIN_ALPHA_TO_STD.items()
        }
        dominant_raw = int(probs_raw.argmax())
        return PredictEmotionResponse(
            emotion=EMOTIONS[SWIN_ALPHA_TO_STD[dominant_raw]],
            confidence=round(float(probs_raw[dominant_raw]), 3),
            probabilities=probabilities,
            source="backend",
            model_name="Swin Transformer",
        )

    if model_name == "cnn":
        model = registry.cnn_baseline
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=registry.load_errors["cnn_baseline"] or "CNN Baseline is not loaded.",
            )
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
            ]
        )
        tensor = transform(face_image).unsqueeze(0).to(next(model.parameters()).device)
        with torch.inference_mode():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        dominant = int(probs.argmax())
        return PredictEmotionResponse(
            emotion=EMOTIONS[dominant],
            confidence=round(float(probs[dominant]), 3),
            probabilities={name: round(float(probs[i]) * 100, 1) for i, name in enumerate(EMOTIONS)},
            source="backend",
            model_name="CNN Baseline",
        )

    raise ValueError(f"Unsupported classifier: {model_name}")


def analyze_image_faces(image: "Image.Image", model_name: str) -> List[dict]:
    detections = detect_faces_in_image(image)
    output = []
    selected_models = ("swin", "cnn") if model_name == "compare" else (model_name,)
    for detection in detections:
        face_crop = crop_face_for_classification(image, detection["xyxy"])
        predictions = {}
        errors = {}
        for selected in selected_models:
            try:
                predictions[selected] = classify_face_image(face_crop, selected).model_dump()
            except Exception as exc:
                errors[selected] = str(exc)
        output.append(
            {
                "box": detection["box"],
                "detector_confidence": round(detection["detector_confidence"], 4),
                "predictions": predictions,
                "model_errors": errors,
            }
        )
    return output


def analyze_image_faces_response(image: "Image.Image", model_name: str) -> AnalyzeFacesResponse:
    results = analyze_image_faces(image, model_name)
    faces = [
        FaceAnalysisResponse(
            face_id=f"Face {index + 1}",
            box=BoundingBox(**result["box"]),
            detector_confidence=result["detector_confidence"],
            predictions={key: PredictEmotionResponse(**value) for key, value in result["predictions"].items()},
            model_errors=result["model_errors"],
        )
        for index, result in enumerate(results)
    ]
    model_errors = {}
    for face in faces:
        for model, error in face.model_errors.items():
            model_errors.setdefault(model, error)
    return AnalyzeFacesResponse(model=model_name, face_count=len(faces), faces=faces, model_errors=model_errors)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "project": "Improved Facial Emotion Recognition Using Swin Transformer and YOLOv8 in Comparison with CNN",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health")
def health_check():
    """Reports status of YOLOv8 face detector, Swin Transformer, and CNN baseline."""
    torch_available = HAS_TORCH
    cuda_available = torch.cuda.is_available() if HAS_TORCH else False
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    models = registry.status()
    core_ready = models["yolov8_face"]["loaded"] and models["swin_transformer"]["loaded"]
    return {
        "status": "healthy" if core_ready else "degraded",
        "ready": core_ready,
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_installed": torch_available,
            "cuda_available": cuda_available,
            "compute_device": device_name,
            "pillow_installed": HAS_PIL,
            "ultralytics_installed": HAS_ULTRALYTICS,
            "timm_installed": HAS_TIMM,
        },
        "weights_directory": str(WEIGHTS_DIR),
        "models": models,
    }


@app.get("/api/evaluation-metrics")
def get_evaluation_metrics():
    """Returns actual measured evaluation results on the FER2013 test set for both models."""
    results = {}
    if SWIN_METRICS_PATH.is_file():
        with open(SWIN_METRICS_PATH, "r", encoding="utf-8") as f:
            results["swin_transformer"] = json.load(f)
    if CNN_METRICS_PATH.is_file():
        with open(CNN_METRICS_PATH, "r", encoding="utf-8") as f:
            results["cnn_baseline"] = json.load(f)
    return results


@app.post(
    "/api/detect-face",
    response_model=DetectFaceResponse,
    summary="YOLOv8 Face Detection",
)
def detect_face(req: DetectFaceRequest):
    """
    Real face detection using YOLOv8 (yolov8n-face.pt).
    Accepts: { "image": "data:image/jpeg;base64,..." }
    Returns: Normalized bounding box coordinates [0, 1] and face detection count.
    """
    image = decode_base64_image(req.image)

    try:
        detections = detect_faces_in_image(image)
        boxes_out = [BoundingBox(**face["box"]) for face in detections]
        highest_conf = max((face["detector_confidence"] for face in detections), default=0.0)

        detected = len(boxes_out) > 0
        return DetectFaceResponse(
            detected=detected,
            faceCount=len(boxes_out),
            confidence=round(highest_conf, 3) if detected else None,
            box=boxes_out[0] if detected else None,
            boxes=boxes_out,
            source="backend",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during YOLOv8 face detection: {str(e)}",
        )


@app.post(
    "/api/analyze-faces",
    response_model=AnalyzeFacesResponse,
    summary="Detect and classify every face in an image",
)
def analyze_faces(req: AnalyzeFacesRequest):
    image = decode_base64_image(req.image)
    try:
        return analyze_image_faces_response(image, req.model)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multiple-face analysis failed: {exc}",
        )


@app.post(
    "/api/predict-emotion",
    response_model=PredictEmotionResponse,
    summary="Swin Transformer Emotion Classification",
)
def predict_emotion(req: PredictEmotionRequest):
    """
    Real emotion classification using Swin Transformer fine-tuned on FER2013.
    Accepts: { "image": "data:image/jpeg;base64,..." } (cropped face patch)
    Returns: Dominant emotion label, confidence, and full 7-class distribution.
    """
    face_image = decode_base64_image(req.image)

    try:
        return classify_face_image(face_image, "swin")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during Swin Transformer classification: {str(e)}",
        )


@app.post(
    "/api/compare-models",
    response_model=ModelComparisonResponse,
    summary="Compare Swin Transformer vs. CNN Baseline",
)
def compare_models(req: PredictEmotionRequest):
    """
    Side-by-side comparison of the proposed Swin Transformer against the CNN Baseline
    on the exact same facial image patch.
    """
    swin_res = None
    cnn_res = None
    model_errors: Dict[str, str] = {}

    # 1. Swin Transformer prediction
    if registry.swin_classifier is not None:
        try:
            swin_res = predict_emotion(req)
        except HTTPException as exc:
            model_errors["swin_transformer"] = str(exc.detail)
        except Exception as exc:
            model_errors["swin_transformer"] = f"Inference failed: {exc}"
    else:
        model_errors["swin_transformer"] = registry.load_errors["swin_transformer"] or "Model is not loaded."

    # 2. CNN Baseline prediction
    if registry.cnn_baseline is not None and HAS_TORCH:
        try:
            face_image = decode_base64_image(req.image)
            from torchvision import transforms

            cnn_transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((48, 48)),
                    transforms.ToTensor(),
                ]
            )
            tensor_cnn = cnn_transform(face_image).unsqueeze(0)
            device = next(registry.cnn_baseline.parameters()).device
            tensor_cnn = tensor_cnn.to(device)

            with torch.no_grad():
                logits = registry.cnn_baseline(tensor_cnn)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            probabilities = {
                emotion: round(float(probs[i]) * 100, 1)
                for i, emotion in enumerate(EMOTIONS)
            }
            dom_idx = int(probs.argmax())
            cnn_res = PredictEmotionResponse(
                emotion=EMOTIONS[dom_idx],
                confidence=round(float(probs[dom_idx]), 3),
                probabilities=probabilities,
                source="backend",
                model_name="CNN Baseline",
            )
        except Exception as exc:
            model_errors["cnn_baseline"] = f"Inference failed: {exc}"
    else:
        model_errors["cnn_baseline"] = registry.load_errors["cnn_baseline"] or "Model is not loaded."

    if swin_res and cnn_res:
        if swin_res.emotion == cnn_res.emotion:
            summary = (
                f"Both models agreed on '{swin_res.emotion}' "
                f"(Swin: {int(swin_res.confidence * 100)}%, CNN: {int(cnn_res.confidence * 100)}%)."
            )
        else:
            summary = (
                f"Models differed: Swin Transformer predicted '{swin_res.emotion}' ({int(swin_res.confidence * 100)}%) "
                f"while CNN Baseline predicted '{cnn_res.emotion}' ({int(cnn_res.confidence * 100)}%)."
            )
    elif swin_res:
        summary = f"Swin Transformer predicted '{swin_res.emotion}' ({int(swin_res.confidence * 100)}%). CNN baseline unavailable."
    elif cnn_res:
        summary = f"CNN baseline predicted '{cnn_res.emotion}' ({int(cnn_res.confidence * 100)}%). Swin Transformer unavailable."
    else:
        summary = "Neither model is currently available for comparison."

    return ModelComparisonResponse(
        source="backend",
        swin_transformer=swin_res,
        cnn_baseline=cnn_res,
        model_errors=model_errors,
        comparison_summary=summary,
    )


MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = 600


@app.post("/api/video-jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_video_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: Literal["swin", "cnn", "compare"] = Form("swin"),
    interval_ms: int = Form(1000, ge=500, le=10000),
):
    """Upload a bounded video and analyze sampled frames in a background task."""
    if not HAS_CV2:
        raise HTTPException(status_code=503, detail="OpenCV video decoding is unavailable.")
    if registry.yolo_detector is None:
        raise HTTPException(status_code=503, detail="YOLOv8 face detector model is not loaded.")
    required_models = ("swin", "cnn") if model == "compare" else (model,)
    unavailable = [
        name for name in required_models
        if (name == "swin" and registry.swin_classifier is None)
        or (name == "cnn" and registry.cnn_baseline is None)
    ]
    if unavailable:
        raise HTTPException(status_code=503, detail=f"Requested model is not loaded: {', '.join(unavailable)}")

    filename = Path(file.filename or "video").name
    extension = Path(filename).suffix.lower()
    if extension not in {".mp4", ".mov", ".webm"}:
        raise HTTPException(status_code=415, detail="Supported video extensions are MP4, MOV and WebM.")

    temporary_path = None
    total_bytes = 0
    try:
        with tempfile.NamedTemporaryFile(prefix="fer-video-", suffix=extension, delete=False) as temporary:
            temporary_path = temporary.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the 200 MB upload limit.")
                temporary.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="The uploaded video is empty.")

        capture = cv2.VideoCapture(temporary_path)
        try:
            if not capture.isOpened():
                raise HTTPException(status_code=415, detail="This video codec could not be decoded by the backend.")
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
            ok, first_frame = capture.read()
            if duration <= 0 or not ok or first_frame is None:
                raise HTTPException(status_code=400, detail="The video has no readable frames or duration metadata.")
            if duration > MAX_VIDEO_DURATION_SECONDS:
                raise HTTPException(status_code=413, detail="Video duration exceeds the 10 minute limit.")
        finally:
            capture.release()

        job_id = uuid.uuid4().hex
        sample_total = max(1, int((duration * 1000 + interval_ms - 1) // interval_ms))
        video_jobs.create(
            job_id,
            {
                "job_id": job_id,
                "status": "queued",
                "file_name": filename,
                "file_size_bytes": total_bytes,
                "duration_seconds": round(duration, 3),
                "model": model,
                "interval_ms": interval_ms,
                "sampled_frames_total": sample_total,
                "frames_processed": 0,
                "face_detections": 0,
                "current_timestamp_seconds": None,
                "processing_seconds": None,
                "events": [],
                "frame_records": [],
                "reliability": None,
                "error": None,
            },
        )
        background_tasks.add_task(process_video_job, job_id, temporary_path, interval_ms, analyze_image_faces)
        temporary_path = None
        return {"job_id": job_id, "status": "queued", "duration_seconds": round(duration, 3), "sampled_frames_total": sample_total}
    finally:
        await file.close()
        if temporary_path:
            try:
                os.remove(temporary_path)
            except OSError:
                pass


@app.get("/api/video-jobs/{job_id}")
def get_video_job(job_id: str):
    job = video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video analysis job was not found or has expired.")
    return job


@app.delete("/api/video-jobs/{job_id}", status_code=status.HTTP_202_ACCEPTED)
def cancel_video_job(job_id: str):
    job = video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video analysis job was not found or has expired.")
    if not video_jobs.cancel(job_id):
        return {"job_id": job_id, "status": job["status"]}
    return {"job_id": job_id, "status": "cancelling"}
