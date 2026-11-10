"""In-process asynchronous video analysis jobs with sampled-frame inference."""

import copy
import os
import threading
import time
from typing import Callable, Dict, Optional

import cv2
from PIL import Image

try:
    from .tracking import IoUFaceTracker
    from .reliability_analysis import analyze_reliability, measure_face_quality
except ImportError:
    from tracking import IoUFaceTracker
    from reliability_analysis import analyze_reliability, measure_face_quality


class VideoJobManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: Dict[str, dict] = {}
        self._cancel: Dict[str, threading.Event] = {}

    def create(self, job_id: str, job: dict) -> None:
        with self._lock:
            self._jobs[job_id] = job
            self._cancel[job_id] = threading.Event()

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job is not None else None

    def get_model(self, job_id: str) -> Optional[str]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.get("model") if job is not None else None

    def update(self, job_id: str, **changes) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(changes)

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            event = self._cancel.get(job_id)
            return event is None or event.is_set()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            event = self._cancel.get(job_id)
            if job is None or event is None:
                return False
            if job["status"] not in ("queued", "processing"):
                return False
            event.set()
            self._jobs[job_id]["status"] = "cancelling"
            return True


video_jobs = VideoJobManager()


def process_video_job(
    job_id: str,
    path: str,
    interval_ms: int,
    analyze_frame: Callable[[Image.Image, str], list],
) -> None:
    started = time.perf_counter()
    capture = None
    try:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError("The uploaded video could not be decoded by the installed OpenCV codecs.")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
        if duration <= 0:
            raise ValueError("The video has no readable duration or frames.")

        sample_count = max(1, int((duration * 1000 + interval_ms - 1) // interval_ms))
        video_jobs.update(
            job_id,
            status="processing",
            duration_seconds=round(duration, 3),
            sampled_frames_total=sample_count,
        )
        tracker = IoUFaceTracker(max_missed=2)
        events = []
        frame_records = []
        analyzed = 0
        face_detections = 0

        for sample_index in range(sample_count):
            if video_jobs.is_cancelled(job_id):
                video_jobs.update(job_id, status="cancelled", processing_seconds=round(time.perf_counter() - started, 3))
                return

            target_ms = sample_index * interval_ms
            capture.set(cv2.CAP_PROP_POS_MSEC, target_ms)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            actual_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or target_ms)
            timestamp = max(0.0, min(duration, actual_ms / 1000.0))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).convert("RGB")
            model_name = video_jobs.get_model(job_id)
            if model_name is None:
                raise ValueError("Video analysis job no longer exists.")
            faces = analyze_frame(image, model_name)
            ids = tracker.update(face["box"] for face in faces)
            frame_records.append({
                "sample_index": sample_index,
                "timestamp_seconds": round(timestamp, 3),
                "face_count": len(faces),
            })

            for face, face_id in zip(faces, ids):
                try:
                    quality = measure_face_quality(image, face["box"], face["detector_confidence"])
                except Exception:
                    quality = {
                        "face_visibility": None,
                        "face_visibility_note": "Not available: no visibility estimator is configured.",
                        "face_crop_width_px": None,
                        "face_crop_height_px": None,
                        "blur_laplacian_variance": None,
                        "brightness_mean": None,
                        "detection_confidence": face.get("detector_confidence"),
                        "measurement_error": "Not available: image-quality measurement failed.",
                    }
                events.append(
                    {
                        "timestamp_seconds": round(timestamp, 3),
                        "sample_index": sample_index,
                        "face_id": face_id,
                        "box": face["box"],
                        "detector_confidence": face["detector_confidence"],
                        "predictions": face["predictions"],
                        "quality": quality,
                    }
                )

            analyzed += 1
            face_detections += len(faces)
            video_jobs.update(
                job_id,
                frames_processed=analyzed,
                face_detections=face_detections,
                current_timestamp_seconds=round(timestamp, 3),
                events=events,
                frame_records=frame_records,
            )

        if video_jobs.is_cancelled(job_id):
            video_jobs.update(job_id, status="cancelled", processing_seconds=round(time.perf_counter() - started, 3))
        elif analyzed == 0:
            raise ValueError("No frames could be extracted from this video.")
        else:
            reliability = analyze_reliability(events, frame_records, video_jobs.get_model(job_id) or "swin")
            video_jobs.update(
                job_id,
                status="completed",
                frames_processed=analyzed,
                face_detections=face_detections,
                events=events,
                frame_records=frame_records,
                reliability=reliability,
                processing_seconds=round(time.perf_counter() - started, 3),
                current_timestamp_seconds=None,
            )
    except Exception as exc:
        video_jobs.update(
            job_id,
            status="failed",
            error=str(exc),
            processing_seconds=round(time.perf_counter() - started, 3),
        )
    finally:
        if capture is not None:
            capture.release()
        try:
            os.remove(path)
        except OSError:
            pass
