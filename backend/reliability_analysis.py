"""Evidence summaries for video predictions; these are not emotion ground truth."""

import os
from typing import Any, Dict, List


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(high, max(low, value))


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(high, max(low, value))


def get_reliability_rules() -> Dict[str, Any]:
    """Return configurable screening thresholds; confidence is deliberately unused."""
    return {
        "minimum_usable_frames": _env_int("FER_RELIABILITY_MIN_USABLE_FRAMES", 3, 1, 100),
        "minimum_model_agreement_percent": _env_float("FER_RELIABILITY_MIN_AGREEMENT_PERCENT", 80.0, 0.0, 100.0),
        "minimum_temporal_consistency_percent": _env_float("FER_RELIABILITY_MIN_TEMPORAL_PERCENT", 80.0, 0.0, 100.0),
        "maximum_low_quality_frame_percent": _env_float("FER_RELIABILITY_MAX_LOW_QUALITY_PERCENT", 30.0, 0.0, 100.0),
        "minimum_face_crop_side_px": _env_int("FER_QUALITY_MIN_FACE_SIDE_PX", 48, 16, 512),
        "minimum_laplacian_variance": _env_float("FER_QUALITY_MIN_LAPLACIAN_VARIANCE", 20.0, 0.0, 10000.0),
        "minimum_brightness": _env_float("FER_QUALITY_MIN_BRIGHTNESS", 35.0, 0.0, 255.0),
        "maximum_brightness": _env_float("FER_QUALITY_MAX_BRIGHTNESS", 220.0, 0.0, 255.0),
        "minimum_detection_confidence": _env_float("FER_QUALITY_MIN_DETECTION_CONFIDENCE", 0.5, 0.0, 1.0),
        "threshold_note": "These configurable screening heuristics are not validated clinical or accuracy thresholds. Model confidence is not used as accuracy.",
    }


def measure_face_quality(image, box: Dict[str, float], detection_confidence: float) -> Dict[str, Any]:
    """Measure pixel-level face crop size, grayscale sharpness and brightness."""
    import cv2
    import numpy as np

    image_width, image_height = image.size
    x1 = max(0, min(image_width, int(box["x"] * image_width)))
    y1 = max(0, min(image_height, int(box["y"] * image_height)))
    x2 = max(x1, min(image_width, int((box["x"] + box["width"]) * image_width)))
    y2 = max(y1, min(image_height, int((box["y"] + box["height"]) * image_height)))
    crop = np.asarray(image.convert("RGB"))[y1:y2, x1:x2]
    crop_height, crop_width = crop.shape[:2]
    if crop_width == 0 or crop_height == 0:
        return {
            "face_visibility": None,
            "face_visibility_note": "Not available: no valid face crop was produced.",
            "face_crop_width_px": crop_width,
            "face_crop_height_px": crop_height,
            "blur_laplacian_variance": None,
            "brightness_mean": None,
            "detection_confidence": round(float(detection_confidence), 4),
            "quality_flags": ["invalid_face_crop"],
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    return {
        "face_visibility": None,
        "face_visibility_note": "Not available: the project has no occlusion or pose-based visibility estimator.",
        "face_crop_width_px": crop_width,
        "face_crop_height_px": crop_height,
        "blur_laplacian_variance": round(sharpness, 3),
        "brightness_mean": round(brightness, 3),
        "detection_confidence": round(float(detection_confidence), 4),
    }


def _quality_flags(quality: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
    flags = []
    width = quality.get("face_crop_width_px")
    height = quality.get("face_crop_height_px")
    if width is None or height is None or min(width, height) < rules["minimum_face_crop_side_px"]:
        flags.append("face_crop_too_small")
    blur = quality.get("blur_laplacian_variance")
    if blur is None or blur < rules["minimum_laplacian_variance"]:
        flags.append("possible_blur")
    brightness = quality.get("brightness_mean")
    if brightness is None or not rules["minimum_brightness"] <= brightness <= rules["maximum_brightness"]:
        flags.append("brightness_outside_screening_range")
    score = quality.get("detection_confidence")
    if score is None or score < rules["minimum_detection_confidence"]:
        flags.append("low_detection_confidence")
    return flags


def analyze_reliability(
    events: List[Dict[str, Any]], frame_records: List[Dict[str, Any]], model: str,
) -> Dict[str, Any]:
    """Summarize paired model agreement, per-track changes and measurable quality."""
    rules = get_reliability_rules()
    enriched_events = []
    for event in events:
        row = dict(event)
        quality = dict(row.get("quality") or {})
        quality["quality_flags"] = _quality_flags(quality, rules)
        row["quality"] = quality
        enriched_events.append(row)

    comparable = 0
    agreeing = 0
    disagreeing = 0
    unavailable = 0
    disagreement_timeline = []
    for event in enriched_events:
        swin = (event.get("predictions") or {}).get("swin")
        cnn = (event.get("predictions") or {}).get("cnn")
        if swin and cnn:
            comparable += 1
            is_agreement = swin.get("emotion") == cnn.get("emotion")
            if is_agreement:
                agreeing += 1
            else:
                disagreeing += 1
        else:
            unavailable += 1
            is_agreement = None
        disagreement_timeline.append({
            "sample_index": event["sample_index"],
            "timestamp_seconds": event["timestamp_seconds"],
            "face_id": event["face_id"],
            "cnn_emotion": cnn.get("emotion") if cnn else None,
            "swin_emotion": swin.get("emotion") if swin else None,
            "agree": is_agreement,
        })

    primary_model = "swin" if model in ("swin", "compare") else "cnn"
    by_track: Dict[str, List[Dict[str, Any]]] = {}
    for event in enriched_events:
        prediction = (event.get("predictions") or {}).get(primary_model)
        if prediction and prediction.get("emotion"):
            by_track.setdefault(event["face_id"], []).append(event)

    expression_counts: Dict[str, int] = {}
    expression_changes = 0
    adjacent_pairs = 0
    stable_sequences = []
    unstable_sequences = []
    for face_id, track_events in by_track.items():
        track_events.sort(key=lambda item: item["sample_index"])
        sequence = [(event["sample_index"], event["timestamp_seconds"], event["predictions"][primary_model]["emotion"]) for event in track_events]
        for _, _, emotion in sequence:
            expression_counts[emotion] = expression_counts.get(emotion, 0) + 1
        for left, right in zip(sequence, sequence[1:]):
            if right[0] != left[0] + 1:
                continue
            adjacent_pairs += 1
            if left[2] != right[2]:
                expression_changes += 1

        run_start = 0
        while run_start < len(sequence):
            run_end = run_start + 1
            while (
                run_end < len(sequence)
                and sequence[run_end][0] == sequence[run_end - 1][0] + 1
                and sequence[run_end][2] == sequence[run_start][2]
            ):
                run_end += 1
            run = sequence[run_start:run_end]
            summary = {
                "face_id": face_id,
                "emotion": run[0][2],
                "start_sample_index": run[0][0],
                "end_sample_index": run[-1][0],
                "start_timestamp_seconds": run[0][1],
                "end_timestamp_seconds": run[-1][1],
                "frames": len(run),
            }
            (stable_sequences if len(run) >= 2 else unstable_sequences).append(summary)
            run_start = run_end

    temporal_consistency = (
        100.0 * (adjacent_pairs - expression_changes) / adjacent_pairs if adjacent_pairs else None
    )
    most_frequent = (
        max(expression_counts, key=expression_counts.get) if expression_counts else None
    )

    frame_quality = {}
    for event in enriched_events:
        frame_quality.setdefault(event["sample_index"], []).append(event["quality"])
    insufficient_frame_indexes = set()
    for frame in frame_records:
        qualities = frame_quality.get(frame["sample_index"], [])
        if frame.get("face_count", 0) == 0 or not qualities or all(q["quality_flags"] for q in qualities):
            insufficient_frame_indexes.add(frame["sample_index"])
    low_quality_count = len(insufficient_frame_indexes)

    usable_frame_indexes = {
        event["sample_index"]
        for event in enriched_events
        if not event["quality"]["quality_flags"]
        and ((event.get("predictions") or {}).get("swin") or (event.get("predictions") or {}).get("cnn"))
    }
    usable_frames = len(usable_frame_indexes)
    frame_total = len(frame_records)
    agreement_percent = 100.0 * agreeing / comparable if comparable else None
    low_quality_percent = 100.0 * low_quality_count / frame_total if frame_total else 100.0

    if usable_frames < rules["minimum_usable_frames"]:
        status = "insufficient_visual_evidence"
        reasons = [f"Only {usable_frames} usable sampled frames; {rules['minimum_usable_frames']} are required."]
    else:
        reasons = []
        if comparable == 0:
            reasons.append("CNN–Swin agreement is unavailable because no paired predictions were returned.")
        elif comparable < rules["minimum_usable_frames"]:
            reasons.append(
                f"Only {comparable} paired CNN–Swin frames; {rules['minimum_usable_frames']} are required for a reliable decision."
            )
        elif agreement_percent < rules["minimum_model_agreement_percent"]:
            reasons.append("CNN and Swin disagree more often than the configured rule allows.")
        if temporal_consistency is None:
            reasons.append("Temporal consistency is unavailable because no consecutive predictions exist for a face track.")
        elif temporal_consistency < rules["minimum_temporal_consistency_percent"]:
            reasons.append("Temporal consistency is below the configured rule.")
        if low_quality_percent > rules["maximum_low_quality_frame_percent"]:
            reasons.append("The proportion of low-quality or face-missing sampled frames exceeds the configured rule.")
        status = "reliable" if not reasons else "uncertain"

    quality_rows = []
    for event in enriched_events:
        q = event["quality"]
        quality_rows.append({
            "sample_index": event["sample_index"],
            "timestamp_seconds": event["timestamp_seconds"],
            "face_id": event["face_id"],
            "face_visibility": q.get("face_visibility"),
            "face_visibility_note": q.get("face_visibility_note", "Not available."),
            "face_crop_width_px": q.get("face_crop_width_px"),
            "face_crop_height_px": q.get("face_crop_height_px"),
            "blur_laplacian_variance": q.get("blur_laplacian_variance"),
            "brightness_mean": q.get("brightness_mean"),
            "detection_confidence": q.get("detection_confidence"),
            "quality_flags": q["quality_flags"],
            "measurement_error": q.get("measurement_error"),
        })

    return {
        "status": status,
        "reasons": reasons,
        "model_agreement": {
            "agreeing_frames": agreeing,
            "disagreeing_frames": disagreeing,
            "paired_prediction_frames": comparable,
            "unavailable_prediction_frames": unavailable,
            "agreement_percent": round(agreement_percent, 2) if agreement_percent is not None else None,
            "timeline": disagreement_timeline,
        },
        "temporal_stability": {
            "primary_model": primary_model,
            "most_frequent_expression": most_frequent,
            "expression_counts": expression_counts,
            "expression_changes": expression_changes,
            "consecutive_prediction_pairs": adjacent_pairs,
            "stable_sequences": stable_sequences,
            "unstable_sequences": unstable_sequences,
            "temporal_consistency_percent": round(temporal_consistency, 2) if temporal_consistency is not None else None,
            "note": "Repeated model outputs describe prediction consistency only, not a person's true emotional state.",
        },
        "face_quality": {
            "insufficient_quality_frames": low_quality_count,
            "frame_count": frame_total,
            "low_quality_frame_percent": round(low_quality_percent, 2),
            "usable_frames": usable_frames,
            "measurements": quality_rows,
            "no_face_frames": [frame for frame in frame_records if frame.get("face_count", 0) == 0],
            "visibility_note": "Occlusion/pose visibility is not available without a dedicated estimator.",
        },
        "rules": rules,
        "confidence_note": "Classifier confidence is not treated as prediction accuracy or reliability.",
    }
