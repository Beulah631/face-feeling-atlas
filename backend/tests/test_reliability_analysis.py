import sys
import unittest
from pathlib import Path

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from reliability_analysis import analyze_reliability


def quality(**overrides):
    result = {
        "face_visibility": None,
        "face_visibility_note": "Not available in this test.",
        "face_crop_width_px": 80,
        "face_crop_height_px": 80,
        "blur_laplacian_variance": 90.0,
        "brightness_mean": 120.0,
        "detection_confidence": 0.9,
    }
    result.update(overrides)
    return result


def event(index, swin="Happy", cnn="Happy", quality_data=None):
    predictions = {}
    if swin is not None:
        predictions["swin"] = {"emotion": swin, "confidence": 0.72}
    if cnn is not None:
        predictions["cnn"] = {"emotion": cnn, "confidence": 0.68}
    return {
        "sample_index": index,
        "timestamp_seconds": float(index),
        "face_id": "Face 1",
        "predictions": predictions,
        "quality": quality_data or quality(),
    }


def frames(count):
    return [{"sample_index": i, "timestamp_seconds": float(i), "face_count": 1} for i in range(count)]


class ReliabilityAnalysisTests(unittest.TestCase):
    def test_model_agreement_and_disagreement_are_counted_from_labels(self):
        report = analyze_reliability(
            [event(0), event(1, swin="Sad", cnn="Sad"), event(2, swin="Angry", cnn="Fear")],
            frames(3), "compare",
        )
        agreement = report["model_agreement"]
        self.assertEqual(agreement["agreeing_frames"], 2)
        self.assertEqual(agreement["disagreeing_frames"], 1)
        self.assertAlmostEqual(agreement["agreement_percent"], 66.67, places=2)
        self.assertEqual(agreement["timeline"][2]["agree"], False)

    def test_missing_model_prediction_is_unavailable_not_invented(self):
        report = analyze_reliability([event(0, cnn=None)], frames(1), "compare")
        agreement = report["model_agreement"]
        self.assertEqual(agreement["paired_prediction_frames"], 0)
        self.assertEqual(agreement["unavailable_prediction_frames"], 1)
        self.assertIsNone(agreement["timeline"][0]["cnn_emotion"])
        self.assertIsNone(agreement["agreement_percent"])

    def test_stable_and_changing_predictions_are_summarized(self):
        stable = analyze_reliability([event(0), event(1), event(2)], frames(3), "compare")
        temporal = stable["temporal_stability"]
        self.assertEqual(temporal["most_frequent_expression"], "Happy")
        self.assertEqual(temporal["expression_changes"], 0)
        self.assertEqual(temporal["temporal_consistency_percent"], 100.0)
        self.assertEqual(temporal["stable_sequences"][0]["frames"], 3)

        changing = analyze_reliability(
            [event(0, swin="Happy"), event(1, swin="Sad"), event(2, swin="Happy")], frames(3), "swin"
        )
        self.assertEqual(changing["temporal_stability"]["expression_changes"], 2)
        self.assertEqual(changing["temporal_stability"]["temporal_consistency_percent"], 0.0)

    def test_poor_face_quality_causes_insufficient_visual_evidence(self):
        poor = quality(face_crop_width_px=20, face_crop_height_px=18, blur_laplacian_variance=2, brightness_mean=8)
        report = analyze_reliability([event(0, quality_data=poor)], frames(1), "compare")
        self.assertEqual(report["status"], "insufficient_visual_evidence")
        self.assertIn("face_crop_too_small", report["face_quality"]["measurements"][0]["quality_flags"])
        self.assertIsNone(report["face_quality"]["measurements"][0]["face_visibility"])

    def test_empty_result_returns_insufficient_visual_evidence(self):
        report = analyze_reliability([], [], "compare")
        self.assertEqual(report["status"], "insufficient_visual_evidence")
        self.assertIsNone(report["model_agreement"]["agreement_percent"])
        self.assertIsNone(report["temporal_stability"]["most_frequent_expression"])

    def test_no_face_sample_counts_as_insufficient_quality(self):
        report = analyze_reliability([], [{"sample_index": 0, "timestamp_seconds": 0.0, "face_count": 0}], "swin")
        self.assertEqual(report["face_quality"]["insufficient_quality_frames"], 1)
        self.assertEqual(len(report["face_quality"]["no_face_frames"]), 1)

    def test_temporal_comparison_does_not_bridge_missing_sampled_frames(self):
        spaced_events = [event(0), event(2), event(4)]
        sampled = [
            {"sample_index": i, "timestamp_seconds": float(i), "face_count": int(i in (0, 2, 4))}
            for i in range(5)
        ]
        temporal = analyze_reliability(spaced_events, sampled, "swin")["temporal_stability"]
        self.assertEqual(temporal["consecutive_prediction_pairs"], 0)
        self.assertIsNone(temporal["temporal_consistency_percent"])
        self.assertEqual(temporal["stable_sequences"], [])


if __name__ == "__main__":
    unittest.main()
