import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app


class MultiFaceAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (320, 240), color=(90, 90, 90))
        self.detections = [
            {"box": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}, "detector_confidence": 0.91, "xyxy": (32, 24, 96, 72)},
            {"box": {"x": 0.6, "y": 0.15, "width": 0.18, "height": 0.18}, "detector_confidence": 0.84, "xyxy": (192, 36, 250, 79)},
        ]

    def test_all_detected_faces_are_classified_independently(self):
        predicted = app.PredictEmotionResponse(
            emotion="Happy",
            confidence=0.73,
            probabilities={name: 0.0 for name in app.EMOTIONS},
            source="backend",
            model_name="Swin Transformer",
        )
        with patch.object(app, "detect_faces_in_image", return_value=self.detections), patch.object(
            app, "classify_face_image", return_value=predicted
        ) as classify:
            result = app.analyze_image_faces_response(self.image, "swin")
        self.assertEqual(result.face_count, 2)
        self.assertEqual([face.face_id for face in result.faces], ["Face 1", "Face 2"])
        self.assertEqual(classify.call_count, 2)

    def test_no_faces_returns_empty_results_without_classifier_call(self):
        with patch.object(app, "detect_faces_in_image", return_value=[]), patch.object(
            app, "classify_face_image"
        ) as classify:
            result = app.analyze_image_faces_response(self.image, "swin")
        self.assertEqual(result.face_count, 0)
        self.assertEqual(result.faces, [])
        classify.assert_not_called()

    def test_model_failure_is_reported_for_face(self):
        with patch.object(app, "detect_faces_in_image", return_value=self.detections[:1]), patch.object(
            app, "classify_face_image", side_effect=RuntimeError("test model failure")
        ):
            result = app.analyze_image_faces_response(self.image, "swin")
        self.assertEqual(result.face_count, 1)
        self.assertEqual(result.faces[0].predictions, {})
        self.assertIn("test model failure", result.faces[0].model_errors["swin"])


if __name__ == "__main__":
    unittest.main()
