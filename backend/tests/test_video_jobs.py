import os
import tempfile
import unittest
import uuid
from pathlib import Path

import cv2
import numpy as np

from video_jobs import process_video_job, video_jobs


class VideoJobTests(unittest.TestCase):
    def make_video(self):
        handle = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
        path = handle.name
        handle.close()
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (64, 48))
        if not writer.isOpened():
            os.remove(path)
            self.skipTest("OpenCV MJPG video writer is unavailable")
        for frame_index in range(4):
            writer.write(np.full((48, 64, 3), frame_index * 30, dtype=np.uint8))
        writer.release()
        return path

    def create_job(self, job_id):
        video_jobs.create(
            job_id,
            {
                "job_id": job_id,
                "status": "queued",
                "model": "swin",
                "events": [],
                "frames_processed": 0,
                "face_detections": 0,
            },
        )

    def test_video_is_sampled_and_tracking_id_persists(self):
        path = self.make_video()
        job_id = uuid.uuid4().hex
        self.create_job(job_id)
        face = {
            "box": {"x": 0.2, "y": 0.2, "width": 0.3, "height": 0.3},
            "detector_confidence": 0.9,
            "predictions": {"swin": {"emotion": "Happy", "confidence": 0.8}},
        }
        process_video_job(job_id, path, 500, lambda _image, _model: [face])
        result = video_jobs.get(job_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["frames_processed"], 4)
        self.assertEqual(result["face_detections"], 4)
        self.assertEqual(result["reliability"]["model_agreement"]["paired_prediction_frames"], 0)
        self.assertEqual(result["reliability"]["model_agreement"]["unavailable_prediction_frames"], 4)
        self.assertEqual({event["face_id"] for event in result["events"]}, {"Face 1"})
        self.assertTrue(all(event["timestamp_seconds"] >= 0 for event in result["events"]))
        self.assertFalse(os.path.exists(path), "temporary video must be removed")

    def test_no_face_frames_complete_without_fake_events(self):
        path = self.make_video()
        job_id = uuid.uuid4().hex
        self.create_job(job_id)
        process_video_job(job_id, path, 1000, lambda _image, _model: [])
        result = video_jobs.get(job_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["face_detections"], 0)
        self.assertEqual(result["events"], [])

    def test_model_failure_marks_job_failed_and_cleans_upload(self):
        path = self.make_video()
        job_id = uuid.uuid4().hex
        self.create_job(job_id)

        def fail(_image, _model):
            raise RuntimeError("inference failed")

        process_video_job(job_id, path, 500, fail)
        result = video_jobs.get(job_id)
        self.assertEqual(result["status"], "failed")
        self.assertIn("inference failed", result["error"])
        self.assertFalse(os.path.exists(path), "temporary video must be removed after failure")


if __name__ == "__main__":
    unittest.main()
