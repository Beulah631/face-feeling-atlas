import unittest

from tracking import IoUFaceTracker


class FaceTrackerTests(unittest.TestCase):
    def test_empty_frame_has_no_faces(self):
        tracker = IoUFaceTracker()
        self.assertEqual(tracker.update([]), [])

    def test_nearby_boxes_keep_temporary_id(self):
        tracker = IoUFaceTracker()
        first = {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}
        moved = {"x": 0.12, "y": 0.11, "width": 0.2, "height": 0.2}
        self.assertEqual(tracker.update([first]), ["Face 1"])
        self.assertEqual(tracker.update([moved]), ["Face 1"])

    def test_unrelated_box_gets_new_id_after_gap(self):
        tracker = IoUFaceTracker(max_missed=1)
        first = {"x": 0.05, "y": 0.05, "width": 0.15, "height": 0.15}
        unrelated = {"x": 0.75, "y": 0.75, "width": 0.15, "height": 0.15}
        self.assertEqual(tracker.update([first]), ["Face 1"])
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.update([]), [])
        self.assertEqual(tracker.update([unrelated]), ["Face 2"])


if __name__ == "__main__":
    unittest.main()
