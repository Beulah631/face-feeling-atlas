"""Temporary, frame-local face tracking based only on bounding-box geometry."""

from dataclasses import dataclass
from math import hypot
from typing import Dict, Iterable, List, Tuple


Box = Dict[str, float]


@dataclass
class Track:
    track_id: str
    box: Box
    missed: int = 0


class IoUFaceTracker:
    """Greedy one-to-one box association; does not identify people across sessions."""

    def __init__(self, max_missed: int = 2, min_iou: float = 0.08, max_center_distance: float = 0.16):
        self.max_missed = max_missed
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self._tracks: Dict[str, Track] = {}
        self._next_id = 1

    @staticmethod
    def _center(box: Box) -> Tuple[float, float]:
        return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        left = max(a["x"], b["x"])
        top = max(a["y"], b["y"])
        right = min(a["x"] + a["width"], b["x"] + b["width"])
        bottom = min(a["y"] + a["height"], b["y"] + b["height"])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        area_a = max(0.0, a["width"]) * max(0.0, a["height"])
        area_b = max(0.0, b["width"]) * max(0.0, b["height"])
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0

    def update(self, boxes: Iterable[Box]) -> List[str]:
        current = list(boxes)
        candidates = []
        for index, box in enumerate(current):
            center = self._center(box)
            for track_id, track in self._tracks.items():
                iou = self._iou(box, track.box)
                old_center = self._center(track.box)
                distance = hypot(center[0] - old_center[0], center[1] - old_center[1])
                old_area = max(track.box["width"] * track.box["height"], 1e-6)
                new_area = max(box["width"] * box["height"], 1e-6)
                scale_ratio = new_area / old_area
                if (
                    (iou >= self.min_iou or distance <= self.max_center_distance)
                    and 0.25 <= scale_ratio <= 4.0
                ):
                    candidates.append((iou - distance * 0.1, index, track_id))

        matched_boxes = set()
        matched_tracks = set()
        assignments: List[str | None] = [None] * len(current)
        for _, index, track_id in sorted(candidates, reverse=True):
            if index in matched_boxes or track_id in matched_tracks:
                continue
            matched_boxes.add(index)
            matched_tracks.add(track_id)
            assignments[index] = track_id

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_tracks:
                track.missed += 1
                if track.missed > self.max_missed:
                    del self._tracks[track_id]

        for index, box in enumerate(current):
            track_id = assignments[index]
            if track_id is None:
                track_id = f"Face {self._next_id}"
                self._next_id += 1
                assignments[index] = track_id
                self._tracks[track_id] = Track(track_id, box)
            else:
                self._tracks[track_id] = Track(track_id, box)

        return [track_id for track_id in assignments if track_id is not None]
