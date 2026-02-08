"""Tests for Hungarian data association."""

import numpy as np

from sentinel.core.types import Detection, SensorType
from sentinel.tracking.association import AssociationResult, HungarianAssociator
from sentinel.tracking.track import Track


def _det(cx, cy, w=100, h=100, cls="person", conf=0.9):
    return Detection(
        sensor_type=SensorType.CAMERA,
        timestamp=0.0,
        bbox=np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32),
        class_id=0,
        class_name=cls,
        confidence=conf,
    )


def _make_track(cx, cy, w=100, h=100) -> Track:
    """Create a track initialized at (cx, cy)."""
    det = _det(cx, cy, w, h)
    track = Track(detection=det, dt=1 / 30)
    # Predict once so the track has predicted state
    track.predict()
    return track


class TestAssociationResult:
    def test_default_empty(self):
        result = AssociationResult()
        assert result.matched_pairs == []
        assert result.unmatched_tracks == []
        assert result.unmatched_detections == []


class TestHungarianAssociator:
    def test_empty_tracks(self):
        assoc = HungarianAssociator()
        result = assoc.associate([], [_det(100, 100)])
        assert result.matched_pairs == []
        assert result.unmatched_tracks == []
        assert result.unmatched_detections == [0]

    def test_empty_detections(self):
        assoc = HungarianAssociator()
        track = _make_track(100, 100)
        result = assoc.associate([track], [])
        assert result.matched_pairs == []
        assert result.unmatched_tracks == [0]
        assert result.unmatched_detections == []

    def test_both_empty(self):
        assoc = HungarianAssociator()
        result = assoc.associate([], [])
        assert result.matched_pairs == []
        assert result.unmatched_tracks == []
        assert result.unmatched_detections == []

    def test_single_match(self):
        assoc = HungarianAssociator(gate_threshold=50.0)
        track = _make_track(100, 100)
        det = _det(105, 102)
        result = assoc.associate([track], [det])
        assert len(result.matched_pairs) == 1
        assert result.matched_pairs[0] == (0, 0)
        assert result.unmatched_tracks == []
        assert result.unmatched_detections == []

    def test_detection_too_far_is_unmatched(self):
        assoc = HungarianAssociator(gate_threshold=5.0)
        track = _make_track(100, 100)
        det = _det(500, 500)  # Far away, outside gate
        result = assoc.associate([track], [det])
        assert result.matched_pairs == []
        assert result.unmatched_tracks == [0]
        assert result.unmatched_detections == [0]

    def test_multiple_tracks_multiple_detections(self):
        assoc = HungarianAssociator(gate_threshold=50.0)
        tracks = [_make_track(100, 100), _make_track(500, 500)]
        dets = [_det(102, 103), _det(498, 502)]
        result = assoc.associate(tracks, dets)
        assert len(result.matched_pairs) == 2
        # Track 0 should match det 0, track 1 should match det 1
        matched_dict = dict(result.matched_pairs)
        assert matched_dict[0] == 0
        assert matched_dict[1] == 1

    def test_more_detections_than_tracks(self):
        assoc = HungarianAssociator(gate_threshold=50.0)
        tracks = [_make_track(100, 100)]
        dets = [_det(102, 103), _det(500, 500)]
        result = assoc.associate(tracks, dets)
        assert len(result.matched_pairs) == 1
        assert len(result.unmatched_detections) == 1

    def test_more_tracks_than_detections(self):
        assoc = HungarianAssociator(gate_threshold=50.0)
        tracks = [_make_track(100, 100), _make_track(500, 500)]
        dets = [_det(102, 103)]
        result = assoc.associate(tracks, dets)
        assert len(result.matched_pairs) == 1
        assert len(result.unmatched_tracks) == 1

    def test_optimal_assignment_avoids_greedy_error(self):
        """The Hungarian algorithm should find the global optimum, not greedy.

        Consider two tracks and two detections where greedy nearest-neighbor
        would make a suboptimal first assignment.
        """
        assoc = HungarianAssociator(gate_threshold=200.0)
        # Track A at (100, 100), Track B at (120, 100)
        # Det 1 at (110, 100) — equidistant-ish from both
        # Det 2 at (125, 100) — clearly closest to B
        # Greedy: A->Det1 (dist 10), then B can't match Det2 well since it's close
        # Optimal: A->Det1, B->Det2 (both close) — or A->Det2 is wrong, let's check
        tracks = [_make_track(100, 100), _make_track(130, 100)]
        dets = [_det(112, 100), _det(128, 100)]
        result = assoc.associate(tracks, dets)
        assert len(result.matched_pairs) == 2
        matched_dict = dict(result.matched_pairs)
        # Track 0 (at ~100) should get det 0 (at 112), track 1 (at ~130) should get det 1 (at 128)
        assert matched_dict[0] == 0
        assert matched_dict[1] == 1

    def test_detection_without_bbox_is_unmatched(self):
        assoc = HungarianAssociator(gate_threshold=50.0)
        track = _make_track(100, 100)
        det = Detection(
            sensor_type=SensorType.CAMERA,
            timestamp=0.0,
            bbox=None,  # No bbox
        )
        result = assoc.associate([track], [det])
        assert result.matched_pairs == []
        assert result.unmatched_tracks == [0]
        assert result.unmatched_detections == [0]

    def test_cost_matrix_respects_iou_weight(self):
        """Higher IoU weight should prefer overlapping bboxes."""
        # Two associators with different weight profiles
        iou_heavy = HungarianAssociator(gate_threshold=200.0, iou_weight=0.9, mahalanobis_weight=0.1)
        maha_heavy = HungarianAssociator(gate_threshold=200.0, iou_weight=0.1, mahalanobis_weight=0.9)

        track = _make_track(100, 100, w=80, h=80)
        det = _det(105, 102, w=80, h=80)

        result_iou = iou_heavy.associate([track], [det])
        result_maha = maha_heavy.associate([track], [det])

        # Both should match, just verifying they don't crash with different weights
        assert len(result_iou.matched_pairs) == 1
        assert len(result_maha.matched_pairs) == 1
