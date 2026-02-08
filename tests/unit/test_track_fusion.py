"""Tests for track-level fusion."""

import numpy as np
import pytest

from sentinel.core.types import Detection, SensorType
from sentinel.fusion.track_fusion import FusedTrack, TrackFusion
from sentinel.tracking.radar_track import RadarTrack
from sentinel.tracking.track import Track


def _cam_det(cx, cy, w=100, h=100, cls="person", conf=0.9):
    return Detection(
        sensor_type=SensorType.CAMERA,
        timestamp=0.0,
        bbox=np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32),
        class_id=0,
        class_name=cls,
        confidence=conf,
    )


def _radar_det(range_m=3000.0, azimuth_deg=0.0):
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=0.0,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        velocity_mps=0.0,
        rcs_dbsm=15.0,
    )


def _make_cam_track(cx, cy) -> Track:
    """Create a confirmed camera track at pixel (cx, cy)."""
    det = _cam_det(cx, cy)
    track = Track(detection=det, dt=1 / 30, confirm_hits=1)
    track.predict()
    track.update(det)
    return track


def _make_radar_track(range_m, azimuth_deg) -> RadarTrack:
    """Create a confirmed radar track."""
    det = _radar_det(range_m, azimuth_deg)
    track = RadarTrack(detection=det, dt=0.1, confirm_hits=1)
    track.predict()
    track.update(det)
    return track


class TestFusedTrack:
    def test_dual_sensor_flag(self):
        cam = _make_cam_track(640, 360)
        rdr = _make_radar_track(3000, 0.0)
        ft = FusedTrack(
            fused_id="TEST",
            camera_track=cam,
            radar_track=rdr,
            sensor_sources={SensorType.CAMERA, SensorType.RADAR},
        )
        assert ft.is_dual_sensor

    def test_camera_only(self):
        cam = _make_cam_track(640, 360)
        ft = FusedTrack(fused_id="TEST", camera_track=cam)
        assert not ft.is_dual_sensor

    def test_to_dict(self):
        ft = FusedTrack(
            fused_id="TEST",
            range_m=3000.0,
            azimuth_deg=15.0,
            sensor_sources={SensorType.RADAR},
            fusion_quality=0.75,
        )
        d = ft.to_dict()
        assert d["fused_id"] == "TEST"
        assert d["range_m"] == 3000.0
        assert "radar" in d["sources"]


class TestTrackFusion:
    def test_no_tracks_returns_empty(self):
        fusion = TrackFusion()
        result = fusion.fuse([], [])
        assert result == []

    def test_camera_only_tracks(self):
        fusion = TrackFusion()
        cam = [_make_cam_track(640, 360)]
        result = fusion.fuse(cam, [])
        assert len(result) == 1
        assert result[0].camera_track is not None
        assert result[0].radar_track is None
        assert SensorType.CAMERA in result[0].sensor_sources

    def test_radar_only_tracks(self):
        fusion = TrackFusion()
        rdr = [_make_radar_track(3000, 0.0)]
        result = fusion.fuse([], rdr)
        assert len(result) == 1
        assert result[0].radar_track is not None
        assert result[0].camera_track is None
        assert SensorType.RADAR in result[0].sensor_sources

    def test_matching_azimuth_fuses(self):
        """Camera track at image center (az~0) + radar at az=0 should fuse."""
        fusion = TrackFusion(camera_hfov_deg=60.0, image_width_px=1280, azimuth_gate_deg=5.0)
        cam = [_make_cam_track(640, 360)]  # center pixel -> az ~0
        rdr = [_make_radar_track(3000, 0.0)]  # az = 0
        result = fusion.fuse(cam, rdr)
        assert len(result) == 1
        assert result[0].is_dual_sensor

    def test_distant_azimuth_no_fusion(self):
        """Camera at image center (az~0) + radar at az=30 should not fuse."""
        fusion = TrackFusion(camera_hfov_deg=60.0, image_width_px=1280, azimuth_gate_deg=5.0)
        cam = [_make_cam_track(640, 360)]
        rdr = [_make_radar_track(3000, 30.0)]  # 30 degrees away
        result = fusion.fuse(cam, rdr)
        assert len(result) == 2  # separate tracks
        dual = [f for f in result if f.is_dual_sensor]
        assert len(dual) == 0

    def test_multiple_fusions(self):
        """Two camera + two radar with matching azimuths."""
        fusion = TrackFusion(camera_hfov_deg=60.0, image_width_px=1280, azimuth_gate_deg=5.0)
        # Camera at left third (az~-20) and center (az~0)
        cam_left = _make_cam_track(213, 360)  # ~-20 degrees
        cam_center = _make_cam_track(640, 360)  # ~0 degrees
        rdr_left = _make_radar_track(5000, -20.0)
        rdr_center = _make_radar_track(3000, 0.0)
        result = fusion.fuse([cam_left, cam_center], [rdr_left, rdr_center])
        dual = [f for f in result if f.is_dual_sensor]
        assert len(dual) == 2

    def test_pixel_to_azimuth_center(self):
        fusion = TrackFusion(camera_hfov_deg=60.0, image_width_px=1280)
        assert fusion.pixel_to_azimuth(640) == pytest.approx(0.0)

    def test_pixel_to_azimuth_left_edge(self):
        fusion = TrackFusion(camera_hfov_deg=60.0, image_width_px=1280)
        assert fusion.pixel_to_azimuth(0) == pytest.approx(-30.0)

    def test_pixel_to_azimuth_right_edge(self):
        fusion = TrackFusion(camera_hfov_deg=60.0, image_width_px=1280)
        assert fusion.pixel_to_azimuth(1280) == pytest.approx(30.0)

    def test_fused_track_has_class_name(self):
        fusion = TrackFusion(camera_hfov_deg=60.0, image_width_px=1280, azimuth_gate_deg=5.0)
        cam = [_make_cam_track(640, 360)]
        rdr = [_make_radar_track(3000, 0.0)]
        result = fusion.fuse(cam, rdr)
        assert result[0].class_name == "person"  # from camera

    def test_fused_track_has_range(self):
        fusion = TrackFusion(camera_hfov_deg=60.0, image_width_px=1280, azimuth_gate_deg=5.0)
        cam = [_make_cam_track(640, 360)]
        rdr = [_make_radar_track(3000, 0.0)]
        result = fusion.fuse(cam, rdr)
        assert result[0].range_m is not None
        assert result[0].range_m == pytest.approx(3000.0, abs=50.0)
