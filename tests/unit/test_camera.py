"""Tests for camera adapter."""

from sentinel.sensors.camera import CameraAdapter


class TestCameraAdapter:
    def test_init_usb(self):
        cam = CameraAdapter(source=0)
        assert cam._source == 0
        assert not cam.is_connected

    def test_init_file(self):
        cam = CameraAdapter(source="test_video.mp4")
        assert cam._is_file is True

    def test_init_rtsp(self):
        cam = CameraAdapter(source="rtsp://192.168.1.1/stream")
        assert cam._is_file is False

    def test_context_manager_missing_source(self):
        """Connecting to a non-existent source returns False."""
        cam = CameraAdapter(source=999)
        result = cam.connect()
        assert result is False
        assert not cam.is_connected

    def test_disconnect_safe(self):
        """Disconnecting without connecting should not raise."""
        cam = CameraAdapter(source=0)
        cam.disconnect()  # Should not raise

    def test_read_frame_disconnected(self):
        cam = CameraAdapter(source=0)
        assert cam.read_frame() is None

    def test_total_frames_not_file(self):
        cam = CameraAdapter(source=0)
        assert cam.total_frames is None
