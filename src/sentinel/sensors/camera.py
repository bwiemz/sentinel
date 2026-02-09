"""Camera adapter supporting USB webcams, RTSP streams, and video files."""

from __future__ import annotations

import logging
import platform

import cv2

from sentinel.core.clock import Clock, SystemClock
from sentinel.core.types import SensorType
from sentinel.sensors.base import AbstractSensor
from sentinel.sensors.frame import SensorFrame

logger = logging.getLogger(__name__)


class CameraAdapter(AbstractSensor):
    """Unified camera adapter for USB, RTSP, and video file sources.

    Args:
        source: Camera source -- int (USB device index), str (RTSP URL or file path).
        width: Requested frame width (USB cameras only).
        height: Requested frame height (USB cameras only).
        fps: Requested FPS (USB cameras only).
        buffer_size: OpenCV capture buffer size. 1 = minimal latency.
        backend: OpenCV backend hint. "auto", "dshow" (Windows), "v4l2" (Linux).
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        buffer_size: int = 1,
        backend: str = "auto",
        clock: Clock | None = None,
    ):
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._buffer_size = buffer_size
        self._backend = backend
        self._cap: cv2.VideoCapture | None = None
        self._clock = clock if clock is not None else SystemClock()
        self._frame_count = 0
        self._is_file = isinstance(source, str) and not source.startswith("rtsp")

    def connect(self) -> bool:
        """Open the camera/video source."""
        api = self._resolve_backend()
        if api is not None:
            self._cap = cv2.VideoCapture(self._source, api)
        else:
            self._cap = cv2.VideoCapture(self._source)

        if not self._cap.isOpened():
            logger.error("Failed to open camera source: %s", self._source)
            return False

        # Configure USB cameras
        if isinstance(self._source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera connected: %s @ %dx%d %.1f FPS",
            self._source,
            actual_w,
            actual_h,
            actual_fps,
        )
        return True

    def disconnect(self) -> None:
        """Release the camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera disconnected: %s", self._source)

    def read_frame(self) -> SensorFrame | None:
        """Grab and decode one frame."""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            if self._is_file:
                logger.info("Video file ended after %d frames", self._frame_count)
            return None

        self._frame_count += 1
        return SensorFrame(
            data=frame,
            timestamp=self._clock.now(),
            sensor_type=SensorType.CAMERA,
            frame_number=self._frame_count,
            metadata={
                "resolution": (frame.shape[0], frame.shape[1]),
                "source": str(self._source),
            },
        )

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def total_frames(self) -> int | None:
        """Total frame count for video files, None for live sources."""
        if self._cap is None or not self._is_file:
            return None
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return total if total > 0 else None

    def _resolve_backend(self) -> int | None:
        if self._backend == "auto":
            if isinstance(self._source, int) and platform.system() == "Windows":
                return cv2.CAP_DSHOW
            return None
        backends = {
            "dshow": cv2.CAP_DSHOW,
            "v4l2": cv2.CAP_V4L2,
            "ffmpeg": cv2.CAP_FFMPEG,
            "gstreamer": cv2.CAP_GSTREAMER,
        }
        return backends.get(self._backend)
