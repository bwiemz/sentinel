"""MJPEG frame encoding utility."""

from __future__ import annotations

import cv2
import numpy as np


def encode_frame_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode a BGR numpy frame to JPEG bytes.

    Args:
        frame: OpenCV BGR image.
        quality: JPEG quality 0-100.

    Returns:
        JPEG-encoded bytes.

    Raises:
        ValueError: If encoding fails.
    """
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buf = cv2.imencode(".jpg", frame, params)
    if not success:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()
