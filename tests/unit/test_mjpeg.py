"""Tests for MJPEG frame encoding."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.ui.web.mjpeg import encode_frame_jpeg


class TestEncodeFrameJpeg:
    def test_encodes_valid_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = encode_frame_jpeg(frame)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_output_starts_with_jpeg_soi_marker(self):
        frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = encode_frame_jpeg(frame)
        assert result[:2] == b"\xff\xd8"  # JPEG Start of Image

    def test_quality_affects_size(self):
        frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        low = encode_frame_jpeg(frame, quality=10)
        high = encode_frame_jpeg(frame, quality=95)
        assert len(high) > len(low)

    def test_grayscale_frame_encodes(self):
        frame = np.zeros((100, 100), dtype=np.uint8)
        result = encode_frame_jpeg(frame)
        assert result[:2] == b"\xff\xd8"

    def test_1x1_frame(self):
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        result = encode_frame_jpeg(frame)
        assert len(result) > 0
