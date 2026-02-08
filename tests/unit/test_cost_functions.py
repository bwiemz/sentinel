"""Tests for data association cost functions."""

import numpy as np
import pytest

from sentinel.tracking.cost_functions import centroid_distance, iou_bbox
from sentinel.tracking.filters import KalmanFilter


class TestIoUBbox:
    def test_identical_boxes(self):
        box = np.array([10, 20, 50, 60])
        assert iou_bbox(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([20, 20, 30, 30])
        assert iou_bbox(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([5, 5, 15, 15])
        # Intersection: 5x5=25, Union: 100+100-25=175
        assert iou_bbox(a, b) == pytest.approx(25.0 / 175.0)

    def test_one_inside_other(self):
        outer = np.array([0, 0, 100, 100])
        inner = np.array([25, 25, 75, 75])
        # Intersection: 50x50=2500, Union: 10000+2500-2500=10000
        assert iou_bbox(outer, inner) == pytest.approx(2500.0 / 10000.0)

    def test_edge_touching(self):
        a = np.array([0, 0, 10, 10])
        b = np.array([10, 0, 20, 10])
        assert iou_bbox(a, b) == pytest.approx(0.0)

    def test_zero_area_box(self):
        a = np.array([5, 5, 5, 5])  # zero-area
        b = np.array([0, 0, 10, 10])
        assert iou_bbox(a, b) == pytest.approx(0.0)

    def test_symmetry(self):
        a = np.array([0, 0, 20, 15])
        b = np.array([10, 5, 30, 25])
        assert iou_bbox(a, b) == pytest.approx(iou_bbox(b, a))


class TestCentroidDistance:
    def test_same_point(self):
        p = np.array([100.0, 200.0])
        assert centroid_distance(p, p) == pytest.approx(0.0)

    def test_known_distance(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert centroid_distance(a, b) == pytest.approx(5.0)

    def test_symmetry(self):
        a = np.array([10.0, 20.0])
        b = np.array([30.0, 50.0])
        assert centroid_distance(a, b) == pytest.approx(centroid_distance(b, a))
