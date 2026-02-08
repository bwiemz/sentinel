"""Tests for multi-frequency detection correlator."""

import numpy as np
import pytest

from sentinel.core.types import Detection, SensorType
from sentinel.fusion.multifreq_correlator import (
    CorrelatedDetection,
    MultiFreqCorrelator,
)


def _make_radar_det(range_m, azimuth_deg, rcs_dbsm, band):
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=1.0,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        velocity_mps=0.0,
        rcs_dbsm=rcs_dbsm,
        radar_band=band,
    )


class TestMultiFreqCorrelator:
    def test_empty_detections(self):
        corr = MultiFreqCorrelator()
        groups, uncorr = corr.correlate([])
        assert groups == []

    def test_single_band_no_cross_correlation(self):
        dets = [_make_radar_det(5000, 10.0, 10.0, "x_band")]
        corr = MultiFreqCorrelator()
        groups, _ = corr.correlate(dets)
        assert len(groups) == 1
        assert groups[0].num_bands == 1

    def test_two_bands_same_target_correlates(self):
        dets = [
            _make_radar_det(5000, 10.0, 10.0, "x_band"),
            _make_radar_det(5010, 10.2, 15.0, "vhf"),  # Close in range/azimuth
        ]
        corr = MultiFreqCorrelator(range_gate_m=100.0, azimuth_gate_deg=3.0)
        groups, _ = corr.correlate(dets)
        # Should correlate into one group
        assert len(groups) == 1
        assert groups[0].num_bands == 2

    def test_two_bands_different_targets_no_correlation(self):
        dets = [
            _make_radar_det(5000, 10.0, 10.0, "x_band"),
            _make_radar_det(20000, -30.0, 8.0, "vhf"),  # Very different location
        ]
        corr = MultiFreqCorrelator(range_gate_m=100.0, azimuth_gate_deg=3.0)
        groups, _ = corr.correlate(dets)
        assert len(groups) == 2

    def test_three_bands_one_target(self):
        dets = [
            _make_radar_det(5000, 10.0, 10.0, "x_band"),
            _make_radar_det(5005, 10.1, 12.0, "s_band"),
            _make_radar_det(5020, 10.5, 35.0, "vhf"),
        ]
        corr = MultiFreqCorrelator(range_gate_m=100.0, azimuth_gate_deg=3.0)
        groups, _ = corr.correlate(dets)
        assert len(groups) == 1
        assert groups[0].num_bands == 3

    def test_stealth_detection_vhf_only(self):
        """VHF-only detection should be flagged as stealth candidate."""
        dets = [
            _make_radar_det(5000, 10.0, 10.0, "x_band"),  # Conventional target
            _make_radar_det(8000, 20.0, -5.0, "vhf"),      # VHF-only (stealth)
        ]
        corr = MultiFreqCorrelator(range_gate_m=100.0, azimuth_gate_deg=3.0)
        groups, _ = corr.correlate(dets)
        # The VHF-only detection should be separate and flagged
        stealth_groups = [g for g in groups if g.is_stealth_candidate]
        assert len(stealth_groups) >= 1

    def test_primary_selection_prefers_highest_freq(self):
        dets = [
            _make_radar_det(5000, 10.0, 10.0, "x_band"),
            _make_radar_det(5010, 10.1, 30.0, "vhf"),
        ]
        corr = MultiFreqCorrelator(range_gate_m=100.0, azimuth_gate_deg=3.0)
        groups, _ = corr.correlate(dets)
        assert groups[0].primary_detection.radar_band == "x_band"

    def test_rcs_variation_stealth_flag(self):
        """Large RCS variation across bands indicates stealth."""
        dets = [
            _make_radar_det(5000, 10.0, -30.0, "x_band"),  # Very low at X-band
            _make_radar_det(5010, 10.1, -5.0, "vhf"),       # Much higher at VHF
        ]
        corr = MultiFreqCorrelator(range_gate_m=100.0, azimuth_gate_deg=3.0)
        groups, _ = corr.correlate(dets)
        assert groups[0].is_stealth_candidate  # 25 dB variation > 15 dB threshold

    def test_range_gate_respected(self):
        dets = [
            _make_radar_det(5000, 10.0, 10.0, "x_band"),
            _make_radar_det(5200, 10.0, 12.0, "vhf"),  # 200m apart > 100m gate
        ]
        corr = MultiFreqCorrelator(range_gate_m=100.0, azimuth_gate_deg=3.0)
        groups, _ = corr.correlate(dets)
        assert len(groups) == 2  # Should not correlate

    def test_azimuth_gate_respected(self):
        dets = [
            _make_radar_det(5000, 10.0, 10.0, "x_band"),
            _make_radar_det(5010, 20.0, 12.0, "vhf"),  # 10 deg apart > 3 deg gate
        ]
        corr = MultiFreqCorrelator(range_gate_m=100.0, azimuth_gate_deg=3.0)
        groups, _ = corr.correlate(dets)
        assert len(groups) == 2

    def test_combined_rcs_computed(self):
        dets = [
            _make_radar_det(5000, 10.0, 10.0, "x_band"),
            _make_radar_det(5010, 10.1, 20.0, "vhf"),
        ]
        corr = MultiFreqCorrelator(range_gate_m=100.0, azimuth_gate_deg=3.0)
        groups, _ = corr.correlate(dets)
        assert groups[0].combined_rcs_dbsm == pytest.approx(15.0)  # Average of 10, 20
