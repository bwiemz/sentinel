"""Multi-frequency detection correlator.

Groups radar detections from different frequency bands that originate from
the same physical target, using range/azimuth proximity gating.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from sentinel.core.types import Detection, RadarBand


@dataclass
class CorrelatedDetection:
    """A group of detections from multiple frequency bands for one target."""

    primary_detection: Detection
    band_detections: dict[str, Detection] = field(default_factory=dict)
    bands_detected: list[str] = field(default_factory=list)
    combined_pd: float = 0.0
    combined_rcs_dbsm: float = 0.0
    is_stealth_candidate: bool = False
    is_hypersonic_candidate: bool = False

    @property
    def num_bands(self) -> int:
        return len(self.bands_detected)


# Band priority for selecting the primary detection (highest freq = best resolution)
_BAND_PRIORITY = {
    RadarBand.X_BAND.value: 5,
    RadarBand.S_BAND.value: 4,
    RadarBand.L_BAND.value: 3,
    RadarBand.UHF.value: 2,
    RadarBand.VHF.value: 1,
}

# Minimum RCS variation (dB) across bands to flag as stealth candidate
_STEALTH_RCS_VARIATION_DB = 15.0


class MultiFreqCorrelator:
    """Correlates radar detections across frequency bands.

    Uses range and azimuth proximity gating to group detections from
    different bands that likely originate from the same target.

    Args:
        range_gate_m: Maximum range difference for correlation.
        azimuth_gate_deg: Maximum azimuth difference for correlation.
        stealth_rcs_variation_db: Min RCS variation across bands to flag stealth.
    """

    def __init__(
        self,
        range_gate_m: float = 100.0,
        azimuth_gate_deg: float = 3.0,
        stealth_rcs_variation_db: float = 15.0,
    ):
        self._range_gate = range_gate_m
        self._azimuth_gate = azimuth_gate_deg
        self._stealth_rcs_variation_db = stealth_rcs_variation_db

    def correlate(
        self,
        detections: list[Detection],
    ) -> tuple[list[CorrelatedDetection], list[Detection]]:
        """Correlate multi-band detections.

        Returns:
            (correlated_groups, uncorrelated_detections)
        """
        if not detections:
            return [], []

        # Group by band
        by_band: dict[str, list[Detection]] = {}
        for det in detections:
            band = det.radar_band or "unknown"
            by_band.setdefault(band, []).append(det)

        # If only one band, no cross-band correlation possible
        if len(by_band) <= 1:
            groups = []
            for det in detections:
                band = det.radar_band or "unknown"
                groups.append(
                    CorrelatedDetection(
                        primary_detection=det,
                        band_detections={band: det},
                        bands_detected=[band],
                        combined_pd=0.0,
                        combined_rcs_dbsm=det.rcs_dbsm or 0.0,
                    )
                )
            return groups, []

        # Select primary band (highest frequency with detections)
        sorted_bands = sorted(
            by_band.keys(),
            key=lambda b: _BAND_PRIORITY.get(b, 0),
            reverse=True,
        )
        primary_band = sorted_bands[0]
        secondary_bands = sorted_bands[1:]

        # Start with primary band detections as group anchors
        groups: list[dict[str, Detection]] = []
        for det in by_band[primary_band]:
            band = det.radar_band or "unknown"
            groups.append({band: det})

        # Match secondary band detections into groups
        unmatched_secondary: list[Detection] = []
        for sec_band in secondary_bands:
            sec_dets = by_band[sec_band]
            if not groups or not sec_dets:
                unmatched_secondary.extend(sec_dets)
                continue

            # Build cost matrix
            n_groups = len(groups)
            n_sec = len(sec_dets)
            cost = np.full((n_groups, n_sec), 1e5)

            for i, group in enumerate(groups):
                anchor = self._get_anchor(group)
                for j, sd in enumerate(sec_dets):
                    dist = self._distance(anchor, sd)
                    if dist < 1e4:
                        cost[i, j] = dist

            # Hungarian assignment
            row_idx, col_idx = linear_sum_assignment(cost)
            matched_cols = set()
            for r, c in zip(row_idx, col_idx, strict=False):
                if cost[r, c] < 1e4:
                    groups[r][sec_band] = sec_dets[c]
                    matched_cols.add(c)

            for j, sd in enumerate(sec_dets):
                if j not in matched_cols:
                    unmatched_secondary.append(sd)

        # Build CorrelatedDetection objects from groups
        correlated = []
        for group in groups:
            correlated.append(self._build_correlated(group))

        # Unmatched secondary detections become single-band groups
        # (these could be stealth targets visible only at low freq!)
        for det in unmatched_secondary:
            band = det.radar_band or "unknown"
            cd = CorrelatedDetection(
                primary_detection=det,
                band_detections={band: det},
                bands_detected=[band],
                combined_rcs_dbsm=det.rcs_dbsm or 0.0,
            )
            # Low-freq only detection is a stealth candidate
            if _BAND_PRIORITY.get(band, 0) <= 2:  # VHF or UHF only
                cd.is_stealth_candidate = True
            correlated.append(cd)

        return correlated, []

    def _distance(self, a: Detection, b: Detection) -> float:
        """Compute gated distance between two detections."""
        if a.range_m is None or b.range_m is None:
            return 1e5
        if a.azimuth_deg is None or b.azimuth_deg is None:
            return 1e5

        dr = abs(a.range_m - b.range_m)
        daz = abs(a.azimuth_deg - b.azimuth_deg) % 360.0
        daz = min(daz, 360.0 - daz)

        if dr > self._range_gate or daz > self._azimuth_gate:
            return 1e5

        # Normalized distance
        return (dr / self._range_gate) ** 2 + (daz / self._azimuth_gate) ** 2

    def _get_anchor(self, group: dict[str, Detection]) -> Detection:
        """Get the highest-priority detection as anchor."""
        best_band = max(group.keys(), key=lambda b: _BAND_PRIORITY.get(b, 0))
        return group[best_band]

    def _build_correlated(self, group: dict[str, Detection]) -> CorrelatedDetection:
        """Build a CorrelatedDetection from a group of band detections."""
        bands = sorted(group.keys(), key=lambda b: _BAND_PRIORITY.get(b, 0), reverse=True)
        primary = group[bands[0]]

        # Combined RCS: average across bands
        rcs_values = [d.rcs_dbsm for d in group.values() if d.rcs_dbsm is not None]
        avg_rcs = np.mean(rcs_values) if rcs_values else 0.0

        # Check for stealth signature (large RCS variation across bands)
        is_stealth = False
        if len(rcs_values) >= 2:
            rcs_range = max(rcs_values) - min(rcs_values)
            is_stealth = rcs_range >= self._stealth_rcs_variation_db

        return CorrelatedDetection(
            primary_detection=primary,
            band_detections=dict(group),
            bands_detected=bands,
            combined_rcs_dbsm=float(avg_rcs),
            is_stealth_candidate=is_stealth,
        )
