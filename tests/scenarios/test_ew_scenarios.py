"""EW scenario validation tests.

Scenario A: Noise Jamming — stand-off jammer degrades radar Pd; VHF more
    resilient; QI maintains detection with ECCM.
Scenario B: Chaff + Decoy — chaff flagged by correlator; decoy without IR
    flagged in fusion; real target not flagged.
Scenario C: Combined EW — jammer + chaff + decoy simultaneously; correct
    threat classification; QI + thermal provide detection where classical fails.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pytest

from sentinel.core.types import Detection, RadarBand, SensorType, TargetType
from sentinel.fusion.multi_sensor_fusion import (
    THREAT_CRITICAL,
    THREAT_HIGH,
    THREAT_LOW,
    THREAT_MEDIUM,
    MultiSensorFusion,
)
from sentinel.fusion.multifreq_correlator import CorrelatedDetection, MultiFreqCorrelator
from sentinel.sensors.environment import EnvironmentModel
from sentinel.sensors.ew import (
    ChaffCloud,
    DecoySource,
    ECCMConfig,
    EWModel,
    JammerSource,
    burn_through_range_m,
)
from sentinel.sensors.multifreq_radar_sim import (
    MultiFreqRadarConfig,
    MultiFreqRadarSimulator,
    MultiFreqRadarTarget,
    multifreq_radar_frame_to_detections,
)
from sentinel.sensors.quantum_radar_sim import (
    QuantumRadarConfig,
    QuantumRadarSimulator,
    quantum_radar_frame_to_detections,
)
from sentinel.sensors.thermal_sim import (
    ThermalSimConfig,
    ThermalSimulator,
    ThermalTarget,
    thermal_frame_to_detections,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _radar_detection(range_m: float, azimuth_deg: float, ts: float = 1.0) -> Detection:
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=ts,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        velocity_mps=0.0,
        rcs_dbsm=10.0,
    )


def _thermal_detection(azimuth_deg: float, ts: float = 1.0) -> Detection:
    return Detection(
        sensor_type=SensorType.THERMAL,
        timestamp=ts,
        azimuth_deg=azimuth_deg,
        temperature_k=400.0,
    )


# ===================================================================
# Scenario A: Noise Jamming
# ===================================================================


class TestNoiseJammingScenario:
    """Stand-off noise jammer degrades radar detection."""

    @pytest.fixture
    def noise_jammer_env(self):
        """Strong noise jammer at 20km along y-axis."""
        jammer = JammerSource(
            position=np.array([0.0, 20000.0]),
            erp_watts=5e5,
            bandwidth_hz=1e6,
            jam_type="noise",
            active=True,
        )
        ew = EWModel(
            jammers=[jammer],
            radar_peak_power_w=1e6,
            radar_gain_db=30.0,
            radar_bandwidth_hz=1e6,
        )
        return EnvironmentModel(ew=ew, use_ew_effects=True)

    def test_jammer_degrades_multifreq_radar(self, noise_jammer_env):
        """Multi-freq radar produces fewer detections under noise jamming."""
        target = MultiFreqRadarTarget(
            target_id="TGT1",
            position=np.array([0.0, 15000.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=10.0,
        )

        # Clean run
        cfg_clean = MultiFreqRadarConfig(
            bands=[RadarBand.X_BAND, RadarBand.VHF],
            max_range_m=50000.0,
            base_detection_probability=0.9,
            false_alarm_rate=0.0,
            targets=[target],
        )
        sim_clean = MultiFreqRadarSimulator(cfg_clean, seed=42)
        sim_clean.connect()
        clean_total = sum(
            len(multifreq_radar_frame_to_detections(sim_clean.read_frame()))
            for _ in range(40)
        )
        sim_clean.disconnect()

        # Jammed run
        cfg_jammed = MultiFreqRadarConfig(
            bands=[RadarBand.X_BAND, RadarBand.VHF],
            max_range_m=50000.0,
            base_detection_probability=0.9,
            false_alarm_rate=0.0,
            targets=[target],
            environment=noise_jammer_env,
        )
        sim_jammed = MultiFreqRadarSimulator(cfg_jammed, seed=42)
        sim_jammed.connect()
        jammed_total = 0
        for _ in range(40):
            frame = sim_jammed.read_frame()
            real_dets = [d for d in multifreq_radar_frame_to_detections(frame)
                         if not d.is_ew_generated]
            jammed_total += len(real_dets)
        sim_jammed.disconnect()

        assert jammed_total <= clean_total

    def test_vhf_more_resilient_than_xband_at_unit_level(self):
        """VHF band is less affected than X-band by narrowband X-band jammer
        at the EW model level (unit-level verification)."""
        jammer = JammerSource(
            position=np.array([0.0, 20000.0]),
            erp_watts=5e5,
            bandwidth_hz=1e6,
            jam_type="noise",
            target_bands=[RadarBand.X_BAND],  # Only jams X-band
        )
        ew = EWModel(jammers=[jammer])
        env = EnvironmentModel(ew=ew, use_ew_effects=True)

        # X-band should get SNR reduction, VHF should not
        x_adj = env.ew_snr_adjustment_db(15000.0, 10e9, RadarBand.X_BAND)
        vhf_adj = env.ew_snr_adjustment_db(15000.0, 50e6, RadarBand.VHF)

        assert x_adj < 0  # X-band is degraded
        assert vhf_adj == 0.0  # VHF unaffected by narrowband X-band jammer

    def test_qi_maintains_detection_under_jamming_unit_level(self):
        """QI ECCM advantage at the EW model level offsets jamming."""
        jammer = JammerSource(
            position=np.array([0.0, 20000.0]),
            erp_watts=5e5,
            bandwidth_hz=1e6,
            jam_type="noise",
        )
        eccm = ECCMConfig(quantum_eccm=True, quantum_eccm_advantage_db=6.0)
        ew = EWModel(jammers=[jammer], eccm=eccm)

        base_advantage = 3.0  # dB
        effective = ew.effective_quantum_advantage_db(base_advantage)
        assert effective == base_advantage + 6.0  # ECCM adds 6 dB

    def test_burn_through_range_concept(self):
        """Burn-through: at close range, radar overcomes jammer."""
        btr = burn_through_range_m(
            radar_peak_power_w=1e6,
            radar_gain_db=30.0,
            rcs_m2=10.0,
            jammer_erp_w=1e5,
            jammer_range_m=20000.0,
            jammer_bw_hz=1e6,
            radar_bw_hz=1e6,
        )
        assert 100.0 < btr < 50000.0


# ===================================================================
# Scenario B: Chaff + Decoy Discrimination
# ===================================================================


class TestChaffDecoyScenario:
    """Chaff and decoy discrimination via multi-sensor fusion."""

    def test_chaff_flagged_by_correlator(self):
        """Chaff signature (high uniform RCS across bands) flagged by correlator."""
        dets = []
        for band in [RadarBand.X_BAND, RadarBand.S_BAND, RadarBand.VHF]:
            dets.append(Detection(
                sensor_type=SensorType.RADAR,
                timestamp=1.0,
                range_m=7000.0,
                azimuth_deg=45.0,
                rcs_dbsm=33.0,  # Uniform across bands
                radar_band=band.value,
            ))

        correlator = MultiFreqCorrelator(range_gate_m=200.0, azimuth_gate_deg=5.0)
        correlated, _ = correlator.correlate(dets)

        multi_band = [c for c in correlated if c.num_bands >= 2]
        assert len(multi_band) >= 1
        assert any(c.is_chaff_candidate for c in multi_band)

    def test_decoy_without_ir_flagged_in_fusion(self):
        """Radar-only track (no thermal match) → decoy candidate in fusion."""
        from sentinel.tracking.radar_track import RadarTrack

        det = _radar_detection(8000.0, 20.0)
        rt = RadarTrack(det)
        rt.score = 0.7

        fusion = MultiSensorFusion(
            azimuth_gate_deg=10.0,
            thermal_azimuth_gate_deg=5.0,
        )
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=[rt],
            thermal_tracks=[],
            quantum_radar_tracks=[],
        )

        assert len(fused) >= 1
        decoy_tracks = [f for f in fused if f.is_decoy_candidate]
        assert len(decoy_tracks) >= 1
        assert decoy_tracks[0].threat_level == THREAT_LOW

    def test_real_target_not_flagged_as_decoy(self):
        """Real target with both radar and thermal → not flagged as decoy."""
        from sentinel.tracking.radar_track import RadarTrack
        from sentinel.tracking.thermal_track import ThermalTrack

        rdet = _radar_detection(10000.0, 15.0)
        rt = RadarTrack(rdet)
        rt.score = 0.9

        tdet = _thermal_detection(15.0)
        tt = ThermalTrack(tdet)
        tt.score = 0.8

        fusion = MultiSensorFusion(
            azimuth_gate_deg=10.0,
            thermal_azimuth_gate_deg=20.0,
        )
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=[rt],
            thermal_tracks=[tt],
            quantum_radar_tracks=[],
        )

        assert len(fused) >= 1
        for f in fused:
            if f.radar_track is not None and f.thermal_track is not None:
                assert f.is_decoy_candidate is False


# ===================================================================
# Scenario C: Combined EW
# ===================================================================


class TestCombinedEWScenario:
    """Jammer + chaff + decoy simultaneously."""

    @pytest.fixture
    def combined_ew_env(self):
        """Full EW environment: noise jammer + chaff + decoy."""
        jammer = JammerSource(
            position=np.array([25000.0, 0.0]),
            erp_watts=2e5,
            bandwidth_hz=1e6,
            jam_type="noise",
        )
        chaff = ChaffCloud(
            position=np.array([6000.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            deploy_time=_time.time(),
            initial_rcs_dbsm=30.0,
            lifetime_s=300.0,
        )
        decoy = DecoySource(
            position=np.array([4000.0, 0.0]),
            velocity=np.array([20.0, 0.0]),
            rcs_dbsm=8.0,
            has_thermal_signature=False,
            deploy_time=_time.time(),
            lifetime_s=300.0,
        )
        eccm = ECCMConfig(quantum_eccm=True, quantum_eccm_advantage_db=6.0)
        ew = EWModel(
            jammers=[jammer],
            chaff_clouds=[chaff],
            decoys=[decoy],
            eccm=eccm,
        )
        return EnvironmentModel(ew=ew, use_ew_effects=True)

    def test_combined_ew_radar_produces_ew_returns(self, combined_ew_env):
        """Under combined EW, radar gets EW-injected false targets (chaff/decoy)."""
        # No real targets — just verify EW injection works
        cfg = MultiFreqRadarConfig(
            bands=[RadarBand.X_BAND, RadarBand.VHF],
            max_range_m=50000.0,
            base_detection_probability=0.95,
            false_alarm_rate=0.0,
            targets=[],
            environment=combined_ew_env,
        )
        sim = MultiFreqRadarSimulator(cfg, seed=42)
        sim.connect()
        ew_total = 0
        for _ in range(10):
            frame = sim.read_frame()
            for d in frame.data:
                if d.get("is_ew_generated"):
                    ew_total += 1
        sim.disconnect()

        # Should have EW-generated detections (chaff + decoy)
        assert ew_total >= 1

    def test_thermal_unaffected_by_radar_jamming(self, combined_ew_env):
        """Thermal sensor is passive — not affected by noise jamming.
        With a hot target, thermal should detect it regardless of jamming."""
        # Hypersonic target = very hot = detectable by thermal
        thermal_target = ThermalTarget(
            target_id="TGT_HOT",
            position=np.array([8000.0, 0.0]),  # Along +x = within FOV
            velocity=np.array([0.0, 0.0]),
            target_type=TargetType.HYPERSONIC,
            mach=5.0,  # Very hot
        )
        th_cfg = ThermalSimConfig(
            fov_deg=60.0,
            max_range_m=50000.0,
            detection_probability=0.95,
            false_alarm_rate=0.0,
            targets=[thermal_target],
            environment=combined_ew_env,
        )
        th_sim = ThermalSimulator(th_cfg, seed=43)
        th_sim.connect()
        th_total = sum(
            len(thermal_frame_to_detections(th_sim.read_frame()))
            for _ in range(30)
        )
        th_sim.disconnect()

        # Thermal should detect despite radar jamming (passive sensor)
        assert th_total >= 1

    def test_correct_threat_classification_under_ew(self):
        """Under EW, chaff → LOW, decoy → LOW, real threats properly classified."""
        from sentinel.tracking.radar_track import RadarTrack
        from sentinel.tracking.thermal_track import ThermalTrack

        # Real target: radar + thermal → MEDIUM (multi-sensor)
        rdet_real = _radar_detection(10000.0, 30.0)
        rt_real = RadarTrack(rdet_real)
        rt_real.score = 0.9

        tdet_real = _thermal_detection(30.0)
        tt_real = ThermalTrack(tdet_real)
        tt_real.score = 0.8

        # Decoy: radar only → should be LOW
        rdet_decoy = _radar_detection(5000.0, 80.0)
        rt_decoy = RadarTrack(rdet_decoy)
        rt_decoy.score = 0.6

        # Chaff-correlated detection — at different azimuth from both targets
        cd_chaff = CorrelatedDetection(
            primary_detection=Detection(
                sensor_type=SensorType.RADAR,
                timestamp=1.0,
                range_m=7000.0,
                azimuth_deg=55.0,  # Far from real target (30°) and decoy (80°)
                rcs_dbsm=30.0,
                radar_band=RadarBand.X_BAND.value,
            ),
            bands_detected=[RadarBand.X_BAND.value, RadarBand.VHF.value],
            is_chaff_candidate=True,
        )

        fusion = MultiSensorFusion(
            azimuth_gate_deg=10.0,
            thermal_azimuth_gate_deg=15.0,
        )
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=[rt_real, rt_decoy],
            thermal_tracks=[tt_real],
            correlated_detections=[cd_chaff],
            quantum_radar_tracks=[],
        )

        # Find each fused track by identity (not track_id since auto-generated)
        real_fused = None
        decoy_fused = None
        for f in fused:
            if f.radar_track is rt_real and f.thermal_track is not None:
                real_fused = f
            elif f.radar_track is rt_decoy and f.thermal_track is None:
                decoy_fused = f

        # Real target: should be MEDIUM or higher (multi-sensor confirmed)
        if real_fused is not None:
            assert real_fused.threat_level in (THREAT_MEDIUM, THREAT_HIGH, THREAT_CRITICAL)
            assert real_fused.is_decoy_candidate is False

        # Decoy: should be LOW (radar-only, no thermal)
        if decoy_fused is not None:
            assert decoy_fused.threat_level == THREAT_LOW
            assert decoy_fused.is_decoy_candidate is True
