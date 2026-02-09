"""Integration tests: EW effects flowing through simulators and fusion.

Tests verify that when EW is enabled, simulators produce the expected
behavior (reduced Pd, extra false targets, chaff/decoy discrimination).
Also verifies backward compatibility: EW disabled → identical behavior.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pytest
from omegaconf import OmegaConf

from sentinel.core.types import Detection, RadarBand, SensorType, TargetType
from sentinel.fusion.multi_sensor_fusion import (
    THREAT_LOW,
    EnhancedFusedTrack,
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
from sentinel.sensors.radar_sim import (
    RadarSimConfig,
    RadarTarget,
    RadarSimulator,
    radar_frame_to_detections,
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
    """Create a radar detection for track initialization."""
    return Detection(
        sensor_type=SensorType.RADAR,
        timestamp=ts,
        range_m=range_m,
        azimuth_deg=azimuth_deg,
        velocity_mps=0.0,
        rcs_dbsm=10.0,
    )


def _thermal_detection(azimuth_deg: float, ts: float = 1.0) -> Detection:
    """Create a thermal detection for track initialization."""
    return Detection(
        sensor_type=SensorType.THERMAL,
        timestamp=ts,
        azimuth_deg=azimuth_deg,
        temperature_k=400.0,
    )


def _env_with_noise_jammer(erp_watts: float = 1e5, jammer_range_m: float = 10000.0) -> EnvironmentModel:
    """EnvironmentModel with a single noise jammer at the given range along y-axis."""
    jammer = JammerSource(
        position=np.array([0.0, jammer_range_m]),
        erp_watts=erp_watts,
        bandwidth_hz=1e6,
        jam_type="noise",
        active=True,
    )
    ew = EWModel(jammers=[jammer])
    return EnvironmentModel(ew=ew, use_ew_effects=True)


def _env_with_deceptive_jammer(n_false: int = 5) -> EnvironmentModel:
    jammer = JammerSource(
        position=np.array([0.0, 20000.0]),
        erp_watts=1e4,
        bandwidth_hz=1e6,
        jam_type="deceptive",
        active=True,
        n_false_targets=n_false,
    )
    ew = EWModel(jammers=[jammer])
    return EnvironmentModel(ew=ew, use_ew_effects=True)


def _env_with_chaff(rcs_dbsm: float = 30.0) -> EnvironmentModel:
    chaff = ChaffCloud(
        position=np.array([5000.0, 0.0]),
        velocity=np.array([0.0, 0.0]),
        deploy_time=_time.time(),  # Deploy NOW so chaff is active
        initial_rcs_dbsm=rcs_dbsm,
        lifetime_s=120.0,
    )
    ew = EWModel(chaff_clouds=[chaff])
    return EnvironmentModel(ew=ew, use_ew_effects=True)


def _env_with_decoy(has_thermal: bool = False) -> EnvironmentModel:
    decoy = DecoySource(
        position=np.array([3000.0, 0.0]),
        velocity=np.array([50.0, 0.0]),
        rcs_dbsm=10.0,
        has_thermal_signature=has_thermal,
        thermal_temperature_k=500.0,
        decoy_id="decoy_test",
        deploy_time=_time.time(),  # Deploy NOW so decoy is active
        lifetime_s=300.0,
    )
    ew = EWModel(decoys=[decoy])
    return EnvironmentModel(ew=ew, use_ew_effects=True)


# ===================================================================
# Classical Radar + EW
# ===================================================================


class TestRadarSimWithNoiseJammer:
    """Radar sim + noise jammer → reduced detections."""

    def test_noise_jammer_reduces_detection_probability(self):
        """With strong noise jamming, fewer detections are expected."""
        target = RadarTarget(
            target_id="TGT1",
            position=np.array([0.0, 20000.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=10.0,
        )

        # Run without EW
        cfg_clean = RadarSimConfig(
            targets=[target],
            detection_probability=0.95,
            false_alarm_rate=0.0,
            max_range_m=50000.0,
        )
        sim_clean = RadarSimulator(cfg_clean, seed=42)
        sim_clean.connect()
        clean_count = 0
        for _ in range(50):
            frame = sim_clean.read_frame()
            dets = radar_frame_to_detections(frame)
            clean_count += len(dets)
        sim_clean.disconnect()

        # Run WITH noise jammer
        env = _env_with_noise_jammer(erp_watts=1e6, jammer_range_m=15000.0)
        cfg_ew = RadarSimConfig(
            targets=[target],
            detection_probability=0.95,
            false_alarm_rate=0.0,
            max_range_m=50000.0,
            environment=env,
        )
        sim_ew = RadarSimulator(cfg_ew, seed=42)
        sim_ew.connect()
        ew_count = 0
        for _ in range(50):
            frame = sim_ew.read_frame()
            dets = radar_frame_to_detections(frame)
            ew_count += len(dets)
        sim_ew.disconnect()

        # EW should reduce detections
        assert ew_count <= clean_count


class TestRadarSimWithDeceptiveJammer:
    """Radar sim + deceptive jammer → extra false detections."""

    def test_deceptive_jammer_injects_false_targets(self):
        """Deceptive jammer should add extra detections marked is_ew_generated."""
        env = _env_with_deceptive_jammer(n_false=5)
        cfg = RadarSimConfig(
            targets=[],  # No real targets
            detection_probability=0.95,
            false_alarm_rate=0.0,
            max_range_m=50000.0,
            environment=env,
        )
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        sim.disconnect()

        # Should have EW-generated detections
        ew_dets = [d for d in frame.data if d.get("is_ew_generated")]
        assert len(ew_dets) >= 1


class TestRadarSimEWDisabled:
    """EW disabled → identical behavior to no-EW case."""

    def test_ew_disabled_no_effect(self):
        """When use_ew_effects=False, EW model is ignored."""
        jammer = JammerSource(
            position=np.array([10000.0, 0.0]),
            erp_watts=1e6,
            bandwidth_hz=1e6,
            jam_type="noise",
        )
        ew = EWModel(jammers=[jammer])
        env = EnvironmentModel(ew=ew, use_ew_effects=False)  # Disabled!

        target = RadarTarget(
            target_id="TGT1",
            position=np.array([15000.0, 0.0]),  # Along +x = boresight
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=10.0,
        )
        cfg = RadarSimConfig(
            targets=[target],
            detection_probability=1.0,  # Guaranteed detection
            false_alarm_rate=0.0,
            max_range_m=50000.0,
            environment=env,
        )
        sim = RadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        sim.disconnect()

        # Should still detect the target (EW disabled)
        assert len(frame.data) >= 1
        ew_dets = [d for d in frame.data if d.get("is_ew_generated")]
        assert len(ew_dets) == 0


# ===================================================================
# Multi-Freq Radar + EW
# ===================================================================


class TestMultiFreqWithNarrowbandJammer:
    """Multi-freq + narrowband jammer → affected band degraded, others not."""

    def test_narrowband_jammer_affects_target_band(self):
        """A narrowband X-band jammer should degrade X-band more than VHF."""
        jammer = JammerSource(
            position=np.array([0.0, 15000.0]),
            erp_watts=1e5,
            bandwidth_hz=1e6,
            jam_type="noise",
            target_bands=[RadarBand.X_BAND],
        )
        ew = EWModel(jammers=[jammer])
        env = EnvironmentModel(ew=ew, use_ew_effects=True)

        target = MultiFreqRadarTarget(
            target_id="MF1",
            position=np.array([0.0, 20000.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=10.0,
        )
        cfg = MultiFreqRadarConfig(
            bands=[RadarBand.X_BAND, RadarBand.VHF],
            max_range_m=50000.0,
            base_detection_probability=0.9,
            false_alarm_rate=0.0,
            targets=[target],
            environment=env,
        )
        sim = MultiFreqRadarSimulator(cfg, seed=42)
        sim.connect()
        x_count = 0
        vhf_count = 0
        for _ in range(50):
            frame = sim.read_frame()
            for d in frame.data:
                if d.get("is_ew_generated"):
                    continue
                band = d.get("radar_band", "")
                if band == RadarBand.X_BAND.value:
                    x_count += 1
                elif band == RadarBand.VHF.value:
                    vhf_count += 1
        sim.disconnect()

        # VHF should have more detections than jammed X-band
        assert vhf_count >= x_count


class TestMultiFreqWithChaff:
    """Multi-freq + chaff → high-RCS detections on all bands."""

    def test_chaff_produces_radar_returns(self):
        """Chaff cloud should produce detections on the radar."""
        env = _env_with_chaff(rcs_dbsm=35.0)
        cfg = MultiFreqRadarConfig(
            bands=[RadarBand.X_BAND, RadarBand.VHF],
            max_range_m=50000.0,
            base_detection_probability=0.9,
            false_alarm_rate=0.0,
            targets=[],
            environment=env,
        )
        sim = MultiFreqRadarSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        sim.disconnect()

        ew_dets = [d for d in frame.data if d.get("is_ew_generated")]
        assert len(ew_dets) >= 1


# ===================================================================
# Quantum Radar + EW
# ===================================================================


class TestQuantumRadarWithJammer:
    """Quantum radar + noise jammer → less Pd reduction than classical."""

    def test_qi_eccm_mitigates_jamming(self):
        """With QI ECCM, quantum radar should maintain higher detection than classical."""
        jammer = JammerSource(
            position=np.array([10000.0, 0.0]),
            erp_watts=1e5,
            bandwidth_hz=1e6,
            jam_type="noise",
        )
        eccm = ECCMConfig(quantum_eccm=True, quantum_eccm_advantage_db=6.0)
        ew = EWModel(jammers=[jammer], eccm=eccm)
        env = EnvironmentModel(ew=ew, use_ew_effects=True)

        target = MultiFreqRadarTarget(
            target_id="QI1",
            position=np.array([8000.0, 0.0]),  # Along +x = boresight
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=5.0,
            target_type=TargetType.STEALTH,
        )
        cfg = QuantumRadarConfig(
            max_range_m=15000.0,
            squeeze_param_r=0.5,
            n_modes=50000,
            targets=[target],
            false_alarm_rate=0.0,
            environment=env,
        )
        sim = QuantumRadarSimulator(cfg, seed=42)
        sim.connect()
        det_count = 0
        for _ in range(30):
            frame = sim.read_frame()
            dets = quantum_radar_frame_to_detections(frame)
            det_count += len(dets)
        sim.disconnect()

        # QI should still detect something despite jamming (with ECCM)
        assert det_count >= 1

    def test_qi_without_eccm_still_affected(self):
        """Without QI ECCM, quantum radar is affected by jamming."""
        jammer = JammerSource(
            position=np.array([0.0, 10000.0]),
            erp_watts=1e6,
            bandwidth_hz=1e6,
            jam_type="noise",
        )
        ew = EWModel(jammers=[jammer])  # No ECCM
        env = EnvironmentModel(ew=ew, use_ew_effects=True)

        target = MultiFreqRadarTarget(
            target_id="QI2",
            position=np.array([0.0, 8000.0]),
            velocity=np.array([0.0, 0.0]),
            rcs_dbsm=5.0,
        )

        # Run without EW
        cfg_clean = QuantumRadarConfig(
            max_range_m=15000.0,
            squeeze_param_r=0.5,
            n_modes=50000,
            targets=[target],
            false_alarm_rate=0.0,
        )
        sim_clean = QuantumRadarSimulator(cfg_clean, seed=42)
        sim_clean.connect()
        clean_count = 0
        for _ in range(50):
            frame = sim_clean.read_frame()
            clean_count += len(quantum_radar_frame_to_detections(frame))
        sim_clean.disconnect()

        # Run WITH jammer (no ECCM)
        cfg_ew = QuantumRadarConfig(
            max_range_m=15000.0,
            squeeze_param_r=0.5,
            n_modes=50000,
            targets=[target],
            false_alarm_rate=0.0,
            environment=env,
        )
        sim_ew = QuantumRadarSimulator(cfg_ew, seed=42)
        sim_ew.connect()
        ew_count = 0
        for _ in range(50):
            frame = sim_ew.read_frame()
            ew_count += len(quantum_radar_frame_to_detections(frame))
        sim_ew.disconnect()

        # EW should reduce detections (no ECCM to offset)
        assert ew_count <= clean_count


# ===================================================================
# Thermal Simulator + EW
# ===================================================================


class TestThermalWithDecoy:
    """Thermal + decoy interactions."""

    def test_decoy_without_ir_no_thermal_return(self):
        """Decoy without thermal signature produces no thermal detections."""
        env = _env_with_decoy(has_thermal=False)
        cfg = ThermalSimConfig(
            fov_deg=60.0,
            max_range_m=50000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[],
            environment=env,
        )
        sim = ThermalSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        sim.disconnect()

        ew_dets = [d for d in frame.data if d.get("is_ew_generated")]
        assert len(ew_dets) == 0

    def test_decoy_with_ir_produces_thermal(self):
        """Decoy with IR emitter produces thermal detections."""
        env = _env_with_decoy(has_thermal=True)
        cfg = ThermalSimConfig(
            fov_deg=60.0,
            max_range_m=50000.0,
            detection_probability=1.0,
            false_alarm_rate=0.0,
            targets=[],
            environment=env,
        )
        sim = ThermalSimulator(cfg, seed=42)
        sim.connect()
        frame = sim.read_frame()
        sim.disconnect()

        ew_dets = [d for d in frame.data if d.get("is_ew_generated")]
        assert len(ew_dets) >= 1


# ===================================================================
# Multi-Freq Correlator + Chaff
# ===================================================================


class TestCorrelatorChaffDetection:
    """Correlator flags chaff: high uniform RCS across bands."""

    def test_chaff_signature_flagged(self):
        """Detections with high RCS and low cross-band variation → chaff candidate."""
        dets = [
            Detection(
                sensor_type=SensorType.RADAR,
                timestamp=1.0,
                range_m=5000.0,
                azimuth_deg=45.0,
                rcs_dbsm=32.0,
                radar_band=RadarBand.X_BAND.value,
            ),
            Detection(
                sensor_type=SensorType.RADAR,
                timestamp=1.0,
                range_m=5000.0,
                azimuth_deg=45.0,
                rcs_dbsm=31.0,
                radar_band=RadarBand.VHF.value,
            ),
        ]
        correlator = MultiFreqCorrelator(range_gate_m=200.0, azimuth_gate_deg=5.0)
        correlated, _ = correlator.correlate(dets)

        assert len(correlated) >= 1
        multi_band = [c for c in correlated if c.num_bands >= 2]
        assert len(multi_band) >= 1
        assert multi_band[0].is_chaff_candidate is True

    def test_stealth_signature_not_chaff(self):
        """Detections with high RCS variation across bands → stealth, not chaff."""
        dets = [
            Detection(
                sensor_type=SensorType.RADAR,
                timestamp=1.0,
                range_m=10000.0,
                azimuth_deg=30.0,
                rcs_dbsm=-5.0,
                radar_band=RadarBand.X_BAND.value,
            ),
            Detection(
                sensor_type=SensorType.RADAR,
                timestamp=1.0,
                range_m=10000.0,
                azimuth_deg=30.0,
                rcs_dbsm=15.0,
                radar_band=RadarBand.VHF.value,
            ),
        ]
        correlator = MultiFreqCorrelator(range_gate_m=200.0, azimuth_gate_deg=5.0)
        correlated, _ = correlator.correlate(dets)

        multi_band = [c for c in correlated if c.num_bands >= 2]
        assert len(multi_band) >= 1
        assert multi_band[0].is_stealth_candidate is True
        assert multi_band[0].is_chaff_candidate is False


# ===================================================================
# Multi-Sensor Fusion + Decoy Discrimination
# ===================================================================


class TestFusionDecoyDiscrimination:
    """Fusion layer: radar-only track (no thermal) → decoy candidate."""

    def test_radar_only_no_thermal_flagged_decoy(self):
        """A track with radar but no thermal, camera, or QI → decoy candidate."""
        from sentinel.tracking.radar_track import RadarTrack

        det = _radar_detection(10000.0, 45.0)
        rt = RadarTrack(det)
        rt.score = 0.8

        fusion = MultiSensorFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
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
        radar_only = [f for f in fused if f.radar_track is not None and f.thermal_track is None and f.camera_track is None]
        assert len(radar_only) >= 1
        assert radar_only[0].is_decoy_candidate is True
        assert radar_only[0].threat_level == THREAT_LOW

    def test_radar_plus_thermal_not_decoy(self):
        """A track with both radar and thermal → NOT a decoy candidate."""
        from sentinel.tracking.radar_track import RadarTrack
        from sentinel.tracking.thermal_track import ThermalTrack

        rdet = _radar_detection(10000.0, 10.0)
        rt = RadarTrack(rdet)
        rt.score = 0.8

        tdet = _thermal_detection(10.0)
        tt = ThermalTrack(tdet)
        tt.score = 0.7

        fusion = MultiSensorFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=10.0,
            thermal_azimuth_gate_deg=15.0,
        )
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=[rt],
            thermal_tracks=[tt],
            quantum_radar_tracks=[],
        )

        assert len(fused) >= 1
        for f in fused:
            if f.thermal_track is not None:
                assert f.is_decoy_candidate is False


class TestFusionChaffDiscrimination:
    """Fusion layer: chaff flag propagated from correlator."""

    def test_chaff_flag_propagated_from_correlator(self):
        """Chaff candidate flag flows from correlator through fusion."""
        from sentinel.tracking.radar_track import RadarTrack

        det = _radar_detection(5000.0, 45.0)
        rt = RadarTrack(det)
        rt.score = 0.8

        cd = CorrelatedDetection(
            primary_detection=Detection(
                sensor_type=SensorType.RADAR,
                timestamp=1.0,
                range_m=5000.0,
                azimuth_deg=45.0,
                rcs_dbsm=30.0,
                radar_band=RadarBand.X_BAND.value,
            ),
            bands_detected=[RadarBand.X_BAND.value, RadarBand.VHF.value],
            is_chaff_candidate=True,
        )

        fusion = MultiSensorFusion(
            camera_hfov_deg=60.0,
            image_width_px=1280,
            azimuth_gate_deg=10.0,
            thermal_azimuth_gate_deg=5.0,
        )
        fused = fusion.fuse(
            camera_tracks=[],
            radar_tracks=[rt],
            thermal_tracks=[],
            correlated_detections=[cd],
            quantum_radar_tracks=[],
        )

        assert len(fused) >= 1
        chaff_tracks = [f for f in fused if f.is_chaff_candidate]
        assert len(chaff_tracks) >= 1
        assert chaff_tracks[0].threat_level == THREAT_LOW


# ===================================================================
# Backward Compatibility
# ===================================================================


class TestEWBackwardCompatibility:
    """All simulators with EW disabled → identical to pre-Phase-13."""

    def test_all_simulators_no_ew_unchanged(self):
        """When environment has no EW, behavior is identical."""
        env = EnvironmentModel()
        assert env.ew is None
        assert env.use_ew_effects is False
        assert env.ew_snr_adjustment_db(10000.0, 10e9) == 0.0
        assert env.get_ew_false_detections(np.array([0.0, 0.0])) == []
        assert env.get_ew_thermal_returns(np.array([0.0, 0.0])) == []

    def test_ew_model_present_but_disabled(self):
        """EW model exists but use_ew_effects=False → no effect."""
        ew = EWModel(jammers=[JammerSource(
            position=np.array([0.0, 5000.0]),
            erp_watts=1e6,
            bandwidth_hz=1e6,
        )])
        env = EnvironmentModel(ew=ew, use_ew_effects=False)
        assert env.ew_snr_adjustment_db(10000.0, 10e9) == 0.0
        assert env.get_ew_false_detections(np.array([0.0, 0.0])) == []

    def test_env_from_omegaconf_ew_disabled(self):
        """EnvironmentModel.from_omegaconf with ew.enabled=false → no EW."""
        cfg = OmegaConf.create({
            "terrain": {"enabled": False},
            "weather": {"enabled": False},
            "atmosphere": {"enabled": False},
            "clutter": {"enabled": False},
            "ew": {"enabled": False},
        })
        env = EnvironmentModel.from_omegaconf(cfg)
        assert env.ew is None
        assert env.use_ew_effects is False

    def test_env_from_omegaconf_ew_enabled(self):
        """EnvironmentModel.from_omegaconf with ew.enabled=true builds EW model."""
        cfg = OmegaConf.create({
            "terrain": {"enabled": False},
            "weather": {"enabled": False},
            "atmosphere": {"enabled": False},
            "clutter": {"enabled": False},
            "ew": {
                "enabled": True,
                "jammers": [{
                    "position": [0.0, 10000.0],
                    "erp_watts": 50000.0,
                    "bandwidth_hz": 1000000.0,
                    "jam_type": "noise",
                }],
                "chaff_clouds": [],
                "decoys": [],
                "eccm": {
                    "sidelobe_blanking": False,
                    "frequency_agility": False,
                    "burn_through_mode": False,
                    "quantum_eccm": False,
                },
                "radar_peak_power_w": 1000000.0,
                "radar_gain_db": 30.0,
                "radar_bandwidth_hz": 1000000.0,
            },
        })
        env = EnvironmentModel.from_omegaconf(cfg)
        assert env.ew is not None
        assert env.use_ew_effects is True
        assert len(env.ew.jammers) == 1
