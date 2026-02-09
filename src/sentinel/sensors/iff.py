"""IFF (Identification Friend or Foe) interrogation simulation.

Models a secondary surveillance radar (SSR) operating at 1030/1090 MHz
that interrogates aircraft transponders to determine identity. Supports
Modes 1-5 including cryptographic military modes.

All IFF features default OFF. When ``IFFConfig.enabled`` is False or no
IFF interrogator is created, the system behaves identically to pre-Phase-19.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig

from sentinel.core.types import IFFCode, IFFMode


# ===================================================================
# Physics
# ===================================================================


def iff_interrogation_probability(
    range_m: float,
    max_range_m: float,
    base_probability: float = 0.98,
) -> float:
    """Probability of successful IFF interrogation at given range.

    Models the 1030 MHz interrogation link budget. Probability drops
    off quadratically with range (free-space path loss).

    Returns:
        Probability in [0, 1].
    """
    if range_m <= 0 or max_range_m <= 0:
        return 0.0
    if range_m >= max_range_m:
        return 0.0
    ratio = range_m / max_range_m
    return base_probability * (1.0 - ratio * ratio)


def iff_spoof_detection_score(
    crypto_valid: bool,
    code_consistent: bool,
    kinematic_plausible: bool,
) -> float:
    """Compute spoof detection score from multiple indicators.

    Higher score = more likely spoofed. Each failed check adds 0.33.

    Returns:
        Score in [0, 1]. Above threshold (~0.7) triggers SPOOF_SUSPECT.
    """
    score = 0.0
    if not crypto_valid:
        score += 0.40
    if not code_consistent:
        score += 0.30
    if not kinematic_plausible:
        score += 0.30
    return min(score, 1.0)


def iff_altitude_check(
    mode_c_altitude_ft: float | None,
    radar_range_m: float,
    radar_elevation_deg: float | None,
) -> bool:
    """Cross-check Mode C altitude against radar elevation data.

    Returns True if altitude is consistent (or data unavailable).
    """
    if mode_c_altitude_ft is None or radar_elevation_deg is None:
        return True  # Can't cross-check, assume OK
    # Compute expected altitude from radar geometry
    el_rad = math.radians(radar_elevation_deg)
    expected_alt_m = radar_range_m * math.sin(el_rad)
    expected_alt_ft = expected_alt_m * 3.28084
    # Allow 500 ft tolerance
    return abs(mode_c_altitude_ft - expected_alt_ft) < 500.0


# ===================================================================
# Data Structures
# ===================================================================


@dataclass
class IFFTransponder:
    """IFF transponder configuration for a simulated target.

    Friendly targets have ``enabled=True`` with valid crypto modes.
    Hostile targets typically have ``enabled=False`` or ``is_spoofed=True``.
    """

    enabled: bool = False
    modes: list[IFFMode] = field(default_factory=list)
    mode_3a_code: str = "0000"  # 4-digit octal squawk code
    mode_s_address: str = ""  # 24-bit ICAO hex address
    mode_c_altitude_ft: float | None = None  # Reported altitude
    mode_4_valid: bool = False  # Has valid Mode 4 crypto key
    mode_5_valid: bool = False  # Has valid Mode 5 crypto key
    is_spoofed: bool = False  # Simulation flag: forged codes
    reliability: float = 1.0  # Transponder reply probability (0-1)

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig | dict) -> IFFTransponder:
        """Parse from OmegaConf config dict."""
        modes = [IFFMode(m) for m in cfg.get("modes", [])]
        return cls(
            enabled=cfg.get("enabled", False),
            modes=modes,
            mode_3a_code=str(cfg.get("mode_3a_code", "0000")),
            mode_s_address=str(cfg.get("mode_s_address", "")),
            mode_c_altitude_ft=cfg.get("mode_c_altitude_ft", None),
            mode_4_valid=cfg.get("mode_4_valid", False),
            mode_5_valid=cfg.get("mode_5_valid", False),
            is_spoofed=cfg.get("is_spoofed", False),
            reliability=cfg.get("reliability", 1.0),
        )


@dataclass
class IFFResponse:
    """Result of a single IFF interrogation of one target."""

    target_id: str
    timestamp: float
    responded: bool
    modes_replied: list[IFFMode] = field(default_factory=list)
    mode_3a_code: str | None = None
    mode_s_address: str | None = None
    mode_c_altitude_ft: float | None = None
    mode_4_authenticated: bool = False
    mode_5_authenticated: bool = False
    identification: IFFCode = IFFCode.PENDING
    confidence: float = 0.0
    is_spoof_suspect: bool = False
    interrogation_range_m: float = 0.0


@dataclass
class IFFResult:
    """Aggregated IFF identification for a track over multiple interrogations."""

    identification: IFFCode = IFFCode.UNKNOWN
    confidence: float = 0.0
    consecutive_responses: int = 0
    consecutive_no_responses: int = 0
    mode_3a_code: str | None = None
    mode_s_address: str | None = None
    last_authenticated_mode: IFFMode | None = None
    spoof_indicators: int = 0
    total_interrogations: int = 0


# ===================================================================
# Configuration
# ===================================================================


@dataclass
class IFFConfig:
    """IFF interrogator configuration."""

    enabled: bool = False
    max_interrogation_range_m: float = 400_000.0  # ~400 km
    interrogation_rate_hz: float = 1.0
    base_interrogation_probability: float = 0.98
    modes: list[IFFMode] = field(
        default_factory=lambda: [IFFMode.MODE_3A, IFFMode.MODE_C]
    )
    require_mode_4: bool = False  # Require crypto for FRIENDLY
    require_mode_5: bool = False
    spoof_detection_enabled: bool = True
    spoof_detection_threshold: float = 0.7
    no_response_hostile_threshold: int = 3  # Consecutive misses → ASSUMED_HOSTILE
    friendly_response_threshold: int = 2  # Consecutive hits → ASSUMED_FRIENDLY
    controlled_airspace: bool = False  # No IFF → higher threat
    friendly_codes: list[str] = field(default_factory=list)  # Known-good squawk codes

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig | dict) -> IFFConfig:
        """Parse from OmegaConf config dict."""
        modes = [IFFMode(m) for m in cfg.get("modes", ["mode_3a", "mode_c"])]
        return cls(
            enabled=cfg.get("enabled", False),
            max_interrogation_range_m=cfg.get("max_interrogation_range_m", 400_000.0),
            interrogation_rate_hz=cfg.get("interrogation_rate_hz", 1.0),
            base_interrogation_probability=cfg.get("base_interrogation_probability", 0.98),
            modes=modes,
            require_mode_4=cfg.get("require_mode_4", False),
            require_mode_5=cfg.get("require_mode_5", False),
            spoof_detection_enabled=cfg.get("spoof_detection_enabled", True),
            spoof_detection_threshold=cfg.get("spoof_detection_threshold", 0.7),
            no_response_hostile_threshold=cfg.get("no_response_hostile_threshold", 3),
            friendly_response_threshold=cfg.get("friendly_response_threshold", 2),
            controlled_airspace=cfg.get("controlled_airspace", False),
            friendly_codes=list(cfg.get("friendly_codes", [])),
        )


# ===================================================================
# IFF Interrogator
# ===================================================================


class IFFInterrogator:
    """Simulates IFF interrogation co-located with radar.

    Interrogates all targets within range each cycle. Targets with enabled
    transponders respond; targets without are classified as UNKNOWN and
    eventually escalated to ASSUMED_HOSTILE.

    IFF results are keyed by ``target_id`` and accumulate over multiple
    interrogations to build confidence.
    """

    def __init__(
        self,
        config: IFFConfig,
        seed: int | None = None,
    ):
        self._config = config
        self._rng = np.random.default_rng(seed)
        # Running IFF state per target_id
        self._track_results: dict[str, IFFResult] = {}

    @property
    def config(self) -> IFFConfig:
        return self._config

    @property
    def track_results(self) -> dict[str, IFFResult]:
        """Current IFF identification results by target_id."""
        return dict(self._track_results)

    def interrogate_target(
        self,
        target_id: str,
        transponder: IFFTransponder | None,
        target_range_m: float,
        timestamp: float,
    ) -> IFFResponse:
        """Interrogate a single target and return the response.

        Args:
            target_id: Target identifier.
            transponder: Target's IFF transponder config (None if no transponder).
            target_range_m: Distance from interrogator to target.
            timestamp: Current simulation time.

        Returns:
            IFFResponse with interrogation results.
        """
        cfg = self._config

        # Check if in interrogation range
        p_interrog = iff_interrogation_probability(
            target_range_m,
            cfg.max_interrogation_range_m,
            cfg.base_interrogation_probability,
        )

        # No transponder or transponder disabled → no response
        if transponder is None or not transponder.enabled:
            return IFFResponse(
                target_id=target_id,
                timestamp=timestamp,
                responded=False,
                interrogation_range_m=target_range_m,
            )

        # Interrogation signal reaches target?
        if self._rng.random() > p_interrog:
            return IFFResponse(
                target_id=target_id,
                timestamp=timestamp,
                responded=False,
                interrogation_range_m=target_range_m,
            )

        # Transponder reliability check
        if self._rng.random() > transponder.reliability:
            return IFFResponse(
                target_id=target_id,
                timestamp=timestamp,
                responded=False,
                interrogation_range_m=target_range_m,
            )

        # Transponder responds — determine which modes reply
        modes_replied: list[IFFMode] = []
        for mode in cfg.modes:
            if mode in transponder.modes:
                modes_replied.append(mode)

        if not modes_replied:
            # Transponder active but doesn't support queried modes
            return IFFResponse(
                target_id=target_id,
                timestamp=timestamp,
                responded=False,
                interrogation_range_m=target_range_m,
            )

        # Build response
        mode_3a = transponder.mode_3a_code if IFFMode.MODE_3A in modes_replied else None
        mode_s = transponder.mode_s_address if IFFMode.MODE_S in modes_replied else None
        mode_c_alt = transponder.mode_c_altitude_ft if IFFMode.MODE_C in modes_replied else None

        # Crypto authentication
        mode_4_auth = False
        mode_5_auth = False
        if IFFMode.MODE_4 in modes_replied:
            # For spoofed transponders, crypto is invalid
            mode_4_auth = transponder.mode_4_valid and not transponder.is_spoofed
        if IFFMode.MODE_5 in modes_replied:
            mode_5_auth = transponder.mode_5_valid and not transponder.is_spoofed

        # Determine initial identification from this response
        identification = IFFCode.UNKNOWN
        confidence = 0.0

        # Crypto authentication → strongest identification
        if mode_5_auth:
            identification = IFFCode.FRIENDLY
            confidence = 0.95
        elif mode_4_auth:
            identification = IFFCode.FRIENDLY
            confidence = 0.90
        elif mode_3a is not None:
            # Check against known friendly codes
            if cfg.friendly_codes and mode_3a in cfg.friendly_codes:
                identification = IFFCode.FRIENDLY
                confidence = 0.70
            else:
                # Response but no crypto / unknown code
                identification = IFFCode.UNKNOWN
                confidence = 0.30

        # Spoof detection
        is_spoof = False
        if cfg.spoof_detection_enabled and transponder.is_spoofed:
            # In real system, detect via crypto failure + code analysis
            # Crypto modes requested but authentication failed
            crypto_valid = True
            if (cfg.require_mode_4 and IFFMode.MODE_4 in modes_replied
                    and not mode_4_auth):
                crypto_valid = False
            if (cfg.require_mode_5 and IFFMode.MODE_5 in modes_replied
                    and not mode_5_auth):
                crypto_valid = False

            spoof_score = iff_spoof_detection_score(
                crypto_valid=crypto_valid,
                code_consistent=True,  # Would need history tracking
                kinematic_plausible=True,  # Would need kinematic data
            )
            if spoof_score >= cfg.spoof_detection_threshold:
                is_spoof = True
                identification = IFFCode.SPOOF_SUSPECT
                confidence = spoof_score

        return IFFResponse(
            target_id=target_id,
            timestamp=timestamp,
            responded=True,
            modes_replied=modes_replied,
            mode_3a_code=mode_3a,
            mode_s_address=mode_s,
            mode_c_altitude_ft=mode_c_alt,
            mode_4_authenticated=mode_4_auth,
            mode_5_authenticated=mode_5_auth,
            identification=identification,
            confidence=confidence,
            is_spoof_suspect=is_spoof,
            interrogation_range_m=target_range_m,
        )

    def update_track_identification(
        self,
        target_id: str,
        response: IFFResponse,
    ) -> IFFResult:
        """Update running IFF identification for a track.

        Accumulates responses over time to build confidence. Multiple
        successful interrogations increase confidence; consecutive
        failures escalate toward ASSUMED_HOSTILE.

        Returns:
            Updated IFFResult for this track.
        """
        cfg = self._config

        if target_id not in self._track_results:
            self._track_results[target_id] = IFFResult()

        result = self._track_results[target_id]
        result.total_interrogations += 1

        if response.responded:
            result.consecutive_responses += 1
            result.consecutive_no_responses = 0

            # Update codes
            if response.mode_3a_code is not None:
                result.mode_3a_code = response.mode_3a_code
            if response.mode_s_address is not None:
                result.mode_s_address = response.mode_s_address

            # Update authentication
            if response.mode_5_authenticated:
                result.last_authenticated_mode = IFFMode.MODE_5
            elif response.mode_4_authenticated:
                result.last_authenticated_mode = IFFMode.MODE_4

            # Spoof detection
            if response.is_spoof_suspect:
                result.spoof_indicators += 1
                result.identification = IFFCode.SPOOF_SUSPECT
                result.confidence = min(
                    0.5 + result.spoof_indicators * 0.15, 0.99
                )
            elif response.identification == IFFCode.FRIENDLY:
                # Crypto-authenticated friendly
                if result.identification != IFFCode.SPOOF_SUSPECT:
                    result.identification = IFFCode.FRIENDLY
                    # Confidence grows with consecutive responses
                    result.confidence = min(
                        response.confidence + result.consecutive_responses * 0.02,
                        0.99,
                    )
            elif result.consecutive_responses >= cfg.friendly_response_threshold:
                # Multiple responses without crypto → assumed friendly
                if result.identification not in (
                    IFFCode.FRIENDLY, IFFCode.SPOOF_SUSPECT
                ):
                    result.identification = IFFCode.ASSUMED_FRIENDLY
                    result.confidence = min(
                        0.50 + result.consecutive_responses * 0.05,
                        0.80,
                    )
        else:
            # No response
            result.consecutive_no_responses += 1
            result.consecutive_responses = 0

            if result.consecutive_no_responses >= cfg.no_response_hostile_threshold:
                # Multiple failures → assumed hostile
                if result.identification not in (
                    IFFCode.FRIENDLY, IFFCode.SPOOF_SUSPECT
                ):
                    result.identification = IFFCode.ASSUMED_HOSTILE
                    result.confidence = min(
                        0.50 + result.consecutive_no_responses * 0.05,
                        0.90,
                    )
            elif result.identification == IFFCode.UNKNOWN:
                result.identification = IFFCode.PENDING
                result.confidence = 0.0

        return result

    def interrogate_all(
        self,
        targets: list[tuple[str, IFFTransponder | None, float]],
        timestamp: float,
    ) -> dict[str, IFFResult]:
        """Interrogate all targets and update running identification.

        Args:
            targets: List of (target_id, transponder, range_m) tuples.
            timestamp: Current simulation time.

        Returns:
            Dict of target_id → IFFResult with updated identifications.
        """
        results: dict[str, IFFResult] = {}
        for target_id, transponder, range_m in targets:
            response = self.interrogate_target(
                target_id, transponder, range_m, timestamp,
            )
            result = self.update_track_identification(target_id, response)
            results[target_id] = result
        return results

    def get_result(self, target_id: str) -> IFFResult | None:
        """Get current IFF result for a target."""
        return self._track_results.get(target_id)

    def reset(self) -> None:
        """Clear all accumulated IFF state."""
        self._track_results.clear()
