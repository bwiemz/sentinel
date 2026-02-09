"""Environment modeling: terrain masking, atmospheric propagation, weather, and clutter.

All effects are optional and default OFF. When disabled, simulators behave
identically to pre-Phase-12 code (backward compatible).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig

from sentinel.core.types import RadarBand, ThermalBand

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Approximate center frequencies for each radar band (Hz).
BAND_CENTER_FREQ_HZ: dict[RadarBand, float] = {
    RadarBand.VHF: 150e6,
    RadarBand.UHF: 600e6,
    RadarBand.L_BAND: 1.5e9,
    RadarBand.S_BAND: 3.0e9,
    RadarBand.X_BAND: 10.0e9,
}


# ===================================================================
# Terrain masking
# ===================================================================


@dataclass
class TerrainGrid:
    """Synthetic 2D elevation grid for line-of-sight computation.

    The grid covers a rectangular area.  Cell (0, 0) in the elevation
    array maps to world coordinate (origin_x_m, origin_y_m).
    Elevation values are in meters above sea level (ASL).
    """

    elevation_data: np.ndarray  # shape (ny, nx), float, meters ASL
    resolution_m: float  # grid cell size in meters
    origin_x_m: float = 0.0  # world-x of the grid's left edge
    origin_y_m: float = 0.0  # world-y of the grid's bottom edge

    # -- constructors -------------------------------------------------------

    @classmethod
    def flat(cls, extent_m: float = 50000.0, resolution_m: float = 100.0) -> TerrainGrid:
        """Create a flat terrain (all zeros)."""
        n = max(1, int(extent_m / resolution_m))
        return cls(
            elevation_data=np.zeros((n, n), dtype=np.float64),
            resolution_m=resolution_m,
            origin_x_m=-extent_m / 2.0,
            origin_y_m=-extent_m / 2.0,
        )

    @classmethod
    def procedural_hills(
        cls,
        extent_m: float = 50000.0,
        resolution_m: float = 100.0,
        max_elevation_m: float = 500.0,
        seed: int = 42,
    ) -> TerrainGrid:
        """Generate procedural terrain using superimposed sine waves."""
        rng = np.random.RandomState(seed)
        n = max(1, int(extent_m / resolution_m))
        elev = np.zeros((n, n), dtype=np.float64)
        # Superimpose several sine-wave "hills" at random positions
        n_hills = 8
        for _ in range(n_hills):
            cx = rng.uniform(0, n)
            cy = rng.uniform(0, n)
            wavelength = rng.uniform(n * 0.1, n * 0.4)
            amplitude = rng.uniform(0.2, 1.0) * max_elevation_m / n_hills
            xs = np.arange(n, dtype=np.float64)
            ys = np.arange(n, dtype=np.float64)
            xx, yy = np.meshgrid(xs, ys)
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            elev += amplitude * np.maximum(0.0, np.cos(np.pi * dist / wavelength))
        return cls(
            elevation_data=np.clip(elev, 0.0, max_elevation_m),
            resolution_m=resolution_m,
            origin_x_m=-extent_m / 2.0,
            origin_y_m=-extent_m / 2.0,
        )

    @classmethod
    def ridge(
        cls,
        extent_m: float = 50000.0,
        resolution_m: float = 100.0,
        ridge_x_m: float = 5000.0,
        ridge_height_m: float = 300.0,
        ridge_width_m: float = 500.0,
    ) -> TerrainGrid:
        """Generate a single east-west ridge at a given x position.

        The ridge is a Gaussian cross-section in x, infinite in y.
        """
        n = max(1, int(extent_m / resolution_m))
        origin_x = -extent_m / 2.0
        origin_y = -extent_m / 2.0
        xs = origin_x + np.arange(n) * resolution_m
        sigma = ridge_width_m / 2.0
        ridge_profile = ridge_height_m * np.exp(-0.5 * ((xs - ridge_x_m) / max(sigma, 1.0)) ** 2)
        elev = np.tile(ridge_profile, (n, 1))
        return cls(
            elevation_data=elev,
            resolution_m=resolution_m,
            origin_x_m=origin_x,
            origin_y_m=origin_y,
        )

    # -- queries ------------------------------------------------------------

    def elevation_at(self, x: float, y: float) -> float:
        """Bilinear-interpolated elevation at world coordinates (x, y).

        Returns 0.0 for points outside the grid.
        """
        # Convert to fractional grid indices
        fi = (x - self.origin_x_m) / self.resolution_m
        fj = (y - self.origin_y_m) / self.resolution_m
        ny, nx = self.elevation_data.shape
        if fi < 0 or fi >= nx - 1 or fj < 0 or fj >= ny - 1:
            return 0.0
        i0 = int(fi)
        j0 = int(fj)
        dx = fi - i0
        dy = fj - j0
        e00 = self.elevation_data[j0, i0]
        e10 = self.elevation_data[j0, min(i0 + 1, nx - 1)]
        e01 = self.elevation_data[min(j0 + 1, ny - 1), i0]
        e11 = self.elevation_data[min(j0 + 1, ny - 1), min(i0 + 1, nx - 1)]
        return float(
            e00 * (1 - dx) * (1 - dy)
            + e10 * dx * (1 - dy)
            + e01 * (1 - dx) * dy
            + e11 * dx * dy
        )


def line_of_sight(
    terrain: TerrainGrid,
    sensor_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
    n_samples: int = 50,
) -> bool:
    """Check whether a clear line-of-sight exists between *sensor_pos* and *target_pos*.

    Both positions are (x, y, z_above_sea_level) tuples.  The function
    samples *n_samples* points along the straight line and checks whether
    the terrain elevation at each point exceeds the ray height.

    Returns ``True`` if LoS is clear, ``False`` if blocked.
    """
    sx, sy, sz = sensor_pos
    tx, ty, tz = target_pos
    for i in range(1, n_samples):
        frac = i / n_samples
        px = sx + frac * (tx - sx)
        py = sy + frac * (ty - sy)
        ray_z = sz + frac * (tz - sz)
        terrain_z = terrain.elevation_at(px, py)
        if terrain_z > ray_z:
            return False
    return True


# ===================================================================
# Atmospheric propagation
# ===================================================================


def atmospheric_attenuation_db_per_km(
    freq_hz: float,
    temperature_k: float = 290.0,
    humidity_pct: float = 50.0,
) -> float:
    """Clear-air atmospheric attenuation (dB/km, one-way).

    Simplified model inspired by ITU-R P.676:
    - O₂ absorption peak near 60 GHz
    - H₂O absorption peak near 22 GHz
    - Below 10 GHz: very low (< 0.01 dB/km)
    """
    freq_ghz = freq_hz / 1e9
    if freq_ghz <= 0:
        return 0.0
    # Dry air (oxygen): quadratic ramp peaking at ~60 GHz
    dry = 0.0 if freq_ghz < 1.0 else 6e-5 * freq_ghz ** 2
    # Water vapor: broad peak near 22 GHz, scaled by humidity
    humidity_factor = humidity_pct / 100.0
    wet = humidity_factor * 0.001 * freq_ghz * math.exp(-0.5 * ((freq_ghz - 22.0) / 8.0) ** 2)
    # Temperature scaling (higher temp → slightly less O₂ absorption)
    temp_factor = 290.0 / max(temperature_k, 100.0)
    return max(0.0, (dry + wet) * temp_factor)


def rain_attenuation_db_per_km(freq_hz: float, rain_rate_mm_h: float) -> float:
    """Rain attenuation (dB/km, one-way) using simplified ITU-R P.838.

    Higher frequencies are attenuated much more than lower ones.
    """
    if rain_rate_mm_h <= 0 or freq_hz <= 0:
        return 0.0
    freq_ghz = freq_hz / 1e9
    # k and alpha coefficients (simplified piecewise model)
    if freq_ghz < 1.0:
        k, alpha = 0.0, 1.0  # negligible
    elif freq_ghz < 5.0:
        k = 5e-5 * freq_ghz ** 2
        alpha = 1.0
    elif freq_ghz < 15.0:
        k = 0.001 * (freq_ghz / 10.0) ** 2
        alpha = 1.1 + 0.01 * freq_ghz
    else:
        k = 0.005 * (freq_ghz / 15.0) ** 1.5
        alpha = 1.2
    return k * rain_rate_mm_h ** alpha


def total_propagation_loss_db(
    freq_hz: float,
    range_m: float,
    rain_rate_mm_h: float = 0.0,
    temperature_k: float = 290.0,
    humidity_pct: float = 50.0,
) -> float:
    """Total **two-way** atmospheric propagation loss in dB.

    = 2 × range_km × (clear_air + rain) dB/km.
    """
    range_km = range_m / 1000.0
    atten_per_km = atmospheric_attenuation_db_per_km(freq_hz, temperature_k, humidity_pct)
    atten_per_km += rain_attenuation_db_per_km(freq_hz, rain_rate_mm_h)
    return 2.0 * range_km * atten_per_km


def thermal_atmospheric_transmission(
    band: ThermalBand,
    range_m: float,
    humidity_pct: float = 50.0,
    rain_rate_mm_h: float = 0.0,
    visibility_km: float = 20.0,
) -> float:
    """Atmospheric transmission factor (0–1) for thermal/IR bands.

    MWIR (3-5 µm) transmits better than LWIR (8-12 µm) in humid or
    rainy conditions.  SWIR is most affected by fog/rain.  Low
    visibility degrades all bands via Beer-Lambert scattering.
    """
    range_km = range_m / 1000.0
    humidity_factor = humidity_pct / 100.0

    # Base extinction coefficient (1/km) by band
    if band == ThermalBand.MWIR:
        base_extinction = 0.05 + 0.15 * humidity_factor
    elif band == ThermalBand.LWIR:
        base_extinction = 0.08 + 0.25 * humidity_factor
    else:  # SWIR
        base_extinction = 0.10 + 0.30 * humidity_factor

    # Rain adds scattering
    if rain_rate_mm_h > 0:
        base_extinction += 0.02 * rain_rate_mm_h ** 0.6

    # Visibility reduction (Beer-Lambert)
    if visibility_km > 0:
        # Meteorological visibility relates to extinction at 550nm
        # IR extinction is generally lower, so we scale down
        vis_extinction = 3.912 / visibility_km * 0.3  # ~30% of visible extinction
        base_extinction += vis_extinction

    return max(0.0, min(1.0, math.exp(-base_extinction * range_km)))


# ===================================================================
# Weather effects
# ===================================================================


@dataclass
class WeatherConditions:
    """Current weather state for simulation.  Defaults = clear sky."""

    rain_rate_mm_h: float = 0.0
    humidity_pct: float = 50.0
    temperature_k: float = 290.0
    visibility_km: float = 20.0
    cloud_cover_pct: float = 0.0  # 0-100
    wind_speed_mps: float = 0.0
    sea_state: int = 0  # Douglas scale 0-9


def weather_thermal_contrast_factor(weather: WeatherConditions) -> float:
    """Factor (0–1) reducing thermal contrast due to cloud cover.

    Overcast sky raises effective sky background temperature, reducing
    contrast between target and background.
    """
    # At 100% cloud cover, effective contrast drops to ~60% of clear-sky value
    cloud_fraction = min(weather.cloud_cover_pct, 100.0) / 100.0
    return 1.0 - 0.4 * cloud_fraction


def weather_visibility_range_factor(weather: WeatherConditions, max_range_m: float) -> float:
    """Fraction (0–1) of *max_range_m* usable under current visibility.

    In poor visibility (fog, heavy rain) the effective thermal sensing
    range shrinks proportionally.
    """
    if max_range_m <= 0:
        return 0.0
    vis_range_m = weather.visibility_km * 1000.0
    return min(1.0, vis_range_m / max_range_m)


# ===================================================================
# Clutter model
# ===================================================================


def surface_clutter_snr_reduction_db(
    elevation_angle_deg: float,
    range_m: float,
    beam_width_deg: float = 2.0,
    surface_sigma0_db: float = -30.0,
    sea_state: int = 0,
) -> float:
    """SNR reduction (positive dB value) from surface clutter.

    At low elevation angles the radar beam illuminates the ground/sea,
    generating clutter that competes with target returns.
    Above ~5° elevation, surface clutter is negligible.
    """
    if elevation_angle_deg > 5.0 or range_m <= 0:
        return 0.0
    # Grazing angle factor: more clutter at shallower angles
    graze_factor = max(0.0, 1.0 - elevation_angle_deg / 5.0)
    # Sea state adds extra clutter (up to +10 dB at sea_state 9)
    sea_boost_db = min(sea_state, 9) * 1.1
    # Clutter power scales with illuminated area (beam_width * range)
    illuminated_m = beam_width_deg * (math.pi / 180.0) * range_m
    clutter_db = surface_sigma0_db + 10.0 * math.log10(max(illuminated_m, 1.0)) + sea_boost_db
    # Reduction only meaningful when clutter is significant
    reduction = max(0.0, clutter_db + 50.0) * graze_factor  # +50 offset so σ0=-30 gives ~20*graze
    return min(reduction, 30.0)  # cap at 30 dB


def rain_clutter_snr_reduction_db(
    freq_hz: float,
    rain_rate_mm_h: float,
    range_m: float,
    beam_width_deg: float = 2.0,
) -> float:
    """SNR reduction (positive dB) from rain volume clutter.

    Rain within the radar beam generates backscatter competing with
    target returns.  Higher frequencies produce more rain clutter.
    """
    if rain_rate_mm_h <= 0 or freq_hz <= 0 or range_m <= 0:
        return 0.0
    freq_ghz = freq_hz / 1e9
    # Rain reflectivity scales with Z = 200 * R^1.6 (Marshall-Palmer)
    z_factor = 200.0 * rain_rate_mm_h ** 1.6
    # Frequency-dependent backscatter cross-section
    freq_factor = (freq_ghz / 10.0) ** 2 if freq_ghz < 20.0 else 4.0
    # Volume illuminated by the beam
    beam_rad = beam_width_deg * math.pi / 180.0
    volume_m3 = beam_rad ** 2 * range_m * 150.0  # crude pulse-volume estimate
    clutter_power = z_factor * freq_factor * volume_m3 * 1e-15
    if clutter_power <= 0:
        return 0.0
    return min(20.0, max(0.0, 10.0 * math.log10(1.0 + clutter_power)))


def clutter_false_alarm_rate_multiplier(
    base_far: float,
    elevation_angle_deg: float = 0.0,
    sea_state: int = 0,
    rain_rate_mm_h: float = 0.0,
) -> float:
    """Compute increased false alarm rate due to clutter.

    Returns the *effective* FAR (≥ base_far).
    """
    multiplier = 1.0
    # Surface clutter at low elevation
    if elevation_angle_deg < 5.0:
        graze_factor = 1.0 - elevation_angle_deg / 5.0
        multiplier += 2.0 * graze_factor * max(1.0, sea_state / 3.0)
    # Rain clutter
    if rain_rate_mm_h > 0:
        multiplier += 0.5 * (rain_rate_mm_h / 10.0) ** 0.8
    return base_far * multiplier


# ===================================================================
# EnvironmentModel facade
# ===================================================================


@dataclass
class EnvironmentModel:
    """Unified environment model combining terrain, weather, atmosphere, and clutter.

    Passed as a single optional object to each simulator config.  When
    feature flags are ``False`` (the default), the corresponding effects
    are skipped and simulator behavior is unchanged.
    """

    terrain: TerrainGrid | None = None
    weather: WeatherConditions = field(default_factory=WeatherConditions)
    sensor_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Feature flags (all default OFF)
    use_terrain_masking: bool = False
    use_atmospheric_propagation: bool = False
    use_weather_effects: bool = False
    use_clutter: bool = False

    # -- convenience methods ------------------------------------------------

    def is_target_visible(
        self,
        target_x: float,
        target_y: float,
        target_z: float = 0.0,
    ) -> bool:
        """Check LoS.  Returns ``True`` if terrain masking is disabled or LoS is clear."""
        if not self.use_terrain_masking or self.terrain is None:
            return True
        return line_of_sight(self.terrain, self.sensor_position, (target_x, target_y, target_z))

    def radar_snr_adjustment_db(
        self,
        freq_hz: float,
        range_m: float,
        elevation_angle_deg: float = 0.0,
    ) -> float:
        """Total SNR adjustment from atmosphere + clutter (negative = loss)."""
        total_db = 0.0
        if self.use_atmospheric_propagation:
            total_db -= total_propagation_loss_db(
                freq_hz,
                range_m,
                rain_rate_mm_h=self.weather.rain_rate_mm_h,
                temperature_k=self.weather.temperature_k,
                humidity_pct=self.weather.humidity_pct,
            )
        if self.use_clutter:
            total_db -= surface_clutter_snr_reduction_db(
                elevation_angle_deg,
                range_m,
                sea_state=self.weather.sea_state,
            )
            if self.weather.rain_rate_mm_h > 0:
                total_db -= rain_clutter_snr_reduction_db(
                    freq_hz, self.weather.rain_rate_mm_h, range_m
                )
        return total_db

    def thermal_detection_factor(self, band: ThermalBand, range_m: float) -> float:
        """Combined atmospheric + weather factor for thermal detection (0–1)."""
        factor = 1.0
        if self.use_atmospheric_propagation:
            factor *= thermal_atmospheric_transmission(
                band,
                range_m,
                humidity_pct=self.weather.humidity_pct,
                rain_rate_mm_h=self.weather.rain_rate_mm_h,
                visibility_km=self.weather.visibility_km,
            )
        if self.use_weather_effects:
            factor *= weather_thermal_contrast_factor(self.weather)
        return factor

    def effective_thermal_max_range(self, base_max_range_m: float) -> float:
        """Reduce thermal max range based on visibility."""
        if not self.use_weather_effects:
            return base_max_range_m
        return base_max_range_m * weather_visibility_range_factor(self.weather, base_max_range_m)

    def effective_false_alarm_rate(
        self,
        base_far: float,
        elevation_angle_deg: float = 0.0,
    ) -> float:
        """Adjust false alarm rate for clutter effects."""
        if not self.use_clutter:
            return base_far
        return clutter_false_alarm_rate_multiplier(
            base_far,
            elevation_angle_deg,
            sea_state=self.weather.sea_state,
            rain_rate_mm_h=self.weather.rain_rate_mm_h,
        )

    # -- config loader ------------------------------------------------------

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> EnvironmentModel:
        """Create from the ``sentinel.environment`` config section."""
        terrain_cfg = cfg.get("terrain", {})
        weather_cfg = cfg.get("weather", {})
        atmo_cfg = cfg.get("atmosphere", {})
        clutter_cfg = cfg.get("clutter", {})
        pos_cfg = cfg.get("sensor_position", {})

        # Build terrain
        terrain = None
        terrain_type = terrain_cfg.get("type", "flat") if terrain_cfg.get("enabled", False) else None
        if terrain_type == "flat":
            terrain = TerrainGrid.flat(
                extent_m=terrain_cfg.get("extent_m", 50000.0),
                resolution_m=terrain_cfg.get("resolution_m", 100.0),
            )
        elif terrain_type == "procedural":
            terrain = TerrainGrid.procedural_hills(
                extent_m=terrain_cfg.get("extent_m", 50000.0),
                resolution_m=terrain_cfg.get("resolution_m", 100.0),
                max_elevation_m=terrain_cfg.get("max_elevation_m", 500.0),
                seed=terrain_cfg.get("seed", 42),
            )

        # Build weather
        weather = WeatherConditions(
            rain_rate_mm_h=weather_cfg.get("rain_rate_mm_h", 0.0),
            humidity_pct=weather_cfg.get("humidity_pct", 50.0),
            temperature_k=weather_cfg.get("temperature_k", 290.0),
            visibility_km=weather_cfg.get("visibility_km", 20.0),
            cloud_cover_pct=weather_cfg.get("cloud_cover_pct", 0.0),
            wind_speed_mps=weather_cfg.get("wind_speed_mps", 0.0),
            sea_state=weather_cfg.get("sea_state", 0),
        )

        sensor_pos = (
            float(pos_cfg.get("x_m", 0.0)),
            float(pos_cfg.get("y_m", 0.0)),
            float(pos_cfg.get("altitude_m", 0.0)),
        )

        return cls(
            terrain=terrain,
            weather=weather,
            sensor_position=sensor_pos,
            use_terrain_masking=terrain_cfg.get("enabled", False),
            use_atmospheric_propagation=atmo_cfg.get("enabled", False),
            use_weather_effects=weather_cfg.get("enabled", False),
            use_clutter=clutter_cfg.get("enabled", False),
        )
