"""Launch the SENTINEL web dashboard with simulated live data.

Run:
    python scripts/demo_dashboard.py

Then open http://localhost:8080 in your browser.
"""

from __future__ import annotations

import math
import random
import threading
import time

import uvicorn

from sentinel.ui.web.server import create_app
from sentinel.ui.web.state_buffer import StateBuffer, StateSnapshot

UPDATE_HZ = 10
VIDEO_FPS = 15
PORT = 8080


def _generate_snapshot(t: float) -> StateSnapshot:
    """Create a snapshot with moving tracks that change over time."""
    elapsed = t % 120  # 2-minute cycle

    # Simulated camera tracks
    camera_tracks = [
        {
            "track_id": "CAM-001",
            "state": "confirmed",
            "position": [640 + int(100 * math.sin(t * 0.5)), 360 + int(50 * math.cos(t * 0.3))],
            "velocity": [12.5, -3.2],
            "score": 0.95,
        },
        {
            "track_id": "CAM-002",
            "state": "confirmed",
            "position": [200 + int(30 * math.cos(t * 0.7)), 100 + int(20 * math.sin(t * 0.4))],
            "velocity": [-5.0, 8.1],
            "score": 0.82,
        },
        {
            "track_id": "CAM-003",
            "state": "tentative",
            "position": [900 + int(40 * math.sin(t * 0.2)), 500],
            "velocity": [2.0, 0.0],
            "score": 0.35,
        },
    ]

    # Simulated radar tracks (orbiting)
    radar_tracks = [
        {
            "track_id": "RDR-001",
            "state": "confirmed",
            "range_m": 8000 + 500 * math.sin(t * 0.1),
            "azimuth_deg": (15.0 + t * 2.0) % 360 - 180,
            "velocity_mps": 250.0,
            "score": 0.91,
        },
        {
            "track_id": "RDR-002",
            "state": "confirmed",
            "range_m": 5000 + 300 * math.cos(t * 0.15),
            "azimuth_deg": (-10.0 - t * 1.5) % 360 - 180,
            "velocity_mps": 80.0,
            "score": 0.78,
        },
        {
            "track_id": "RDR-003",
            "state": "coasting",
            "range_m": 15000,
            "azimuth_deg": 45.0 + 5 * math.sin(t * 0.05),
            "velocity_mps": 600.0,
            "score": 0.55,
        },
    ]

    # Simulated thermal tracks
    thermal_tracks = [
        {
            "track_id": "THM-001",
            "state": "confirmed",
            "azimuth_deg": 14.5 + 2 * math.sin(t * 0.3),
            "temperature_k": 1800.0 + 200 * math.sin(t * 0.5),
        },
        {
            "track_id": "THM-002",
            "state": "confirmed",
            "azimuth_deg": -45.0 + 3 * math.cos(t * 0.2),
            "temperature_k": 900.0,
        },
    ]

    # Enhanced fused tracks
    enhanced_fused = [
        {
            "fused_id": "EFT-001",
            "sensor_count": 3,
            "range_m": 8000 + 500 * math.sin(t * 0.1),
            "azimuth_deg": (15.0 + t * 2.0) % 360 - 180,
            "velocity_mps": 250.0,
            "threat_level": "CRITICAL",
            "is_stealth_candidate": False,
            "is_hypersonic_candidate": True,
            "fusion_quality": 0.92,
        },
        {
            "fused_id": "EFT-002",
            "sensor_count": 2,
            "range_m": 5000 + 300 * math.cos(t * 0.15),
            "azimuth_deg": (-10.0 - t * 1.5) % 360 - 180,
            "velocity_mps": 80.0,
            "threat_level": "HIGH",
            "is_stealth_candidate": True,
            "is_hypersonic_candidate": False,
            "fusion_quality": 0.75,
        },
        {
            "fused_id": "EFT-003",
            "sensor_count": 1,
            "range_m": 15000,
            "azimuth_deg": 45.0 + 5 * math.sin(t * 0.05),
            "velocity_mps": 600.0,
            "threat_level": "LOW",
            "is_stealth_candidate": False,
            "is_hypersonic_candidate": False,
            "fusion_quality": 0.50,
        },
    ]

    # Simulated system status
    fps = 28.0 + 4 * math.sin(t * 0.3) + random.uniform(-0.5, 0.5)
    system_status = {
        "fps": round(fps, 1),
        "track_count": len(camera_tracks),
        "confirmed_count": sum(1 for tr in camera_tracks if tr["state"] == "confirmed"),
        "detection_count": len(camera_tracks) + random.randint(0, 3),
        "uptime": elapsed,
        "camera_connected": True,
        "radar_track_count": len(radar_tracks),
        "thermal_track_count": len(thermal_tracks),
        "fused_track_count": len(enhanced_fused),
        "sensor_health": {
            "camera": {"enabled": True, "error_count": 0},
            "radar": {"enabled": True, "error_count": 0},
            "thermal": {"enabled": True, "error_count": 0},
            "quantum_radar": {"enabled": True, "error_count": 0},
            "multifreq_radar": {"enabled": True, "error_count": 0},
            "fusion": {"enabled": True, "error_count": 0},
        },
        "detect_ms": round(6.0 + 3 * math.sin(t * 0.7) + random.uniform(-0.3, 0.3), 1),
        "track_ms": round(1.0 + 0.5 * math.sin(t * 0.5) + random.uniform(-0.1, 0.1), 1),
        "radar_ms": round(0.8 + 0.3 * math.cos(t * 0.4) + random.uniform(-0.1, 0.1), 1),
        "fusion_ms": round(0.4 + 0.2 * math.sin(t * 0.6) + random.uniform(-0.05, 0.05), 1),
        "render_ms": round(2.5 + 1.0 * math.cos(t * 0.8) + random.uniform(-0.2, 0.2), 1),
        "threat_counts": {"CRITICAL": 1, "HIGH": 1, "MEDIUM": 0, "LOW": 1},
    }

    return StateSnapshot(
        timestamp=time.time(),
        system_status=system_status,
        camera_tracks=camera_tracks,
        radar_tracks=radar_tracks,
        thermal_tracks=thermal_tracks,
        fused_tracks=[],
        enhanced_fused_tracks=enhanced_fused,
        hud_frame_jpeg=None,  # No HUD frame in demo mode
    )


def _data_pump(buf: StateBuffer, hz: int) -> None:
    """Background thread that feeds simulated data into the buffer."""
    interval = 1.0 / hz
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        buf.update(_generate_snapshot(t))
        time.sleep(interval)


def main() -> None:
    buf = StateBuffer()
    app = create_app(buf, update_hz=UPDATE_HZ, video_fps=VIDEO_FPS)

    # Start the simulated data pump
    pump = threading.Thread(target=_data_pump, args=(buf, UPDATE_HZ), daemon=True)
    pump.start()

    print(f"\n  SENTINEL Dashboard Demo")
    print(f"  Open http://localhost:{PORT} in your browser\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
