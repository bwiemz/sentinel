"""SENTINEL Track History & Replay demo.

Simulates a multi-sensor pipeline, records track data into a ring buffer,
exports to file (JSON or msgpack), imports back, and replays step-by-step.

Run:
    python scripts/demo_history.py
    python scripts/demo_history.py --duration 60 --format msgpack --compress
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sentinel.core.types import RecordingState, PlaybackState
from sentinel.history.buffer import HistoryBuffer
from sentinel.history.config import HistoryConfig
from sentinel.history.frame import HistoryFrame
from sentinel.history.recorder import HistoryRecorder
from sentinel.history.replay import ReplayController
from sentinel.history.storage import HistoryStorage
from sentinel.ui.web.state_buffer import StateBuffer


# ---------------------------------------------------------------------------
# Mock pipeline that generates moving tracks
# ---------------------------------------------------------------------------


class MockClock:
    def __init__(self, start: float = 1000.0):
        self._now = start
        self._start = start

    def now(self) -> float:
        return self._now

    def elapsed(self) -> float:
        return self._now - self._start

    def advance(self, dt: float) -> None:
        self._now += dt


@dataclass
class MockTrack:
    track_id: str
    score: float
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "score": self.score,
            "position": np.array([self.x, self.y]),
        }


@dataclass
class MockTrackManager:
    active_tracks: list[MockTrack] = field(default_factory=list)


class MockPipeline:
    def __init__(self, clock: MockClock):
        self._clock = clock
        self._cam_tracks = [
            MockTrack("CAM-1", 0.9, 100, 200),
            MockTrack("CAM-2", 0.7, 300, 400),
        ]
        self._radar_tracks = [
            MockTrack("RAD-1", 0.95, 5000, 3000),
            MockTrack("RAD-2", 0.85, 8000, 6000),
        ]
        self._thermal_tracks = [
            MockTrack("THM-1", 0.6, 2000, 1500),
        ]
        self._latest_tracks = self._cam_tracks
        self._latest_fused_tracks = []
        self._latest_enhanced_fused = []
        self._radar_track_manager = MockTrackManager(self._radar_tracks)
        self._thermal_track_manager = MockTrackManager(self._thermal_tracks)
        self._frame = 0

    def step(self, dt: float) -> None:
        """Move tracks to simulate motion."""
        self._frame += 1
        t = self._clock.elapsed()
        for tr in self._cam_tracks:
            tr.x = 200 + 100 * math.sin(t * 0.5)
            tr.y = 300 + 80 * math.cos(t * 0.3)
        for tr in self._radar_tracks:
            tr.x -= 10 * dt  # approaching
            tr.y += 5 * dt
        for tr in self._thermal_tracks:
            tr.x = 2000 + 500 * math.sin(t * 0.1)
            tr.y = 1500 + 200 * math.cos(t * 0.2)

    def get_system_status(self) -> dict[str, Any]:
        return {
            "fps": 30,
            "track_count": len(self._cam_tracks) + len(self._radar_tracks) + len(self._thermal_tracks),
            "uptime_s": self._clock.elapsed(),
            "frame_number": self._frame,
        }


# ---------------------------------------------------------------------------
# Demo stages
# ---------------------------------------------------------------------------

DIVIDER = "=" * 60


def stage_record(duration: float, fps: float, cfg: HistoryConfig, verbose: bool):
    """Stage 1: Record simulated pipeline data."""
    print(f"\n{DIVIDER}")
    print(f"STAGE 1: RECORDING  ({duration}s at {fps} FPS)")
    print(DIVIDER)

    rec = HistoryRecorder(cfg)
    clock = MockClock(1000.0)
    pipe = MockPipeline(clock)

    rec.start()
    dt = 1.0 / fps
    n_frames = int(duration * fps)

    t0 = time.monotonic()
    for i in range(n_frames):
        pipe.step(dt)
        rec.record_frame(pipe)
        clock.advance(dt)

        if verbose and (i + 1) % int(fps * 5) == 0:
            s = rec.get_status()
            print(f"  [{i+1:6d}/{n_frames}] recorded={s['recorded_count']:5d}  "
                  f"buffer={s['buffer_frames']:5d}  mem={s.get('estimated_memory_bytes', 0) / 1024:.1f} KB")

    rec.stop()
    wall = time.monotonic() - t0

    s = rec.get_status()
    print(f"\n  Recording complete:")
    print(f"  - Total frames offered:  {s['frame_counter']}")
    print(f"  - Frames recorded:       {s['recorded_count']}")
    print(f"  - Buffer utilization:    {s['buffer_frames']} / {s['buffer_capacity']}")
    print(f"  - Time range:            {s.get('time_range', 'N/A')}")
    print(f"  - Estimated memory:      {s.get('estimated_memory_bytes', 0) / 1024:.1f} KB")
    print(f"  - Wall-clock time:       {wall:.2f}s")

    return rec


def stage_export(rec: HistoryRecorder, output_path: str, fmt: str, compress: bool):
    """Stage 2: Export to file."""
    print(f"\n{DIVIDER}")
    print(f"STAGE 2: EXPORT  (format={fmt}, compression={compress})")
    print(DIVIDER)

    t0 = time.monotonic()
    HistoryStorage.save(rec.buffer, output_path, fmt=fmt, compression=compress)
    wall = time.monotonic() - t0

    import os
    size = os.path.getsize(output_path)
    print(f"  File: {output_path}")
    print(f"  Size: {size / 1024:.1f} KB")
    print(f"  Time: {wall:.3f}s")

    meta = HistoryStorage.get_metadata(output_path)
    print(f"  Metadata:")
    print(f"    - frame_count: {meta['frame_count']}")
    print(f"    - time_range:  {meta.get('time_range', 'N/A')}")
    print(f"    - version:     {meta.get('version', 'N/A')}")

    return output_path


def stage_import(file_path: str):
    """Stage 3: Import from file."""
    print(f"\n{DIVIDER}")
    print("STAGE 3: IMPORT")
    print(DIVIDER)

    t0 = time.monotonic()
    loaded = HistoryStorage.load(file_path)
    wall = time.monotonic() - t0

    print(f"  Frames loaded:  {loaded.frame_count}")
    print(f"  Time range:     {loaded.time_range}")
    print(f"  Load time:      {wall:.3f}s")

    # Spot check
    f = loaded.get_frame(0)
    print(f"  First frame:    #{f.frame_number}, ts={f.timestamp:.2f}, "
          f"cam={len(f.camera_tracks)}, radar={len(f.radar_tracks)}, "
          f"thermal={len(f.thermal_tracks)}")

    return loaded


def stage_replay(loaded: HistoryBuffer, n_steps: int = 10):
    """Stage 4: Step through loaded data via ReplayController."""
    print(f"\n{DIVIDER}")
    print(f"STAGE 4: REPLAY  ({n_steps} steps)")
    print(DIVIDER)

    state_buf = StateBuffer()
    ctrl = ReplayController(state_buffer=state_buf)
    ctrl.load(loaded)

    s = ctrl.get_status()
    print(f"  Total frames: {s['total_frames']}")
    print(f"  Speed:        {s['speed']}x")

    step_size = max(1, loaded.frame_count // n_steps)
    for i in range(n_steps):
        idx = min(i * step_size, loaded.frame_count - 1)
        ctrl.seek_to_frame(idx)
        s = ctrl.get_status()
        snap = state_buf.snapshot()
        frame_info = ""
        if snap and snap.system_status:
            frame_info = f"  fps={snap.system_status.get('fps', '?')}"
        print(f"  Step {i+1:3d}: frame={s['current_index']:5d}  "
              f"cam_tracks={len(snap.camera_tracks) if snap else 0}  "
              f"radar_tracks={len(snap.radar_tracks) if snap else 0}{frame_info}")

    ctrl.stop()
    print("\n  Replay complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="SENTINEL History & Replay Demo")
    parser.add_argument("--duration", type=float, default=30, help="Recording duration (seconds)")
    parser.add_argument("--fps", type=float, default=30, help="Simulated FPS")
    parser.add_argument("--max-frames", type=int, default=5000, help="Ring buffer capacity")
    parser.add_argument("--interval", type=int, default=1, help="Capture interval (every Nth frame)")
    parser.add_argument("--format", choices=["json", "msgpack"], default="json", help="Export format")
    parser.add_argument("--compress", action="store_true", help="Enable gzip compression")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--steps", type=int, default=10, help="Replay step count")
    parser.add_argument("--verbose", action="store_true", help="Show progress during recording")
    args = parser.parse_args()

    print(DIVIDER)
    print("SENTINEL - Track History & Replay Demo")
    print(DIVIDER)
    print(f"  Duration:    {args.duration}s")
    print(f"  FPS:         {args.fps}")
    print(f"  Max frames:  {args.max_frames}")
    print(f"  Interval:    {args.interval}")
    print(f"  Format:      {args.format}")
    print(f"  Compression: {args.compress}")

    cfg = HistoryConfig(
        max_frames=args.max_frames,
        capture_interval=args.interval,
    )

    # Output path
    ext = args.format
    if args.compress:
        ext += ".gz"
    output = args.output or f"data/recordings/demo_recording.{ext}"

    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Run stages
    rec = stage_record(args.duration, args.fps, cfg, args.verbose)
    stage_export(rec, output, args.format, args.compress)
    loaded = stage_import(output)
    stage_replay(loaded, args.steps)

    print(f"\n{DIVIDER}")
    print("DEMO COMPLETE")
    print(DIVIDER)


if __name__ == "__main__":
    main()
