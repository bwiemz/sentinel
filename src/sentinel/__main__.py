"""SENTINEL CLI entry point.

Usage:
    python -m sentinel                          # Default config, USB webcam
    python -m sentinel --config custom.yaml     # Custom config
    python -m sentinel --source video.mp4       # Override camera source
    python -m sentinel --source 0               # USB camera index
    python -m sentinel --source rtsp://...      # RTSP stream
"""

from __future__ import annotations

import argparse
import sys

from sentinel.core.config import SentinelConfig
from sentinel.core.pipeline import SentinelPipeline
from sentinel.utils.logging import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="SENTINEL - Advanced Multi-Sensor Tracking System",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Path to configuration YAML file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--source", "-s",
        default=None,
        help="Override camera source (0=USB, path=video file, rtsp://=stream)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Override YOLO model path (e.g. yolov8s.pt, yolov8m.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override detection confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override inference device (cpu, cuda:0, mps)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )
    args = parser.parse_args()

    # Load config
    config = SentinelConfig(args.config)
    try:
        cfg = config.load()
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    # Apply CLI overrides
    if args.source is not None:
        # Try to parse as int (USB device index)
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
        config.override("sentinel.sensors.camera.source", source)

    if args.model is not None:
        config.override("sentinel.detection.model", args.model)
    if args.confidence is not None:
        config.override("sentinel.detection.confidence", args.confidence)
    if args.device is not None:
        config.override("sentinel.detection.device", args.device)

    # Setup logging
    log_level = args.log_level or cfg.sentinel.system.get("log_level", "INFO")
    setup_logging(log_level)

    # Run pipeline
    pipeline = SentinelPipeline(cfg)
    pipeline.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
