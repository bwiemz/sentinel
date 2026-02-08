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
        "--config",
        "-c",
        default="config/default.yaml",
        help="Path to configuration YAML file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--source",
        "-s",
        default=None,
        help="Override camera source (0=USB, path=video file, rtsp://=stream)",
    )
    parser.add_argument(
        "--model",
        "-m",
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
    parser.add_argument(
        "--validate-config",
        action="store_true",
        default=False,
        help="Validate config against Pydantic schema before starting",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (default: no file logging)",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        default=False,
        help="Output logs as JSON instead of human-readable",
    )
    args = parser.parse_args()

    # Load config
    config = SentinelConfig(args.config)
    try:
        cfg = config.load(validate=args.validate_config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Config validation failed:\n{e}", file=sys.stderr)
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
    log_file = args.log_file or cfg.sentinel.system.get("log_file", None)
    log_json = args.log_json or cfg.sentinel.system.get("log_json", False)
    setup_logging(log_level, log_file=log_file, log_json=log_json)

    # Run pipeline
    pipeline = SentinelPipeline(cfg)
    pipeline.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
