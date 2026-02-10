"""HistoryStorage — export/import HistoryBuffer to/from disk files."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.history.buffer import HistoryBuffer
from sentinel.history.frame import HistoryFrame

logger = logging.getLogger(__name__)

try:
    import msgpack

    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False

# File format version (for forward compatibility)
HISTORY_FORMAT_VERSION = 1


def _default_serializer(obj: Any) -> Any:
    """Convert non-serializable objects for msgpack/JSON."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Cannot serialize {type(obj)}")


def _walk_object_hook(obj: Any) -> Any:
    """Recursively reconstruct numpy arrays from deserialized dicts."""
    if isinstance(obj, dict):
        obj = {k: _walk_object_hook(v) for k, v in obj.items()}
        if obj.get("__ndarray__"):
            return np.array(obj["data"], dtype=obj["dtype"])
        return obj
    if isinstance(obj, list):
        return [_walk_object_hook(item) for item in obj]
    return obj


class HistoryStorage:
    """Export/import HistoryBuffer to/from disk files.

    File format (msgpack or JSON)::

        {
            "version": 1,
            "metadata": {
                "frame_count": N,
                "time_range": [t_start, t_end],
                "config_snapshot": {...},
            },
            "frames": [ ... HistoryFrame dicts ... ]
        }

    With ``compression=True`` the file is gzip-wrapped.
    """

    @staticmethod
    def save(
        buffer: HistoryBuffer,
        filepath: str | Path,
        fmt: str = "msgpack",
        compression: bool = False,
        config_snapshot: dict | None = None,
    ) -> Path:
        """Export all buffer frames to a file.  Returns the resolved path."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        frames = buffer.get_all_frames()
        time_range = buffer.time_range

        data = {
            "version": HISTORY_FORMAT_VERSION,
            "metadata": {
                "frame_count": len(frames),
                "time_range": list(time_range) if time_range else [0.0, 0.0],
                "config_snapshot": config_snapshot or {},
            },
            "frames": [f.to_dict() for f in frames],
        }

        raw: bytes
        if fmt == "msgpack" and _HAS_MSGPACK:
            raw = msgpack.packb(data, default=_default_serializer, use_bin_type=True)
        else:
            raw = json.dumps(data, default=_default_serializer).encode("utf-8")
            if fmt == "msgpack" and not _HAS_MSGPACK:
                logger.warning("msgpack not installed, falling back to JSON format")

        if compression:
            raw = gzip.compress(raw)

        filepath.write_bytes(raw)
        logger.info("Saved %d frames to %s (%d bytes)", len(frames), filepath, len(raw))
        return filepath

    @staticmethod
    def load(filepath: str | Path, max_frames: int = 0) -> HistoryBuffer:
        """Load frames from a file into a new HistoryBuffer.

        Args:
            filepath: Path to the recording file.
            max_frames: If > 0, override buffer capacity.  If 0, uses
                the number of loaded frames.
        """
        filepath = Path(filepath)
        raw = filepath.read_bytes()

        # Try gzip decompression
        try:
            raw = gzip.decompress(raw)
        except gzip.BadGzipFile:
            pass

        # Try msgpack first, fall back to JSON
        data: dict
        if _HAS_MSGPACK:
            try:
                data = msgpack.unpackb(raw, raw=False)
            except Exception:
                data = json.loads(raw.decode("utf-8"))
        else:
            data = json.loads(raw.decode("utf-8"))

        data = _walk_object_hook(data)

        version = data.get("version", 1)
        if version > HISTORY_FORMAT_VERSION:
            logger.warning(
                "File version %d > supported %d, some data may be lost",
                version,
                HISTORY_FORMAT_VERSION,
            )

        frame_dicts = data.get("frames", [])
        frames = [HistoryFrame.from_dict(fd) for fd in frame_dicts]

        capacity = max_frames if max_frames > 0 else max(len(frames), 1)
        buf = HistoryBuffer(max_frames=capacity)
        buf.load_frames(frames)

        logger.info("Loaded %d frames from %s", len(frames), filepath)
        return buf

    @staticmethod
    def get_metadata(filepath: str | Path) -> dict:
        """Read only the metadata header from a file."""
        filepath = Path(filepath)
        raw = filepath.read_bytes()
        try:
            raw = gzip.decompress(raw)
        except gzip.BadGzipFile:
            pass

        if _HAS_MSGPACK:
            try:
                data = msgpack.unpackb(raw, raw=False)
            except Exception:
                data = json.loads(raw.decode("utf-8"))
        else:
            data = json.loads(raw.decode("utf-8"))

        return data.get("metadata", {})
