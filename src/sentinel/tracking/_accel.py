"""C++ acceleration dispatch.

Tries to import the compiled _sentinel_core extension module.
If not available (not compiled, wrong platform, etc.), falls back
to pure Python implementations. All existing Python code continues
to work unchanged.
"""
from __future__ import annotations

_HAS_CPP: bool = False
_sentinel_core = None

try:
    from sentinel import _sentinel_core as _mod  # type: ignore[attr-defined]
    _sentinel_core = _mod
    _HAS_CPP = True
except ImportError:
    pass


def has_cpp_acceleration() -> bool:
    """Return True if C++ extension is available."""
    return _HAS_CPP
