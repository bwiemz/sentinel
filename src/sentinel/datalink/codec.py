"""Bit-level codec for Link 16 message encoding/decoding.

Link 16 J-series messages use sub-byte field widths (3-bit identity,
13-bit track number, 23-bit latitude, etc.).  BitWriter/BitReader
provide efficient packing and unpacking of arbitrary-width fields
into byte buffers.
"""

from __future__ import annotations


class BitWriter:
    """Writes fields of arbitrary bit widths into a byte buffer.

    Bits are packed MSB-first within each byte.
    """

    __slots__ = ("_buf", "_bit_pos")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._bit_pos = 0

    # ------------------------------------------------------------------

    def write_unsigned(self, value: int, bits: int) -> None:
        """Write an unsigned integer using *bits* bits (MSB first)."""
        if bits <= 0:
            raise ValueError("bits must be > 0")
        max_val = (1 << bits) - 1
        if value < 0:
            raise ValueError(f"Unsigned value must be >= 0, got {value}")
        value = value & max_val  # clamp silently to field width
        for i in range(bits - 1, -1, -1):
            self._write_bit((value >> i) & 1)

    def write_signed(self, value: int, bits: int) -> None:
        """Write a signed integer (two's complement) using *bits* bits."""
        if bits <= 0:
            raise ValueError("bits must be > 0")
        min_val = -(1 << (bits - 1))
        max_val = (1 << (bits - 1)) - 1
        if value < min_val or value > max_val:
            value = max(min_val, min(max_val, value))
        unsigned = value & ((1 << bits) - 1)
        self.write_unsigned(unsigned, bits)

    def write_bool(self, value: bool) -> None:
        """Write a single boolean bit."""
        self._write_bit(1 if value else 0)

    # ------------------------------------------------------------------

    def pad_to_byte(self) -> None:
        """Pad remaining bits in current byte with zeros."""
        remainder = self._bit_pos % 8
        if remainder != 0:
            for _ in range(8 - remainder):
                self._write_bit(0)

    def to_bytes(self) -> bytes:
        """Return the buffer as an immutable bytes object.

        Pads to the next byte boundary if needed.
        """
        self.pad_to_byte()
        return bytes(self._buf)

    @property
    def bit_position(self) -> int:
        return self._bit_pos

    @property
    def byte_length(self) -> int:
        return (self._bit_pos + 7) // 8

    # ------------------------------------------------------------------

    def _write_bit(self, bit: int) -> None:
        byte_idx = self._bit_pos // 8
        bit_idx = 7 - (self._bit_pos % 8)
        if byte_idx >= len(self._buf):
            self._buf.append(0)
        if bit:
            self._buf[byte_idx] |= (1 << bit_idx)
        self._bit_pos += 1


class BitReader:
    """Reads fields of arbitrary bit widths from a byte buffer."""

    __slots__ = ("_data", "_bit_pos", "_total_bits")

    def __init__(self, data: bytes | bytearray) -> None:
        self._data = bytes(data)
        self._bit_pos = 0
        self._total_bits = len(self._data) * 8

    # ------------------------------------------------------------------

    def read_unsigned(self, bits: int) -> int:
        """Read an unsigned integer of *bits* width (MSB first)."""
        if bits <= 0:
            raise ValueError("bits must be > 0")
        if self._bit_pos + bits > self._total_bits:
            raise ValueError(
                f"Cannot read {bits} bits: only {self.bits_remaining} remaining"
            )
        value = 0
        for _ in range(bits):
            value = (value << 1) | self._read_bit()
        return value

    def read_signed(self, bits: int) -> int:
        """Read a signed (two's complement) integer of *bits* width."""
        unsigned = self.read_unsigned(bits)
        if unsigned >= (1 << (bits - 1)):
            unsigned -= (1 << bits)
        return unsigned

    def read_bool(self) -> bool:
        """Read a single boolean bit."""
        return self._read_bit() == 1

    # ------------------------------------------------------------------

    @property
    def bits_remaining(self) -> int:
        return self._total_bits - self._bit_pos

    @property
    def bit_position(self) -> int:
        return self._bit_pos

    # ------------------------------------------------------------------

    def _read_bit(self) -> int:
        byte_idx = self._bit_pos // 8
        bit_idx = 7 - (self._bit_pos % 8)
        self._bit_pos += 1
        return (self._data[byte_idx] >> bit_idx) & 1
