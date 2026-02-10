"""Unit tests for BitWriter / BitReader codec."""

from __future__ import annotations

import pytest

from sentinel.datalink.codec import BitWriter, BitReader


# ---------------------------------------------------------------------------
# BitWriter basics
# ---------------------------------------------------------------------------


class TestBitWriterUnsigned:
    def test_write_1_bit_zero(self):
        w = BitWriter()
        w.write_unsigned(0, 1)
        assert w.to_bytes() == b"\x00"

    def test_write_1_bit_one(self):
        w = BitWriter()
        w.write_unsigned(1, 1)
        assert w.to_bytes() == b"\x80"

    def test_write_3_bits(self):
        w = BitWriter()
        w.write_unsigned(0b101, 3)
        # 101_00000 = 0xA0
        assert w.to_bytes() == b"\xa0"

    def test_write_8_bits(self):
        w = BitWriter()
        w.write_unsigned(0xAB, 8)
        assert w.to_bytes() == b"\xab"

    def test_write_13_bits(self):
        w = BitWriter()
        w.write_unsigned(8191, 13)  # max 13-bit = 0x1FFF
        # 1_1111_1111_1111_000 = 0xFF 0xF8
        assert w.to_bytes() == b"\xff\xf8"

    def test_write_23_bits(self):
        w = BitWriter()
        w.write_unsigned(0, 23)
        assert w.to_bytes() == b"\x00\x00\x00"

    def test_write_24_bits(self):
        w = BitWriter()
        w.write_unsigned(0xABCDEF, 24)
        assert w.to_bytes() == b"\xab\xcd\xef"

    def test_negative_value_raises(self):
        w = BitWriter()
        with pytest.raises(ValueError, match="Unsigned value must be >= 0"):
            w.write_unsigned(-1, 8)

    def test_zero_bits_raises(self):
        w = BitWriter()
        with pytest.raises(ValueError, match="bits must be > 0"):
            w.write_unsigned(0, 0)

    def test_value_clamped_to_field_width(self):
        w = BitWriter()
        w.write_unsigned(0xFF, 4)  # only lower 4 bits
        # 1111_0000 = 0xF0
        assert w.to_bytes() == b"\xf0"


class TestBitWriterSigned:
    def test_positive_value(self):
        w = BitWriter()
        w.write_signed(3, 4)  # 0011
        assert w.to_bytes() == b"\x30"

    def test_negative_value(self):
        w = BitWriter()
        w.write_signed(-1, 4)  # 1111 in twos complement
        assert w.to_bytes() == b"\xf0"

    def test_zero(self):
        w = BitWriter()
        w.write_signed(0, 8)
        assert w.to_bytes() == b"\x00"

    def test_min_value(self):
        w = BitWriter()
        w.write_signed(-4, 3)  # -4 is min for 3-bit signed: 100
        assert w.to_bytes() == b"\x80"

    def test_max_value(self):
        w = BitWriter()
        w.write_signed(3, 3)  # 011
        assert w.to_bytes() == b"\x60"

    def test_clamp_overflow(self):
        w = BitWriter()
        w.write_signed(100, 3)  # clamped to 3 (max for 3-bit signed)
        r = BitReader(w.to_bytes())
        assert r.read_signed(3) == 3

    def test_clamp_underflow(self):
        w = BitWriter()
        w.write_signed(-100, 3)  # clamped to -4 (min for 3-bit signed)
        r = BitReader(w.to_bytes())
        assert r.read_signed(3) == -4


class TestBitWriterBool:
    def test_true(self):
        w = BitWriter()
        w.write_bool(True)
        assert w.to_bytes() == b"\x80"

    def test_false(self):
        w = BitWriter()
        w.write_bool(False)
        assert w.to_bytes() == b"\x00"


class TestBitWriterPadding:
    def test_pad_to_byte(self):
        w = BitWriter()
        w.write_unsigned(0b11, 2)
        w.pad_to_byte()
        assert w.bit_position == 8
        assert w.to_bytes() == b"\xc0"

    def test_pad_already_aligned(self):
        w = BitWriter()
        w.write_unsigned(0xFF, 8)
        w.pad_to_byte()
        assert w.bit_position == 8

    def test_byte_length(self):
        w = BitWriter()
        w.write_unsigned(0, 13)
        assert w.byte_length == 2


# ---------------------------------------------------------------------------
# BitReader basics
# ---------------------------------------------------------------------------


class TestBitReaderUnsigned:
    def test_read_1_bit(self):
        r = BitReader(b"\x80")
        assert r.read_unsigned(1) == 1

    def test_read_3_bits(self):
        r = BitReader(b"\xa0")
        assert r.read_unsigned(3) == 0b101

    def test_read_8_bits(self):
        r = BitReader(b"\xab")
        assert r.read_unsigned(8) == 0xAB

    def test_read_13_bits_max(self):
        r = BitReader(b"\xff\xf8")
        assert r.read_unsigned(13) == 8191

    def test_read_24_bits(self):
        r = BitReader(b"\xab\xcd\xef")
        assert r.read_unsigned(24) == 0xABCDEF

    def test_read_past_end_raises(self):
        r = BitReader(b"\x00")
        r.read_unsigned(8)
        with pytest.raises(ValueError, match="Cannot read"):
            r.read_unsigned(1)

    def test_bits_remaining(self):
        r = BitReader(b"\x00\x00")
        assert r.bits_remaining == 16
        r.read_unsigned(5)
        assert r.bits_remaining == 11


class TestBitReaderSigned:
    def test_positive(self):
        r = BitReader(b"\x30")
        assert r.read_signed(4) == 3

    def test_negative(self):
        r = BitReader(b"\xf0")
        assert r.read_signed(4) == -1

    def test_min_3bit(self):
        r = BitReader(b"\x80")
        assert r.read_signed(3) == -4

    def test_max_3bit(self):
        r = BitReader(b"\x60")
        assert r.read_signed(3) == 3


class TestBitReaderBool:
    def test_true(self):
        r = BitReader(b"\x80")
        assert r.read_bool() is True

    def test_false(self):
        r = BitReader(b"\x00")
        assert r.read_bool() is False


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_multi_field_roundtrip(self):
        """Pack multiple fields, read them back."""
        w = BitWriter()
        w.write_unsigned(0x2, 4)      # 4 bits
        w.write_unsigned(1234, 13)    # 13 bits
        w.write_unsigned(6, 3)        # 3 bits
        w.write_signed(-12345, 23)    # 23 bits
        w.write_signed(54321, 24)     # 24 bits
        w.write_unsigned(16000, 14)   # 14 bits
        data = w.to_bytes()

        r = BitReader(data)
        assert r.read_unsigned(4) == 0x2
        assert r.read_unsigned(13) == 1234
        assert r.read_unsigned(3) == 6
        assert r.read_signed(23) == -12345
        assert r.read_signed(24) == 54321
        assert r.read_unsigned(14) == 16000

    def test_128_bits_is_16_bytes(self):
        """Full J2.2-like field layout produces exactly 16 bytes."""
        w = BitWriter()
        w.write_unsigned(0x2, 4)     # sub-type
        w.write_unsigned(100, 13)    # track number
        w.write_unsigned(6, 3)       # identity
        w.write_signed(0, 23)        # latitude
        w.write_signed(0, 24)        # longitude
        w.write_unsigned(0, 14)      # altitude
        w.write_unsigned(0, 10)      # speed
        w.write_unsigned(0, 9)       # course
        w.write_unsigned(0, 3)       # quality
        w.write_unsigned(0, 3)       # strength
        w.write_unsigned(0, 12)      # iff 3a
        w.write_unsigned(0, 2)       # threat
        w.write_unsigned(0, 4)       # env flags
        w.write_unsigned(0, 4)       # reserved
        data = w.to_bytes()
        assert len(data) == 16

    def test_all_ones(self):
        w = BitWriter()
        w.write_unsigned(0xFF, 8)
        w.write_unsigned(0xFF, 8)
        data = w.to_bytes()
        assert data == b"\xff\xff"

    def test_all_zeros(self):
        w = BitWriter()
        w.write_unsigned(0, 16)
        assert w.to_bytes() == b"\x00\x00"

    def test_boundary_values(self):
        """Test boundary values for each common width."""
        for bits in [1, 3, 4, 8, 10, 13, 14, 23, 24]:
            max_val = (1 << bits) - 1
            w = BitWriter()
            w.write_unsigned(max_val, bits)
            data = w.to_bytes()
            r = BitReader(data)
            assert r.read_unsigned(bits) == max_val
