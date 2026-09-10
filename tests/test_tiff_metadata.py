"""Hand-built synthetic TIFF headers; no slide files or network required."""
import importlib
import struct

import pytest


def _probe(blob, reader=None):
    module = importlib.import_module("data.tiff_metadata")
    return module.probe_aperio_calibration(
        reader or (lambda start, length: blob[start:start + length]), len(blob)
    )


def _tiff(description=b"Aperio| AppMag = 20 | MPP = 0.5\0", endian="<", big=False,
          tag=270, field_type=2, count=None, offset=None):
    marker = b"II" if endian == "<" else b"MM"
    count = len(description) if count is None else count
    if big:
        header = marker + struct.pack(endian + "HHHQ", 43, 8, 0, 16)
        value = struct.pack(endian + "Q", 52 if offset is None else offset)
        if count <= 8 and offset is None:
            value = description.ljust(8, b"\0")
        return header + struct.pack(endian + "QHHQ", 1, tag, field_type, count) + value + bytes(8) + description
    header = marker + struct.pack(endian + "HI", 42, 8)
    value = struct.pack(endian + "I", 26 if offset is None else offset)
    if count <= 4 and offset is None:
        value = description.ljust(4, b"\0")
    return header + struct.pack(endian + "HHHI", 1, tag, field_type, count) + value + bytes(4) + description


@pytest.mark.parametrize("endian,big", [("<", False), (">", False), ("<", True), (">", True)])
def test_reads_classic_and_bigtiff_calibration_with_bounded_requests(endian, big):
    blob = _tiff(endian=endian, big=big)
    calls = []

    def read(start, length):
        calls.append((start, length))
        assert 0 <= start < len(blob) and 0 < length <= len(blob) - start
        return blob[start:start + length]

    assert _probe(blob, read) == {"objective_power": 20.0, "mpp": 0.5, "vendor": "aperio"}
    assert calls[0] == (0, 16)
    assert len(calls) <= 4
    assert sum(length for _, length in calls) <= 70792


@pytest.mark.parametrize("description,expected", [
    (b"Aperio|MPP=0.25\0", {"objective_power": None, "mpp": 0.25, "vendor": "aperio"}),
    (b"Aperio\0", {"objective_power": None, "mpp": None, "vendor": "aperio"}),
    (b"Other|AppMag=20\0", {"objective_power": None, "mpp": None, "vendor": "unsupported"}),
    (b"Aperio|AppMag=20|AppMag=20.0\0", {"objective_power": 20.0, "mpp": None, "vendor": "aperio"}),
])
def test_missing_calibration_is_not_inferred(description, expected):
    assert _probe(_tiff(description, big=True)) == expected


@pytest.mark.parametrize("field", [b"AppMag=20|AppMag=40", b"AppMag=nan", b"AppMag=inf",
    b"AppMag=9", b"AppMag=81", b"AppMag=unknown", b"AppMag=", b"MPP=0",
    b"MPP=-1", b"MPP=inf", b"MPP=0.5|MPP=0.25"])
def test_invalid_or_conflicting_calibration_fails_closed(field):
    module = importlib.import_module("data.tiff_metadata")
    with pytest.raises(module.TIFFMetadataError):
        _probe(_tiff(b"Aperio|" + field + b"\0"))


@pytest.mark.parametrize("kwargs", [{"count": 65537}, {"offset": 99999},
    {"field_type": 3}, {"count": 0}])
def test_rejects_invalid_description_size_offset_and_type(kwargs):
    module = importlib.import_module("data.tiff_metadata")
    with pytest.raises(module.TIFFMetadataError):
        _probe(_tiff(**kwargs))


def test_rejects_short_callback_reads():
    module = importlib.import_module("data.tiff_metadata")
    with pytest.raises(module.TIFFMetadataError, match="read"):
        _probe(_tiff(), lambda start, length: b"\0" * (length - 1))


def test_rejects_oversized_ifd_before_reading_table():
    module = importlib.import_module("data.tiff_metadata")
    blob = b"II" + struct.pack("<HIH", 42, 8, 257) + bytes(20)
    with pytest.raises(module.TIFFMetadataError, match="entries"):
        _probe(blob)


def test_rejects_non_ascii_description_without_exposing_it():
    module = importlib.import_module("data.tiff_metadata")
    with pytest.raises(module.TIFFMetadataError) as error:
        _probe(_tiff(b"Aperio|private-identifier-\xff\0"))
    assert "private-identifier" not in str(error.value)


def test_absent_description_is_unsupported():
    assert _probe(_tiff(tag=271)) == {"objective_power": None, "mpp": None, "vendor": "unsupported"}


def test_transport_failure_is_not_reclassified_as_ineligible_metadata():
    def failed_read(start, length):
        raise ConnectionError('synthetic transport unavailable')
    with pytest.raises(ConnectionError):
        _probe(_tiff(), failed_read)
