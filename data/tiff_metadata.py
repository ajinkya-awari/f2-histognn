"""Bounded, network-free Aperio calibration preflight from TIFF metadata.

This is not slide validation: full-file checksum and OpenSlide validation remain
mandatory before extraction. Vendor is a description-prefix indication only.
Field mapping: https://openslide.org/formats/aperio/ (AppMag and MPP).
"""
from __future__ import annotations

import math
import struct
from collections.abc import Callable


class TIFFMetadataError(ValueError):
    """Malformed, ambiguous, or out-of-budget TIFF calibration metadata."""


def probe_aperio_calibration(
    read_range: Callable[[int, int], bytes], file_size: int,
) -> dict[str, float | str | None]:
    """Read only the header, first IFD, and bounded ASCII ImageDescription.

    At most four callback reads and 70,688 requested bytes are permitted.
    Missing calibration is None; invalid/conflicting declared values fail closed.
    No raw description or slide identifiers are returned or included in errors.
    """
    if isinstance(file_size, bool) or not isinstance(file_size, int) or file_size < 16:
        raise TIFFMetadataError("invalid TIFF file size")
    calls = 0
    requested = 0

    def read(start: int, length: int) -> bytes:
        nonlocal calls, requested
        if start < 0 or length <= 0 or start > file_size - length:
            raise TIFFMetadataError("TIFF read outside file bounds")
        calls += 1
        requested += length
        if calls > 4 or requested > 70688:
            raise TIFFMetadataError("TIFF metadata read budget exceeded")
        # Transport errors must remain distinct: availability is not an
        # eligibility criterion. The transport owns sanitized network failures.
        value = read_range(start, length)
        if not isinstance(value, bytes) or len(value) != length:
            raise TIFFMetadataError("TIFF metadata read returned invalid byte count")
        return value

    result = {"objective_power": None, "mpp": None, "vendor": "unsupported"}
    header = read(0, 16)
    if header[:2] not in (b"II", b"MM"):
        return result
    endian = "<" if header[:2] == b"II" else ">"
    version = struct.unpack_from(endian + "H", header, 2)[0]
    if version == 42:
        first_ifd = struct.unpack_from(endian + "I", header, 4)[0]
        count_format, count_size, entry_size, inline_size, header_size = "H", 2, 12, 4, 8
        entry_format = "HHI"
    elif version == 43:
        offset_size, reserved, first_ifd = struct.unpack_from(endian + "HHQ", header, 4)
        if offset_size != 8 or reserved != 0:
            raise TIFFMetadataError("invalid BigTIFF header")
        count_format, count_size, entry_size, inline_size, header_size = "Q", 8, 20, 8, 16
        entry_format = "HHQ"
    else:
        return result
    if first_ifd < header_size:
        raise TIFFMetadataError("invalid first TIFF directory offset")
    entries = struct.unpack(endian + count_format, read(first_ifd, count_size))[0]
    if not 1 <= entries <= 256:
        raise TIFFMetadataError("invalid TIFF directory entries count")
    table = read(first_ifd + count_size, entries * entry_size + inline_size)
    descriptions = []
    for index in range(entries):
        start = index * entry_size
        tag, field_type, count = struct.unpack_from(endian + entry_format, table, start)
        if tag == 270:
            if field_type != 2 or not 1 <= count <= 65536:
                raise TIFFMetadataError("invalid TIFF ASCII description type or size")
            descriptions.append((count, table[start + entry_size - inline_size:start + entry_size]))
    if not descriptions:
        return result
    if len(descriptions) != 1:
        raise TIFFMetadataError("duplicate TIFF descriptions")
    count, field = descriptions[0]
    if count <= inline_size:
        raw = field[:count]
    else:
        offset = int.from_bytes(field, "little" if endian == "<" else "big")
        if offset < header_size:
            raise TIFFMetadataError("invalid TIFF description offset")
        raw = read(offset, count)
    if not raw.endswith(b"\0") or b"\0" in raw[:-1]:
        raise TIFFMetadataError("invalid TIFF ASCII termination")
    try:
        description = raw[:-1].decode("ascii")
    except UnicodeDecodeError:
        raise TIFFMetadataError("invalid TIFF ASCII description") from None
    if not description.startswith("Aperio"):
        return result
    result["vendor"] = "aperio"
    values = {}
    for field in description.split("|")[1:]:
        key, separator, value = field.partition("=")
        key = key.strip()
        if key not in ("AppMag", "MPP"):
            continue
        try:
            number = float(value.strip()) if separator else float("nan")
        except ValueError:
            raise TIFFMetadataError("invalid Aperio calibration value") from None
        if not math.isfinite(number) or (not 10 <= number <= 80 if key == "AppMag" else number <= 0):
            raise TIFFMetadataError("invalid Aperio calibration value")
        if key in values and values[key] != number:
            raise TIFFMetadataError("conflicting Aperio calibration values")
        values[key] = number
    result["objective_power"] = values.get("AppMag")
    result["mpp"] = values.get("MPP")
    return result
