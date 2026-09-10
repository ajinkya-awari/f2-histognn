"""Bounded, checksum-verified single-slide HTTPS Range transport.

No request is made at import time. The caller owns cohort and disk budgets;
this module owns one exclusively created destination and never substitutes
files. Range downloads follow the GDC transfer tool's single-file strategy.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from http.client import IncompleteRead
import math
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from data.gdc import GDCSlideRecord


class RangeTransferError(RuntimeError):
    """Sanitized failure without request URLs or private identifiers."""


class _RetryableTransfer(Exception):
    pass


def read_header_range(record, start: int, length: int, *, opener=urlopen) -> bytes:
    """Read at most 64 KiB of TIFF metadata, with strict GDC byte identity.

    This does not establish whole-file integrity. Full MD5 remains mandatory
    before any slide is used for extraction. Transport failure is not missing
    calibration and must never be converted into an eligibility exclusion.
    """
    if (not isinstance(record, GDCSlideRecord)
        or type(start) is not int or type(length) is not int
        or start < 0 or not 1 <= length <= 65536 or start + length > record.file_size
        or re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', record.file_id) is None):
        raise ValueError('invalid bounded header range')
    end = start + length - 1
    for attempt in range(3):
        request = Request(f'https://api.gdc.cancer.gov/data/{record.file_id}', headers={
            'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
            'User-Agent': 'f2-histognn/0.1'})
        try:
            with opener(request, timeout=30) as response:
                match = re.fullmatch(r'(?:bytes )?([0-9]+)-([0-9]+)/([0-9]+)',
                                     response.headers.get('Content-Range', ''))
                if response.status != 206 or match is None or tuple(map(int, match.groups())) != (start, end, record.file_size):
                    raise RangeTransferError('header response byte range does not match the manifest')
                if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                    raise RangeTransferError('encoded header response is not allowed')
                body = response.read(length + 1)
                if len(body) != length:
                    raise RangeTransferError('header response byte count is invalid')
                return body
        except HTTPError as error:
            retryable = error.code in (408, 429, 500, 502, 503, 504)
            error.close()
            if not retryable or attempt == 2:
                raise RangeTransferError('header request failed') from None
        except (OSError, URLError, IncompleteRead):
            if attempt == 2:
                raise RangeTransferError('header transport retries exhausted') from None
    raise RangeTransferError('header transport retries exhausted')


def download_slide_ranges(
    record: GDCSlideRecord,
    destination: str | Path,
    *,
    opener=urlopen,
    workers: int = 4,
    chunk_bytes: int = 32 * 1024 * 1024,
    socket_timeout: float = 60,
    max_retries: int = 2,
    deadline_seconds: float = 600,
    max_transfer_bytes: int | None = None,
) -> dict[str, int]:
    """Download one slide, returning aggregate transport counts after MD5.

    At most four workers write disjoint ranges to independent file handles.
    Retries overwrite only their own range. Received bytes, including retries,
    cannot exceed twice the manifest size or a smaller caller-supplied budget.
    Failures join workers before cleanup.
    ``opener`` injection supports entirely offline synthetic byte-stream tests.
    """

    if (
        not isinstance(record, GDCSlideRecord)
        or type(record.file_size) is not int or record.file_size <= 0
        or re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", record.file_id) is None
        or re.fullmatch(r"[0-9a-fA-F]{32}", record.md5sum) is None
    ):
        raise ValueError("valid slide identity, size, and checksum are required")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("workers must be between one and four")
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 32 * 1024 * 1024:
        raise ValueError("chunk_bytes must be positive and at most 32 MiB")
    if type(max_retries) is not int or not 0 <= max_retries <= 2:
        raise ValueError("max_retries must be between zero and two")
    if max_transfer_bytes is None:
        max_transfer_bytes = 2 * record.file_size
    if type(max_transfer_bytes) is not int or not record.file_size <= max_transfer_bytes <= 2 * record.file_size:
        raise ValueError("max_transfer_bytes must be between file size and twice file size")
    for value in (socket_timeout, deadline_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("timeouts must be finite and positive")

    path = Path(destination)
    expires = time.monotonic() + deadline_seconds
    cancelled = threading.Event()
    budget_lock = threading.Lock()
    available_bytes = max_transfer_bytes
    received_bytes = 0

    def check_running():
        if cancelled.is_set():
            raise RangeTransferError("slide transfer cancelled")
        if time.monotonic() >= expires:
            raise RangeTransferError("slide transfer deadline exceeded")

    def read_bounded(response, requested):
        nonlocal available_bytes, received_bytes
        check_running()
        with budget_lock:
            allowed = min(requested, available_bytes)
            if allowed <= 0:
                raise RangeTransferError("slide transfer byte budget exhausted")
            available_bytes -= allowed
        data = b""
        try:
            try:
                data = response.read(allowed)
            except IncompleteRead as error:
                # Account for bytes delivered with the exception; the next EOF
                # triggers a bounded retry rather than hiding received bytes.
                data = error.partial
            except (OSError, URLError):
                raise _RetryableTransfer() from None
            if len(data) > allowed:
                raise RangeTransferError("response exceeded its requested read boundary")
            return data
        finally:
            with budget_lock:
                received_bytes += len(data)
                available_bytes += allowed - len(data)

    def transfer_range(start, end):
        try:
            for attempt in range(max_retries + 1):
                check_running()
                request = Request(
                    f"https://api.gdc.cancer.gov/data/{record.file_id}",
                    headers={"Range": f"bytes={start}-{end}",
                             "Accept-Encoding": "identity", "User-Agent": "f2-histognn/0.1"},
                )
                try:
                    try:
                        response = opener(request, timeout=min(socket_timeout, expires - time.monotonic()))
                    except HTTPError as error:
                        code = error.code
                        error.close()
                        if code in (408, 429, 500, 502, 503, 504):
                            raise _RetryableTransfer() from None
                        raise RangeTransferError("slide range request rejected") from None
                    except (OSError, URLError):
                        raise _RetryableTransfer() from None
                    with response:
                        if response.status in (408, 429, 500, 502, 503, 504):
                            raise _RetryableTransfer()
                        if response.status != 206:
                            raise RangeTransferError("server did not honor the requested byte range")
                        # GDC also emits the observed bare numeric form without
                        # the standard 'bytes ' prefix; byte identity is unchanged.
                        observed_range = re.fullmatch(
                            r"(?:bytes )?([0-9]+)-([0-9]+)/([0-9]+)",
                            response.headers.get("Content-Range", ""),
                        )
                        if observed_range is None or tuple(map(int, observed_range.groups())) != (start, end, record.file_size):
                            raise RangeTransferError("response byte range does not match the manifest")
                        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                            raise RangeTransferError("encoded range response is not allowed")
                        length = response.headers.get("Content-Length")
                        if length is not None and length != str(end - start + 1):
                            raise RangeTransferError("response content length does not match its range")
                        remaining = end - start + 1
                        with path.open("r+b") as handle:
                            handle.seek(start)
                            while remaining:
                                data = read_bounded(response, min(1024 * 1024, remaining))
                                if not data:
                                    raise _RetryableTransfer()
                                handle.write(data)
                                remaining -= len(data)
                            # HTTP Content-Length frames the body; no extra
                            # byte budget is needed to prove EOF when present.
                            if length is None and read_bounded(response, 1):
                                raise RangeTransferError("response contains bytes beyond its range")
                    return
                except _RetryableTransfer:
                    if attempt == max_retries:
                        raise RangeTransferError("slide range retries exhausted") from None
        except BaseException:
            cancelled.set()
            raise

    # Exclusive creation sits outside cleanup: pre-existing data is never ours.
    handle = path.open("xb")
    try:
        with handle:
            handle.truncate(record.file_size)
        intervals = [(start, min(start + chunk_bytes, record.file_size) - 1)
                     for start in range(0, record.file_size, chunk_bytes)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(transfer_range, start, end) for start, end in intervals]
            try:
                for future in as_completed(futures):
                    future.result()
            finally:
                cancelled.set()
        # All workers are joined here; hashing and deletion cannot race writers.
        if path.stat().st_size != record.file_size:
            raise RangeTransferError("slide size verification failed")
        digest = hashlib.md5()
        with path.open("rb") as handle:
            while True:
                if time.monotonic() >= expires:
                    raise RangeTransferError("slide transfer deadline exceeded")
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        if digest.hexdigest() != record.md5sum.lower():
            raise RangeTransferError("slide checksum verification failed")
        return {"transferred_bytes": received_bytes, "range_count": len(intervals),
                "workers": min(workers, len(intervals))}
    except BaseException as error:
        cancelled.set()
        path.unlink(missing_ok=True)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(error, RangeTransferError):
            raise error from None
        raise RangeTransferError("slide transfer failed") from None
