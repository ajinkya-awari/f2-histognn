"""Offline byte-stream tests; no GDC requests or real slide data."""

import hashlib
from http.client import IncompleteRead
import importlib
import io
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from data.gdc import GDCSlideRecord


def downloader():
    assert importlib.util.find_spec("data.transfer") is not None, "bounded downloader is missing"
    return importlib.import_module("data.transfer").download_slide_ranges


def record(payload):
    return GDCSlideRecord(
        "00000000-0000-0000-0000-000000000001", "synthetic.svs", len(payload),
        hashlib.md5(payload).hexdigest(), "TCGA-LUAD", "LUAD", "synthetic-case",
        "synthetic-case", "synthetic-slide",
    )


class Response(io.BytesIO):
    def __init__(self, payload, start, end, total, *, status=206, content_range=None):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": content_range or f"bytes {start}-{end}/{total}"}


def ranges(request):
    assert request.get_header("Accept-encoding") == "identity"
    assert request.full_url.startswith("https://api.gdc.cancer.gov/data/")
    start, end = request.get_header("Range").removeprefix("bytes=").split("-")
    return int(start), int(end)


def test_out_of_order_ranges_produce_exact_verified_single_file(tmp_path):
    payload = b"abcdefghijklmnop"
    late = threading.Event()
    completed = []
    active = 0
    peak = 0
    lock = threading.Lock()

    def opener(request, timeout):
        nonlocal active, peak
        start, end = ranges(request)
        with lock:
            active += 1
            peak = max(peak, active)
        if start == 0:
            assert late.wait(5)
        else:
            late.set()
        with lock:
            completed.append(start)
            active -= 1
        return Response(payload[start:end + 1], start, end, len(payload))

    destination = tmp_path / "slide.svs"
    downloader()(record(payload), destination, opener=opener, chunk_bytes=4)
    assert destination.read_bytes() == payload
    assert sorted(completed) == [0, 4, 8, 12]
    assert completed[0] != 0
    assert 2 <= peak <= 4
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("mode", ["ignored", "wrong-range", "encoding", "overflow"])
def test_invalid_range_response_fails_closed_and_cleans_created_file(tmp_path, mode):
    payload = b"abcd"
    reads = []

    class Observed(Response):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    def opener(request, timeout):
        start, end = ranges(request)
        response = Observed(payload + (b"!" if mode == "overflow" else b""), start, end, 4,
                            status=200 if mode == "ignored" else 206,
                            content_range="bytes 1-4/5" if mode == "wrong-range" else None)
        if mode == "encoding":
            response.headers["Content-Encoding"] = "gzip"
        return response

    destination = tmp_path / "slide.svs"
    with pytest.raises(RuntimeError):
        downloader()(record(payload), destination, opener=opener)
    assert not destination.exists()
    if mode != "overflow":
        assert reads == []


def test_truncated_range_retries_then_overwrites_partial_bytes(tmp_path):
    payload = b"abcd"
    attempts = []

    def opener(request, timeout):
        start, end = ranges(request)
        attempts.append(start)
        return Response(payload[:2] if len(attempts) == 1 else payload, start, end, 4)

    destination = tmp_path / "slide.svs"
    downloader()(record(payload), destination, opener=opener)
    assert attempts == [0, 0]
    assert destination.read_bytes() == payload


def test_transient_errors_stop_after_three_attempts_without_url_exposure(tmp_path):
    attempts = []

    def opener(request, timeout):
        attempts.append(1)
        raise URLError("https://private-identifier.invalid/secret")

    destination = tmp_path / "slide.svs"
    with pytest.raises(RuntimeError) as error:
        downloader()(record(b"abcd"), destination, opener=opener)
    assert len(attempts) == 3
    assert "private-identifier" not in str(error.value)
    assert not destination.exists()


def test_checksum_mismatch_cleans_file(tmp_path):
    def opener(request, timeout):
        start, end = ranges(request)
        return Response(b"xxxx", start, end, 4)

    destination = tmp_path / "slide.svs"
    with pytest.raises(RuntimeError, match="checksum"):
        downloader()(record(b"abcd"), destination, opener=opener)
    assert not destination.exists()


def test_existing_destination_is_never_modified_or_requested(tmp_path):
    destination = tmp_path / "slide.svs"
    destination.write_bytes(b"keep")

    def opener(request, timeout):
        pytest.fail("existing destination must fail before network access")

    with pytest.raises(FileExistsError):
        downloader()(record(b"abcd"), destination, opener=opener)
    assert destination.read_bytes() == b"keep"


def test_transfer_byte_budget_bounds_retried_partial_responses(tmp_path):
    transferred = []

    class Partial(Response):
        def read(self, size=-1):
            data = super().read(size)
            transferred.append(len(data))
            return data

    def opener(request, timeout):
        start, end = ranges(request)
        return Partial(b"abc", start, end, 4)

    destination = tmp_path / "slide.svs"
    with pytest.raises(RuntimeError):
        downloader()(record(b"abcd"), destination, opener=opener)
    assert sum(transferred) <= 8
    assert not destination.exists()


def test_http_incomplete_read_counts_partial_bytes_before_retry(tmp_path):
    attempts = []

    class Incomplete(Response):
        def read(self, size=-1):
            self.seek(4)
            if not getattr(self, "raised", False):
                self.raised = True
                raise IncompleteRead(b"ab", 2)
            return b""

    def opener(request, timeout):
        start, end = ranges(request)
        attempts.append(1)
        cls = Incomplete if len(attempts) == 1 else Response
        return cls(b"abcd", start, end, 4)

    destination = tmp_path / "slide.svs"
    result = downloader()(record(b"abcd"), destination, opener=opener)
    assert len(attempts) == 2
    assert result["transferred_bytes"] == 6
    assert destination.read_bytes() == b"abcd"


def test_deadline_is_checked_before_reading_response_body(tmp_path, monkeypatch):
    transfer = downloader()
    module = importlib.import_module("data.transfer")
    now = [0.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0]))

    class MustNotRead(Response):
        def read(self, size=-1):
            pytest.fail("expired transfer must not read more bytes")

    def opener(request, timeout):
        assert 0 < timeout <= 60
        now[0] = 601.0
        return MustNotRead(b"abcd", 0, 3, 4)

    destination = tmp_path / "slide.svs"
    with pytest.raises(RuntimeError, match="deadline"):
        transfer(record(b"abcd"), destination, opener=opener)
    assert not destination.exists()


def test_failure_joins_other_workers_before_deleting_destination(tmp_path):
    transfer = downloader()
    started = threading.Event()
    bad_closed = threading.Event()
    release = threading.Event()
    done = threading.Event()
    failures = []
    destination = tmp_path / "slide.svs"

    class Bad(Response):
        def close(self):
            super().close()
            bad_closed.set()

    def opener(request, timeout):
        start, end = ranges(request)
        if start == 0:
            started.set()
            assert release.wait(5)
            assert destination.exists()
            return Response(b"abcd", start, end, 8)
        assert started.wait(5)
        return Bad(b"efgh", start, end, 8, status=200)

    def run():
        try:
            transfer(record(b"abcdefgh"), destination, opener=opener, chunk_bytes=4)
        except RuntimeError as error:
            failures.append(error)
        finally:
            done.set()

    caller = threading.Thread(target=run)
    caller.start()
    try:
        assert bad_closed.wait(5)
        assert not done.wait(0.1)
        assert destination.exists()
    finally:
        release.set()
        caller.join(5)
    assert done.is_set()
    assert len(failures) == 1
    assert not destination.exists()


@pytest.mark.parametrize("header", ["bytes 0-3/4", "0-3/4"])
def test_standard_and_observed_gdc_range_headers_preserve_exact_bytes(tmp_path, header):
    def opener(request, timeout):
        return Response(b"abcd", 0, 3, 4, content_range=header)

    destination = tmp_path / "slide.svs"
    downloader()(record(b"abcd"), destination, opener=opener)
    assert destination.read_bytes() == b"abcd"


@pytest.mark.parametrize("header", ["1-4/4", "0-3/5", "0-2/4", "0-3/*", "junk0-3/4"])
def test_bare_header_numeric_mismatch_or_invalid_format_is_rejected(tmp_path, header):
    def opener(request, timeout):
        return Response(b"abcd", 0, 3, 4, content_range=header)

    destination = tmp_path / "slide.svs"
    with pytest.raises(RuntimeError, match="range"):
        downloader()(record(b"abcd"), destination, opener=opener)
    assert not destination.exists()


def test_caller_budget_limits_partial_retry_network_bytes(tmp_path):
    attempts = []
    transferred = []

    class Counted(Response):
        def read(self, size=-1):
            data = super().read(size)
            transferred.append(len(data))
            return data

    def opener(request, timeout):
        start, end = ranges(request)
        attempts.append(1)
        return Counted(b"ab" if len(attempts) == 1 else b"abcd", start, end, 4)

    destination = tmp_path / "slide.svs"
    with pytest.raises(RuntimeError, match="budget"):
        downloader()(record(b"abcd"), destination, opener=opener, max_transfer_bytes=5)
    assert sum(transferred) <= 5
    assert not destination.exists()


@pytest.mark.parametrize("budget", [3, 9, 4.5, True])
def test_caller_budget_cannot_be_less_than_file_or_exceed_twice_file(tmp_path, budget):
    def opener(request, timeout):
        pytest.fail("invalid budget must fail before network access")

    destination = tmp_path / "slide.svs"
    with pytest.raises(ValueError, match="max_transfer_bytes"):
        downloader()(record(b"abcd"), destination, opener=opener, max_transfer_bytes=budget)
    assert not destination.exists()


def test_exact_file_size_budget_accepts_complete_content_length_framed_response(tmp_path):
    def opener(request, timeout):
        response = Response(b"abcd", 0, 3, 4)
        response.headers["Content-Length"] = "4"
        return response

    destination = tmp_path / "slide.svs"
    result = downloader()(record(b"abcd"), destination, opener=opener, max_transfer_bytes=4)
    assert result["transferred_bytes"] == 4
    assert destination.read_bytes() == b"abcd"
