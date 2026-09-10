import importlib

import pytest

from tests.test_transfer_contracts import Response, record


def reader():
    value = getattr(importlib.import_module('data.transfer'), 'read_header_range', None)
    assert callable(value), 'bounded header reader missing'
    return value


@pytest.mark.parametrize('prefix', ['bytes ', ''])
def test_header_reader_accepts_only_exact_verified_interval(prefix):
    r = record(b'abcdefghijklmnop')
    def opener(request, timeout):
        assert request.get_header('Range') == 'bytes=4-7'
        return Response(b'efgh',4,7,16,content_range=prefix+'4-7/16')
    assert reader()(r,4,4,opener=opener) == b'efgh'


@pytest.mark.parametrize('status,header,body', [(200,'4-7/16',b'efgh'),
    (206,'4-7/17',b'efgh'), (206,'0-3/16',b'abcd'),
    (206,'4-7/16',b'efg'), (206,'4-7/16',b'efghi')])
def test_header_reader_rejects_ignored_ranges_wrong_identity_and_bad_lengths(status,header,body):
    def opener(request, timeout):
        return Response(body,4,7,16,status=status,content_range=header)
    with pytest.raises(RuntimeError):
        reader()(record(b'abcdefghijklmnop'),4,4,opener=opener)


def test_header_reader_rejects_out_of_bounds_without_network():
    def opener(*args, **kwargs):
        pytest.fail('out of bounds reached network')
    with pytest.raises(ValueError):
        reader()(record(b'abcdefghijklmnop'),14,4,opener=opener)
