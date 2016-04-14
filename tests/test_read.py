"""Test reading batches of character streams."""
from samyro import read

from cStringIO import StringIO

import numpy


def test_patch_batches():
    rr = read.CharacterPatchSampler(filehandle=StringIO(u'foobar\n'),
                                    sample_length=3, batch_size=2)
    batches = rr.batches(shuffle=False)

    first = batches.next()
    assert isinstance(first, read.Sample)
    assert isinstance(first.input, numpy.ndarray)

    batches = rr.batches(shuffle=False, to_numpy=False)

    first = list(batches.next())
    assert len(first) == 2
    assert first[0] == read.Sample(input="\x7ffo", output="foo")


def test_patch_strings():
    rr = read.CharacterPatchSampler(filehandle=StringIO(u'foobar\n'),
                                    sample_length=3, batch_size=2)
    strings = rr.string_samples()

    first = strings.next()
    assert isinstance(first, read.Sample)
    assert read.Sample(input="\x7ffo", output="foo") == first

    second = strings.next()
    assert isinstance(second, read.Sample)
    assert read.Sample(input="oba", output="bar") == second

    third = strings.next()
    assert isinstance(third, read.Sample)
    assert read.Sample(input="r\n\x01", output="\n\x01\x01") == third

    # assert raises EndOfInput thereafter?


def test_lines_strings():
    rr = read.LineSampler(filehandle=StringIO(u'foobar\nbaz\nwikiwiki\n'),
                          sample_length=6, batch_size=2)
    strings = rr.string_samples()

    first = strings.next()
    assert isinstance(first, read.Sample)
    assert read.Sample(input="\x7ffooba", output="foobar") == first

    second = strings.next()
    assert isinstance(second, read.Sample)
    assert read.Sample(input="\x7fbaz\n\x01", output="baz\n\x01\x01") == second

    third = strings.next()
    assert isinstance(third, read.Sample)
    assert read.Sample(input="\x7fwikiw", output="wikiwi") == third

    # assert raises EndOfInput thereafter?


def test_paras_strings():
    rr = read.ParagraphSampler(
        filehandle=StringIO(u'foobar\nbaz\n\n\nwikiwiki\nyap\n\nwooo'),
        sample_length=15, batch_size=2)
    strings = rr.string_samples()

    first = strings.next()
    assert isinstance(first, read.Sample)
    assert read.Sample(input="\x7ffoobar\nbaz\n\x01\x01\x01",
                       output="foobar\nbaz\n\x01\x01\x01\x01") == first

    second = strings.next()
    assert isinstance(second, read.Sample)
    assert read.Sample(input="\x7fwikiwiki\nyap\n\x01",
                       output="wikiwiki\nyap\n\x01\x01") == second

    third = strings.next()
    assert isinstance(third, read.Sample)
    assert read.Sample(input="\x7fwooo" + "\x01" * 10,
                       output="wooo" + "\x01" * 11) == third

    # assert raises EndOfInput thereafter?
