"""Test reading batches of character streams."""
from six.moves import cStringIO as StringIO
from samyro import integerize, read

import numpy

import re


def repair(string):
    string = re.sub(r'<BOS>', integerize.BOS_CHAR, string)
    string = re.sub(r'<EOS>', integerize.EOS_CHAR, string)
    return string


def test_patch_batches():
    rr = read.CharacterPatchSampler(filehandles=[StringIO(u'foobar\n')],
                                    sample_length=3, batch_size=2)
    batches = rr.batches(shuffle=False)

    first = next(batches)
    assert isinstance(first, read.Sample)
    assert isinstance(first.input, numpy.ndarray)

    batches = rr.batches(shuffle=False, to_numpy=False)

    first = list(next(batches))
    assert len(first) == 2
    assert first[0] == read.Sample(input=repair("<BOS>fo"), output="foo")


def test_patch_strings():
    rr = read.CharacterPatchSampler(filehandles=[StringIO(u'foobar\n')],
                                    sample_length=3, batch_size=2)
    strings = rr.string_samples()

    first = next(strings)
    assert isinstance(first, read.Sample)
    assert read.Sample(input=repair("<BOS>fo"), output="foo") == first

    second = next(strings)
    assert isinstance(second, read.Sample)
    assert read.Sample(input="oba", output="bar") == second

    third = next(strings)
    assert isinstance(third, read.Sample)
    assert read.Sample(input=repair("r\n<EOS>"),
                       output=repair("\n<EOS><EOS>")) == third

    # assert raises EndOfInput thereafter?


def test_lines_strings():
    rr = read.LineSampler(filehandles=[StringIO(u'foobar\nbaz\nwikiwiki\n')],
                          sample_length=6, batch_size=2)
    strings = rr.string_samples()

    first = next(strings)
    assert isinstance(first, read.Sample)
    assert read.Sample(input=repair("<BOS>fooba"), output="foobar") == first

    second = next(strings)
    assert isinstance(second, read.Sample)
    assert read.Sample(input=repair("<BOS>baz\n<EOS>"),
                       output=repair("baz\n<EOS><EOS>")) == second

    third = next(strings)
    assert isinstance(third, read.Sample)
    assert read.Sample(input=repair("<BOS>wikiw"), output="wikiwi") == third

    # assert raises EndOfInput thereafter?


def test_paras_strings():
    rr = read.ParagraphSampler(
        filehandles=[StringIO(u'foobar\nbaz\n\n\nwikiwiki\nyap\n\nwooo')],
        sample_length=15, batch_size=2)
    strings = rr.string_samples()

    first = next(strings)
    assert isinstance(first, read.Sample)
    assert read.Sample(
        input=repair("<BOS>foobar\nbaz\n<EOS><EOS><EOS>"),
        output=repair("foobar\nbaz\n<EOS><EOS><EOS><EOS>")) == first

    second = next(strings)
    assert isinstance(second, read.Sample)
    assert read.Sample(input=repair("<BOS>wikiwiki\nyap\n<EOS>"),
                       output=repair("wikiwiki\nyap\n<EOS><EOS>")) == second

    third = next(strings)
    assert isinstance(third, read.Sample)
    assert read.Sample(input=repair("<BOS>wooo" + "<EOS>" * 10),
                       output=repair("wooo" + "<EOS>" * 11)) == third

    # assert raises EndOfInput thereafter?
