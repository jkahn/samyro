"""Tzara functions for loading up training batches from input text."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import itertools
import random

import numpy

import tensorflow as tf

# TODO(jkahn): remove dependency
from prettytensor.tutorial import data_utils

from samyro import integerize


class Sample(collections.namedtuple('Sample', ['input', 'output'])):
    """Contains an input/output pair (any format)."""
    @classmethod
    def to_numpy_batch(cls, batch, dtype=numpy.int32):
        """Takes an iterable of Sample objects; produces a numpy rectangle."""
        # assert that lengths are all the same?
        batch = [s.to_numpy_arrays(dtype=dtype) for s in batch]
        return cls(input=numpy.array([s.input for s in batch],
                                     dtype=dtype),
                   output=numpy.array([s.output for s in batch],
                                      dtype=dtype))

    def to_numpy_arrays(self, dtype=numpy.int32):
        """Returns numpy-ified version of string inputs for a single sample."""
        return Sample(input=numpy.array([integerize.convert_to_int(c)
                                         for c in self.input], dtype=dtype),
                      output=numpy.array([integerize.convert_to_int(c)
                                          for c in self.output], dtype=dtype))

    @classmethod
    def from_text(cls, text, length, right_pad=integerize.EOS_CHAR):
        # Pad out end of line with EOS_CHAR
        if len(text) < length:
            n_pad_chars = length - len(text)
            text += (right_pad * n_pad_chars)

        # Trim back overlong lines
        if len(text) > length:
            text = text[:length]

        # Apply offset, padding with BOS_CHAR
        return cls(input=integerize.BOS_CHAR + text[:-1],
                   output=text)

    def pretty(self):
        raise NotImplementedError("implement aligned output")


# TODO(jkahn): make this an ABC?
class Sampler(object):
    """An ABC for any sampler."""
    def __init__(self, sample_length, batch_size,
                 dummy_value=integerize.EOS_CHAR):
        self.sample_length = sample_length
        self.batch_size = batch_size
        self.dummy_value = dummy_value

    def __iter__(self):
        return self.stream()

    def string_samples(self):
        raise NotImplementedError("This is an abstract base class."
                                  " Implement 'string_samples' in subclasses.")

    def stream(self):
        for s in self.string_samples():
            yield s.to_numpy_arrays()

    @property
    def dummy_sequence(self):
        return Sample(input=self.dummy_value * self.sample_length,
                      output=self.dummy_value * self.sample_length)

    def batches(self, shuffle, to_numpy=True):
        seqs = self.string_samples()
        if shuffle:
            seqs = list(seqs)
            random.shuffle(seqs)
        batches = itertools.izip_longest(fillvalue=self.dummy_sequence,
                                         *([iter(seqs)] * self.batch_size))
        if to_numpy:
            return (Sample.to_numpy_batch(b) for b in batches)
        else:
            return batches

    def placeholder(self):
        return integerize.placeholder(self.batch_size, self.sample_length)

    def as_labels(self, output_placeholder):
        tokens_per_batch = self.batch_size * self.sample_length
        _t = tf.concat(1,
                       [tf.constant(numpy.arange(tokens_per_batch).reshape(
                           (tokens_per_batch, 1)), dtype=tf.int32),
                        data_utils.reshape_data(output_placeholder)])
        return tf.sparse_to_dense(_t, [tokens_per_batch,
                                       integerize.CHARS],
                                  1.0, 0.0)


class FileSampler(Sampler):
    """File(handle) sampler base class."""
    def __init__(self, sample_length, batch_size,
                 encoding='utf-8',
                 filename=None, filehandle=None):
        assert filename is not None or filehandle is not None
        assert not all([filename, filehandle])
        if not filehandle:
            filehandle = codecs.open(filename, 'rb', encoding)
        self.filehandle = filehandle
        super(FileSampler, self).__init__(
            sample_length=sample_length, batch_size=batch_size,
            dummy_value=integerize.EOS_CHAR)

    @property
    def characters(self):
        """Returns iterable of characters."""
        self.filehandle.seek(0)
        return self.filehandle.read()

    @property
    def lines(self):
        """Returns iterable of lines.

        Default implementation uses file's linebreaks.
        """
        self.filehandle.seek(0)
        return self.filehandle

    # TODO(jkahn): make paragraph_sep into a regex
    def paragraphs(self, paragraph_sep=""):
        """Returns iterable of non-empty line groups.

        File is thereby separated by empty lines.
        """

        self.filehandle.seek(0)
        current = []
        for line in self.lines:
            if line.strip() == paragraph_sep:
                if current:
                    yield ''.join(current)
                    current = []
            else:
                current.append(line)
        if current:
            yield ''.join(current)


class LineSampler(FileSampler):
    """A file(handle) sampler that returns fixed-length lines.

    Each sample is trimmed or padded to the target line length.
    Each sample pads the leftmost input with one integerize.BOS_CHAR token
    and pads the right input with integerize.EOS_CHAR.

    Whitespace-only lines are ignored.
    """
    def string_samples(self):
        for line in self.lines:
            if len(line.strip()):
                yield Sample.from_text(line, length=self.sample_length)


class ParagraphSampler(FileSampler):
    """A file(handle) sampler that returns fixed-length paragraph samples.

    Same padding rules as LineSampler, but inputs are paragraphs instead.
    """
    def string_samples(self):
        for line in self.paragraphs():
            yield Sample.from_text(line, length=self.sample_length)


class CharacterPatchSampler(FileSampler):
    """A file(handle) sampler that returns patches of given sample_length.

    Input file uses integerize.BOS_CHAR as leftmost input token for
    first sample, but all other samples use an input and an output
    with an offset of 1.
    """
    def string_samples(self):
        char_iter = self.characters
        prev = integerize.BOS_CHAR
        for b in itertools.izip_longest(
                fillvalue=integerize.EOS_CHAR,
                *([iter(char_iter)] * self.sample_length)):
            yield Sample(input=prev + ''.join(b[:-1]),
                         output=''.join(b))
            prev = b[-1]
        if prev != integerize.EOS_CHAR:
            # last sample not rounded evenly
            yield Sample(
                input=prev + integerize.EOS_CHAR * (self.sample_length - 1),
                output=integerize.EOS_CHAR * self.sample_length)

# TODO(jeremy): generate training examples from each *line* or perhaps sliding
# pair thereof. Consider weighting the second line 1.0 but the first 0.0?
# What happens when we give it couplets?
