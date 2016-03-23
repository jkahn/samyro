"""Tzara functions for loading up training batches from input text."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import random

import numpy

import tensorflow as tf

# TODO(jkahn): remove dependency
from prettytensor.tutorial import data_utils

from samyro import integerize


class IOPair(collections.namedtuple('IOPair', ['input', 'output'])):
    """Contains an input/output pair in numpy array formats."""
    dtype = numpy.int32

    @classmethod
    def _dummy_sequence(cls, length):
        return numpy.array((integerize.EOS,) * length, dtype=cls.dtype)

    @classmethod
    def _seq_from_tok(cls, toks, timesteps):
        """Draw sequences of length timesteps from toks.

        Returns iterator over input/output pairs representing in-out.
        First pair is always BOS/?; last pair is always ?/EOS.
        """
        prev = integerize.BOS
        for b in itertools.izip_longest(fillvalue=integerize.EOS,
                                        *([iter(toks)] * (timesteps))):
            yield cls(input=numpy.array((prev,) + b[:-1], dtype=cls.dtype),
                      output=numpy.array(b, dtype=cls.dtype))
            prev = b[-1]
        if prev != integerize.EOS:
            # Last token not rounded evenly.
            yield cls(input=numpy.array((prev,), dtype=cls.dtype),
                      output=numpy.array((integerize.EOS,), dtype=cls.dtype))

    @classmethod
    def _batch_from_seq(cls, seqs, batch_size, timesteps):
        """Stack sequences (as drawn from _seq_from_tok) into rectangular
        batches of size batch_size x timesteps.
        """
        batches = itertools.izip_longest(
            fillvalue=cls._dummy_sequence(timesteps),
            *([iter(seqs)] * batch_size)
        )
        for b in batches:
            yield cls(input=numpy.array([s.input for s in b],
                                        dtype=cls.dtype),
                      output=numpy.array([s.output for s in b],
                                         dtype=cls.dtype))

    @classmethod
    def batchstream_from_tokenstream(cls, toks, timesteps, batch_size,
                                     shuffle=True):
        """Provides batch_size x timesteps chunks.

        Within each row (per batch_size), tokens are guaranteed to remain in
        the same order (they are drawn from adjacent tokens).

        If shuffle, break toks into timesteps-length chunks and shuffle first.
        """
        seqs_iter = cls._seq_from_tok(toks, timesteps=timesteps)
        if shuffle:
            seqs_iter = list(seqs_iter)
            # TODO(jeremy): optionally seed the shuffle for testing &
            # reproducibility.
            random.shuffle(seqs_iter)
        return cls._batch_from_seq(seqs_iter, batch_size=batch_size,
                                   timesteps=timesteps)


def character_patches(filehandle, timesteps, batch_size, shuffle=True):
    """Returns iterator over character patch batches."""
    # fh = open(filename, 'rb')
    char_iter = (integerize.convert_to_int(c) for c in filehandle.read())
    return IOPair.batchstream_from_tokenstream(
        char_iter,
        timesteps=timesteps,
        batch_size=batch_size,
        shuffle=True)


class Reader(collections.namedtuple('Reader', ['batch_size', 'timesteps'])):
    def stream(self, text_file, shuffle):
        return character_patches(open(text_file, 'rb'),
                                 batch_size=self.batch_size,
                                 timesteps=self.timesteps,
                                 shuffle=shuffle)

    def placeholder(self):
        return integerize.placeholder(self.batch_size, self.timesteps)

    def as_labels(self, output_placeholder):
        tokens_per_batch = self.batch_size * self.timesteps
        _t = tf.concat(1,
                       [tf.constant(numpy.arange(tokens_per_batch).reshape(
                           (tokens_per_batch, 1)), dtype=tf.int32),
                        data_utils.reshape_data(output_placeholder)])
        return tf.sparse_to_dense(_t, [tokens_per_batch,
                                       integerize.CHARS],
                                  1.0, 0.0)


# TODO(jeremy): generate training examples from each *line* or perhaps sliding
# pair thereof. Consider weighting the second line 1.0 but the first 0.0?
# What happens when we give it couplets?
