"""Utility functions and definitions for converting to/from integers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import random

import prettytensor
import tensorflow

UNK = 0
BOS = 2  # officially STX "START TEXT"
EOS = 3  # officially ETX "END TEXT"

UNK_CHAR = u'\ufeff'
BOS_CHAR = chr(BOS)
EOS_CHAR = chr(EOS)

CHARS = 128


def reshape_cleavable(tensor, per_example_length=1):
    """Reshapes input so that it is appropriate for sequence_lstm..

    The expected format for sequence lstms is [timesteps * batch,
    per_example_length] and the data produced by the readers in
    samyro.read is [batch, timestep, *optional* expected_length].  The
    result can be cleaved so that there is a Tensor per timestep.

    Args:
      tensor: The tensor to reshape.
      per_example_length: The number of examples at each timestep.
    Returns:
      A Pretty Tensor that is compatible with cleave and then sequence_lstm.

    Largely borrowed from prettytensor.tutorial.data_utils.
    """
    # We can put the data into a format that can be easily cleaved by
    # transposing it (so that it varies fastest in batch) and then making each
    # component have a single value.
    # This will make it compatible with the Pretty Tensor function
    # cleave_sequence.
    dims = [1, 0]
    for i in xrange(2, tensor.get_shape().ndims):
        dims.append(i)
    transposed = prettytensor.wrap(
        tensorflow.transpose(tensor, dims))
    return transposed.reshape([-1, per_example_length])


def placeholder(batch_size, timesteps):
    return tensorflow.placeholder(tensorflow.int32, [batch_size, timesteps])


def convert_to_int(char, ordmax=128):
    i = ord(char)
    if i >= ordmax:
        return UNK
    return i


def random_seed_char():
    return chr(ord('A') + random.randint(0, 25))


def convert_to_symbol(integer, ordmax=128):
    if integer >= ordmax:
        return UNK_CHAR
    return chr(integer)
