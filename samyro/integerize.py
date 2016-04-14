"""Utility functions and definitions for converting to/from integers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import random

import tensorflow as tf

UNK = 0
BOS = 127
EOS = 1

UNK_CHAR = u'\ufeff'
BOS_CHAR = "\x7f"
EOS_CHAR = "\x01"

CHARS = 128


def placeholder(batch_size, timesteps):
    return tf.placeholder(tf.int32, [batch_size, timesteps])


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
