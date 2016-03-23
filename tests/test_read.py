"""Test reading batches of character streams."""
from samyro import read

from cStringIO import StringIO

import numpy


def test_read():
    batch_itr = read.character_patches(StringIO(u'foobar\n'),
                                       timesteps=3, batch_size=2,
                                       shuffle=False)
    first = batch_itr.next()
    assert isinstance(first, read.IOPair)
    assert isinstance(first.input, numpy.ndarray)
