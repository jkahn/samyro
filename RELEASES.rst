Release notes
=============

v0.0.3 - 28 April 2016
----------------------

Python 3 support fixes (tested with 3.4).

Changed index of `<BOS>` and `<EOS>` tokens to be \\x03 `STX` and \\x04 `ETX`
since the input here is much closer to TTY behavior than traditional
word vocabularies (in which 1 and 2 are usually reserved for `<S>` and
`</S>` respectively).

Added some doc support tools to a doc extra.


v0.0.2 - 25 April 2016
----------------------

Support seed text for samyro write.

Support samyro sample, and other forms of patch generation.

Improved examples.


v0.0.1 - 23 March 2016
----------------------

Initial release.
