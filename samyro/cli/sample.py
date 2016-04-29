from __future__ import print_function

import argparse
# import tensorflow as tf

import itertools

from samyro.cli import positive_int
import samyro.cli.shared
import samyro.integerize


def set_sampler_args(parser):
    samyro.cli.shared.set_sampler_shared_args(parser, default_batch_size=1)

    parser.add_argument('--shuffle', help="whether to shuffle samples",
                        default=True, type=bool)

    parser.add_argument('--batches', help="how many samples to draw",
                        default=5,
                        type=positive_int)

    parser.add_argument('--format', help="how to format examples",
                        choices=['text', 'numpy'],
                        default='text')

    # Positional argument for text file to read.
    parser.add_argument(
        'files', metavar="FILEGLOB", help="text files to sample", type=str,
        nargs="*",
        default=['/opt/data/texts/shakespeare_input.txt'])

    parser.set_defaults(execute=execute)


def execute(args):
    """Actual execution of the sampler."""
    sampler = samyro.cli.shared.get_sampler(args)

    print("args: ", args)

    if args.format == 'text':
        stream = sampler.batches(shuffle=args.shuffle, to_numpy=False)
    elif args.format == 'numpy':
        stream = sampler.batches(shuffle=args.shuffle, to_numpy=True)
    else:
        assert False, "whoa nelly"

    for b_i, b in enumerate(itertools.islice(stream, args.batches)):
        print("# batch %s" % b_i)
        for i in b:
            print(i)


def main():
    """Entry point for main samyro-write script."""

    # shared_parent = samyro.cli.shared.set_shared_args()

    parser = argparse.ArgumentParser(prog='samyro-sample',
                                     fromfile_prefix_chars='@',)

    set_sampler_args(parser)

    args = parser.parse_args()

    execute(args)
