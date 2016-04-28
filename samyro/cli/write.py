from __future__ import print_function

import argparse
import tensorflow as tf

from samyro.cli import positive_int, positive_float
import samyro.cli.shared
import samyro.integerize


def execute(args):
    """Actual execution of the model in the checkpoint_pattern."""
    model = samyro.cli.shared.get_model(args)
    inference_i, inference_o = model.inference_io(reuse=None)
    with tf.Session() as sess:
        runner = samyro.cli.shared.get_runner(args)
        runner.load_from_checkpoint(sess)
        text = samyro.write.write(inference_i, inference_o,
                                  seed=args.seed,
                                  max_length=args.max_length,
                                  temperature=args.temperature)
        print(text)


def set_writer_args(writer_parser):
    """The write subcommand uses the shared arguments but also these."""

    writer_parser.add_argument('--seed', type=str,
                               default=samyro.integerize.BOS_CHAR,
                               help="seed string to bootstrap hidden state")

    writer_parser.add_argument('--max_length', type=positive_int, default=1024,
                               help="max length in characters")

    writer_parser.add_argument('--temperature', type=positive_float,
                               default=0.5,
                               help="temperature (positive float)")

    writer_parser.set_defaults(execute=execute)


def main():
    """Entry point for main samyro-write script."""

    shared_parent = samyro.cli.shared.set_shared_args()

    parser = argparse.ArgumentParser(prog='samyro-write',
                                     fromfile_prefix_chars='@',
                                     parents=[shared_parent])

    set_writer_args(parser)

    args = parser.parse_args()

    execute(args)
