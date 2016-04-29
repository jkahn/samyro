import argparse

import prettytensor as pt

import samyro.learn
import samyro.model
import samyro.read
import samyro.write

import samyro.cli.learn
import samyro.cli.write
import samyro.cli.sample

from samyro.cli import positive_int


def get_model(args):
    """Summarize model according to user specified params."""
    return samyro.model.Model(name=args.name,
                              embedding=args.embedding_size,
                              lstms=args.lstm_dims)


def get_runner(args):
    return pt.train.Runner(save_path=args.checkpoint_pattern)


def set_shared_args():
    """All subcommands use these arguments."""

    # TODO: add a verbosity level.
    shared_parent = argparse.ArgumentParser(add_help=False)

    # Model config arguments
    model_group = shared_parent.add_argument_group('model configuration')
    model_group.add_argument(
        '--embedding_size',
        metavar="N_DIMS",
        help="size of input character embedding",
        default=16,
        type=int)

    model_group.add_argument(
        "--lstm_dims", metavar="STACK_OF_LSTM_DIMS",
        help="comma-separated list of LSTM sizes",
        type=lambda x: tuple(int(y.strip()) for y in x.split(",")),
        default=(128, 256))

    model_group.add_argument(
        "--name", metavar="namespace prefix",
        help="namespace prefix for this model; stored internally",
        default='shakespeare', type=str)

    checkpoint_group = shared_parent.add_argument_group(
        'parameter configuration')

    checkpoint_group.add_argument(
        '--checkpoint_pattern',
        metavar="FILE_PREFIX",
        help="where to find checkpoints",
        type=str,
        default='/home/jeremy/notebooks/shakespeare/shakespeare')

    return shared_parent


def set_sampler_shared_args(parser, default_batch_size=8):
    reader_group = parser.add_argument_group(
        'sampler read configuration')

    reader_group.add_argument(
        '--batch_size', type=positive_int, default=default_batch_size,
        help="how many segments should be read per backprop batch")

    reader_group.add_argument(
        '--steps_per_sample',
        type=positive_int, default=100,
        help="how many timesteps per sample")

    reader_group.add_argument(
        '--sampler',
        choices=['patches', 'lines', 'paragraphs'],
        default='patches',
        help="select the class of sampler to use on the file")


def get_sampler(args):
    if args.sampler == 'patches':
        sampler_class = samyro.read.CharacterPatchSampler
    elif args.sampler == 'lines':
        sampler_class = samyro.read.LineSampler
    elif args.sampler == 'paragraphs':
        sampler_class = samyro.read.ParagraphSampler
    else:
        assert False, "unrecognized --sampler %s" % args.sampler

    return sampler_class(
        filenames=args.files,
        sample_length=args.steps_per_sample,
        batch_size=args.batch_size)


def main():
    """Entry point for main samyro script with subcommands."""

    shared_parent = set_shared_args()

    # Main argument parsers.
    parser = argparse.ArgumentParser(prog='samyro', fromfile_prefix_chars='@')

    subparsers = parser.add_subparsers(help="samyro subcommands",
                                       dest="subcommand_name")

    learner_parser = subparsers.add_parser("learn", help="reading a text",
                                           parents=[shared_parent])

    samyro.cli.learn.set_learner_args(learner_parser)

    writer_parser = subparsers.add_parser("write", help="generate new text",
                                          parents=[shared_parent])
    samyro.cli.write.set_writer_args(writer_parser)

    sample_parser = subparsers.add_parser("sample",
                                          help="debug sampling from files")
    samyro.cli.sample.set_sampler_args(sample_parser)

    # Actual work done by the execute function declared inside each subcommand

    # TODO: This would be a great place to put a shared call to argcomplete.
    args = parser.parse_args()

    args.execute(args)
