import argparse

import tensorflow as tf

from samyro.cli import positive_int, positive_float

import samyro.cli.shared
import samyro.read
import samyro.learn


def execute(args):
    """The actual execution of learning (training)."""
    print(args)
    model = samyro.cli.shared.get_model(args)
    reader = samyro.read.Reader(batch_size=args.batch_size,
                                timesteps=args.steps_per_sample)
    trainer = samyro.learn.Trainer(model_instance=model,
                                   reader=reader, reuse=None)

    with tf.Session():   # do we need to call out this sess variable?
        runner = samyro.cli.shared.get_runner(args)
        for epoch in xrange(args.epochs):
            trainer.train_pass(
                runner,
                print_every=args.print_every,
                batch_iterator=reader.stream(args.file.name,
                                             shuffle=args.shuffle_train),
                num_batches=args.batches_per_epoch)

            classification_accuracy = trainer.eval_pass(
                runner,
                batch_iterator=reader.stream(args.file.name,
                                             shuffle=args.shuffle_eval),
                num_batches=args.eval_per_epoch)
            print('Next character accuracy after epoch %d: %g%%'
                  % (epoch + 1, classification_accuracy * 100))

            print(samyro.write.write(*(model.inference_io(reuse=True)),
                                     max_length=args.sample_length,
                                     temperature=args.sample_temperature))


def set_learner_args(learner_parser):
    """samyro learn uses these arguments and defines the learn function."""

    # Learner-specific arguments
    reader_group = learner_parser.add_argument_group(
        'learner read configuration')

    reader_group.add_argument(
        '--batch_size', type=positive_int, default=8,
        help="how many segments should be read per backprop batch")

    reader_group.add_argument(
        '--steps_per_sample', type=positive_int, default=100,
        help="how many timesteps per sample")

    regime_group = learner_parser.add_argument_group(
        'training regime configuration')
    regime_group.add_argument('--epochs',
                              type=positive_int,
                              default=10,
                              help="how many training epochs to run")
    regime_group.add_argument('--batches_per_epoch',
                              type=positive_int,
                              default=300,
                              help="how many batches per epoch")
    regime_group.add_argument('--shuffle_train', default=True,
                              type=bool,
                              help="if each epoch should be a new random set")
    regime_group.add_argument(
        '--print_every', default=25,
        type=int,
        help="how many batches between training loss updates")

    eval_group = learner_parser.add_argument_group(
        'eval regime configuration: 1x per epoch')
    eval_group.add_argument('--eval_per_epoch',
                            type=positive_int,
                            default=300,
                            help="how many eval batches per epoch")
    eval_group.add_argument('--shuffle_eval', default=False,
                            type=bool,
                            help="if each eval should be a new random draw")

    sample_group = learner_parser.add_argument_group(
        'draw a sample from the model every epoch')
    sample_group.add_argument('--sample_length',
                              type=positive_int,
                              default=128,
                              help="max length of eval sample")
    sample_group.add_argument('--sample_temperature',
                              type=positive_float,
                              default=0.5,
                              help="temperature for eval samples")

    # Positional argument for text file to read.
    learner_parser.add_argument(
        'file', help="text file", type=argparse.FileType('r'),
        nargs="?",
        default='/opt/data/texts/shakespeare_input.txt')

    learner_parser.set_defaults(execute=execute)


def main():
    """Entry point for main samyro-learn script."""

    shared_parent = samyro.cli.shared.set_shared_args()

    parser = argparse.ArgumentParser(prog='samyro-learn',
                                     fromfile_prefix_chars='@',
                                     parents=[shared_parent])

    set_learner_args(parser)

    args = parser.parse_args()

    execute(args)
