"""Tzara learner parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf

from samyro import read, write, model


class Trainer(object):
    """A training regime."""
    def __init__(self, model_instance, reader, reuse):
        self.model = model_instance
        self.reader = reader

        assert isinstance(model_instance, model.Model)
        assert isinstance(reader, read.Reader)

        input_placeholder = reader.placeholder()
        output_placeholder = reader.placeholder()
        labels = reader.as_labels(output_placeholder)

        self.feed_vars = (input_placeholder, output_placeholder)
        self.train_op, self.train_loss = model_instance.train_op_loss(
            input_placeholder,
            labels, reuse=reuse)
        self.eval_accuracy = model_instance.eval_accuracy(
            input_placeholder,
            labels, reuse=True)
        self.inference_io = model_instance.inference_io(reuse=True)

    def train_pass(self, runner, batch_iterator, num_batches=300,
                   print_every=25):
        runner.train_model(train_op=self.train_op,
                           cost_to_log=self.train_loss,
                           num_steps=num_batches,
                           feed_vars=self.feed_vars,
                           feed_data=batch_iterator,
                           print_every=print_every)

    def eval_pass(self, runner, batch_iterator, num_batches=300):
        return runner.evaluate_model(accuracy=self.eval_accuracy,
                                     num_steps=num_batches,
                                     feed_vars=self.feed_vars,
                                     feed_data=batch_iterator)

    def run_cycle(self, checkpoint_path, text_file,
                  shuffle_between_sequences=True,
                  epochs=10, batches_per_epoch=300,
                  print_every=25):
        with tf.Session():
            runner = pt.train.Runner(save_path=checkpoint_path)
            # TODO: yield at each epoch
            for epoch in xrange(epochs):
                self.train_pass(
                    runner,
                    batch_iterator=self.reader.stream(
                        text_file,
                        shuffle=shuffle_between_sequences),
                    num_batches=batches_per_epoch,
                    print_every=print_every)

                classification_accuracy = self.eval_pass(
                    runner,
                    batch_iterator=self.reader.stream(
                        text_file,
                        shuffle=False),
                    num_batches=batches_per_epoch)
                print('Next character accuracy after epoch %d: %g%%'
                      % (epoch + 1, classification_accuracy * 100))

                print(write.write(*(self.inference_io),
                                  max_length=128, temperature=0.5))

            print(write.write(*(self.inference_io)))
