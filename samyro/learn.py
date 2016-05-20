"""Samyro learner parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from samyro import read, model


class Trainer(object):
    """A training regime."""
    def __init__(self, model_instance, sampler, batch_size, reuse):

        assert isinstance(model_instance, model.Model)
        assert isinstance(sampler, read.Sampler)

        input_placeholder = sampler.placeholder()
        output_placeholder = sampler.placeholder()
        labels = sampler.as_labels(output_placeholder)

        self.feed_vars = (input_placeholder, output_placeholder)
        self.train_op, self.train_loss = model_instance.train_op_loss(
            input_placeholder,
            labels, reuse=reuse)
        self.eval_accuracy = model_instance.eval_accuracy(
            input_placeholder,
            labels, reuse=True)

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
