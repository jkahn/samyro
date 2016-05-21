"""Defines models for reading that save out to checkpoints."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from samyro import integerize

import prettytensor as pt
import tensorflow as tf


class Model(object):
    """A model shape.

    TODO(jkahn): convert to collections.namedtuple?
    """
    def __init__(self, name, embedding=16, lstms=(128, 256)):
        self.name = name
        self.embedding = embedding
        self.lstms = lstms

    def create(self, input_placeholder, phase):
        """Creates a 2 layer LSTM model with dropout.

        Args:
          input_placeholder: placeholder of timesteps x sequences

          phase: Phase controls whether or not dropout is active.  In
            training mode we want to perform dropout, but in test we
            want to disable it.

        Returns: The logits layer.

        """
        timesteps = input_placeholder.get_shape()[1].value
        text_in = integerize.reshape_cleavable(input_placeholder)
        with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
            # The embedding lookup must be placed on a cpu.
            with tf.device('/cpu:0'):
                embedded = text_in.embedding_lookup(integerize.CHARS,
                                                    [self.embedding])
        # Because the sequence LSTM expects each timestep to be its
        # own Tensor, we need to cleave the sequence.  Below we can
        # build a stacked 2 layer LSTM by just chaining them together.
        # You can stack as many layers as you want.
        lstm = embedded.cleave_sequence(timesteps)
        assert len(self.lstms)
        for lstm_size in self.lstms:
            lstm = lstm.sequence_lstm(lstm_size)

        # The classifier is much more efficient if it runs across the entire
        # dataset at once, so we want to squash (i.e. uncleave).
        # Note: if phase is test, dropout is a noop.
        return (lstm.squash_sequence()
                .dropout(keep_prob=0.8, phase=phase)
                .fully_connected(integerize.CHARS, activation_fn=None))

    def train_op_loss(self, input_placeholder, labels, reuse=None):
        # Training and eval graph
        with tf.variable_scope(self.name, reuse=reuse):
            # Core train graph
            result = self.create(input_placeholder,
                                 pt.Phase.train).softmax(labels)

            train_op = pt.apply_optimizer(tf.train.AdagradOptimizer(0.5),
                                          losses=[result.loss])
            return train_op, result.loss

    def eval_accuracy(self, input_placeholder, labels, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            # Eval graph
            eval_result = self.create(input_placeholder,
                                      pt.Phase.test).softmax(labels)

        # Accuracy creates variables, so do it outside of scope
        return eval_result.softmax.evaluate_classifier(labels,
                                                       phase=pt.Phase.test)

    def inference_io(self, reuse):
        with tf.variable_scope(self.name, reuse=reuse), pt.defaults_scope(
                summary_collections=['INFERENCE_SUMMARIES']):
            inf_input = tf.placeholder(tf.int32, [])
            inf_logits = self.create(pt.wrap(inf_input).reshape([1, 1]),
                                     pt.Phase.infer)
            return inf_input, inf_logits
