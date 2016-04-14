"""Functions for writing, given a samyro model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import prettytensor as pt
import six

from samyro import integerize


def write(input_placeholder, output_activations,
          seed, max_length=1024, temperature=1.0):
    """Samples from the model

    Sampling is done by first running either the seed or an arbitrary character
    through the model and then drawing the next character from the probability
    distribution definted by `softmax`.

    Args:
      input_placeholder: tensorflow input placeholder, provided by model
      output_activations: tensorflow output activations tensor, provided by
        model
      seed: A non-empty string of characters to prime the network.
      max_length: The maximum length to draw in case EOS is not reached.
      temperature: A positive value used to renormalize the inputs.  A higher
        value selects less likely choices. 1.0 leaves the distribution alone.
    Returns:
      A string that was sampled from the model.
    """
    assert temperature > 0, 'Temperature must be greater than 0.'
    assert isinstance(seed, six.string_types) and len(seed)
    # The recurrent runner takes care of tracking the model's state at
    # each step and provides a reset call to zero it out for each query.
    recurrent_runner = pt.train.RecurrentRunner()

    # We need to reset the hidden state for each query.
    recurrent_runner.reset()
    # Initialize the system
    result = ''
    for c in seed[:-1]:
        recurrent_runner.run([output_activations],
                             {input_placeholder: integerize.convert_to_int(c)})
        result += c

    # Start sampling!
    ci = integerize.convert_to_int(seed[-1])
    while len(result) < max_length and ci != integerize.EOS:
        result += chr(ci)
        # The softmax is probability normalized and would have been
        # appropriate here if we weren't applying the temperature
        # (temperature could also be done in TensorFlow).
        logit_result = recurrent_runner.run([output_activations],
                                            {input_placeholder: ci})[0][0]
        logit_result /= temperature

        # Apply the softmax in numpy to convert from logits to
        # probabilities.  Subtract off the max for numerical stability
        # -- logits are invariant to additive scaling and this
        # eliminates overflows.
        logit_result -= logit_result.max()

        distribution = numpy.exp(logit_result)
        distribution /= distribution.sum()

        # Numpy multinomial needs the value to be strictly < 1
        distribution -= .00000001
        ci = numpy.argmax(numpy.random.multinomial(1, distribution))
    result += integerize.convert_to_symbol(ci)  # Add the last letter.
    return result.replace(
        integerize.BOS_CHAR, '').replace(integerize.EOS_CHAR, '')
