import time
from typing import Dict

import sonnet as snt
import tensorflow as tf
from acme.agents.tf import dqn
from trfl import base_ops


class DQNLearnerWithAE(dqn.DQNLearner):
    """DQN Learner with DBC loss."""

    def __init__(self, **kwargs) -> None:
        """Initializes the learner, with a seperate optimiser for DBC loss."""
        self._ae_optimiser = snt.optimizers.Adam(kwargs.get("learning_rate"))
        self._wandb_logger = kwargs.pop("wandb")
        kwargs.pop("beta")
        super().__init__(**kwargs)

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        """Do a step of SGD and update the priorities."""

        # Run a DQN training step.
        fetches = super()._step()
        fetches["loss/dqn"] = fetches["loss"]
        del fetches["loss"]

        # Pull out the data needed for updates/priorities.
        inputs = next(self._iterator)
        o_tm1, a_tm1, r_t, d_t, o_t = inputs.data

        with tf.GradientTape() as tape:
            z_tm1 = self._network.encode(o_tm1)
            reconstruction = self._network.decode(z_tm1)
            reconstruction_l2 = tf.reduce_sum(
                tf.math.squared_difference(o_tm1, reconstruction), axis=-1
            )
            reconstruction_l2 = tf.reduce_mean(reconstruction_l2)  # []

        # Do a step of SGD.
        gradients = tape.gradient(reconstruction_l2, self._network.trainable_variables)
        self._ae_optimiser.apply(gradients, self._network.trainable_variables)

        fetches["loss/reconstruction_l2"] = reconstruction_l2

        return fetches

    def step(self):
        # Do a batch of SGD.
        result = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        result.update(counts)

        # Snapshot and attempt to write logs.
        if self._snapshotter is not None:
            self._snapshotter.save()
        self._logger.write(result)
        # Log losses.
        self._wandb_logger.log(result)
