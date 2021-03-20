import time
from typing import Dict

import sonnet as snt
import tensorflow as tf
from acme.agents.tf import dqn
from trfl import base_ops


class DQNLearnerWithVAE(dqn.DQNLearner):
    """DQN Learner with VAE."""

    def __init__(self, **kwargs) -> None:
        """Initializes the learner, with a seperate optimiser for DBC loss."""
        self._vae_optimiser = snt.optimizers.Adam(kwargs.get("learning_rate"))
        self._wandb_logger = kwargs.pop("wandb")
        self.beta = kwargs.pop("beta")

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
            loss, vae_loss_breakdown = self.compute_loss(self._network, o_tm1)
        gradients = tape.gradient(loss, self._network.trainable_variables)
        self._vae_optimiser.apply(gradients, self._network.trainable_variables)

        fetches["loss/vae_loss"] = loss

        return {**fetches, **vae_loss_breakdown}

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

    @tf.function
    def compute_loss(self, model, x):
        mean, logvar = model._encoder(x, reparametrize=False)
        z = model._encoder.reparameterize(mean, logvar)
        reconstruction = model.decode(z)

        # TODO: for atari use cross entropy https://www.tensorflow.org/tutorials/generative/cvae

        # latent space loss. KL divergence between latent space distribution and unit gaussian, for each batch.
        # first half of eq 10. in https://arxiv.org/abs/1312.6114
        kl_loss = -0.5 * tf.reduce_sum(
            1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1
        )
        kl_loss *= self.beta

        # reconstruction error, using pixel-wise L2 loss, for each batch
        rec_loss = tf.reduce_sum(
            tf.math.squared_difference(reconstruction, x), axis=[1]
        )  # TODO: change axes for atari
        vae_loss_breakdown = {
            "kl_mean": tf.reduce_mean(kl_loss),
            "rec_mean": tf.reduce_mean(rec_loss),
        }
        # sum the two and average over batches
        loss = tf.reduce_mean(
            kl_loss + rec_loss
        )  # TODO: adding KL loss results in terrible performance
        return loss, vae_loss_breakdown
