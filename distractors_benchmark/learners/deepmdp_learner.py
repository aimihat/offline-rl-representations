import time
from typing import Dict

import sonnet as snt
import tensorflow as tf
from acme.agents.tf import dqn
from networks import DeterministicTransitionModel
from trfl import base_ops


class DQNLearnerWithForwardLatent(dqn.DQNLearner):
    """A learner for model-based representation learning.
    Encompasses forward models, inverse models, as well as latent models like
    DeepMDP.
    """

    def __init__(self, **kwargs) -> None:
        self._wandb_logger = kwargs.pop("wandb")
        kwargs.pop("beta")
        super().__init__(**kwargs)
        self.reward_weight = 1.0
        self.forward_weight = 1.0
        learning_rate = 1e-4

        self.reward_decoder = snt.Sequential(
            [snt.Linear(64), tf.nn.relu, snt.Linear(1), tf.nn.relu]
        )
        self.forward_decoder = DeterministicTransitionModel(
            encoder_feature_dim=self._network._encoder_feature_dim, layer_width=64
        )

        self._deepmdp_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        # Run a DQN training step.
        fetches = super()._step()
        fetches["loss/dqn"] = fetches["loss"]
        del fetches["loss"]

        # Pull out the data needed for updates/priorities.
        inputs = next(self._iterator)
        o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
        a_tm1 = tf.expand_dims(tf.cast(a_tm1, tf.float32), -1)
        r_t = tf.expand_dims(r_t, axis=-1)

        with tf.GradientTape() as tape:
            z_tm1 = self._network.encode(o_tm1)

            reward_decoder_in = tf.concat([z_tm1, a_tm1], -1)
            reward_pred = self.reward_decoder(reward_decoder_in)
            reward_loss = tf.square(r_t - reward_pred)

            forward_decoder_in = reward_decoder_in
            forward_pred_mu, _ = self.forward_decoder(forward_decoder_in)

            z_t = self._network.encode(o_t)
            forward_loss = tf.reduce_mean(tf.square(z_t - forward_pred_mu), axis=-1)

            loss = tf.reduce_mean(
                self.reward_weight * reward_loss + self.forward_weight * forward_loss
            )

        trainable_variables = (
            list(self._network._encoder.trainable_variables)
            + list(self.reward_decoder.trainable_variables)
            + list(self.forward_decoder.trainable_variables)
        )
        grads = tape.gradient(loss, trainable_variables)

        self._deepmdp_optimizer.apply_gradients(zip(grads, trainable_variables))

        return {
            "embed_loss": loss,
            "reward_loss": tf.reduce_mean(reward_loss),
            "forward_loss": tf.reduce_mean(forward_loss),
        }

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
