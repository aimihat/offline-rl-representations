import time
from typing import Dict

import sonnet as snt
import tensorflow as tf
from acme.agents.tf import dqn
from acme.tf.networks.continuous import LayerNormMLP
from networks import DeterministicTransitionModel
from trfl import base_ops


def custom_init(self, **kwargs) -> None:
    custom_kwargs = kwargs.get("custom_kwargs")
    self.reward_weight = 1.0
    self.forward_weight = 1.0

    self._deepmdp_weight = custom_kwargs.get("deepmdp_weight")

    self.forward_decoder = custom_kwargs.get("forward_decoder")
    self.reward_decoder = custom_kwargs.get("reward_decoder")

    self._abstraction_optimizer = self._deepmdp_optimizer = snt.optimizers.Adam(
        learning_rate=custom_kwargs.get("deepmdp_learning_rate")
    )

    # create variables for `reward` and `forward` decoders
    self._pretrain_step(False, next(self._iterator).data)

    self._trainable_variables = (
        list(self._network.trainable_variables)
        + list(self.reward_decoder.trainable_variables)
        + list(self.forward_decoder.trainable_variables)
    )


@tf.function
def pretraining(self, joint_training, data, *args) -> Dict[str, tf.Tensor]:
    # Pull out the data needed for updates/priorities.
    o_tm1, a_tm1, r_t, d_t, o_t = data
    expand_dims = lambda x: tf.expand_dims(x, -1)
    to_float = lambda x: tf.cast(x, tf.float32)
    a_tm1 = expand_dims(to_float(a_tm1))
    r_t = expand_dims(to_float(r_t))

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
    weighted_abstr_loss = loss * self._deepmdp_weight

    fetches = {
        "loss/embed": loss,
        "loss/reward": tf.reduce_mean(reward_loss),
        "loss/forward": tf.reduce_mean(forward_loss),
    }
    return weighted_abstr_loss, fetches
