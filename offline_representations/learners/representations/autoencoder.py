"""A DQN with an AE representation"""
from typing import Dict, Optional, Tuple

import sonnet as snt
import tensorflow as tf


def custom_init(self, **kwargs) -> None:
    """Initializes the learner, with a seperate optimiser for reconstruction loss."""
    custom_kwargs = kwargs.get("custom_kwargs")
    self._abstraction_optimizer = self._ae_optimiser = snt.optimizers.Adam(
        custom_kwargs.get("ae_learning_rate")
    )
    self._decoder_network = custom_kwargs.get("decoder_network")

    self._ae_weight = custom_kwargs.get("ae_weight")
    if custom_kwargs.get("n_distractors", None) is not None:
        n_distractors = custom_kwargs.get("n_distractors")
        scaling_factor = 106.0 / (n_distractors + 6.0)
        self._ae_weight *= 106.0 / (n_distractors + 6.0)
        print(f"Scaling AE WEIGHT by {str(scaling_factor)} to {str(self._ae_weight)}")

    self._trainable_variables = list(self._network.trainable_variables) + list(
        self._decoder_network.trainable_variables
    )


@tf.function
def pretraining(self, joint_training, data, *args) -> Dict[str, tf.Tensor]:
    o_tm1, a_tm1, r_t, d_t, o_t = data

    fetches = {}

    o_tm1 = tf.image.convert_image_dtype(o_tm1, tf.float32)

    z_tm1 = self._network.encode(o_tm1)
    reconstruction = self._decoder_network(z_tm1)
    reconstruction_l2 = tf.norm(o_tm1 - reconstruction)
    reconstruction_l2 = tf.reduce_mean(reconstruction_l2)
    weighted_abstr_loss = reconstruction_l2 * self._ae_weight

    # Do a step of SGD.

    fetches["loss/reconstruction_l2"] = reconstruction_l2

    return weighted_abstr_loss, fetches
