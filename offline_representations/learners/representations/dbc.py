from typing import Dict

import sonnet as snt
import tensorflow as tf
from acme.agents.tf import dqn
from trfl import base_ops


def custom_init(self, **kwargs) -> None:
    """Initializes the learner, with a seperate optimiser for DBC loss."""
    custom_kwargs = kwargs.get("custom_kwargs")
    self._abstraction_optimizer = self._dbc_optimiser = snt.optimizers.Adam(
        custom_kwargs.get("dbc_learning_rate")
    )
    self._projector = custom_kwargs.get("projection_network")
    self._projection_feature_dim = custom_kwargs.get("projection_feature_dim")
    self._dbc_weight = custom_kwargs.get("dbc_weight")
    self._trainable_variables = list(self._network.trainable_variables) + list(
        self._projector.trainable_variables
    )
    # if not hasattr(self, '_target_network'): # If running DBC on BC
    #     self._target_network = kwargs.get('network')
    self.projection_decoder = custom_kwargs.get("projection_decoder")


# TODO: setting this to tf.function fails at tf.gather (know issue?)
def pretraining(self, joint_training, data, *args) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""
    o_tm1, a_tm1, r_t, d_t, o_t = data
    expand_dims = lambda x: tf.expand_dims(x, -1)
    to_float = lambda x: tf.cast(x, tf.float32)
    a_tm1 = expand_dims(to_float(a_tm1))
    if joint_training:
        # Using target_network for DBC potentially adds extra stability
        dbc_target = self._target_network
    else:
        # target network doesn't get updated -> we cannot use it here
        dbc_target = self._network

    z_tm1 = self._projector(self._network.encode(o_tm1))
    z_t = tf.stop_gradient(self._projector(dbc_target.encode(o_t)))

    forward_decoder_in = tf.concat([z_tm1, a_tm1], -1)
    forward_pred_mu, _ = self.projection_decoder(forward_decoder_in)
    forward_loss = tf.reduce_mean(tf.square(z_t - forward_pred_mu), axis=-1)

    dbc_loss = bisim(z_tm1, forward_pred_mu, r_t, d_t, self._projection_feature_dim)
    dbc_loss = tf.reduce_mean(dbc_loss, axis=[0])  # []
    weighted_abstr_loss = (
        dbc_loss + 5 * tf.reduce_mean(forward_loss)
    ) * self._dbc_weight

    # Do a step of SGD.
    fetches = {"loss/dbc": dbc_loss, "loss/forward": tf.reduce_mean(forward_loss)}

    return weighted_abstr_loss, fetches


@tf.function
def bisim(
    z_tm1: tf.Tensor,
    z_t: tf.Tensor,
    r_t: tf.Tensor,
    pcont_t: tf.Tensor,
    projection_feature_dim: int,
    name: str = "bisim",
) -> tf.Tensor:
    """Implements the bisimulation metric loss as a TensorFlow op.

    See "Learning Invariant Representations for Reinforcement Learning without
    Reconstruction"(https://arxiv.org/abs/2006.10742).

    Args:
      z_tm1: The latent state for the first timestep in a batch of transitions, with
        shape `[B, D]`.
      z_t: The latent state for the second timestep in a batch of transitions, with
        shape `[B, D]`.
      r_t: The rewards, with shape `[B]`.
      pcont_t: Pcontinue values (i.e., discount), with shape `[B]`.

    Returns:
      The bisimulation metric loss.
    """
    base_ops.wrap_rank_shape_assert([[z_tm1, z_t], [r_t, pcont_t]], [2, 1], name)
    tf.debugging.assert_shapes(
        [
            (z_tm1, (256, projection_feature_dim)),
        ]
    )
    tf.debugging.assert_shapes(
        [
            (z_tm1, (256, projection_feature_dim)),
        ]
    )
    tf.debugging.assert_shapes(
        [
            (r_t, (256,)),
        ]
    )

    # Add dummy dimensions for `r_t` and `pcont_t`.
    add_dummy_dimension = lambda x: x[..., None]
    r_t = add_dummy_dimension(r_t)
    pcont_t = add_dummy_dimension(pcont_t)

    # Permute data in the batch dimension for calculating the contrastive loss.
    batch_size = z_tm1.shape[0]
    indices = tf.range(start=0, limit=batch_size, dtype=tf.int32)
    permutated_indices = tf.random.shuffle(indices)
    permute = lambda x: tf.gather(x, permutated_indices)
    z_tm1_ = permute(z_tm1)
    z_t_ = permute(z_t)
    r_t_ = permute(r_t)

    # Calculate the L1 distance in:
    distance = lambda x, x_: tf.linalg.norm(x - x_, ord=2, axis=-1)
    # 1. latent states.
    z_dist = distance(z_tm1, z_tm1_)
    # 2. transition dynamics (calculated with samples).
    p_dist = distance(z_t, z_t_)
    # 3. immediate (one-step) rewards.
    r_dist = distance(r_t, r_t_)

    # Target distance.
    bisimilarity = tf.stop_gradient(r_dist + 0.99 * p_dist)

    return (z_dist - bisimilarity) ** 2
