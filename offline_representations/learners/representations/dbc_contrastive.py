"""Learning DBC as a Contrastive Metric Embedding."""

import time
from typing import Dict

import numpy as np
import sonnet as snt
import tensorflow as tf
from acme.agents.tf import dqn
from trfl import base_ops


def custom_init(self, **kwargs) -> None:
    """Initializes the learner, with a seperate optimiser for DBC loss."""
    custom_kwargs = kwargs.get("custom_kwargs")
    self._projector = custom_kwargs.get("projection_network")

    self._abstraction_optimizer = self._contrastive_optimizer = snt.optimizers.Adam(
        custom_kwargs.get("contrastive_dbc_learning_rate")
    )
    self._contrastive_loss_weight = custom_kwargs.get("contrastive_loss_weight")
    self._contrastive_loss_temperature = custom_kwargs.get(
        "contrastive_loss_temperature"
    )
    self._use_coupling_weights = custom_kwargs.get("use_coupling_weights", True)
    self._trainable_variables = list(self._network.trainable_variables) + list(
        self._projector.trainable_variables
    )


@tf.function
def pretraining(self, joint_training, batch1, batch2, *args) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""

    fetches = {}

    # Contrastive loss for DBC
    assert self._contrastive_loss_weight > 0

    weighted_abstr_loss = (
        contrastive_loss
    ) = self._contrastive_loss_weight * contrastive_metric_loss(
        batch1,
        batch2,
        self._network,
        self._projector,
        self._contrastive_loss_temperature,
        self._use_coupling_weights,
    )

    tf.debugging.check_numerics(contrastive_loss, "Contrastive loss is inf or nan.")

    fetches["loss/contrastive"] = contrastive_loss
    return weighted_abstr_loss, fetches


@tf.function
def contrastive_metric_loss(
    batch1,
    batch2,
    network,
    projector,
    contrastive_loss_temperature,
    use_coupling_weights,
    temperature: float = 1,
):
    """DBC contrastive loss."""

    # TODO: choose batch sizes
    # TODO: does he use entire replay buffer then samples a few reference points for each step? https://github.com/google-research/google-research/blob/cf6dea3992b102ab6c845debb66c061970f4f6f9/pse/dm_control/utils/helper_utils.py#L59
    o_tm1, _, r_t, _, o_t = batch1
    o_tm1_, _, r_t_, _, o_t_ = batch2

    z_tm1, z_tm1_ = projector(network.encode(o_tm1)), projector(network.encode(o_tm1_))
    z_t, z_t_ = projector(network.encode(o_t)), projector(
        network.encode(o_t_)
    )  # TODO: he doesn't put stop gradient/or target network here, but we did in regular dbc

    # TODO: dealing with repeated samples (if at all) ?

    metric_vals = tf.stop_gradient(bisim(z_t, z_t_, r_t, r_t_))

    similarity_matrix = cosine_similarity(z_tm1, z_tm1_)
    alignment_loss = contrastive_loss(
        similarity_matrix,
        metric_vals,
        temperature,
        coupling_temperature=contrastive_loss_temperature,
        use_coupling_weights=use_coupling_weights,
    )

    return alignment_loss


@tf.function
def contrastive_loss(
    similarity_matrix: tf.Tensor,
    metric_values: tf.Tensor,
    temperature: float,
    coupling_temperature: float = 1.0,
    use_coupling_weights: bool = True,
) -> tf.Tensor:
    """Contrative Loss with soft coupling."""

    metric_shape = tf.shape(metric_values)
    similarity_matrix /= temperature
    neg_logits = similarity_matrix

    col_indices = tf.cast(tf.argmin(metric_values, axis=1), dtype=tf.int32)
    pos_indices = tf.stack(
        (tf.range(metric_shape[0], dtype=tf.int32), col_indices), axis=1
    )
    pos_logits = tf.gather_nd(similarity_matrix, pos_indices)

    if use_coupling_weights:
        # Do not penalise as much for states that are selected 'different' but in reality are close.
        metric_values /= coupling_temperature
        coupling = tf.exp(-metric_values)
        pos_weights = -tf.gather_nd(metric_values, pos_indices)
        pos_logits += pos_weights
        negative_weights = tf.math.log((1.0 - coupling) + 1e-9)
        neg_logits += tf.tensor_scatter_nd_update(
            negative_weights, pos_indices, pos_weights
        )
    neg_logits = tf.math.reduce_logsumexp(neg_logits, axis=1)  # not excluding pos?
    return tf.reduce_mean(neg_logits - pos_logits)


@tf.function
def bisim(
    z_t: tf.Tensor,
    z_t_: tf.Tensor,
    r_t: tf.Tensor,
    r_t_: tf.Tensor,
    name: str = "bisim",
) -> tf.Tensor:
    """Computes the bisimulation metric, pairwise for two set of observations."""

    r_t = tf.cast(r_t, dtype=tf.float32)
    r_t_ = tf.cast(r_t_, dtype=tf.float32)

    # Add dummy dimensions for `r_t`.
    r_t = add_dummy_dimension(r_t)
    r_t_ = add_dummy_dimension(r_t_)

    # 1. transition dynamics (calculated with samples).
    p_dist = pdist_l2(z_t, z_t_)
    # 2. immediate (one-step) rewards.
    r_dist = pdist_l2(r_t, r_t_)

    # Target distance.
    bisimilarity = r_dist + 0.99 * p_dist
    return bisimilarity


############## Utils ##############


@tf.function
def add_dummy_dimension(x):
    return x[..., None]


@tf.function
def cosine_similarity(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Computes cosine similarity between all pairs of vectors in x and y."""
    x_expanded, y_expanded = x[:, tf.newaxis], y[tf.newaxis, :]
    similarity_matrix = tf.reduce_sum(x_expanded * y_expanded, axis=-1)
    similarity_matrix /= (
        tf.norm(x_expanded, axis=-1) * tf.norm(y_expanded, axis=-1) + 1e-9
    )
    return similarity_matrix


@tf.function
def pdist_l2(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
    """Computes pairwise euclidian distances between the rows of two tensors.

    Args:
        A (tf.Tensor): An n by k tensor.
        B (tf.Tensor): An m by k tensor.

    Returns:
        tf.Tensor: An n by m tensor.
    """
    assert A.shape.as_list() == B.shape.as_list()

    row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
    row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

    row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
    row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

    pdist_l2 = row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B
    pdist_l2 = tf.clip_by_value(
        pdist_l2, clip_value_min=0, clip_value_max=tf.float32.max
    )
    return tf.sqrt(pdist_l2) + 1e-6


# @tf.function
# def pdist_l1(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
#     """Computes pairwise euclidian distances between the rows of two tensors.

#     Args:
#         A (tf.Tensor): An n by k tensor.
#         B (tf.Tensor): An m by k tensor.

#     Returns:
#         tf.Tensor: An n by m tensor.
#     """
#     assert A.shape.as_list() == B.shape.as_list()
#     import pdb
#     pdb.set_trace()
#     row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
#     row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

#     row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
#     row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

#     return row_norms_A - 2 * tf.matmul(A, tf.transpose(B)) + row_norms_B
