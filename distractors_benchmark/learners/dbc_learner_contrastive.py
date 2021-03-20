"""Learning DBC as a Contrastive Metric Embedding."""

import time
from typing import Dict

import numpy as np
import sonnet as snt
import tensorflow as tf
from acme.agents.tf import dqn
from trfl import base_ops

from .utils import add_dummy_dimension, cosine_similarity, pdist_l2


class DQNLearnerWithContrastive(dqn.DQNLearner):
    """DQN Learner with a contrastive metric (DBC) embeddings."""

    def __init__(self, **kwargs) -> None:
        """Initializes the learner, with a seperate optimiser for DBC loss."""
        self._contrastive_optimizer = snt.optimizers.Adam(kwargs.get("learning_rate"))
        self._contrastive_loss_weight = kwargs.get("contrastive_loss_weight", 1.0)
        self._contrastive_loss_temperature = kwargs.get(
            "contrastive_loss_temperature", 0.5
        )

        self._use_coupling_weights = kwargs.get("use_coupling_weights", True)

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

        # Contrastive loss for DBC
        contrastive_loss = 0
        if self._contrastive_loss_weight > 0:
            contrastive_vars = self._network.trainable_variables
            with tf.GradientTape() as tape:
                batch1 = next(self._iterator)
                batch2 = next(self._iterator)
                contrastive_loss = (
                    self._contrastive_loss_weight
                    * self.contrastive_metric_loss(batch1, batch2)
                )

                tf.debugging.check_numerics(
                    contrastive_loss, "Contrastive loss is inf or nan."
                )

            # Do a step of Adam.
            contrastive_grads = tape.gradient(contrastive_loss, contrastive_vars)
            self._contrastive_optimizer.apply(contrastive_grads, contrastive_vars)

        fetches["loss/contrastive"] = contrastive_loss

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

    @tf.function
    def contrastive_metric_loss(
        self,
        batch1,
        batch2,
        temperature: float = 1,
    ):
        """DBC contrastive loss."""

        # TODO: choose batch sizes
        # TODO: does he use entire replay buffer then samples a few reference points for each step? https://github.com/google-research/google-research/blob/cf6dea3992b102ab6c845debb66c061970f4f6f9/pse/dm_control/utils/helper_utils.py#L59
        o_tm1, _, r_t, _, o_t = batch1.data
        o_tm1_, _, r_t_, _, o_t_ = batch2.data

        z_tm1, z_tm1_ = self._network.latent_state(o_tm1), self._network.latent_state(
            o_tm1_
        )
        z_t, z_t_ = self._network.latent_state(o_t), self._network.latent_state(o_t_)

        # TODO: dealing with repeated samples (if at all) ?

        metric_vals = tf.stop_gradient(bisim(z_t, z_t_, r_t, r_t_))

        similarity_matrix = cosine_similarity(z_tm1, z_tm1_)
        alignment_loss = contrastive_loss(
            similarity_matrix,
            metric_vals,
            temperature,
            coupling_temperature=self._contrastive_loss_temperature,
            use_coupling_weights=self._use_coupling_weights,
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
