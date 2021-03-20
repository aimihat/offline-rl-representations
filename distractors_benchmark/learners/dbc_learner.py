import time
from typing import Dict

import sonnet as snt
import tensorflow as tf
from acme.agents.tf import dqn
from trfl import base_ops


class DQNLearnerWithDBC(dqn.DQNLearner):
    """DQN Learner with DBC loss."""

    def __init__(self, **kwargs) -> None:
        """Initializes the learner, with a seperate optimiser for DBC loss."""
        self._dbc_optimiser = snt.optimizers.Adam(kwargs.get("learning_rate"))
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
        keys, probs = inputs.info[:2]

        with tf.GradientTape() as tape:
            z_tm1 = self._network.latent_state(o_tm1)
            # Using target_network potentially adds stability
            z_t = tf.stop_gradient(self._target_network.latent_state(o_t))
            dbc_loss = bisim(z_tm1, z_t, r_t, d_t)
            dbc_loss = tf.reduce_mean(dbc_loss, axis=[0])  # []

        # Do a step of SGD.
        gradients = tape.gradient(dbc_loss, self._network.trainable_variables)
        self._dbc_optimiser.apply(gradients, self._network.trainable_variables)

        fetches["loss/dbc"] = dbc_loss

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
def bisim(
    z_tm1: tf.Tensor,
    z_t: tf.Tensor,
    r_t: tf.Tensor,
    pcont_t: tf.Tensor,
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

    # Add dummy dimensions for `r_t` and `pcont_t`.
    add_dummy_dimension = lambda x: x[..., None]
    r_t = add_dummy_dimension(r_t)
    pcont_t = add_dummy_dimension(pcont_t)

    # Permute data in the batch dimension for calculating the contrastive loss.
    batch_size = z_tm1.shape[0]
    indices = tf.range(start=0, limit=batch_size, dtype=tf.int32)
    permutated_indices = tf.random.shuffle(indices)
    permute = lambda x: tf.gather(
        x, permutated_indices
    )  # pylint: disable=no-value-for-parameter
    z_tm1_ = permute(z_tm1)
    z_t_ = permute(z_t)
    r_t_ = permute(r_t)

    # Calculate the L1 distance in:
    distance = lambda x, x_: tf.linalg.norm(x - x_, ord=1, axis=-1)
    # 1. latent states.
    z_dist = distance(z_tm1, z_tm1_)
    # 2. transition dynamics (calculated with samples).
    p_dist = distance(z_t, z_t_)
    # 3. immediate (one-step) rewards.
    r_dist = distance(r_t, r_t_)

    # Target distance.
    bisimilarity = tf.stop_gradient(r_dist + 0.99 * p_dist)

    return (z_dist - bisimilarity) ** 2
