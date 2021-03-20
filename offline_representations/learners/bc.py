import time
from typing import Dict, Optional, Tuple

import tensorflow as tf
import wandb
from acme.agents.tf.bc import learning as bc
from utils import save_checkpoint

from .representations import autoencoder, dbc, dbc_contrastive, deepmdp


class BCBaseOffline(bc.BCLearner):
    """A base class for building offline learners on top of Acme's BC."""

    def __init__(self, *args, **kwargs) -> None:
        custom_kwargs = kwargs.pop("custom_kwargs", None)

        super().__init__(*args, **kwargs)
        self._custom_init(
            custom_kwargs=custom_kwargs, **kwargs
        )  # representation-specific

        self._timestamp = None

    def _pretrain_finished(self, checkpoint_path, freeze):
        print("Pretraining Finished.")

        # freeze encoder
        self._network.encoder_frozen = freeze

        # checkpoint encoder (for glass-box analysis)
        if hasattr(self._network, "_encoder"):
            save_checkpoint(self._network._encoder, checkpoint_path, name="Encoder")
        if hasattr(self, "_decoder_network"):
            save_checkpoint(self._decoder_network, checkpoint_path, name="Decoder")

    def step(self, pretraining, joint_training=False):
        # Do a batch of SGD.
        result = self._step(pretraining, joint_training)

        # Update our counts and record it.
        counts = self._counter.increment(steps=1)
        result.update(counts)

        # Snapshot and attempt to write logs.
        if self._snapshotter is not None:
            self._snapshotter.save()
            self._logger.write(result, commit=True)

    @tf.function
    def compute_bc_loss(self, o_tm1, a_tm1):
        logits = self._network(o_tm1)
        # wandb.log({'bc_logits_0': wandb.Histogram(logits[:, 0]),'bc_logits_1': wandb.Histogram(logits[:, 1]),'bc_logits_2': wandb.Histogram(logits[:, 2])}, commit=False)
        cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return cce(a_tm1, logits)

    @tf.function
    def _step(self, pretraining, joint_training) -> Dict[str, tf.Tensor]:
        """Do a step of SGD and update the priorities."""

        # Pull out the data needed for updates/priorities.
        inputs = next(self._iterator)
        o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
        del r_t, d_t, o_t

        with tf.GradientTape() as tape:
            if pretraining and not joint_training:
                bc_loss = 0
                weighted_abstr_loss, abstr_fetches = self._pretrain_step(
                    joint_training, inputs.data
                )
                total_loss = weighted_abstr_loss
                optimizer = self._abstraction_optimizer
                trainable_variables = self._trainable_variables
            elif not pretraining and joint_training:
                bc_loss = self.compute_bc_loss(o_tm1, a_tm1)
                weighted_abstr_loss, abstr_fetches = self._pretrain_step(
                    joint_training, inputs.data
                )
                total_loss = bc_loss + weighted_abstr_loss
                optimizer = self._abstraction_optimizer
                trainable_variables = self._trainable_variables
            elif not pretraining and not joint_training:
                abstr_fetches = {}
                bc_loss = total_loss = self.compute_bc_loss(o_tm1, a_tm1)
                optimizer = self._optimizer
                trainable_variables = self._network.trainable_variables
            else:
                raise Exception()

        # Do a step of SGD.
        gradients = tape.gradient(total_loss, trainable_variables)
        # gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
        optimizer.apply(gradients, trainable_variables)

        self._num_steps.assign_add(1)

        # Compute the global norm of the gradients for logging.
        global_gradient_norm = tf.linalg.global_norm(gradients)
        fetches = {
            "loss/bc": bc_loss,
            "loss/weighted_sum": total_loss,
            "gradient_norm": global_gradient_norm,
            **abstr_fetches,
        }

        return fetches


class BCLearnerPlain(BCBaseOffline):
    def _custom_init(self, **kwargs):
        return

    def _pretrain_step(self, joint_training, data):
        raise Exception()


BCLearnerWithAE = type(
    "BCLearnerWithAE",
    (BCBaseOffline,),
    {
        "_custom_init": autoencoder.custom_init,
        "_pretrain_step": autoencoder.pretraining,
    },
)
BCLearnerWithDBC = type(
    "BCLearnerWithDBC",
    (BCBaseOffline,),
    {"_custom_init": dbc.custom_init, "_pretrain_step": dbc.pretraining},
)
BCLearnerWithContrastiveDBC = type(
    "BCLearnerWithContrastiveDBC",
    (BCBaseOffline,),
    {
        "_custom_init": dbc_contrastive.custom_init,
        "_pretrain_step": dbc_contrastive.pretraining,
    },
)
BCLearnerWithForwardLatent = type(
    "BCLearnerWithForwardLatent",
    (BCBaseOffline,),
    {"_custom_init": deepmdp.custom_init, "_pretrain_step": deepmdp.pretraining},
)
