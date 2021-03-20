import time
from typing import Dict, Optional, Tuple

import tensorflow as tf
import trfl
from acme.adders import reverb as adders
from acme.agents.tf.dqn import learning as dqn
from acme.tf import losses
from utils import print_vars, save_checkpoint

from .representations import autoencoder, dbc, dbc_contrastive, deepmdp


class DQNBaseOffline(dqn.DQNLearner):
    """A base class for building offline learners on top of Acme's DQN."""

    def __init__(self, *args, **kwargs) -> None:
        custom_kwargs = kwargs.pop("custom_kwargs", None)

        super().__init__(*args, **kwargs)
        self._custom_init(
            custom_kwargs=custom_kwargs, **kwargs
        )  # representation-specific

        # create variables for both optimizers
        self._step(None, None, create_vars=True)

        if hasattr(self, "_trainable_variables"):
            for v in self._trainable_variables:
                print(v.name, v.shape)

    def _pretrain_finished(self, checkpoint_path, freeze):
        print("Pretraining Finished.")
        # freeze encoder
        self._network.encoder_frozen = freeze
        print(f"set encoder_frozen to {str(freeze)}")
        assert len(self._network.variables) == len(
            self._target_network.variables
        )  # pretraining could initialize extra weights

        print_vars(self._network, trainable=True)
        print_vars(self._target_network, trainable=True)

        # checkpoint encoder (for glass-box analysis)
        # if hasattr(self._network, "_encoder"):
        #     save_checkpoint(self._network._encoder, checkpoint_path, name="Encoder")
        # if hasattr(self, "_decoder_network"):
        #     save_checkpoint(self._decoder_network, checkpoint_path, name="Decoder")

    def step(self, pretraining, joint_training=False):
        # Do a batch of SGD.
        result = self._step(pretraining, joint_training)

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
        self._logger.write(result, commit=True)

    @tf.function
    def compute_dqn_loss(self, o_tm1, a_tm1, r_t, d_t, o_t, probs):
        # Evaluate our networks.
        q_tm1 = self._network(o_tm1)
        q_t_value = self._target_network(o_t)
        q_t_selector = self._network(o_t)

        # The rewards and discounts have to have the same type as network values.
        r_t = tf.cast(r_t, q_tm1.dtype)
        # r_t = tf.clip_by_value(r_t, -1.0, 1.0)
        d_t = tf.cast(d_t, q_tm1.dtype) * tf.cast(self._discount, q_tm1.dtype)

        # Compute the loss.
        _, extra = trfl.double_qlearning(
            q_tm1, a_tm1, r_t, d_t, q_t_value, q_t_selector
        )
        loss = losses.huber(extra.td_error, self._huber_loss_parameter)

        # Get the importance weights.
        importance_weights = 1.0 / probs  # [B]
        importance_weights **= self._importance_sampling_exponent
        importance_weights /= tf.reduce_max(importance_weights)

        # Reweight.
        loss *= tf.cast(importance_weights, loss.dtype)  # [B]
        return tf.reduce_mean(loss, axis=[0])

    @tf.function
    def _step(
        self, pretraining, joint_training, create_vars=False
    ) -> Dict[str, tf.Tensor]:
        """Do a step of SGD and update the priorities."""
        # Pull out the data needed for updates/priorities.
        inputs = next(self._iterator)
        o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
        keys, probs = inputs.info[:2]

        if hasattr(
            self, "_contrastive_loss_weight"
        ):  # Have to put this here, as we cannot access _iterator in GradientTape
            second_batch = next(self._iterator).data
        else:
            second_batch = None

        ################## Hacky Solutions ##################
        # This is required as the two optimizers must be created in the same call to `_step`, within a @tf.function
        if create_vars:
            with tf.GradientTape() as tape:
                test_loss = self.compute_dqn_loss(o_tm1, a_tm1, r_t, d_t, o_t, probs)

            gradients = tape.gradient(test_loss, self._network.trainable_variables)
            self._optimizer.apply(gradients, self._network.trainable_variables)

            if hasattr(self, "_abstraction_optimizer"):
                with tf.GradientTape() as tape:
                    test_loss, _ = self._pretrain_step(
                        joint_training, inputs.data, second_batch
                    )

                gradients = tape.gradient(test_loss, self._trainable_variables)
                self._abstraction_optimizer.apply(gradients, self._trainable_variables)

            return

        # Compute the loss, depending on `joint_training` and `pretraining`
        with tf.GradientTape() as tape:
            if pretraining and not joint_training:
                dqn_loss = 0
                weighted_abstr_loss, abstr_fetches = self._pretrain_step(
                    joint_training, inputs.data, second_batch
                )
                total_loss = weighted_abstr_loss
                optimizer = self._abstraction_optimizer
                trainable_variables = self._trainable_variables
            elif joint_training:
                dqn_loss = self.compute_dqn_loss(o_tm1, a_tm1, r_t, d_t, o_t, probs)
                weighted_abstr_loss, abstr_fetches = self._pretrain_step(
                    joint_training, inputs.data, second_batch
                )
                total_loss = dqn_loss + weighted_abstr_loss
                optimizer = self._abstraction_optimizer
                trainable_variables = self._trainable_variables
            elif not pretraining and not joint_training:
                abstr_fetches = {}
                dqn_loss = total_loss = self.compute_dqn_loss(
                    o_tm1, a_tm1, r_t, d_t, o_t, probs
                )
                optimizer = self._optimizer
                trainable_variables = self._network.trainable_variables
            else:
                raise Exception()

        # Do a step of SGD.
        gradients = tape.gradient(total_loss, trainable_variables)
        optimizer.apply(gradients, trainable_variables)

        # Update the priorities in the replay buffer.
        if self._replay_client:
            priorities = tf.cast(tf.abs(extra.td_error), tf.float64)
            self._replay_client.update_priorities(
                table=adders.DEFAULT_PRIORITY_TABLE, keys=keys, priorities=priorities
            )

        # Periodically update the target network.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(
                self._network.variables, self._target_network.variables
            ):
                dest.assign(src)
        self._num_steps.assign_add(1)

        # Report loss & statistics for logging.
        fetches = {
            "loss/weighted_sum": total_loss,
            "loss/dqn": dqn_loss,
            **abstr_fetches,
        }

        return fetches


class DQNLearnerPlain(DQNBaseOffline):
    def _custom_init(self, **kwargs):
        return

    def _pretrain_step(self, joint_training, data, *args):
        raise Exception()


DQNLearnerWithAE = type(
    "DQNLearnerWithAE",
    (DQNBaseOffline,),
    {
        "_custom_init": autoencoder.custom_init,
        "_pretrain_step": autoencoder.pretraining,
    },
)
DQNLearnerWithDBC = type(
    "DQNLearnerWithDBC",
    (DQNBaseOffline,),
    {"_custom_init": dbc.custom_init, "_pretrain_step": dbc.pretraining},
)


DQNLearnerWithContrastiveDBC = type(
    "DQNLearnerWithContrastiveDBC",
    (DQNBaseOffline,),
    {
        "_custom_init": dbc_contrastive.custom_init,
        "_pretrain_step": dbc_contrastive.pretraining,
    },
)

DQNLearnerWithForwardLatent = type(
    "DQNLearnerWithForwardLatent",
    (DQNBaseOffline,),
    {"_custom_init": deepmdp.custom_init, "_pretrain_step": deepmdp.pretraining},
)
