"""
Modified Acme DQN agent to record 5-step transitions.
"""

import copy
import itertools
import operator
from typing import Optional

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme import datasets, specs, types
from acme.adders import reverb as adders
from acme.adders.reverb import base, utils
from acme.agents import agent
from acme.agents.tf import actors
from acme.agents.tf.dqn import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import loggers, tree_utils


class CustomNStepAdder(adders.NStepTransitionAdder):
    def __init__(self, *args, **kwargs):
        self._tf_record_writer = kwargs["tf_record_writer"]
        kwargs.pop("tf_record_writer")
        super().__init__(*args, **kwargs)

    def _write(self):
        # NOTE: we do not check that the buffer is of length N here. This means
        # that at the beginning of an episode we will add the initial N-1
        # transitions (of size 1, 2, ...) and at the end of an episode (when
        # called from write_last) we will write the final transitions of size (N,
        # N-1, ...). See the Note in the docstring.

        # Form the n-step transition given the steps.
        observation = self._buffer[0].observation
        action = self._buffer[0].action
        extras = self._buffer[0].extras
        next_observation = self._next_observation

        # Give the same tree structure to the n-step return accumulator,
        # n-step discount accumulator, and self.discount, so that they can be
        # iterated in parallel using tree.map_structure.
        (
            n_step_return,
            total_discount,
            self_discount,
        ) = tree_utils.broadcast_structures(
            self._buffer[0].reward, self._buffer[0].discount, self._discount
        )

        # Copy total_discount, so that accumulating into it doesn't affect
        # _buffer[0].discount.
        total_discount = tree.map_structure(np.copy, total_discount)

        # Broadcast n_step_return to have the broadcasted shape of
        # reward * discount. Also copy, to avoid accumulating into
        # _buffer[0].reward.
        n_step_return = tree.map_structure(
            lambda r, d: np.copy(np.broadcast_to(r, np.broadcast(r, d).shape)),
            n_step_return,
            total_discount,
        )

        # NOTE: total discount will have one less discount than it does
        # step.discounts. This is so that when the learner/update uses an additional
        # discount we don't apply it twice. Inside the following loop we will
        # apply this right before summing up the n_step_return.
        for step in itertools.islice(self._buffer, 1, None):
            (
                step_discount,
                step_reward,
                total_discount,
            ) = tree_utils.broadcast_structures(
                step.discount, step.reward, total_discount
            )

            # Equivalent to: `total_discount *= self._discount`.
            tree.map_structure(operator.imul, total_discount, self_discount)

            # Equivalent to: `n_step_return += step.reward * total_discount`.
            tree.map_structure(
                lambda nsr, sr, td: operator.iadd(nsr, sr * td),
                n_step_return,
                step_reward,
                total_discount,
            )

            # Equivalent to: `total_discount *= step.discount`.
            tree.map_structure(operator.imul, total_discount, step_discount)

        if extras:
            transition = (
                observation,
                action,
                n_step_return,
                total_discount,
                next_observation,
                extras,
            )
        else:
            transition = (
                observation,
                action,
                n_step_return,
                total_discount,
                next_observation,
            )

        ##### WE ONLY ADD THIS LOGGING HERE #####
        self._tf_record_writer(
            observation=observation,
            action=action,
            reward=n_step_return,
            discount=total_discount,
            next_observation=next_observation,
        )

        # Create a list of steps.
        final_step = utils.final_step_like(self._buffer[0], next_observation)
        steps = list(self._buffer) + [final_step]

        # Calculate the priority for this transition.
        table_priorities = utils.calculate_priorities(self._priority_fns, steps)

        # Insert the transition into replay along with its priority.
        self._writer.append(transition)
        for table, priority in table_priorities.items():
            self._writer.create_item(table=table, num_timesteps=1, priority=priority)


class DQN(agent.Agent):
    """DQN agent.
    This implements a single-process DQN agent. This is a simple Q-learning
    algorithm that inserts N-step transitions into a replay buffer, and
    periodically updates its policy by sampling these transitions using
    prioritization.
    """

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        network: snt.Module,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        samples_per_insert: float = 32.0,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        importance_sampling_exponent: float = 0.2,
        priority_exponent: float = 0.6,
        n_step: int = 5,
        epsilon: Optional[tf.Tensor] = None,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/acme/",
        policy_network: Optional[snt.Module] = None,
        tf_record_writer=None,
    ):
        """Initialize the agent.
        Args:
          environment_spec: description of the actions, observations, etc.
          network: the online Q network (the one being optimized)
          batch_size: batch size for updates.
          prefetch_size: size to prefetch from replay.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          samples_per_insert: number of samples to take from replay for every insert
            that is made.
          min_replay_size: minimum replay size before updating. This and all
            following arguments are related to dataset construction and will be
            ignored if a dataset argument is passed.
          max_replay_size: maximum replay size.
          importance_sampling_exponent: power to which importance weights are raised
            before normalizing.
          priority_exponent: exponent used in prioritized sampling.
          n_step: number of steps to squash into a single transition.
          epsilon: probability of taking a random action; ignored if a policy
            network is given.
          learning_rate: learning rate for the q-network update.
          discount: discount to use for TD updates.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
          checkpoint_subpath: directory for the checkpoint.
          policy_network: if given, this will be used as the policy network.
            Otherwise, an epsilon greedy policy using the online Q network will be
            created. Policy network is used in the actor to sample actions.
        """

        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        replay_table = reverb.Table(
            name=adders.DEFAULT_PRIORITY_TABLE,
            sampler=reverb.selectors.Prioritized(priority_exponent),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders.NStepTransitionAdder.signature(environment_spec),
        )
        self._server = reverb.Server([replay_table], port=None)

        # The adder is used to insert observations into replay.
        address = f"localhost:{self._server.port}"
        adder = CustomNStepAdder(
            client=reverb.Client(address),
            n_step=n_step,
            discount=discount,
            tf_record_writer=tf_record_writer,
        )

        # The dataset provides an interface to sample from replay.
        replay_client = reverb.TFClient(address)
        dataset = datasets.make_reverb_dataset(
            server_address=address, batch_size=batch_size, prefetch_size=prefetch_size
        )

        # Create epsilon greedy policy network by default.
        if policy_network is None:
            # Use constant 0.05 epsilon greedy policy by default.
            if epsilon is None:
                epsilon = tf.Variable(0.05, trainable=False)
            policy_network = snt.Sequential(
                [
                    network,
                    lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
                ]
            )

        # Create a target network.
        target_network = copy.deepcopy(network)

        # Ensure that we create the variables before proceeding (maybe not needed).
        tf2_utils.create_variables(network, [environment_spec.observations])
        tf2_utils.create_variables(target_network, [environment_spec.observations])

        # Create the actor which defines how we take actions.
        actor = actors.FeedForwardActor(policy_network, adder)

        # The learner updates the parameters (and initializes them).
        learner = learning.DQNLearner(
            network=network,
            target_network=target_network,
            discount=discount,
            importance_sampling_exponent=importance_sampling_exponent,
            learning_rate=learning_rate,
            target_update_period=target_update_period,
            dataset=dataset,
            replay_client=replay_client,
            logger=logger,
            checkpoint=checkpoint,
        )

        if checkpoint:
            self._checkpointer = tf2_savers.Checkpointer(
                directory=checkpoint_subpath,
                objects_to_save=learner.state,
                subdirectory="dqn_learner",
                time_delta_minutes=60.0,
            )
        else:
            self._checkpointer = None

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert,
        )

    def update(self):
        super().update()
        if self._checkpointer is not None:
            self._checkpointer.save()
