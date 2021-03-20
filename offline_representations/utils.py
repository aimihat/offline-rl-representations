"""Helper functions"""
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import acme
import imageio
import numpy as np
import sonnet as snt
import tensorflow as tf
import wandb
import yaml
from acme.agents.tf import actors
from acme.tf import utils as tf2_utils
from acme.tf.utils import zeros_like
from dm_env import specs

sys.path.append("../deepmind-research")
from rl_unplugged import atari


def save_checkpoint(encoder, checkpoint_path, name="Encoder"):
    """Saving encoder weights to use for glass-box analysis."""
    print(f"Checkpointing {name} network")
    save_prefix = os.path.join(checkpoint_path, f"{name}{wandb.run.name}")
    checkpoint = tf.train.Checkpoint(module=encoder)
    checkpoint.save(save_prefix)


def discard_extras(sample):
    """Provided in sample notebook: https://github.com/deepmind/deepmind-research/blob/master/rl_unplugged/atari_dqn.ipynb"""
    return sample._replace(data=sample.data[:5])


def print_vars(snt_module, trainable: bool = False):
    """Displays all variables owned by the module."""
    print("=======================")
    if trainable:
        variables = snt_module.trainable_variables
    else:
        variables = snt_module.variables
    for v in variables:
        print(v.name, v.shape)
    print("=======================")


def encoding_spec(network, observation_spec, n_actions_concat=False):
    """Returns the Acme spec for the encoder output."""
    dummy_input = zeros_like(observation_spec)

    if type(dummy_input) == list:
        dummy_input = tf.expand_dims(dummy_input[0], 0)
    encoding = network.encode(dummy_input)

    shape = encoding.shape
    if n_actions_concat:
        shape = shape[:-1] + (shape[-1] + n_actions_concat,)
    return specs.Array(
        shape=encoding.shape, dtype=encoding.dtype.as_numpy_dtype(), name="Encoding"
    )


def load_encoder_weights(learner, path) -> None:
    """Loads the weights of the encoder from `path`.
    NOTE: WE ARE NOT SETTING THE WEIGHTS OF TARGET NETWORK (IN THE CASE OF DQN).
    This should be ok, after the first target_update."""

    print(f"Loading encoder from {path}")
    checkpoint = tf.train.Checkpoint(module=learner._network._encoder)
    checkpoint.restore(path)


def load_decoder_weights(learner, path) -> None:
    """Loads the weights of the _decoder_network from `path`."""

    print(f"Loading decoder from {path}")
    checkpoint = tf.train.Checkpoint(module=learner._decoder_network)
    checkpoint.restore(path)


def log_batch_images(dataset, n_batches=3):
    """For debugging purposes: ensure data is loaded correctly"""
    print(f"Showing a o_tm1 from {n_batches} batches")
    it = iter(dataset)
    for _ in range(n_batches):
        o_tm1 = next(it)[1][0]
        wandb.log(
            {
                f"o_tm1[...,{f}]": [wandb.Image(im) for im in o_tm1[100:105, ..., f]]
                for f in range(4)
            }
        )


def evaluate(
    loop, environment, conf, game, n_episodes=1, current_step="NA", learner=None
) -> None:
    """Runs the environment for `n_episodes` using the given Q-Network.
    Also saves a short video render of the current policy to Google Drive"""

    loop.run(n_episodes)

    try:
        if "cartpole" not in game and (current_step % 15000 == 0):
            # Render `n_steps_render` steps of online interaction
            replays_path = Path(os.environ.get("ATARI_REPLAYS_PATH"))
            filename = (
                replays_path / f"{game}---{wandb.run.name}---step-{current_step}.mp4"
            )
            run_episode(
                loop._actor, environment, conf["n_steps_render"], filename, learner
            )

            # A separate script uploads videos in `ATARI_REPLAYS_PATH` to Google Drive
    except:
        pass


class FeedForwardActorLoggingQ(actors.FeedForwardActor):
    def __init__(
        self, *args, n_values=1e4, algorithm="DQN", log_states=False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._n_values = n_values
        self._algorithm = algorithm
        self._log_states = log_states
        self._last_n_values = []
        self._last_states = []
        self._counter_states_log = 0

    @tf.function
    def _policy(self, observation):
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # Compute the policy, conditioned on the observation.
        policy = self._policy_network(batched_observation)

        return policy

    def select_action(self, observation):
        policy = self._policy(observation)

        if self._algorithm == "DQN":
            action, q_value = policy
            if len(self._last_n_values) < self._n_values:
                self._last_n_values.append(q_value)

                if self._log_states:
                    self._last_states.append((observation, q_value))
            else:
                wandb.log(
                    {"q-values": wandb.Histogram(self._last_n_values)}, commit=False
                )
                self._last_n_values = []

                # Logging for state distribution analysis
                if self._log_states:
                    with open(
                        f"/vol/bitbucket/ac7117/last_states{self._counter_states_log}.pkl",
                        "wb",
                    ) as f:
                        self._counter_states_log += 1
                        pickle.dump(self._last_states, f)
                        self._last_states = []
        else:
            action = policy

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(action)


def get_evaluation_actor(learner, algorithm, distractors):
    network = lambda q: learner._network(q, is_training=False)

    # Evaluation loop (actor regularly updated)
    if algorithm == "DQN":
        policy_network = snt.Sequential(
            [
                distractors.distract_observation,
                network,
                lambda q: (tf.argmax(q, axis=-1), tf.reduce_max(q)),
            ]
        )
    elif algorithm == "BC":
        policy_network = snt.Sequential(
            [distractors.distract_observation, network, lambda q: tf.argmax(q, axis=-1)]
        )
    else:
        raise Exception()

    return FeedForwardActorLoggingQ(
        policy_network=policy_network, algorithm=algorithm, log_states=False
    )


def memory_growth():
    """Enable memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def load_dataset(arguments, params):
    """Load offline atari data"""
    print(f"Loading offline data for game of {arguments.game}")
    dataset = atari.dataset(
        path=os.environ.get("ATARI_DATA_PATH"),
        game=arguments.game,
        run=arguments.data_run,
        num_shards=arguments.num_shards,
        background_replace=arguments.background_replace != "False",
        params=params,
    )

    return dataset


def run_episode(actor, environment, n_steps: int, filename: str, learner) -> None:
    """Run the agent in the environment for `n_steps`, recording frames to a gif.
    If the encoder uses a reconstruction loss. We concatenate a second video, corresponding to the reconstruction."""
    print(f"Rendering {n_steps} steps to video.")

    frames = []
    save_recons = hasattr(learner, "_decoder_network")

    # remove stacking + add channel dimension
    to_video = lambda x: np.expand_dims(x, axis=1)[..., 0]

    timestep = environment.reset()
    for _ in range(n_steps):
        # Save observation
        frames.append(timestep.observation)
        action = actor.select_action(timestep.observation)
        timestep = environment.step(action)

    frames = np.stack(frames)

    # Also display reconstructions
    if save_recons:
        recons = np.empty((0,) + frames.shape[1:])

        for b in batch(frames, n=256):
            recons = np.append(
                recons,
                tf.image.convert_image_dtype(
                    learner._decoder_network(learner._network.encode(b)), "uint8"
                ),
                axis=0,
            )

        # wandb.log({'reconstruction': wandb.Video(to_video(frames[:250,...]), fps=60, format='gif')})
        frames = np.concatenate([frames, recons], axis=2)

    wandb.log({"observation": wandb.Video(to_video(frames), fps=60, format="gif")})


def load_conf(filename: str = "config.yaml"):
    """Loads a YAML config file."""
    with open(filename, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def set_seeds(seed=1234) -> None:
    """Set various random seeds."""
    print(f"Setting `tf`, `random`, `np` seeds to {seed}")
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_GPU() -> None:
    """Asserts a gpu is available."""
    if tf.test.gpu_device_name():
        print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
    else:
        raise Exception("Please install GPU version of TF")


def batch(iterable, n=1):
    """Transforms an iterable into an generator size `n` batches"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l), ...]
