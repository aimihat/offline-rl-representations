# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Atari RL Unplugged datasets.

Examples in the dataset represent SARSA transitions stored during a
DQN training run as described in https://arxiv.org/pdf/1907.04543.

For every training run we have recorded all 50 million transitions corresponding
to 200 million environment steps (4x factor because of frame skipping). There
are 5 separate datasets for each of the 45 games.

Every transition in the dataset is a tuple containing the following features:

* o_t: Observation at time t. Observations have been processed using the
    canonical Atari frame processing, including 4x frame stacking. The shape
    of a single observation is [84, 84, 4].
* a_t: Action taken at time t.
* r_t: Reward after a_t.
* d_t: Discount after a_t.
* o_tp1: Observation at time t+1.
* a_tp1: Action at time t+1.
* extras:
  * episode_id: Episode identifier.
  * episode_return: Total episode return computed using per-step [-1, 1]
      clipping.
"""
import functools
import os
from typing import Dict

import dm_env
import reverb
import tensorflow as tf
from acme import wrappers
from dm_env import specs
from dopamine.discrete_domains import atari_lib

base_path = os.path.dirname(os.path.realpath(__file__))
import sys

sys.path.append(base_path)
from background_video import RandomVideoSource

video_path = base_path + "/backgroundvideo.mp4"
print(f"loading video from {video_path}")
random_video = RandomVideoSource([video_path])

# 9 tuning games.
TUNING_SUITE = [
    "BeamRider",
    "DemonAttack",
    "DoubleDunk",
    "IceHockey",
    "MsPacman",
    "Pooyan",
    "RoadRunner",
    "Robotank",
    "Zaxxon",
]

# 36 testing games.
TESTING_SUITE = [
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Atlantis",
    "BankHeist",
    "BattleZone",
    "Boxing",
    "Breakout",
    "Carnival",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "Enduro",
    "FishingDerby",
    "Freeway",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "Jamesbond",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "NameThisGame",
    "Phoenix",
    "Pong",
    "Qbert",
    "Riverraid",
    "Seaquest",
    "SpaceInvaders",
    "StarGunner",
    "TimePilot",
    "UpNDown",
    "VideoPinball",
    "WizardOfWor",
    "YarsRevenge",
]

# Total of 45 games.
ALL = TUNING_SUITE + TESTING_SUITE


def _decode_frames(pngs: tf.Tensor):
    """Decode PNGs.

    Args:
      pngs: String Tensor of size (4,) containing PNG encoded images.

    Returns:
      4 84x84 grayscale images packed in a (84, 84, 4) uint8 Tensor.
    """
    # Statically unroll png decoding
    frames = [tf.image.decode_png(pngs[i], channels=1) for i in range(4)]
    frames = tf.concat(frames, axis=2)
    frames.set_shape((84, 84, 4))
    return frames


def _make_reverb_sample(
    o_t: tf.Tensor, a_t: tf.Tensor, r_t: tf.Tensor, d_t: tf.Tensor, o_tp1: tf.Tensor
) -> reverb.ReplaySample:
    """Create Reverb sample with offline data.

    Args:
      o_t: Observation at time t.
      a_t: Action at time t.
      r_t: Reward at time t.
      d_t: Discount at time t.
      o_tp1: Observation at time t+1.
      a_tp1: Action at time t+1.
      extras: Dictionary with extra features.

    Returns:
      Replay sample with fake info: key=0, probability=1, table_size=0.
    """
    info = reverb.SampleInfo(
        key=tf.constant(0, tf.uint64),
        probability=tf.constant(1.0, tf.float64),
        table_size=tf.constant(0, tf.int64),
        priority=tf.constant(1.0, tf.float64),
    )
    data = (o_t, a_t, r_t, d_t, o_tp1)
    return reverb.ReplaySample(info=info, data=data)


def _tf_example_to_reverb_sample(
    tf_example: tf.train.Example, background_replace: bool
) -> reverb.ReplaySample:
    """Create a Reverb replay sample from a TF example."""

    # Parse tf.Example.
    feature_description = {
        "o_t": tf.io.FixedLenFeature([4], tf.string),
        "o_tp1": tf.io.FixedLenFeature([4], tf.string),
        "a_t": tf.io.FixedLenFeature([], tf.int64),
        "a_tp1": tf.io.FixedLenFeature([], tf.int64),
        "r_t": tf.io.FixedLenFeature([], tf.float32),
        "d_t": tf.io.FixedLenFeature([], tf.float32),
        "episode_id": tf.io.FixedLenFeature([], tf.int64),
        "episode_return": tf.io.FixedLenFeature([], tf.float32),
    }
    data = tf.io.parse_single_example(tf_example, feature_description)

    # Process data.
    o_t = _decode_frames(data["o_t"])
    o_tp1 = _decode_frames(data["o_tp1"])

    if background_replace:
        o_t = random_video.replace_background(background_color=87, frame=o_t)
        o_tp1 = random_video.replace_background(background_color=87, frame=o_tp1)
    a_t = tf.cast(data["a_t"], tf.int32)

    return _make_reverb_sample(o_t, a_t, data["r_t"], data["d_t"], o_tp1)


def dataset(
    path: str,
    game: str,
    run: str,
    num_shards: int = 100,
    shuffle_buffer_size: int = 100000,
    background_replace=False,
    params=None,
) -> tf.data.Dataset:
    """TF dataset of Atari SARSA tuples."""

    from_run = to_run = run
    if "-" in run:  # Range of runs
        from_run, to_run = run.split("-")
    from_run, to_run = int(from_run), int(to_run)
    run_range = range(from_run, to_run + 1)
    print(f"Using runs {str(run_range)}.")

    path = os.path.join(path, f"{game}/run_")
    filenames = [
        f"{path}{r}-{i:05d}-of-{num_shards:05d}"
        for i in range(num_shards)
        for r in run_range
    ]
    total_shards = len(filenames)
    print(f"Found {total_shards} total shards.")

    file_ds = tf.data.Dataset.from_tensor_slices(filenames)
    file_ds = file_ds.repeat().shuffle(total_shards, seed=42)
    example_ds = file_ds.interleave(
        functools.partial(tf.data.TFRecordDataset, compression_type="GZIP"),
        cycle_length=tf.data.experimental.AUTOTUNE,
        # deterministic=False,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        block_length=5,
    )
    example_ds = example_ds.map(
        lambda x: _tf_example_to_reverb_sample(
            x, background_replace=background_replace
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    example_ds = example_ds.shuffle(shuffle_buffer_size, seed=42)

    # Optimizations (likely to improve performance)
    dataset = example_ds.batch(params["batch_size"], drop_remainder=True)

    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


class AtariDopamineWrapper(dm_env.Environment):
    """Wrapper for Atari Dopamine environmnet."""

    def __init__(self, env, max_episode_steps=108000, replace_background=False):
        self._env = env
        self._max_episode_steps = max_episode_steps
        self._episode_steps = 0
        self._reset_next_episode = True
        self._replace_background = replace_background
        if self._replace_background:
            self.random_video = RandomVideoSource([video_path])

    def reset(self):
        self._episode_steps = 0
        self._reset_next_step = False
        observation = self._env.reset()

        if self._replace_background:
            observation = self.random_video.replace_background(
                background_color=87, frame=observation
            ).numpy()

        return dm_env.restart(observation.squeeze(-1))

    def step(self, action):
        if self._reset_next_step:
            return self.reset()

        observation, reward, terminal, _ = self._env.step(action.item())
        if self._replace_background:
            observation = self.random_video.replace_background(
                background_color=87, frame=observation
            ).numpy()
        observation = observation.squeeze(-1)

        discount = 1 - float(terminal)
        self._episode_steps += 1
        if terminal:
            self._reset_next_episode = True
            return dm_env.termination(reward, observation)
        elif self._episode_steps == self._max_episode_steps:
            self._reset_next_episode = True
            return dm_env.truncation(reward, observation, discount)
        else:
            return dm_env.transition(reward, observation, discount)

    def observation_spec(self):
        space = self._env.observation_space
        return specs.Array(space.shape[:-1], space.dtype)

    def action_spec(self):
        return specs.DiscreteArray(self._env.action_space.n)


def environment(game: str, replace_background=False) -> dm_env.Environment:
    """Atari environment."""
    env = atari_lib.create_atari_environment(game_name=game, sticky_actions=True)
    env = AtariDopamineWrapper(env, replace_background=replace_background)
    env = wrappers.FrameStackingWrapper(env, num_frames=4)
    return wrappers.SinglePrecisionWrapper(env)


if __name__ == "__main__":
    dataset = dataset(
        path="/vol/bitbucket/ac7117/Pong_offline/",
        game="Pong",
        run="1",
        background_replace=True,
        params={"batch_size": 256},
    )
    diter = iter(dataset)
    for b in range(10):
        obs = next(diter).data[0]

        from PIL import Image

        for p in range(10):
            for j in range(1):
                im = Image.fromarray(obs[p, ..., j].numpy())
                im.save(f"delete/new_{b}_{p}.jpeg")
