import argparse
import os
import pickle

import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Before importing TF
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import copy
import sys
from functools import partial
from multiprocessing.dummy import Pool
from pathlib import Path
from subprocess import call

import custom_dqn_agent
import numpy as np
import tensorflow as tf
import tree
from acme import EnvironmentLoop, specs, wrappers
from acme.agents.tf import dqn

sys.path.append("../offline-representations")
from networks import QNetworkWithEncoder
from utils import memory_growth

sys.path.append("../distractors_benchmarks")
from envs.cartpole_distractors import cartpole, sweep
from wandb_logger import WandBLogger


def _float_feature(value):
    """Returns a float_list from a float / double."""
    try:
        value[0]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    except:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    try:
        value[0]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    except:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(r_t, o_tm1, a_tm1, o_t, d_t):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """

    feature = {
        "r_t": _float_feature(r_t),
        "o_tm1": _float_feature(o_tm1),
        "a_tm1": _int64_feature(a_tm1),
        "o_t": _float_feature(o_t),
        "d_t": _float_feature(d_t),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def run_experiment(args):
    """Evaluate a single bsuite_id/learner"""
    memory_growth()

    # Grab environment
    raw_environment = cartpole.Cartpole(
        n_distractors=0, distractors_type="gaussian", seed=args.run_id
    )
    environment = wrappers.SinglePrecisionWrapper(raw_environment)
    environment_spec = specs.make_environment_spec(environment)
    # Grab agent

    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            params = config["hyperparams"]
        except yaml.YAMLError as exc:
            print(exc)

    network = QNetworkWithEncoder(
        params["feature_dim"],
        environment_spec.actions.num_values,
        encoder_type=params["encoder_type"],
        hidden_dims=params["dqn_hidden_dims"],
    )

    print(params)

    # Initialize wandb logger
    logger = WandBLogger(project="Low_D_Distractors_Benchmark")
    logger.start(
        {"agent": "DQN_Base", "offline_generation": True},
    )

    filename = f'{os.environ.get("CARTPOLE_DATA_PATH")}/five-step-transitions/cartpole_run_{args.run_id}.tfrecord'

    with tf.io.TFRecordWriter(filename) as writer:

        def tf_record_writer(observation, action, reward, discount, next_observation):
            example = serialize_example(
                r_t=reward,
                o_tm1=observation,
                a_tm1=action,
                o_t=next_observation,
                d_t=discount,
            )
            writer.write(example)

        # Use 5-step transitions
        agent = custom_dqn_agent.DQN(
            environment_spec,
            network,
            learning_rate=1e-4,
            epsilon=0.3,
            tf_record_writer=tf_record_writer,
        )

        loop = EnvironmentLoop(
            environment=environment,
            actor=agent,
        )

        loop.run(num_episodes=1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--n-episodes", type=int, required=True)
    args = parser.parse_args()
    run_experiment(args)
