import argparse
import copy
import os
import sys

import sonnet as snt
import tensorflow as tf
from acme.agents.tf import actors
from acme.agents.tf.dqn import learning as dqn
from acme.tf import utils as acme_utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Before importing TF

import sonnet as snt
import tensorflow as tf
from acme import EnvironmentLoop
from acme.agents.tf import actors
from utils import (
    check_GPU,
    evaluate,
    load_conf,
    load_dataset,
    log_batch_images,
    set_seeds,
)

sys.path.append("../deepmind-research")
sys.path.append("../distractors_benchmarks")
from rl_unplugged import atari
from wandb_logger import WandBLogger


def main(arguments, conf):  # TODO: think offline policy selection?
    # Preliminaries
    set_seeds()
    check_GPU()
    params = conf["hyperparams"]

    logger = WandBLogger(project="Offline_Atari_Representations")

    # Initialize dataset/env/learner
    dataset = load_dataset(arguments, params)

    environment = atari.environment(game=arguments.game)
    # Get total number of actions.
    num_actions = environment.action_spec().num_values

    # Create the Q network.
    network = snt.Sequential(
        [
            lambda x: tf.image.convert_image_dtype(x, tf.float32),
            snt.Conv2D(32, [8, 8], [4, 4]),
            tf.nn.relu,
            snt.Conv2D(64, [4, 4], [2, 2]),
            tf.nn.relu,
            snt.Conv2D(64, [3, 3], [1, 1]),
            tf.nn.relu,
            snt.Flatten(),
            snt.nets.MLP([512, num_actions]),
        ]
    )
    acme_utils.create_variables(network, [environment.observation_spec()])

    # Evaluation loop (actor regularly updated)
    policy_network = snt.Sequential(
        [
            network,
            lambda q: tf.argmax(q, axis=-1),
        ]
    )
    eval_loop = EnvironmentLoop(
        environment=environment,
        actor=actors.FeedForwardActor(policy_network=policy_network),
        logger=logger,
    )

    # Start logging
    logger.start({**vars(args)})
    # log_batch_images(dataset)

    # Create the DQN learner.
    learner = dqn.DQNLearner(
        network=network,
        target_network=copy.deepcopy(network),
        discount=0.99,
        learning_rate=3e-4,
        importance_sampling_exponent=0.2,
        target_update_period=2500,
        dataset=dataset,
        logger=logger,
    )

    # Training
    for i in range(2000000):
        learner.step()
        if i % conf["evaluation_frequency"] == 0:
            evaluate(
                eval_loop,
                environment,
                conf,
                arguments.game,
                n_episodes=conf["evaluation_episodes"],
                current_step=i,
            )

    # Final evaluation
    evaluate(
        eval_loop,
        environment,
        conf,
        arguments.game,
        n_episodes=conf["final_evaluation_episodes"],
        current_step="Final",
    )


if __name__ == "__main__":
    assert all(
        required_var in os.environ
        for required_var in ("ATARI_REPLAYS_PATH", "ATARI_DATA_PATH", "CHECKPOINT_PATH")
    )

    # Load configuration
    conf = load_conf()

    # Parse runtime arguments
    parser = argparse.ArgumentParser(
        description="Run an experiment"
    )  # TODO: use all runs?

    parser.add_argument(
        "--game",
        type=str,
        help="Atari game",
        choices=conf["supported_games"],
        required=True,
    )

    parser.add_argument("--num-shards", type=int, default=100, nargs="?")
    parser.add_argument(
        "--data-run", type=str, required=True, help="Run(s) to use. E.g. `1` or `1-3`"
    )

    args = parser.parse_args()

    # Run experiment
    main(args, conf)
