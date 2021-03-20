import argparse
import copy
import os
import sys
from typing import Dict, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Before importing TF

import sonnet as snt
import tensorflow as tf
from acme import EnvironmentLoop
from acme.agents.tf import actors
from learners import get_learner
from utils import (
    check_GPU,
    evaluate,
    get_evaluation_actor,
    load_conf,
    load_dataset,
    load_decoder_weights,
    load_encoder_weights,
    memory_growth,
    set_seeds,
)

sys.path.append("../deepmind-research")
sys.path.append("../distractors_benchmarks")
import wandb
from rl_unplugged import atari
from wandb_logger import WandBLogger


def main(arguments, params, conf, logger):
    set_seeds(seed=arguments.seed)
    memory_growth()
    check_GPU()

    # Initialize dataset/env/learner
    dataset = load_dataset(arguments, params)
    environment = atari.environment(
        game=arguments.game, replace_background=arguments.background_replace != "False"
    )
    learner = get_learner(arguments, environment, dataset, logger, params)

    # Evaluation loop (actor regularly updated)
    policy_network = snt.Sequential(
        [
            learner._network,
            lambda q: tf.argmax(q, axis=-1),
        ]
    )
    eval_loop = EnvironmentLoop(
        environment=environment,
        actor=actors.FeedForwardActor(policy_network=policy_network),
        logger=logger,
    )

    # Pretraining
    if not args.joint_training:
        if arguments.abstraction != "None":
            for _ in range(params["pretrain_steps"]):
                learner.step(pretraining=True)

    learner._pretrain_finished(os.environ.get("CHECKPOINT_PATH"), freeze=args.freeze)

    # Training
    for i in range(params["steps"]):
        learner.step(pretraining=False, joint_training=args.joint_training)
        if i % conf["evaluation_frequency"] == 0:
            evaluate(
                eval_loop,
                environment,
                conf,
                arguments.game,
                n_episodes=conf["evaluation_episodes"],
                current_step=i,
                learner=learner,
            )

    # Final evaluation
    evaluate(
        eval_loop,
        environment,
        conf,
        arguments.game,
        n_episodes=conf["final_evaluation_episodes"],
        current_step="Final",
        learner=learner,
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

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-replace", type=str, default="False")
    parser.add_argument("--algorithm", type=str, help="RL method to use")
    parser.add_argument(
        "--abstraction", type=str, help="State abstraction method to use", required=True
    )
    parser.add_argument(
        "--freeze",
        type=str,
        help="Whether to propagate the DQN loss to the encoder",
        required=True,
        choices=["False", "True"],
    )
    parser.add_argument(
        "--joint_training",
        type=str,
        required=True,
        help="""
        'True': We train the encoder by alternating DQN/BC steps with representation-learning steps. 
        'False': We train the encoder separately in a `pre-training` step. The DQN/BC MLP is trained as a second step.
        
        `--freeze` determines if the gradient from Q-function is allowed to propagate to encoder.
        """,
        choices=["False", "True"],
    )

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
    parser.add_argument(
        "--data-path",
        type=str,
        help='Parent folder containing game data (e.g. data-path = "~/", which contains "~/Pong" and "~/Breakout")',
    )

    args = parser.parse_known_args()[0]
    args.joint_training = (args.joint_training != "False") and (
        args.abstraction != "None"
    )
    args.freeze = args.freeze != "False"

    wandb_params = {**vars(args), **conf["hyperparams"]}
    conf.pop("hyperparams")  # ensure we use dynamic hyperparams when doing a sweep

    logger = WandBLogger(project="Offline_Atari_Representations")
    logger.start(wandb_params)

    # Run experiment
    print(wandb.config)
    print(conf)
    main(args, wandb.config, conf, logger)
