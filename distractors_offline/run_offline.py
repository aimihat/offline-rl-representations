import argparse
import copy
import os
import sys
from typing import Dict, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Before importing TF

sys.path.append("../offline-representations")
import cartpole_dataset
import wandb
from acme import EnvironmentLoop, wrappers
from learners import get_learner
from utils import (
    check_GPU,
    evaluate,
    get_evaluation_actor,
    load_conf,
    memory_growth,
    save_checkpoint,
    set_seeds,
)

sys.path.append("../deepmind-research")
sys.path.append("../distractors_benchmarks")
from envs.cartpole_distractors import cartpole
from wandb_logger import WandBLogger


def main(arguments, params, conf, logger):
    set_seeds(seed=arguments.seed)
    memory_growth()
    check_GPU()

    raw_environment = cartpole.Cartpole(
        observation_dim_for_spec=6 + arguments.n_distractors
    )
    environment = wrappers.SinglePrecisionWrapper(raw_environment)
    distractors = cartpole_dataset.VectorizedDistractors(
        arguments.n_distractors, arguments.distractors_type
    )

    dataset = cartpole_dataset.dataset(
        distractors.add_distractors, arguments.n_distractors
    )

    learner = get_learner(arguments, environment, dataset, logger, params)
    eval_actor = get_evaluation_actor(learner, arguments.algorithm, distractors)
    eval_loop = EnvironmentLoop(
        environment=environment,
        actor=eval_actor,
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
                f"cartpole_{arguments.n_distractors}_{arguments.distractors_type}",
                n_episodes=conf["evaluation_episodes"],
                current_step=i,
                learner=learner,
            )

    # Final evaluation
    evaluate(
        eval_loop,
        environment,
        conf,
        f"cartpole_{arguments.n_distractors}_{arguments.distractors_type}",
        n_episodes=conf["final_evaluation_episodes"],
        current_step="Final",
        learner=learner,
    )

    # Save final encoder for glass box analysis
    print("Checkpointing encoder")
    save_checkpoint(
        learner._network._encoder, os.environ.get("CHECKPOINT_PATH"), name="Encoder"
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
    parser.add_argument(
        "--n_distractors", type=int, help="Number of distractors", required=True
    )
    parser.add_argument(
        "--filter_episodes",
        type=int,
        help="Skip episodes with id lower than this integer",
        default=0,
    )
    parser.add_argument(
        "--distractors_type",
        type=str,
        help="Type of distractors",
        choices=["gaussian", "sine", "action-walk"],
        required=True,
    )
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

    args = parser.parse_known_args()[0]
    args.joint_training = (args.joint_training != "False") and (
        args.abstraction != "None"
    )
    args.freeze = args.freeze != "False"

    env_kwargs = {
        "n_distractors": args.n_distractors,
        "distractors_type": args.distractors_type,
        "seed": None,
    }

    wandb_params = {**vars(args), **conf["hyperparams"], **env_kwargs}
    conf.pop("hyperparams")  # ensure we use dynamic hyperparams when doing a sweep

    logger = WandBLogger(project="Low_D_Distractors_Offline_Benchmark")
    logger.start(wandb_params)

    # Run experiment
    print(wandb.config)
    print(conf)
    main(args, wandb.config, conf, logger)
