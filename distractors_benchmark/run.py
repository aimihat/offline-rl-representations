import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Before importing TF

from functools import partial
from multiprocessing.dummy import Pool
from pathlib import Path
from subprocess import call

from acme import EnvironmentLoop, specs, wrappers
from configs import AGENT_CONFIGS
from envs.cartpole_distractors import cartpole, sweep
from wandb_logger import WandBLogger
import tensorflow as tf

logger = WandBLogger(project="Low_D_Distractors_Benchmark")

# Enable memory growth
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def perform_sweep(args):
    """Run a full sweep over bsuite_ids/learners"""

    run_commands = []
    # Run every cartpole experiment in the benchmark
    for bsuite_id in sweep.SWEEP:
        # Evaluate all abstractions in `agents/config.py` w/ DQN
        for i, _ in enumerate(AGENT_CONFIGS):
            temp_path = (
                "_".join(map(str, sweep._SETTINGS[bsuite_id].values())) + f"_agent{i}"
            )
            run_commands.append(
                f'python3 run.py --bsuite_id="{bsuite_id}" --agent_id={i} > /tmp/{temp_path}.log'
            )

    pool = Pool(args.n_concurrent)
    for i, return_code in enumerate(pool.map(partial(call, shell=True), run_commands)):
        print(f"Command completed with return code {return_code}: `{run_commands[i]}`")


def run_experiment(args):
    """Evaluate a single bsuite_id/learner"""

    # Grab environment
    env_kwargs = sweep._SETTINGS[args.bsuite_id]
    raw_environment = cartpole.Cartpole(**env_kwargs)
    environment = wrappers.SinglePrecisionWrapper(raw_environment)
    environment_spec = specs.make_environment_spec(environment)
    # Grab agent
    agent = AGENT_CONFIGS[args.agent_id]
    # Initialize wandb logger
    logger.start(
        {
            "agent": agent._name,
            "bsuite_id": args.bsuite_id,
            **agent._parameters,
            **env_kwargs,
        },
    )
    # Run the environment loop.
    EnvironmentLoop(
        environment,
        agent.agent(environment_spec, logger=logger),
        logger=logger,
    ).run(num_episodes=sweep.NUM_EPISODES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsuite_id", type=str)
    parser.add_argument("--agent_id", type=int)
    parser.add_argument("--n_concurrent", type=int, default=2)
    args = parser.parse_args()

    if args.bsuite_id or args.agent_id:
        err = "You must pass both bsuite and agent ids for a single run."
        assert args.bsuite_id is not None and args.agent_id is not None, err
        run_experiment(args)
    else:
        perform_sweep(args)
