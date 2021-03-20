import argparse
import os
import pickle
import sys

import imageio
import numpy as np
import sonnet as snt
import tensorflow as tf
import tqdm
from acme import specs
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(".")
sys.path.append("../deepmind-research")

from networks import QNetworkWithEncoder
from rl_unplugged import atari
from utils import batch, check_GPU, encoding_spec, load_conf, print_vars, set_seeds


def generate_online_data(steps, environment, environment_spec, actor=None):
    # Generate lots of (RAM, observation) pairs
    observations = []
    ram = []

    timestep = environment.reset()
    for i in tqdm.tqdm(range(steps)):
        if actor and i > steps // 2:
            # Mixed policy: 50% actor, 50% random
            action = actor.select_action(timestep.observation)
        else:
            action = np.random.randint(
                low=0, high=environment_spec.actions.num_values - 1
            )
        timestep = environment.step(np.array(action))
        observations.append(timestep.observation)
        ram.append(environment._env.environment.unwrapped._get_ram())

    observations = np.array(observations)
    ram = np.array(ram)
    return observations, ram


def main(args):
    set_seeds()
    # check_GPU()

    if args.load_path:
        with open(args.load_path, "rb") as f:
            data = pickle.load(f)
        abstractions = data["abstractions"]
        ram = data["ram"]
    else:
        environment = atari.environment(game=args.game)
        environment_spec = specs.make_environment_spec(environment)
        # Load encoder
        checkpoint_root = os.environ.get("CHECKPOINT_PATH")
        save_prefix = os.path.join(checkpoint_root, args.checkpoint_name)
        encoder_network = QNetworkWithEncoder(  # TODO: update to softmax
            params["feature_dim"],
            environment_spec.actions.num_values,
            use_encoder=True,
        )._encoder
        checkpoint = tf.train.Checkpoint(module=encoder_network)
        checkpoint.restore(save_prefix)
        # Generate (obs, ram) pairs
        observations, ram = generate_online_data(
            args.n_steps, environment, environment_spec
        )
        # Get representations
        abstractions = encode(encoder_network, params, observations)
        # Save
        with open(args.save_path, "wb") as f:
            pickle.dump(
                {
                    "abstractions": abstractions,
                    "ram": ram,
                    "observations": observations,
                },
                f,
            )

    # Compute abstractions from observations and evaluate
    train_decoder(abstractions, ram)


# @tf.function
def encode(network, params, observations):
    abstractions = np.empty((0, params["feature_dim"]))
    print(observations)
    for b in tqdm(batch(observations, 256)):
        abstractions = np.append(abstractions, network(b), 0)
    return abstractions


def save_video(observations):
    """debugging"""

    # Write video
    with imageio.get_writer(
        "/home/hcr-118/atari_replays/Pong--debug.mp4", fps=60
    ) as video:
        for frame in observations:
            video.append_data(frame)


def train_decoder(abstractions, ram):
    X_train, X_test, y_train, y_test = train_test_split(
        abstractions, ram, test_size=0.2, random_state=42
    )
    decoder = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(params["feature_dim"],)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(128),
        ]
    )
    loss_fn = tf.keras.losses.MeanSquaredError(
        reduction="auto", name="mean_squared_error"
    )
    decoder.compile(
        optimizer="adam",
        loss=loss_fn,
        metrics=["mse"],
    )
    decoder.fit(
        X_train,
        y_train,
        epochs=5,
        validation_data=(
            X_test,
            y_test,
        ),
    )

    decoder.evaluate(X_test, y_test, verbose=2)


if __name__ == "__main__":
    conf = load_conf()
    params = conf["hyperparams"]
    # Parsing args
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name", type=str)
    parser.add_argument("--game", type=str)
    parser.add_argument("--load-path", nargs="?", type=str, default=None)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--n-steps", type=int)
    args = parser.parse_args()
    main(args)
