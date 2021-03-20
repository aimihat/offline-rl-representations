from typing import Optional, Tuple

import sonnet as snt
import tensorflow as tf
from acme.tf import networks
from acme.tf.networks.continuous import LayerNormMLP


class PixelEncoder(snt.Module):
    def __init__(self, feature_dim) -> None:
        super().__init__()

        self._encoder = snt.Sequential(
            [
                lambda x: tf.image.convert_image_dtype(x, tf.float32),
                networks.AtariTorso(),
                snt.Linear(output_size=feature_dim),
                snt.LayerNorm(
                    axis=-1, create_scale=False, create_offset=False
                ),  # TODO: Why no relu here? + this is correct?
            ]
        )

    def __call__(self, x: tf.Tensor, is_training=None) -> tf.Tensor:
        return self._encoder(x)


class LowDimEncoder(snt.Module):
    def __init__(self, encoder_feature_dim: int, dropout_rate: float) -> None:
        """Initializes an `encoder_type` encoder network."""

        super().__init__()

        print(f"Creating encoder with {encoder_feature_dim} output dimensions")
        # TODO: Add Layernorm Back?
        self._flatten = snt.Flatten()
        self._encoder = snt.nets.MLP(
            [128, encoder_feature_dim], activate_final=False, dropout_rate=dropout_rate
        )

    def __call__(self, x: tf.Tensor, is_training: bool = True) -> tf.Tensor:
        return self._encoder(self._flatten(x), is_training=is_training)


class LinearProjection(snt.Module):
    def __init__(self, projection_feature_dim) -> None:
        print(f"Initializing projection with {projection_feature_dim} dimensions")
        super().__init__()

        self._projector = snt.nets.MLP([projection_feature_dim])

    def __call__(self, encoding: tf.Tensor) -> tf.Tensor:
        """Returns a projection of the encoded inputs."""
        return self._projector(encoding)


class Decoder(snt.Module):
    def __init__(
        self,
        encoder_type,
        observation_dim=None,
        torso_output_size=11,
        torso_output_n_filters=64,
    ) -> None:
        super().__init__()

        if encoder_type == "Pixel":
            self._decoder = snt.Sequential(
                [
                    snt.Linear(
                        output_size=torso_output_size
                        * torso_output_size
                        * torso_output_n_filters
                    ),
                    lambda x: tf.reshape(
                        x,
                        [
                            -1,
                            torso_output_size,
                            torso_output_size,
                            torso_output_n_filters,
                        ],
                    ),
                    snt.Conv2DTranspose(64, [3, 3], stride=[1, 1]),
                    tf.nn.relu,
                    snt.Conv2DTranspose(
                        64, [4, 4], stride=[2, 2], output_shape=(21, 21)
                    ),
                    tf.nn.relu,
                    snt.Conv2DTranspose(32, [8, 8], stride=[4, 4]),
                    tf.nn.relu,
                    snt.Conv2D(4, [3, 3]),
                    tf.nn.sigmoid,
                ]
            )
        elif encoder_type == "LowDim":
            assert observation_dim is not None
            self._decoder = snt.nets.MLP(
                [128, observation_dim], activate_final=False, name="decoder"
            )

    def __call__(self, encoding: tf.Tensor) -> tf.Tensor:
        """Returns a decoded reconstruction of the inputs."""
        return self._decoder(encoding)


class QNetworkWithEncoder(snt.Module):
    """A simple interface for deep Q-network with an `encode` function."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        encoder_type: str = "Pixel",
        hidden_dims: int = None,
        name: Optional[str] = None,
        dropout_rate=0,
    ) -> None:
        """Initialise a DQN with a `encoder_type` encoder."""
        super().__init__(name=name)
        print(
            f"Initializing QNetwork with {action_dim} actions, {feature_dim} output dims and {dropout_rate} dropout"
        )
        self.dropout_rate = dropout_rate
        self.encoder_frozen = False

        if encoder_type == "Pixel":
            self._encoder = PixelEncoder(feature_dim)
        elif encoder_type == "LowDim":
            self._encoder = LowDimEncoder(feature_dim, dropout_rate=dropout_rate)
        else:
            raise Exception()

        self._mlp = snt.nets.MLP([*hidden_dims, action_dim], dropout_rate=dropout_rate)

    def encode(self, inputs: tf.Tensor, is_training: bool = True) -> tf.Tensor:
        """Return the representation/abstraction for `inputs`."""
        if self.dropout_rate == 0:
            is_training = None
        if self.encoder_frozen:
            return tf.stop_gradient(self._encoder(inputs, is_training=is_training))
        else:
            return self._encoder(inputs, is_training=is_training)

    def __call__(self, inputs: tf.Tensor, is_training: bool = True) -> tf.Tensor:
        """Return the action-values for state `inputs`."""
        if self.dropout_rate == 0:
            is_training = None

        return self._mlp(
            self.encode(inputs, is_training=is_training), is_training=is_training
        )


class DeterministicTransitionModel(snt.Module):
    """A deterministic state transition model."""

    def __init__(self, encoder_feature_dim: int, layer_width: int) -> None:
        super(DeterministicTransitionModel, self).__init__()
        self.layernorm_mlp = LayerNormMLP(
            [layer_width, encoder_feature_dim], activate_final=False
        )
        print("Deterministic transition model chosen.")

    def __call__(self, x: tf.Tensor) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        mu = self.layernorm_mlp(x)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x: tf.Tensor) -> tf.Tensor:
        mu, _ = self(x)
        return mu
