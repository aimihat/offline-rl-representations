from typing import Optional, Tuple

import sonnet as snt
import tensorflow as tf
from acme.tf import networks


class Encoder(snt.Module):
    def __init__(self, encoder_type: str, encoder_feature_dim: int) -> None:
        """Initializes an `encoder_type` encoder network.

        Args:
          encoder_type: One of {"mlp"}.
          encoder_feature_dim: Size of the output layer.
        """

        super().__init__()

        print(
            f"Creating {encoder_type} encoder with {encoder_feature_dim} output dimensions"
        )

        self._encoder_type = encoder_type
        self._encoder_feature_dim = encoder_feature_dim

        # TODO: increase encoder size + parametrize.
        if self._encoder_type == "mlp":
            self._encoder = snt.Sequential(
                [
                    snt.Flatten(),
                    networks.LayerNormMLP(
                        [128, self._encoder_feature_dim], activate_final=False
                    ),
                ]
            )
        elif self._encoder_type == "stochastic":
            self._encoder = snt.Sequential(
                [
                    snt.Flatten(),
                    networks.LayerNormMLP(
                        [128, self._encoder_feature_dim + self._encoder_feature_dim],
                        activate_final=False,
                    ),
                ]
            )
        else:
            raise NotImplementedError(self._encoder_type)

    def reparameterize(self, mean, logvar):
        """VAE parametrization trick."""

        if mean.shape[0] == None:  # when initializing snapshotter
            print("Reparametrizing with no batch dimension")
            return mean
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def __call__(self, x: tf.Tensor, reparametrize: bool = True) -> tf.Tensor:
        if self._encoder_type == "stochastic":
            mean, logvar = tf.split(self._encoder(x), num_or_size_splits=2, axis=1)
            if reparametrize:
                return self.reparameterize(
                    mean, logvar
                )  # TODO: DQN pass uses reparametrize?
            else:
                return mean, logvar
        else:
            return self._encoder(x)


class QNetworkWithEncoder(snt.Module):
    """A simple interface for deep Q-network with an `encode` function."""

    def __init__(
        self,
        action_dim: int,
        observation_dim: int,
        encoder_type: Optional[str],
        encoder_feature_dim: int,
        projection_feature_dim: int,
        name: Optional[str] = None,
    ) -> None:
        """Initialise a DQN with a `encoder_type` encoder.

        Args:
          encoder_type: One of {"mlp", None}.
          action_dim: The number of available actions (applicable only to discrete As).
          encoder_feature_dim: Size of the encoded latent space.
        """
        super().__init__(name=name)
        print(f"Creating QNetwork with {action_dim} actions")
        self._encoder_feature_dim = encoder_feature_dim

        if encoder_type:
            self._encoder = Encoder(encoder_type, encoder_feature_dim)
            self._projector = snt.Linear(projection_feature_dim, name="projector")
        else:
            self._encoder = snt.Flatten()

        self._decoder = snt.nets.MLP(
            [128, observation_dim], activate_final=False, name="decoder"
        )

        self._mlp = snt.nets.MLP([128, 64, action_dim], activate_final=False)

    def encode(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return the representation/abstraction for `inputs`."""
        return self._encoder(inputs)

    def decode(self, encoding: tf.Tensor) -> tf.Tensor:
        """Returns the reconstructed observation."""
        return self._decoder(encoding)

    def latent_state(self, inputs: tf.Tensor) -> tf.Tensor:
        """Returns a projection of the encoded `inputs`."""
        return self._projector(self.encode(inputs))

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return the action-values for state `inputs`."""
        return self._mlp(self.encode(inputs))


class DeterministicTransitionModel(snt.Module):
    """A deterministic state transition model."""

    def __init__(self, encoder_feature_dim: int, layer_width: int) -> None:
        super(DeterministicTransitionModel, self).__init__()
        self.fc = snt.Linear(layer_width)
        self.ln = snt.LayerNorm(0, True, True)
        self.fc_mu = snt.Linear(encoder_feature_dim)
        print("Deterministic transition model chosen.")

    def __call__(self, x: tf.Tensor) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        x = self.fc(x)
        x = self.ln(x)
        x = tf.nn.relu(x)

        mu = self.fc_mu(x)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x: tf.Tensor) -> tf.Tensor:
        mu, sigma = self(x)
        return mu
