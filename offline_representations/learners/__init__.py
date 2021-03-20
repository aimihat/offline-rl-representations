"""RL learner algorithms with various abstractions"""

import copy

import tensorflow as tf
from acme import specs
from acme.agents.tf.dqn import learning as acme_dqn
from acme.tf import utils as acme_utils
from acme.tf.networks.continuous import LayerNormMLP
from networks import (
    Decoder,
    DeterministicTransitionModel,
    LinearProjection,
    QNetworkWithEncoder,
)
from utils import encoding_spec, print_vars

from .bc import (
    BCLearnerPlain,
    BCLearnerWithAE,
    BCLearnerWithContrastiveDBC,
    BCLearnerWithDBC,
    BCLearnerWithForwardLatent,
)
from .dqn import (
    DQNLearnerPlain,
    DQNLearnerWithAE,
    DQNLearnerWithContrastiveDBC,
    DQNLearnerWithDBC,
    DQNLearnerWithForwardLatent,
)


# TODO: Decide which combinations to implement
def get_learner(arguments, environment, dataset, logger, params):
    """Constructs an Acme learner, using a given RL method & abstraction."""

    environment_spec = specs.make_environment_spec(environment)
    use_encoder = arguments.abstraction != "None"

    dropout_rate = params["bc_dropout_rate"] if arguments.algorithm == "BC" else 0.0

    # Necessary as updating .encoder_frozen doesn't work in a tf function
    if True:
        print("DISABLING ALL TF.FUNCTION")
        tf.config.run_functions_eagerly(True)

    network = QNetworkWithEncoder(
        params["feature_dim"],
        environment_spec.actions.num_values,
        encoder_type=params["encoder_type"],
        hidden_dims=params["dqn_hidden_dims"],
        dropout_rate=dropout_rate,
    )

    # Abstraction-associated networks
    decoder_network = Decoder(
        encoder_type=params["encoder_type"],
        observation_dim=environment.observation_spec().shape[0],
    )  # AE
    projection_network = LinearProjection(
        projection_feature_dim=params["projection_feature_dim"]
    )  # DBC
    reward_decoder = LayerNormMLP(
        [params["reward_decoder_layer_width"], 1], activate_final=False
    )  # DeepMDP
    projection_decoder = DeterministicTransitionModel(
        encoder_feature_dim=params["projection_feature_dim"],
        layer_width=params["transition_model_layer_width"],
    )  # DeepMDP
    forward_decoder = DeterministicTransitionModel(
        encoder_feature_dim=params["feature_dim"],
        layer_width=params["transition_model_layer_width"],
    )  # DeepMDP

    create_vars(
        use_encoder,
        network,
        decoder_network,
        projection_network,
        reward_decoder,
        forward_decoder,
        projection_decoder,
        environment,
        environment_spec,
    )

    shared_params = dict(
        dataset=dataset,
        logger=logger,
        custom_kwargs=dict(
            **params,
            decoder_network=decoder_network,
            projection_network=projection_network,
            reward_decoder=reward_decoder,
            forward_decoder=forward_decoder,
            projection_decoder=projection_decoder
        ),
    )

    if arguments.algorithm == "DQN":
        learner_params = dict(
            network=network,
            target_network=copy.deepcopy(network),
            discount=params["discount_rate"],
            learning_rate=params["dqn_learning_rate"],
            importance_sampling_exponent=params["importance_sampling_exponent"],
            target_update_period=params["target_update_period"],
        )

        abstraction_map = {
            "None": DQNLearnerPlain,
            "AE": DQNLearnerWithAE,
            "DBC": DQNLearnerWithDBC,
            "Contrastive_DBC": DQNLearnerWithContrastiveDBC,
            "DeepMDP": DQNLearnerWithForwardLatent,
        }

    elif arguments.algorithm == "BC":
        learner_params = dict(network=network, learning_rate=params["bc_learning_rate"])

        abstraction_map = {
            "None": BCLearnerPlain,
            "AE": BCLearnerWithAE,
            "DBC": BCLearnerWithDBC,
            "Contrastive_DBC": BCLearnerWithContrastiveDBC,
            "DeepMDP": BCLearnerWithForwardLatent,
        }

    else:
        raise Exception()

    return abstraction_map[arguments.abstraction](**learner_params, **shared_params)


def create_vars(
    use_encoder,
    network,
    decoder_network,
    projection_network,
    reward_decoder,
    forward_decoder,
    projection_decoder,
    environment,
    environment_spec,
):
    """Create all relevant network variables."""
    acme_utils.create_variables(network, [environment.observation_spec()])
    if use_encoder:
        encoding_specification = encoding_spec(
            network, [environment.observation_spec()]
        )
        # encoding_and_action_spec = encoding_spec(network, [environment.observation_spec()], n_actions_concat=environment_spec.actions.num_values)

        acme_utils.create_variables(decoder_network, encoding_specification)
        acme_utils.create_variables(projection_network, encoding_specification)
        # acme_utils.create_variables(
        #     reward_decoder,
        #     encoding_and_action_spec
        # )
        # acme_utils.create_variables(
        #     forward_decoder,
        #     encoding_and_action_spec
        # )
