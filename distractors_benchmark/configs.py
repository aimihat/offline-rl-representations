import wandb
from agent import DQNAgent
from learners.aenn_learner import DQNLearnerWithAE
from learners.dbc_learner import DQNLearnerWithDBC
from learners.dbc_learner_contrastive import DQNLearnerWithContrastive
from learners.deepmdp_learner import DQNLearnerWithForwardLatent
from learners.dqn_learner import DQNLearner
from learners.vae_learner import DQNLearnerWithVAE


class AgentConf:
    def __init__(self, agent, learner, parameters={}):
        self._name = f"{agent.__name__}/{learner.__name__}"
        self._agent, self._learner = agent, learner
        self._parameters = parameters

    def agent(self, *args, **kwargs):
        # Initialize agent with learner and parameters from config.
        return self._agent(self._learner, wandb, *args, **kwargs, **self._parameters)


########## Define all agent configs to run. ##########

AGENT_CONFIGS = []

AGENT_CONFIGS.append(
    AgentConf(
        agent=DQNAgent,
        learner=DQNLearnerWithForwardLatent,
        parameters={
            "encoder_type": "mlp",
            "learning_rate": 1e-3,
            "encoder_feature_dim": 64,
            "projection_feature_dim": 32,
            "batch_size": 512,
        },
    )
)
AGENT_CONFIGS.append(
    AgentConf(
        agent=DQNAgent,
        learner=DQNLearnerWithVAE,
        parameters={
            "encoder_type": "stochastic",
            "learning_rate": 1e-3,
            "encoder_feature_dim": 64,
            "projection_feature_dim": 32,
            "batch_size": 512,
            "beta": 1,
        },
    )
)


AGENT_CONFIGS.append(
    AgentConf(
        agent=DQNAgent,
        learner=DQNLearnerWithContrastive,
        parameters={
            "encoder_type": "mlp",
            "learning_rate": 1e-3,
            "encoder_feature_dim": 64,
            "projection_feature_dim": 32,
            "batch_size": 512,
        },
    )
)

AGENT_CONFIGS.append(
    AgentConf(
        agent=DQNAgent,
        learner=DQNLearnerWithAE,
        parameters={
            "encoder_type": "mlp",
            "learning_rate": 1e-3,
            "encoder_feature_dim": 64,
        },
    )
)

AGENT_CONFIGS.append(
    AgentConf(
        agent=DQNAgent,
        learner=DQNLearnerWithDBC,
        parameters={
            "encoder_type": "mlp",
            "learning_rate": 1e-3,
            "encoder_feature_dim": 64,
            "projection_feature_dim": 32,
        },
    )
)

AGENT_CONFIGS.append(
    AgentConf(
        agent=DQNAgent,
        learner=DQNLearner,
    )
)
