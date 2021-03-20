from typing import Dict

import tensorflow as tf
from acme.agents.tf import dqn


class DQNLearner(dqn.DQNLearner):
    """DQN Learner"""

    def __init__(self, **kwargs) -> None:
        """Initializes the learner."""
        self._wandb_logger = kwargs.pop("wandb")
        kwargs.pop("beta")
        super().__init__(**kwargs)

    def _step(self) -> Dict[str, tf.Tensor]:
        fetches = super()._step()
        fetches["loss/dqn"] = fetches["loss"]  # for consistency
        del fetches["loss"]

        # Log losses.
        self._wandb_logger.log(fetches)

        return fetches
