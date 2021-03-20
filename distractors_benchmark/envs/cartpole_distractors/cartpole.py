"""The Cartpole reinforcement learning environment."""

import collections

import dm_env
import numpy as np
from bsuite.environments import base
from dm_env import specs

from . import sweep

CartpoleState = collections.namedtuple(
    "CartpoleState", ["x", "x_dot", "theta", "theta_dot", "time_elapsed"]
)

CartpoleConfig = collections.namedtuple(
    "CartpoleConfig", ["mass_cart", "mass_pole", "length", "force_mag", "gravity"]
)


def step_cartpole(
    action: int, timescale: float, state: CartpoleState, config: CartpoleConfig
) -> CartpoleState:
    """Helper function to step cartpole state under given config."""
    # Unpack variables into "short" names for mathematical equation
    force = (action - 1) * config.force_mag
    cos = np.cos(state.theta)
    sin = np.sin(state.theta)
    pl = config.mass_pole * config.length
    l = config.length
    m_pole = config.mass_pole
    m_total = config.mass_cart + config.mass_pole
    g = config.gravity

    # Compute the physical evolution
    temp = (force + pl * state.theta_dot**2 * sin) / m_total
    theta_acc = (g * sin - cos * temp) / (l * (4 / 3 - m_pole * cos**2 / m_total))
    x_acc = temp - pl * theta_acc * cos / m_total

    # Update states according to discrete dynamics
    x = state.x + timescale * state.x_dot
    x_dot = state.x_dot + timescale * x_acc
    theta = np.remainder(state.theta + timescale * state.theta_dot, 2 * np.pi)
    theta_dot = state.theta_dot + timescale * theta_acc
    time_elapsed = state.time_elapsed + timescale

    return CartpoleState(x, x_dot, theta, theta_dot, time_elapsed)


class Cartpole(base.Environment):
    """This implements a version of the classic Cart Pole task.

    For more information see:
    https://webdocs.cs.ualberta.ca/~sutton/papers/barto-sutton-anderson-83.pdf
    The observation is a vector representing:
      `(x, x_dot, sin(theta), cos(theta), theta_dot, time_elapsed)`

    The actions are discrete ['left', 'stay', 'right']. Episodes start with the
    pole close to upright. Episodes end when the pole falls, the cart falls off
    the table, or the max_time is reached.
    """

    def __init__(
        self,
        height_threshold: float = 0.8,
        x_threshold: float = 3.0,
        timescale: float = 0.01,
        max_time: float = 10.0,
        init_range: float = 0.05,
        seed: int = None,
        n_distractors: int = 0,
        distractors_type: str = None,
        observation_dim_for_spec: int = 6,
    ):
        # Setup.
        self._state = CartpoleState(0, 0, 0, 0, 0)
        self._observation_dim_for_spec = observation_dim_for_spec
        super().__init__()
        self._rng = np.random.RandomState(seed)
        self._init_fn = lambda: self._rng.uniform(low=-init_range, high=init_range)

        # Logging info
        self._raw_return = 0.0
        self._best_episode = 0.0
        self._episode_return = 0.0

        # Reward/episode logic
        self._height_threshold = height_threshold
        self._x_threshold = x_threshold
        self._timescale = timescale
        self._max_time = max_time

        # Problem config
        self._cartpole_config = CartpoleConfig(
            mass_cart=1.0,
            mass_pole=0.1,
            length=0.5,
            force_mag=10.0,
            gravity=9.8,
        )

        # Distractors config and initialization
        self._n_distractors = n_distractors
        self._distractors_type = distractors_type

        if self._n_distractors:
            # Ensure valid distractor-type
            assert self._distractors_type in (
                "gaussian",
                "sine",
                "action-walk",
            ), f"You have {self._n_distractors} distractors but `_distractors_type` is not a valid option."
            if self._distractors_type == "sine":
                self._distractor_phases = init_phases(self._n_distractors)
            elif self._distractors_type == "action-walk":
                self._distractor_starting, self._distractor_step = init_walk_params(
                    self._n_distractors
                )
                self._distractor_state = np.copy(self._distractor_starting)

        # Public attributes.
        self.bsuite_num_episodes = sweep.NUM_EPISODES

    # Overrides the super method.
    def reset(self):
        self._reset_next_step = False
        self._state = CartpoleState(
            x=self._init_fn(),
            x_dot=self._init_fn(),
            theta=self._init_fn(),
            theta_dot=self._init_fn(),
            time_elapsed=0.0,
        )
        self._episode_return = 0

        # Distractors re-initialization
        if self._n_distractors:
            if self._distractors_type == "sine":
                self._distractor_phases = init_phases(self._n_distractors)
            elif self._distractors_type == "action-walk":
                self._distractor_state = np.copy(self._distractor_starting)

        return dm_env.restart(self.observation)

    # Overrides the super method (we implement special auto-reset behavior here).
    def step(self, action):
        if self._reset_next_step:
            return self.reset()

        self._state = step_cartpole(
            action=action,
            timescale=self._timescale,
            state=self._state,
            config=self._cartpole_config,
        )

        # Update action-dependent distractions
        if self._n_distractors:
            if self._distractors_type == "action-walk":
                self._distractor_state += (action - 1) * self._distractor_step
                self._distractor_state = (
                    np.mod(self._distractor_state + 2, 4) - 2
                )  # cycle around -2, 2

        # Rewards only when the pole is central and balanced
        is_reward = (
            np.cos(self._state.theta) > self._height_threshold
            and np.abs(self._state.x) < self._x_threshold
        )
        reward = 1.0 if is_reward else 0.0
        self._raw_return += reward
        self._episode_return += reward

        if self._state.time_elapsed > self._max_time or not is_reward:
            self._best_episode = max(self._episode_return, self._best_episode)
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=self.observation)
        return dm_env.transition(reward=reward, observation=self.observation)

    def _step(self, action: int) -> dm_env.TimeStep:
        raise NotImplementedError("This environment implements its own auto-reset.")

    def _reset(self) -> dm_env.TimeStep:
        raise NotImplementedError("This environment implements its own auto-reset.")

    def action_spec(self):
        return specs.DiscreteArray(dtype=np.int, num_values=3, name="action")

    def observation_spec(self):
        return specs.Array(
            shape=(self._observation_dim_for_spec + self._n_distractors,),
            dtype=np.float32,
            name="state",
        )

    @property
    def observation(self) -> np.ndarray:
        """Approximately normalize output."""

        # Append `n_distractors` random variables to output
        obs = np.zeros((6 + self._n_distractors,), dtype=np.float32)

        obs[0] = self._state.x / self._x_threshold
        obs[1] = self._state.x_dot / self._x_threshold
        obs[2] = np.sin(self._state.theta)
        obs[3] = np.cos(self._state.theta)
        obs[4] = self._state.theta_dot
        obs[5] = self._state.time_elapsed / self._max_time

        # Generate gaussian noise for distractors
        if self._n_distractors:
            if self._distractors_type == "gaussian":
                obs[6 : 6 + self._n_distractors + 1] = np.random.normal(
                    0, 1, self._n_distractors
                )
            elif self._distractors_type == "sine":
                # Each distractor is a discretized sinusoid, with a phase that is initialized on .reset()
                # TODO: variable frequency
                obs[6 : 6 + self._n_distractors + 1] = np.sin(
                    2 * self._state.time_elapsed + self._distractor_phases
                )
            elif self._distractors_type == "action-walk":
                # Distractor `i` is a action-based walk: distractor[i] += step * action (action = 0, 1, 2)
                obs[6 : 6 + self._n_distractors + 1] = self._distractor_state

        return obs

    def bsuite_info(self):
        return dict(raw_return=self._raw_return, best_episode=self._best_episode)


################ Distractor Helpers ################


def init_phases(n):
    """Returns `n` starting sine phases in the range [0, 2pi]"""
    return np.random.uniform(low=0, high=2 * np.pi, size=(n,))


def init_walk_params(n):
    """Returns `n` starting values and step sizes for the action-walk distrators"""
    distractor_starting = np.random.uniform(low=-2, high=2, size=(n,))
    distractor_step = np.random.uniform(low=-0.1, high=0.1, size=(n,))
    return distractor_starting, distractor_step
