"""Sweep definition for cartpole_distractors experiment."""
from typing import Any, Dict, Mapping

import frozendict
from bsuite.experiments.cartpole import sweep as cartpole_sweep
from bsuite.sweep import _parse_sweep

NUM_EPISODES = 1000
SEPARATOR = "/"
_SWEEP = []
_SETTINGS = {}

_all_settings = []
for n_distractors in [0, 30, 100, 1000]:
    for distractors_type in ("gaussian", "sine", "action-walk"):
        for seed in range(1):
            _all_settings.append(
                {
                    "n_distractors": n_distractors,
                    "distractors_type": distractors_type,
                    "seed": seed,
                }
            )

# Construct bsuite_ids for each setting defined by the experiment.
for i, setting in enumerate(_all_settings):
    bsuite_id = f"cartpole_distractors/{i}"
    _SWEEP.append(bsuite_id)
    _SETTINGS[bsuite_id] = setting

SWEEP = tuple(_SWEEP)
SETTINGS: Mapping[str, Dict[str, Any]] = frozendict.frozendict(**_SETTINGS)
