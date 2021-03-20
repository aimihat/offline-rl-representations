# Copyright 2020 MOA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Weights & Biases logger."""

import wandb
from acme.utils.loggers import base, terminal


def _format_key(key: str) -> str:
    """Internal function for formatting keys in Weight & Biases format."""
    return key.title().replace("_", "")


class WandBLogger(terminal.TerminalLogger):
    """Logs to a `wandb` dashboard created in a given `run_id`.

    If multiple `WandBLogger` are created with the same job/run id, results will be
    categorized by labels.
    """

    def __init__(self, project: str):
        """Initializes the logger.

        Args:
          label: label string to use when logging.
        """

        super().__init__(time_delta=1)
        self._label = "WandBLogger"
        self._project = project
        self._run = False

    def start(self, config):
        if self._run:
            self._run.finish()
        self._run = wandb.init(project=self._project, config=config, reinit=True)

    def write(self, values: base.LoggingData, commit=False) -> None:
        super().write(values)
        wandb.log(values, commit=commit)
