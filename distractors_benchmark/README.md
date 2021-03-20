## Installation

```shell
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

## Run Experiments

1. Define learners and their params in `configs.py`.
2. Define environment config sweep (e.g. `n_distractors`, `distractor_type`, etc.) in `envs/cartpole_distractors/sweep.py`.
3. Login to W&B with `wandb login`.

#### Perform a full-sweep
```shell
python run.py --n_concurrent=`nproc --all`
```

#### Single Experiment
```shell
python run.py --bsuite_id="cartpole_distractors/0" --agent_id=0
```