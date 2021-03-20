

## Installation

```shell
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

## Environment

1. Run `export ATARI_DATA_PATH=/path/to/atari/data/[GAME]`
1. Run `export ATARI_REPLAYS_PATH=/path/to/atari/replays`


## Downloading offline data

1. Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install)
2. `gsutil -m cp gs://rl_unplugged/atari/[GAME]/run_* /path/to/atari/data/[GAME]`


## Run Experiment

1. Define training params in `configs.py`.
2. Define environment config sweep (e.g. `n_distractors`, `distractor_type`, etc.) in `envs/cartpole_distractors/sweep.py`.
3. Login to W&B with `wandb login`.


## Sync renders to Google Drive

1. Add API access on Google Account.
2. Run `python google_drive.py`
3. Complete *OAuth2*