# EFFICIENT VECTOR SIMILARITY SEARCH FOR AI SYSTEMS

Compare HNSW, IVF, and NSG on QQP, CIFAR-10, and UCR.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

- QQP: `Dataset/questions.csv` with `question1`, `question2`.
- CIFAR-10: auto-downloaded by torchvision.
- UCR: put files in `Dataset/ucr_cache/` as:
  - `<NAME>_TRAIN.txt`
  - `<NAME>_TEST.txt`

As the dataset size is huge, it is uploaded in google drive and link is provided below:

https://drive.google.com/drive/folders/1aWqKCUFeBkqJWp14cUCAz5EPjhCkourn?usp=drive_link

## Run

```bash
python run_experiments.py
```

Useful options:

```bash
python run_experiments.py --quick
python run_experiments.py --datasets UCR_TimeSeries --ucr-name GunPoint --overwrite
python run_experiments.py --datasets QQP CIFAR10 --qqp-max-unique 10000 --cifar-max-train 10000 --overwrite
python run_experiments.py --scalability
```

## Plots and Report

```bash
python generate_plots.py
python generate_report.py
```

## Outputs

Per dataset (`Output/<QQP|CIFAR10|UCR_TimeSeries>/`):
- `comparison.csv`
- `hnsw_sweep.csv`
- `ivf_sweep.csv`
- `nsg_sweep.csv`
- `best_configs.csv`

Cross-dataset (`Output/`):
- `all_datasets_comparison.csv`
- `final_summary.csv`
- `final_report.md`
- `plots/*.png`
