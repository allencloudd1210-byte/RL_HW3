# RL HW3: DQN and Its Variants

This repository contains the implementation and reports for HW3: DQN and its variants.

## Contents

- `hw3_1/`: Naive DQN for static mode with experience replay.
- `hw3_2/`: Basic DQN, Double DQN, and Dueling DQN comparison for player mode.
- `hw3_3/`: PyTorch Lightning enhanced DQN for random mode.
- `hw3_4/`: Bonus Rainbow DQN for random mode.
- `reports/`: Individual reports plus the combined report PDF/TXT.
- `outputs/`: Training metrics and evaluation CSV/JSON outputs.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\verify_env.py
```

## Main Report

- `reports/HW3_combined_report.txt`
- `reports/HW3_combined_report.pdf`

## Reproduce Runs

```powershell
python -m hw3_1.naive_dqn_static --episodes 1500 --eval-episodes 50 --seed 7
python -m hw3_2.player_mode_variants --episodes 2200 --eval-repeats 10 --seed 11
python -m hw3_3.lightning_random_dqn --episodes 5000 --eval-episodes 300 --seed 23
python -m hw3_4.rainbow_random_dqn --episodes 7000 --eval-episodes 300 --seed 31
```
