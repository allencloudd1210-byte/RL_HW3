# HW3 Environment Setup

This workspace keeps the DRL in Action repository as a reference under:

```powershell
reference\DeepReinforcementLearningInAction
```

The instructor-provided updated starter code should still be used as the implementation baseline when it is available. Keep the reference repo separate and copy only the ideas or required snippets into the starter-code files.

## PowerShell Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\verify_env.py
```

## Jupyter Kernel

```powershell
python -m ipykernel install --user --name rl-hw3 --display-name "Python (RL HW3)"
```

## Optional Atari Dependencies

Install these only if the assignment or starter code requires Atari/image observations:

```powershell
python -m pip install -r requirements-atari.txt
```

## Reference Notes

- Chapter 3 in the reference repo covers basic Deep Q-learning.
- Later chapters contain DQN extensions and other DRL algorithms.
- The root `requirements.txt` here is intentionally smaller than the reference repo requirements so the HW3 environment stays focused and easier to install on Python 3.12.
