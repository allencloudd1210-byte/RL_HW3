"""Quick environment smoke test for HW3 DQN work."""

from __future__ import annotations

import importlib
import sys


def version_of(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def main() -> None:
    import gymnasium as gym
    import numpy as np
    import torch

    print(f"python: {sys.version.split()[0]}")
    print(f"numpy: {np.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"gymnasium: {version_of('gymnasium')}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=0)
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    env.close()

    assert obs.shape == next_obs.shape
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    q = torch.nn.Sequential(
        torch.nn.Linear(obs.shape[0], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, env.action_space.n),
    )
    with torch.no_grad():
        q_values = q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))

    assert q_values.shape == (1, env.action_space.n)
    print("environment smoke test: ok")


if __name__ == "__main__":
    main()
