"""Naive DQN with experience replay for HW3-1 static Gridworld.

Run from the project root:

    python -m hw3_1.naive_dqn_static
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, List, NamedTuple

import numpy as np
import torch
from torch import nn

from hw3_1.gridworld import StaticGridworld


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size replay memory that returns random minibatches."""

    def __init__(self, capacity: int) -> None:
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass
class DQNConfig:
    episodes: int = 1500
    max_moves: int = 50
    replay_capacity: int = 5000
    batch_size: int = 64
    gamma: float = 0.95
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 900
    seed: int = 7
    eval_episodes: int = 50
    output_dir: str = "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_by_episode(episode: int, cfg: DQNConfig) -> float:
    fraction = min(1.0, episode / max(1, cfg.epsilon_decay_episodes))
    return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(model: QNetwork, state: np.ndarray, epsilon: float, n_actions: int) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def optimize_model(
    model: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    cfg: DQNConfig,
) -> float | None:
    if len(replay) < cfg.batch_size:
        return None

    batch = replay.sample(cfg.batch_size)
    states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32)
    actions = torch.as_tensor([t.action for t in batch], dtype=torch.long).unsqueeze(1)
    rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
    next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32)
    dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32)

    current_q = model(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q = model(next_states).max(dim=1).values
        target_q = rewards + cfg.gamma * (1.0 - dones) * next_q

    loss = nn.functional.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def train(cfg: DQNConfig) -> tuple[QNetwork, dict[str, object]]:
    set_seed(cfg.seed)
    env = StaticGridworld()
    model = QNetwork(env.state_dim, env.n_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    replay = ReplayBuffer(cfg.replay_capacity)

    episode_rows: list[dict[str, float | int | bool]] = []
    recent_successes: Deque[bool] = deque(maxlen=100)
    losses: list[float] = []

    for episode in range(cfg.episodes):
        state = env.reset()
        epsilon = epsilon_by_episode(episode, cfg)
        total_reward = 0.0
        won = False

        for move in range(1, cfg.max_moves + 1):
            action = select_action(model, state, epsilon, env.n_actions)
            result = env.step(action)
            replay.push(Transition(state, action, result.reward, result.state, result.terminated))
            loss = optimize_model(model, optimizer, replay, cfg)
            if loss is not None:
                losses.append(loss)

            state = result.state
            total_reward += result.reward
            if result.terminated:
                won = result.reward > 0
                break

        recent_successes.append(won)
        episode_rows.append(
            {
                "episode": episode + 1,
                "epsilon": round(epsilon, 5),
                "moves": move,
                "return": total_reward,
                "won": won,
                "replay_size": len(replay),
                "loss": losses[-1] if losses else "",
                "rolling_success_100": sum(recent_successes) / len(recent_successes),
            }
        )

    eval_stats = evaluate(model, cfg.eval_episodes, cfg.max_moves)
    metrics: dict[str, object] = {
        "config": asdict(cfg),
        "training": {
            "episodes": cfg.episodes,
            "final_replay_size": len(replay),
            "final_rolling_success_100": episode_rows[-1]["rolling_success_100"],
            "updates": len(losses),
            "last_loss": losses[-1] if losses else None,
        },
        "evaluation": eval_stats,
    }

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_training_csv(output_dir / "hw3_1_static_training.csv", episode_rows)
    (output_dir / "hw3_1_static_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), output_dir / "hw3_1_static_dqn.pt")
    return model, metrics


def evaluate(model: QNetwork, episodes: int, max_moves: int) -> dict[str, object]:
    env = StaticGridworld()
    wins = 0
    returns: list[float] = []
    lengths: list[int] = []
    example_actions: list[str] = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        path_actions: list[str] = []
        won = False

        for move in range(1, max_moves + 1):
            action = select_action(model, state, epsilon=0.0, n_actions=env.n_actions)
            result = env.step(action)
            path_actions.append(env.action_names[action])
            state = result.state
            total_reward += result.reward
            if result.terminated:
                won = result.reward > 0
                break

        wins += int(won)
        returns.append(total_reward)
        lengths.append(move)
        if episode == 0:
            example_actions = path_actions

    return {
        "episodes": episodes,
        "wins": wins,
        "win_rate": wins / episodes,
        "average_return": float(np.mean(returns)),
        "average_length": float(np.mean(lengths)),
        "example_greedy_actions": example_actions,
    }


def save_training_csv(path: Path, rows: list[dict[str, float | int | bool]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> DQNConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DQNConfig.episodes)
    parser.add_argument("--eval-episodes", type=int, default=DQNConfig.eval_episodes)
    parser.add_argument("--seed", type=int, default=DQNConfig.seed)
    parser.add_argument("--output-dir", type=str, default=DQNConfig.output_dir)
    args = parser.parse_args()
    return DQNConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    _, metrics = train(cfg)
    print(json.dumps(metrics["evaluation"], indent=2))


if __name__ == "__main__":
    main()
