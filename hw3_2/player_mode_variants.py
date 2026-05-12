"""Compare Basic DQN, Double DQN, and Dueling DQN on player-mode Gridworld.

Run from the project root:

    python -m hw3_2.player_mode_variants
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Iterable, NamedTuple

import numpy as np
import torch
from torch import nn

from hw3_1.gridworld import PlayerGridworld, Position


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DuelingQNetwork(nn.Module):
    """Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feature = self.feature(state)
        value = self.value_stream(feature)
        advantage = self.advantage_stream(feature)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


@dataclass
class VariantConfig:
    episodes: int = 2200
    max_moves: int = 40
    replay_capacity: int = 8000
    batch_size: int = 64
    gamma: float = 0.95
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1400
    target_sync_freq: int = 100
    seed: int = 11
    eval_repeats: int = 10
    output_dir: str = "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_by_episode(episode: int, cfg: VariantConfig) -> float:
    fraction = min(1.0, episode / max(1, cfg.epsilon_decay_episodes))
    return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(model: nn.Module, state: np.ndarray, epsilon: float, n_actions: int) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        return int(torch.argmax(model(state_tensor), dim=1).item())


def make_model(variant: str, state_dim: int, n_actions: int) -> nn.Module:
    if variant == "dueling_dqn":
        return DuelingQNetwork(state_dim, n_actions)
    return QNetwork(state_dim, n_actions)


def compute_targets(
    variant: str,
    online_model: nn.Module,
    target_model: nn.Module,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    with torch.no_grad():
        if variant == "basic_dqn":
            next_q = online_model(next_states).max(dim=1).values
        elif variant == "double_dqn":
            next_actions = online_model(next_states).argmax(dim=1, keepdim=True)
            next_q = target_model(next_states).gather(1, next_actions).squeeze(1)
        elif variant == "dueling_dqn":
            next_q = target_model(next_states).max(dim=1).values
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        return rewards + gamma * (1.0 - dones) * next_q


def optimize_model(
    variant: str,
    online_model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    cfg: VariantConfig,
) -> float | None:
    if len(replay) < cfg.batch_size:
        return None

    batch = replay.sample(cfg.batch_size)
    states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32)
    actions = torch.as_tensor([t.action for t in batch], dtype=torch.long).unsqueeze(1)
    rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
    next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32)
    dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32)

    current_q = online_model(states).gather(1, actions).squeeze(1)
    target_q = compute_targets(
        variant,
        online_model,
        target_model,
        rewards,
        next_states,
        dones,
        cfg.gamma,
    )
    loss = nn.functional.mse_loss(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_model.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def train_variant(variant: str, cfg: VariantConfig) -> tuple[nn.Module, dict[str, object], list[dict[str, object]]]:
    set_seed(cfg.seed)
    env = PlayerGridworld(seed=cfg.seed)
    online_model = make_model(variant, env.state_dim, env.n_actions)
    target_model = make_model(variant, env.state_dim, env.n_actions)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(online_model.parameters(), lr=cfg.learning_rate)
    replay = ReplayBuffer(cfg.replay_capacity)

    episode_rows: list[dict[str, object]] = []
    recent_successes: Deque[bool] = deque(maxlen=100)
    losses: list[float] = []
    update_steps = 0
    first_solved_episode: int | None = None

    for episode in range(cfg.episodes):
        state = env.reset()
        epsilon = epsilon_by_episode(episode, cfg)
        total_reward = 0.0
        won = False

        for move in range(1, cfg.max_moves + 1):
            action = select_action(online_model, state, epsilon, env.n_actions)
            result = env.step(action)
            replay.push(Transition(state, action, result.reward, result.state, result.terminated))

            loss = optimize_model(variant, online_model, target_model, optimizer, replay, cfg)
            if loss is not None:
                losses.append(loss)
                update_steps += 1
                if variant != "basic_dqn" and update_steps % cfg.target_sync_freq == 0:
                    target_model.load_state_dict(online_model.state_dict())

            state = result.state
            total_reward += result.reward
            if result.terminated:
                won = result.reward > 0
                break

        recent_successes.append(won)
        rolling_success = sum(recent_successes) / len(recent_successes)
        if first_solved_episode is None and len(recent_successes) == 100 and rolling_success >= 0.90:
            first_solved_episode = episode + 1

        episode_rows.append(
            {
                "variant": variant,
                "episode": episode + 1,
                "epsilon": round(epsilon, 5),
                "moves": move,
                "return": total_reward,
                "won": won,
                "replay_size": len(replay),
                "loss": losses[-1] if losses else "",
                "rolling_success_100": rolling_success,
            }
        )

    eval_stats = evaluate_all_starts(online_model, cfg.eval_repeats, cfg.max_moves)
    metrics: dict[str, object] = {
        "variant": variant,
        "config": asdict(cfg),
        "training": {
            "episodes": cfg.episodes,
            "updates": update_steps,
            "final_replay_size": len(replay),
            "final_rolling_success_100": episode_rows[-1]["rolling_success_100"],
            "first_episode_rolling_success_ge_90": first_solved_episode,
            "last_loss": losses[-1] if losses else None,
        },
        "evaluation": eval_stats,
    }
    return online_model, metrics, episode_rows


def evaluate_all_starts(model: nn.Module, repeats: int, max_moves: int) -> dict[str, object]:
    env = PlayerGridworld(seed=0)
    starts = env.valid_start_positions()
    rows: list[dict[str, object]] = []
    wins = 0
    returns: list[float] = []
    lengths: list[int] = []
    example_paths: dict[str, list[str]] = {}

    for repeat in range(repeats):
        for start in starts:
            state = env.reset(start_pos=start)
            total_reward = 0.0
            won = False
            actions: list[str] = []

            for move in range(1, max_moves + 1):
                action = select_action(model, state, epsilon=0.0, n_actions=env.n_actions)
                result = env.step(action)
                actions.append(env.action_names[action])
                state = result.state
                total_reward += result.reward
                if result.terminated:
                    won = result.reward > 0
                    break

            wins += int(won)
            returns.append(total_reward)
            lengths.append(move)
            rows.append(
                {
                    "repeat": repeat + 1,
                    "start": str(start),
                    "won": won,
                    "return": total_reward,
                    "moves": move,
                    "actions": " ".join(actions),
                }
            )
            if repeat == 0:
                example_paths[str(start)] = actions

    total = len(rows)
    return {
        "episodes": total,
        "starts": [str(start) for start in starts],
        "wins": wins,
        "win_rate": wins / total,
        "average_return": float(np.mean(returns)),
        "average_length": float(np.mean(lengths)),
        "per_start_rows": rows,
        "example_paths_first_repeat": example_paths,
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compare_variants(cfg: VariantConfig, variants: list[str]) -> dict[str, object]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, object]] = []
    all_training_rows: list[dict[str, object]] = []

    for variant in variants:
        variant_cfg = VariantConfig(**asdict(cfg))
        model, metrics, training_rows = train_variant(variant, variant_cfg)
        all_metrics.append(metrics)
        all_training_rows.extend(training_rows)

        torch.save(model.state_dict(), output_dir / f"hw3_2_{variant}.pt")
        (output_dir / f"hw3_2_{variant}_metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        write_csv(output_dir / f"hw3_2_{variant}_training.csv", training_rows)
        write_csv(output_dir / f"hw3_2_{variant}_eval_by_start.csv", metrics["evaluation"]["per_start_rows"])  # type: ignore[index]

    summary_rows = [
        {
            "variant": metrics["variant"],
            "train_final_success_100": metrics["training"]["final_rolling_success_100"],  # type: ignore[index]
            "first_episode_rolling_success_ge_90": metrics["training"]["first_episode_rolling_success_ge_90"],  # type: ignore[index]
            "eval_win_rate": metrics["evaluation"]["win_rate"],  # type: ignore[index]
            "eval_average_return": metrics["evaluation"]["average_return"],  # type: ignore[index]
            "eval_average_length": metrics["evaluation"]["average_length"],  # type: ignore[index]
            "updates": metrics["training"]["updates"],  # type: ignore[index]
        }
        for metrics in all_metrics
    ]
    write_csv(output_dir / "hw3_2_comparison_summary.csv", summary_rows)
    write_csv(output_dir / "hw3_2_all_training.csv", all_training_rows)

    comparison = {"config": asdict(cfg), "summary": summary_rows, "metrics": all_metrics}
    (output_dir / "hw3_2_comparison_metrics.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    return comparison


def parse_args() -> tuple[VariantConfig, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=VariantConfig.episodes)
    parser.add_argument("--eval-repeats", type=int, default=VariantConfig.eval_repeats)
    parser.add_argument("--seed", type=int, default=VariantConfig.seed)
    parser.add_argument("--output-dir", type=str, default=VariantConfig.output_dir)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["basic_dqn", "double_dqn", "dueling_dqn"],
        choices=["basic_dqn", "double_dqn", "dueling_dqn"],
    )
    args = parser.parse_args()
    cfg = VariantConfig(
        episodes=args.episodes,
        eval_repeats=args.eval_repeats,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    return cfg, args.variants


def main() -> None:
    cfg, variants = parse_args()
    comparison = compare_variants(cfg, variants)
    print(json.dumps(comparison["summary"], indent=2))


if __name__ == "__main__":
    main()
