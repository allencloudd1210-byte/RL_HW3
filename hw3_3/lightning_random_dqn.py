"""PyTorch Lightning DQN for random-mode Gridworld with training tips.

Run from the project root:

    python -m hw3_3.lightning_random_dqn
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

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from hw3_1.gridworld import RandomGridworld


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


class EpisodeDataset(IterableDataset[int]):
    """Dummy iterable that lets Lightning drive one RL episode per batch."""

    def __init__(self, episodes: int) -> None:
        self.episodes = episodes

    def __iter__(self) -> Iterable[int]:
        yield from range(self.episodes)

    def __len__(self) -> int:
        return self.episodes


class ConvolutionalDuelingQNetwork(nn.Module):
    """Small convolutional dueling network for the 4x4x4 Gridworld state."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if state_dim != 64:
            raise ValueError(f"Expected 64-dimensional flattened Gridworld state, got {state_dim}.")
        self.encoder = nn.Sequential(
            nn.Unflatten(1, (4, 4, 4)),
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_dim),
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
        feature = self.encoder(state)
        value = self.value_stream(feature)
        advantage = self.advantage_stream(feature)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


@dataclass
class LightningDQNConfig:
    episodes: int = 5000
    max_moves: int = 45
    replay_capacity: int = 20000
    warmup_steps: int = 512
    batch_size: int = 96
    gamma: float = 0.95
    learning_rate: float = 8e-4
    min_learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 3500
    target_sync_freq: int = 250
    gradient_clip_val: float = 10.0
    reward_shaping_scale: float = 0.2
    seed: int = 23
    eval_episodes: int = 300
    output_dir: str = "outputs"


class LightningRandomDQN(L.LightningModule):
    """LightningModule implementing an enhanced DQN training loop."""

    def __init__(self, cfg: LightningDQNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(asdict(cfg))
        probe_env = RandomGridworld(seed=cfg.seed)
        self.online_net = ConvolutionalDuelingQNetwork(probe_env.state_dim, probe_env.n_actions)
        self.target_net = ConvolutionalDuelingQNetwork(probe_env.state_dim, probe_env.n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.env = RandomGridworld(seed=cfg.seed)
        self.replay = ReplayBuffer(cfg.replay_capacity)
        self.n_actions = probe_env.n_actions
        self.automatic_optimization = False

        self.total_env_steps = 0
        self.total_updates = 0
        self.episode_rows: list[dict[str, object]] = []
        self.recent_successes: Deque[bool] = deque(maxlen=100)
        self.losses: list[float] = []

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.online_net(state)

    def train_dataloader(self) -> DataLoader[int]:
        return DataLoader(EpisodeDataset(self.cfg.episodes), batch_size=None, num_workers=0)

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = torch.optim.AdamW(
            self.online_net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.episodes,
            eta_min=self.cfg.min_learning_rate,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch: int, batch_idx: int) -> torch.Tensor:
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        episode = int(batch_idx)
        epsilon = self.epsilon_by_episode(episode)
        state = self.env.reset()
        total_raw_reward = 0.0
        total_train_reward = 0.0
        losses: list[torch.Tensor] = []
        won = False

        for move in range(1, self.cfg.max_moves + 1):
            action = self.select_action(state, epsilon)
            previous_distance = self.distance_to_goal()
            result = self.env.step(action)
            next_distance = self.distance_to_goal()
            train_reward = self.shaped_reward(result.reward, previous_distance, next_distance)

            self.replay.push(Transition(state, action, train_reward, result.state, result.terminated))
            self.total_env_steps += 1

            loss = self.optimize_from_replay(optimizer)
            if loss is not None:
                losses.append(loss.detach())

            state = result.state
            total_raw_reward += result.reward
            total_train_reward += train_reward
            if result.terminated:
                won = result.reward > 0
                break

        if scheduler is not None and losses:
            scheduler.step()

        avg_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)
        self.recent_successes.append(won)
        rolling_success = sum(self.recent_successes) / len(self.recent_successes)
        current_lr = optimizer.param_groups[0]["lr"]
        self.episode_rows.append(
            {
                "episode": episode + 1,
                "epsilon": round(epsilon, 5),
                "moves": move,
                "raw_return": total_raw_reward,
                "train_return": round(total_train_reward, 5),
                "won": won,
                "replay_size": len(self.replay),
                "updates": self.total_updates,
                "loss": float(avg_loss.item()),
                "lr": current_lr,
                "rolling_success_100": rolling_success,
            }
        )

        self.log("episode_reward", total_raw_reward, prog_bar=False)
        self.log("rolling_success_100", rolling_success, prog_bar=False)
        self.log("loss", avg_loss, prog_bar=False)
        return avg_loss

    def optimize_from_replay(self, optimizer: torch.optim.Optimizer) -> torch.Tensor | None:
        if len(self.replay) < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None

        batch = self.replay.sample(self.cfg.batch_size)
        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        current_q = self.online_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        optimizer.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.cfg.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )
        optimizer.step()

        self.total_updates += 1
        self.losses.append(float(loss.item()))
        if self.total_updates % self.cfg.target_sync_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.online_net(state_tensor), dim=1).item())

    def epsilon_by_episode(self, episode: int) -> float:
        fraction = min(1.0, episode / max(1, self.cfg.epsilon_decay_episodes))
        return self.cfg.epsilon_start + fraction * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def distance_to_goal(self) -> int:
        player_row, player_col = self.env.player_pos
        goal_row, goal_col = self.env.goal_pos
        return abs(player_row - goal_row) + abs(player_col - goal_col)

    def shaped_reward(self, raw_reward: float, previous_distance: int, next_distance: int) -> float:
        if raw_reward != -1.0:
            return raw_reward
        progress = previous_distance - next_distance
        return raw_reward + self.cfg.reward_shaping_scale * progress


def evaluate(model: LightningRandomDQN, episodes: int, max_moves: int, seed: int) -> dict[str, object]:
    env = RandomGridworld(seed=seed)
    wins = 0
    raw_returns: list[float] = []
    lengths: list[int] = []
    rows: list[dict[str, object]] = []

    model.eval()
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        actions: list[str] = []
        won = False

        for move in range(1, max_moves + 1):
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)
                action = int(torch.argmax(model.online_net(state_tensor), dim=1).item())
            result = env.step(action)
            actions.append(env.action_names[action])
            state = result.state
            total_reward += result.reward
            if result.terminated:
                won = result.reward > 0
                break

        wins += int(won)
        raw_returns.append(total_reward)
        lengths.append(move)
        rows.append(
            {
                "episode": episode + 1,
                "won": won,
                "raw_return": total_reward,
                "moves": move,
                "actions": " ".join(actions),
            }
        )

    return {
        "episodes": episodes,
        "wins": wins,
        "win_rate": wins / episodes,
        "average_raw_return": float(np.mean(raw_returns)),
        "average_length": float(np.mean(lengths)),
        "rows": rows,
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: LightningDQNConfig) -> dict[str, object]:
    L.seed_everything(cfg.seed, workers=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model = LightningRandomDQN(cfg)
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
    )
    trainer.fit(model)

    eval_stats = evaluate(model, cfg.eval_episodes, cfg.max_moves, seed=cfg.seed + 10_000)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "hw3_3_lightning_random_training.csv", model.episode_rows)
    write_csv(output_dir / "hw3_3_lightning_random_eval.csv", eval_stats["rows"])  # type: ignore[arg-type]
    trainer.save_checkpoint(output_dir / "hw3_3_lightning_random_dqn.ckpt")
    torch.save(model.online_net.state_dict(), output_dir / "hw3_3_lightning_random_dqn.pt")

    metrics = {
        "framework": "PyTorch Lightning",
        "config": asdict(cfg),
        "training": {
            "episodes": cfg.episodes,
            "env_steps": model.total_env_steps,
            "updates": model.total_updates,
            "final_replay_size": len(model.replay),
            "final_rolling_success_100": model.episode_rows[-1]["rolling_success_100"],
            "last_loss": model.losses[-1] if model.losses else None,
        },
        "evaluation": {key: value for key, value in eval_stats.items() if key != "rows"},
        "training_tips": [
            "PyTorch Lightning manual optimization",
            "convolutional state encoder for 4x4 spatial structure",
            "Dueling network architecture",
            "Double DQN target computation",
            "target network synchronization",
            "Huber loss",
            "gradient clipping",
            "AdamW optimizer",
            "cosine learning rate scheduling",
            "reward shaping based on Manhattan-distance progress",
            "epsilon-greedy exploration schedule",
        ],
    }
    (output_dir / "hw3_3_lightning_random_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def parse_args() -> LightningDQNConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=LightningDQNConfig.episodes)
    parser.add_argument("--eval-episodes", type=int, default=LightningDQNConfig.eval_episodes)
    parser.add_argument("--seed", type=int, default=LightningDQNConfig.seed)
    parser.add_argument("--output-dir", type=str, default=LightningDQNConfig.output_dir)
    args = parser.parse_args()
    return LightningDQNConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    metrics = run(cfg)
    print(json.dumps(metrics["evaluation"], indent=2))


if __name__ == "__main__":
    main()
