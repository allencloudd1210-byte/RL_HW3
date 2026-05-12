"""Rainbow DQN for random-mode Gridworld.

Run from the project root:

    python -m hw3_4.rainbow_random_dqn
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Iterable, NamedTuple

import numpy as np
import torch
from torch import nn

from hw3_1.gridworld import RandomGridworld


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    discount: float


class RawStep(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    """Proportional prioritized replay buffer."""

    def __init__(self, capacity: int, alpha: float, beta_start: float, beta_frames: int) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.memory: list[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, transition: Transition) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, frame: int) -> tuple[list[Transition], np.ndarray, torch.Tensor]:
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.position]

        probs = priorities**self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)

        beta = self.beta_by_frame(frame)
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return [self.memory[int(index)] for index in indices], indices, torch.as_tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for index, priority in zip(indices, priorities):
            value = float(priority) + 1e-6
            self.priorities[int(index)] = value
            self.max_priority = max(self.max_priority, value)

    def beta_by_frame(self, frame: int) -> float:
        fraction = min(1.0, frame / max(1, self.beta_frames))
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def __len__(self) -> int:
        return len(self.memory)


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet linear layer."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        noise = torch.randn(size)
        return noise.sign() * noise.abs().sqrt()


class RainbowNetwork(nn.Module):
    """Convolutional dueling C51 network with noisy value/advantage heads."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_atoms: int,
        noisy_std: float,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if state_dim != 64:
            raise ValueError(f"Expected 64-dimensional flattened Gridworld state, got {state_dim}.")
        self.n_actions = n_actions
        self.n_atoms = n_atoms
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
        self.value = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_atoms, std_init=noisy_std),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, std_init=noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_actions * n_atoms, std_init=noisy_std),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feature = self.encoder(state)
        value = self.value(feature).view(-1, 1, self.n_atoms)
        advantage = self.advantage(feature).view(-1, self.n_actions, self.n_atoms)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.softmax(logits, dim=2).clamp(min=1e-6)

    def q_values(self, state: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.forward(state) * support.view(1, 1, -1), dim=2)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


@dataclass
class RainbowConfig:
    episodes: int = 7000
    max_moves: int = 45
    replay_capacity: int = 20000
    warmup_steps: int = 600
    batch_size: int = 64
    gamma: float = 0.95
    n_step: int = 5
    learning_rate: float = 8e-4
    min_learning_rate: float = 1e-4
    target_sync_freq: int = 250
    gradient_clip_val: float = 10.0
    priority_alpha: float = 0.4
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 30000
    n_atoms: int = 51
    v_min: float = -50.0
    v_max: float = 15.0
    noisy_std: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 3500
    reward_shaping_scale: float = 0.2
    auxiliary_q_loss_weight: float = 5.0
    seed: int = 31
    eval_episodes: int = 300
    output_dir: str = "outputs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def distance_to_goal(env: RandomGridworld) -> int:
    player_row, player_col = env.player_pos
    goal_row, goal_col = env.goal_pos
    return abs(player_row - goal_row) + abs(player_col - goal_col)


def shaped_reward(raw_reward: float, previous_distance: int, next_distance: int, scale: float) -> float:
    if raw_reward != -1.0:
        return raw_reward
    return raw_reward + scale * (previous_distance - next_distance)


def valid_action_mask_from_state(state: np.ndarray, size: int = 4) -> np.ndarray:
    layers = state.reshape(4, size, size)
    player_index = int(np.argmax(layers[0]))
    player_row, player_col = divmod(player_index, size)
    wall_index = int(np.argmax(layers[3]))
    wall_pos = divmod(wall_index, size)
    mask = np.ones(4, dtype=bool)
    for action, (row_delta, col_delta) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        next_pos = (player_row + row_delta, player_col + col_delta)
        row, col = next_pos
        if row < 0 or row >= size or col < 0 or col >= size or next_pos == wall_pos:
            mask[action] = False
    return mask


def mask_invalid_actions(q_values: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    masks = np.stack([valid_action_mask_from_state(state) for state in states.detach().cpu().numpy()])
    mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=q_values.device)
    return q_values.masked_fill(~mask_tensor, -1e9)


def make_n_step_transition(queue: Deque[RawStep], gamma: float, n_step: int) -> Transition:
    reward = 0.0
    discount = 1.0
    next_state = queue[-1].next_state
    done = queue[-1].done

    for index, step in enumerate(queue):
        reward += (gamma**index) * step.reward
        next_state = step.next_state
        done = step.done
        if done or index + 1 >= n_step:
            discount = gamma ** (index + 1)
            break

    first = queue[0]
    return Transition(first.state, first.action, reward, next_state, done, discount)


def push_n_step(
    n_step_queue: Deque[RawStep],
    replay: PrioritizedReplayBuffer,
    transition: RawStep,
    gamma: float,
    n_step: int,
) -> None:
    n_step_queue.append(transition)
    if len(n_step_queue) < n_step and not transition.done:
        return

    replay.push(make_n_step_transition(n_step_queue, gamma, n_step))
    n_step_queue.popleft()
    if transition.done:
        while n_step_queue:
            replay.push(make_n_step_transition(n_step_queue, gamma, n_step))
            n_step_queue.popleft()


def select_action(
    model: RainbowNetwork,
    state: np.ndarray,
    support: torch.Tensor,
    n_actions: int,
    total_steps: int,
    warmup_steps: int,
    epsilon: float,
) -> int:
    if total_steps < warmup_steps or random.random() < epsilon:
        valid_actions = np.flatnonzero(valid_action_mask_from_state(state))
        return int(np.random.choice(valid_actions))
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        model.reset_noise()
        q_values = model.q_values(state_tensor, support)
        q_values = mask_invalid_actions(q_values, state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def projection_distribution(
    next_dist: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    dones: torch.Tensor,
    support: torch.Tensor,
    cfg: RainbowConfig,
) -> torch.Tensor:
    batch_size = rewards.size(0)
    delta_z = (cfg.v_max - cfg.v_min) / (cfg.n_atoms - 1)
    target_z = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * discounts.unsqueeze(1) * support.unsqueeze(0)
    target_z = target_z.clamp(cfg.v_min, cfg.v_max)
    b = (target_z - cfg.v_min) / delta_z
    lower = b.floor().long()
    upper = b.ceil().long()

    offset = (
        torch.arange(batch_size, device=next_dist.device).unsqueeze(1) * cfg.n_atoms
    )
    projected = torch.zeros(batch_size, cfg.n_atoms, device=next_dist.device)
    projected.view(-1).index_add_(0, (lower + offset).view(-1), (next_dist * (upper.float() - b)).view(-1))
    projected.view(-1).index_add_(0, (upper + offset).view(-1), (next_dist * (b - lower.float())).view(-1))

    equal_mask = upper == lower
    if equal_mask.any():
        projected.view(-1).index_add_(0, (lower[equal_mask] + offset.expand_as(lower)[equal_mask]).view(-1), next_dist[equal_mask].view(-1))
    return projected


def optimize_model(
    online_model: RainbowNetwork,
    target_model: RainbowNetwork,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplayBuffer,
    support: torch.Tensor,
    cfg: RainbowConfig,
    total_steps: int,
) -> float | None:
    if len(replay) < max(cfg.batch_size, cfg.warmup_steps):
        return None

    batch, indices, weights = replay.sample(cfg.batch_size, total_steps)
    states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32)
    actions = torch.as_tensor([t.action for t in batch], dtype=torch.long)
    rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
    next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32)
    dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32)
    discounts = torch.as_tensor([t.discount for t in batch], dtype=torch.float32)

    online_model.reset_noise()
    target_model.reset_noise()
    dist = online_model(states)
    chosen_dist = dist[torch.arange(cfg.batch_size), actions]

    with torch.no_grad():
        next_q_values = online_model.q_values(next_states, support)
        next_actions = mask_invalid_actions(next_q_values, next_states).argmax(dim=1)
        next_dist = target_model(next_states)[torch.arange(cfg.batch_size), next_actions]
        target_dist = projection_distribution(next_dist, rewards, discounts, dones, support, cfg)

    distributional_loss = -(target_dist * chosen_dist.log()).sum(dim=1)
    current_q = torch.sum(chosen_dist * support.view(1, -1), dim=1)
    target_q = torch.sum(target_dist * support.view(1, -1), dim=1)
    q_loss = nn.functional.smooth_l1_loss(current_q, target_q, reduction="none")
    per_sample_loss = distributional_loss + cfg.auxiliary_q_loss_weight * q_loss
    loss = (weights * per_sample_loss).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_model.parameters(), cfg.gradient_clip_val)
    optimizer.step()
    replay.update_priorities(indices, per_sample_loss.detach().cpu().numpy())
    return float(loss.item())


def train(cfg: RainbowConfig) -> tuple[RainbowNetwork, dict[str, object], list[dict[str, object]]]:
    set_seed(cfg.seed)
    env = RandomGridworld(seed=cfg.seed)
    online_model = RainbowNetwork(env.state_dim, env.n_actions, cfg.n_atoms, cfg.noisy_std)
    target_model = RainbowNetwork(env.state_dim, env.n_actions, cfg.n_atoms, cfg.noisy_std)
    target_model.load_state_dict(online_model.state_dict())

    support = torch.linspace(cfg.v_min, cfg.v_max, cfg.n_atoms)
    replay = PrioritizedReplayBuffer(
        cfg.replay_capacity,
        alpha=cfg.priority_alpha,
        beta_start=cfg.priority_beta_start,
        beta_frames=cfg.priority_beta_frames,
    )
    optimizer = torch.optim.AdamW(online_model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.episodes,
        eta_min=cfg.min_learning_rate,
    )

    total_steps = 0
    total_updates = 0
    losses: list[float] = []
    episode_rows: list[dict[str, object]] = []
    recent_successes: Deque[bool] = deque(maxlen=100)

    for episode in range(cfg.episodes):
        epsilon = epsilon_by_episode(episode, cfg)
        state = env.reset()
        n_step_queue: Deque[RawStep] = deque()
        total_raw_reward = 0.0
        total_train_reward = 0.0
        won = False

        for move in range(1, cfg.max_moves + 1):
            action = select_action(online_model, state, support, env.n_actions, total_steps, cfg.warmup_steps, epsilon)
            previous_distance = distance_to_goal(env)
            result = env.step(action)
            next_distance = distance_to_goal(env)
            train_reward = shaped_reward(result.reward, previous_distance, next_distance, cfg.reward_shaping_scale)
            push_n_step(
                n_step_queue,
                replay,
                RawStep(state, action, train_reward, result.state, result.terminated),
                cfg.gamma,
                cfg.n_step,
            )

            total_steps += 1
            state = result.state
            total_raw_reward += result.reward
            total_train_reward += train_reward

            loss = optimize_model(online_model, target_model, optimizer, replay, support, cfg, total_steps)
            if loss is not None:
                losses.append(loss)
                total_updates += 1
                if total_updates % cfg.target_sync_freq == 0:
                    target_model.load_state_dict(online_model.state_dict())

            if result.terminated:
                won = result.reward > 0
                break

        if losses:
            scheduler.step()

        recent_successes.append(won)
        episode_rows.append(
            {
                "episode": episode + 1,
                "epsilon": round(epsilon, 5),
                "moves": move,
                "raw_return": total_raw_reward,
                "train_return": round(total_train_reward, 5),
                "won": won,
                "replay_size": len(replay),
                "updates": total_updates,
                "loss": losses[-1] if losses else "",
                "lr": optimizer.param_groups[0]["lr"],
                "rolling_success_100": sum(recent_successes) / len(recent_successes),
                "beta": replay.beta_by_frame(total_steps),
            }
        )

    eval_stats = evaluate(online_model, cfg.eval_episodes, cfg.max_moves, seed=cfg.seed + 10_000, support=support)
    metrics: dict[str, object] = {
        "algorithm": "Rainbow DQN",
        "config": asdict(cfg),
        "training": {
            "episodes": cfg.episodes,
            "env_steps": total_steps,
            "updates": total_updates,
            "final_replay_size": len(replay),
            "final_rolling_success_100": episode_rows[-1]["rolling_success_100"],
            "last_loss": losses[-1] if losses else None,
        },
        "evaluation": {key: value for key, value in eval_stats.items() if key != "rows"},
        "rainbow_components": [
            "Double DQN",
            "Dueling network",
            "Prioritized Experience Replay",
            "n-step returns",
            "C51 distributional value function",
            "NoisyLinear exploration",
        ],
        "training_stabilizers": [
            "convolutional state encoder",
            "target network synchronization",
            "invalid-action mask for wall/out-of-bounds moves",
            "auxiliary expected-Q Huber loss",
            "AdamW optimizer",
            "cosine learning rate scheduler",
            "gradient clipping",
            "Manhattan-distance reward shaping",
            "small epsilon schedule in addition to NoisyLinear exploration",
        ],
    }

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "hw3_4_rainbow_random_training.csv", episode_rows)
    write_csv(output_dir / "hw3_4_rainbow_random_eval.csv", eval_stats["rows"])  # type: ignore[arg-type]
    (output_dir / "hw3_4_rainbow_random_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    torch.save(online_model.state_dict(), output_dir / "hw3_4_rainbow_random_dqn.pt")
    return online_model, metrics, episode_rows


def epsilon_by_episode(episode: int, cfg: RainbowConfig) -> float:
    fraction = min(1.0, episode / max(1, cfg.epsilon_decay_episodes))
    return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)


def evaluate(
    model: RainbowNetwork,
    episodes: int,
    max_moves: int,
    seed: int,
    support: torch.Tensor,
) -> dict[str, object]:
    env = RandomGridworld(seed=seed)
    model.eval()
    wins = 0
    raw_returns: list[float] = []
    lengths: list[int] = []
    rows: list[dict[str, object]] = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        actions: list[str] = []
        won = False

        for move in range(1, max_moves + 1):
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = model.q_values(state_tensor, support)
                q_values = mask_invalid_actions(q_values, state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())
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


def parse_args() -> RainbowConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=RainbowConfig.episodes)
    parser.add_argument("--eval-episodes", type=int, default=RainbowConfig.eval_episodes)
    parser.add_argument("--seed", type=int, default=RainbowConfig.seed)
    parser.add_argument("--output-dir", type=str, default=RainbowConfig.output_dir)
    args = parser.parse_args()
    return RainbowConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main() -> None:
    cfg = parse_args()
    _, metrics, _ = train(cfg)
    print(json.dumps(metrics["evaluation"], indent=2))


if __name__ == "__main__":
    main()
