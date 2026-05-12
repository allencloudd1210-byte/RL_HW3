"""Static Gridworld environment used for HW3-1.

This is a compact, self-contained version of the Chapter 3 Gridworld idea from
DRL in Action. The static mode fixes every object position so the task remains
small enough for a basic DQN smoke run.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


Position = Tuple[int, int]


@dataclass(frozen=True)
class StepResult:
    state: np.ndarray
    reward: float
    terminated: bool
    info: Dict[str, object]


class StaticGridworld:
    """A deterministic 4x4 Gridworld with Player, Goal, Pit, and Wall layers."""

    action_names = ("u", "d", "l", "r")
    action_deltas: Dict[int, Position] = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    def __init__(self, size: int = 4) -> None:
        if size < 4:
            raise ValueError("StaticGridworld requires size >= 4.")
        self.size = size
        self.goal_pos: Position = (0, 0)
        self.pit_pos: Position = (0, 1)
        self.wall_pos: Position = (1, 1)
        self.start_pos: Position = (0, 3)
        self.player_pos: Position = self.start_pos

    @property
    def n_actions(self) -> int:
        return len(self.action_names)

    @property
    def state_dim(self) -> int:
        return 4 * self.size * self.size

    def reset(self) -> np.ndarray:
        self.player_pos = self.start_pos
        return self.state()

    def state(self) -> np.ndarray:
        layers = np.zeros((4, self.size, self.size), dtype=np.float32)
        layers[0, self.player_pos[0], self.player_pos[1]] = 1.0
        layers[1, self.goal_pos[0], self.goal_pos[1]] = 1.0
        layers[2, self.pit_pos[0], self.pit_pos[1]] = 1.0
        layers[3, self.wall_pos[0], self.wall_pos[1]] = 1.0
        return layers.reshape(-1)

    def step(self, action: int) -> StepResult:
        if action not in self.action_deltas:
            raise ValueError(f"Unknown action index: {action}")

        row_delta, col_delta = self.action_deltas[action]
        row, col = self.player_pos
        next_pos = (row + row_delta, col + col_delta)

        if self._is_valid_move(next_pos):
            self.player_pos = next_pos

        reward = self.reward()
        terminated = reward != -1.0
        return StepResult(
            state=self.state(),
            reward=reward,
            terminated=terminated,
            info={"player_pos": self.player_pos, "action": self.action_names[action]},
        )

    def reward(self) -> float:
        if self.player_pos == self.goal_pos:
            return 10.0
        if self.player_pos == self.pit_pos:
            return -10.0
        return -1.0

    def render(self) -> str:
        board = np.full((self.size, self.size), " ", dtype="<U1")
        board[self.goal_pos] = "+"
        board[self.pit_pos] = "-"
        board[self.wall_pos] = "W"
        board[self.player_pos] = "P"
        return "\n".join(" ".join(row) for row in board)

    def _is_valid_move(self, pos: Position) -> bool:
        row, col = pos
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        return pos != self.wall_pos


class PlayerGridworld(StaticGridworld):
    """Gridworld player mode: player start is random, other objects stay fixed."""

    def __init__(self, size: int = 4, seed: int | None = None) -> None:
        super().__init__(size=size)
        self.rng = np.random.default_rng(seed)

    @property
    def terminal_positions(self) -> set[Position]:
        return {self.goal_pos, self.pit_pos}

    @property
    def blocked_positions(self) -> set[Position]:
        return {self.wall_pos}

    def valid_start_positions(self) -> list[Position]:
        occupied = self.terminal_positions | self.blocked_positions
        return [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
            if (row, col) not in occupied
        ]

    def reset(self, start_pos: Position | None = None) -> np.ndarray:
        if start_pos is None:
            starts = self.valid_start_positions()
            index = int(self.rng.integers(0, len(starts)))
            self.player_pos = starts[index]
        else:
            self._validate_start_pos(start_pos, self.valid_start_positions())
            self.player_pos = start_pos
        return self.state()

    def _validate_start_pos(self, pos: Position, valid_positions: Iterable[Position]) -> None:
        if pos not in set(valid_positions):
            raise ValueError(f"Invalid player-mode start position: {pos}")


class RandomGridworld(StaticGridworld):
    """Gridworld random mode: player, goal, pit, and wall are all randomized."""

    def __init__(self, size: int = 4, seed: int | None = None, ensure_reachable: bool = True) -> None:
        super().__init__(size=size)
        self.rng = np.random.default_rng(seed)
        self.ensure_reachable = ensure_reachable

    def reset(self) -> np.ndarray:
        self._sample_board()
        return self.state()

    def _sample_board(self) -> None:
        cells = [(row, col) for row in range(self.size) for col in range(self.size)]
        while True:
            indices = self.rng.choice(len(cells), size=4, replace=False)
            self.player_pos = cells[int(indices[0])]
            self.goal_pos = cells[int(indices[1])]
            self.pit_pos = cells[int(indices[2])]
            self.wall_pos = cells[int(indices[3])]
            if not self.ensure_reachable or self._has_path_to_goal():
                return

    def _has_path_to_goal(self) -> bool:
        blocked = {self.wall_pos, self.pit_pos}
        queue: deque[Position] = deque([self.player_pos])
        visited = {self.player_pos}

        while queue:
            row, col = queue.popleft()
            if (row, col) == self.goal_pos:
                return True
            for row_delta, col_delta in self.action_deltas.values():
                next_pos = (row + row_delta, col + col_delta)
                next_row, next_col = next_pos
                in_bounds = 0 <= next_row < self.size and 0 <= next_col < self.size
                if in_bounds and next_pos not in blocked and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
        return False
