"""
Shared infrastructure used by all reinforcement-learning agents.

Provides:

    • Reproducibility utilities
    • TorchRL environment wrappers
    • Neural-network builders
    • Policy evaluation
    • Experiment logging
    • Policy persistence (save/load)

This module is intentionally algorithm-agnostic so that DQN,
REINFORCE, PPO, and A2C share the same evaluation protocol,
logging format, and model serialization process.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import (
    GymWrapper,
    RewardSum,
    SerialEnv,
    TransformedEnv,
)

from environment import (
    ACTIONS,
    OBS_DIM,
    IbabaDroneEnv,
)

# Project paths
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "models"

N_ACTIONS = len(ACTIONS)


# Fixed evaluation missions
# ---------------------------------------------------------------------

# Every experiment is evaluated on the same missions to ensure
# fair comparisons across algorithms and hyperparameter settings.

EVAL_SEEDS_TRAIN = list(range(10_000, 10_030))
EVAL_SEEDS_TEST = list(range(20_000, 20_030))
PROGRESS_SEEDS = list(range(30_000, 30_010))

# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Environment creation
# ---------------------------------------------------------------------

def make_env(
    num_envs: int = 4,
    seed: int = 0,
    split: str = "train",
):
    """
    Create a TorchRL-compatible environment.

    Supports either a single environment or a batch of environments.
    Episode rewards are tracked automatically through RewardSum.
    """

    def build_single_env():
        return TransformedEnv(
            GymWrapper(
                IbabaDroneEnv(
                    hospital_split=split,
                ),
                categorical_action_encoding=True,
            ),
            RewardSum(),
        )

    env = (
        SerialEnv(num_envs, build_single_env)
        if num_envs > 1
        else build_single_env()
    )

    env.set_seed(seed)

    return env

# Neural networks
# ---------------------------------------------------------------------

def mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...],
    activation=nn.Tanh,
) -> nn.Sequential:
    """
    Build a fully connected MLP.
    """

    layers = []
    current_dim = input_dim

    for hidden_size in hidden_sizes:
        layers.extend(
            [
                nn.Linear(current_dim, hidden_size),
                activation(),
            ]
        )
        current_dim = hidden_size

    layers.append(
        nn.Linear(current_dim, output_dim)
    )

    return nn.Sequential(*layers)


def logits_module(
    hidden_sizes: tuple[int, ...] = (128, 128),
) -> TensorDictModule:
    """
    TorchRL module producing action logits.
    """

    return TensorDictModule(
        mlp(
            OBS_DIM,
            N_ACTIONS,
            hidden_sizes,
        ),
        in_keys=["observation"],
        out_keys=["logits"],
    )

# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    net: nn.Module,
    seeds: list[int],
    split: str,
    greedy: bool = True,
) -> dict:
    """
    Evaluate a policy on a fixed set of missions.

    The raw Gymnasium environment is used rather than TorchRL
    wrappers so that all algorithms are measured under identical
    deployment conditions.

    Returns aggregate mission statistics used throughout training
    and analysis.
    """

    env = IbabaDroneEnv(hospital_split=split)

    mission_results = []

    for seed in seeds:

        observation, _ = env.reset(seed=seed)

        done = False

        while not done:

            action_scores = net(
                torch.as_tensor(observation).unsqueeze(0)
            )

            if greedy:
                action = int(action_scores.argmax(-1))
            else:
                action = int(
                    torch.distributions.Categorical(
                        logits=action_scores
                    ).sample()
                )

            observation, _, terminated, truncated, info = env.step(
                action
            )

            done = terminated or truncated

        mission_results.append(info)

    returns = [
        result["total_reward"]
        for result in mission_results
    ]

    successful_deliveries = [
        result
        for result in mission_results
        if result["delivered"]
    ]

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(
            np.mean(
                [
                    result["outcome"] == "home"
                    for result in mission_results
                ]
            )
        ),
        "delivery_rate": float(
            np.mean(
                [
                    result["delivered"]
                    for result in mission_results
                ]
            )
        ),
        "on_target_rate": float(
            np.mean(
                [
                    result["drop_result"] == "on_target"
                    for result in mission_results
                ]
            )
        ),
        "crash_rate": float(
            np.mean(
                [
                    result["outcome"]
                    in (
                        "crash",
                        "no_fly",
                        "out_of_bounds",
                    )
                    for result in mission_results
                ]
            )
        ),
        "battery_fail_rate": float(
            np.mean(
                [
                    result["outcome"] == "battery"
                    for result in mission_results
                ]
            )
        ),
        "timeout_rate": float(
            np.mean(
                [
                    result["outcome"] == "timeout"
                    for result in mission_results
                ]
            )
        ),
        "mean_delivery_min": (
            float(
                np.mean(
                    [
                        result["delivery_minutes"]
                        for result in successful_deliveries
                    ]
                )
            )
            if successful_deliveries
            else None
        ),
        "mean_steps": float(
            np.mean(
                [
                    result["steps"]
                    for result in mission_results
                ]
            )
        ),
        "returns": [
            round(value, 2)
            for value in returns
        ],
    }

# Run logging
# ---------------------------------------------------------------------

class RunLogger:
    """
    Collect all information generated during training and
    export it as a single JSON run log.
    """

    def __init__(
        self,
        algo: str,
        run_name: str,
        config: dict,
    ):
        self.algo = algo
        self.run_name = run_name
        self.config = config

        self.start_time = time.time()

        self.episodes: list[list[float]] = []
        self.updates: list[dict] = []
        self.progress: list[dict] = []

    def add_episodes(
        self,
        frames: int,
        returns,
    ) -> None:

        for episode_return in returns:
            self.episodes.append(
                [
                    int(frames),
                    round(float(episode_return), 3),
                ]
            )

    def add_update(
        self,
        frames: int,
        **values,
    ) -> None:

        update_row = {
            "frames": int(frames)
        }

        update_row.update(
            {
                key: (
                    None
                    if (
                        value is None
                        or (
                            isinstance(value, float)
                            and math.isnan(value)
                        )
                    )
                    else round(float(value), 5)
                )
                for key, value in values.items()
            }
        )

        self.updates.append(update_row)

    def add_progress(
        self,
        frames: int,
        result: dict,
    ) -> None:

        self.progress.append(
            {
                "frames": int(frames),
                "episodes": len(self.episodes),
                "mean_return": round(
                    result["mean_return"],
                    2,
                ),
                "success_rate": result["success_rate"],
            }
        )

    def save(
        self,
        final_train: dict,
        final_test: dict,
        model_path: str,
    ) -> Path:

        output = {
            "algo": self.algo,
            "run": self.run_name,
            "config": self.config,
            "wall_seconds": round(
                time.time() - self.start_time,
                1,
            ),
            "episodes": self.episodes,
            "updates": self.updates,
            "progress": self.progress,
            "final_eval_train": final_train,
            "final_eval_test": final_test,
            "model_path": model_path,
        }

        log_path = (
            LOG_DIR
            / self.algo
            / f"{self.run_name}.json"
        )

        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        log_path.write_text(
            json.dumps(output)
        )

        return log_path

# Episode statistics
# ---------------------------------------------------------------------

def finished_episode_returns(data) -> list[float]:
    """
    Extract completed episode returns from a TorchRL batch.
    """
    
    done = data["next", "done"].squeeze(-1)

    if done.any():
        return (
            data["next", "episode_reward"]
            .squeeze(-1)[done]
            .tolist()
        )

    return []

# Model persistence
# ---------------------------------------------------------------------

def save_policy(
    net: nn.Module,
    algo: str,
    run_name: str,
    config: dict,
    subdirectory: str,
) -> str:
    """
    Save a trained policy checkpoint.
    """

    model_directory = MODEL_DIR / subdirectory

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_directory
        / f"{run_name}.pt"
    )

    torch.save(
        {
            "algo": algo,
            "state_dict": net.state_dict(),
            "hidden": config.get(
                "hidden",
                [128, 128],
            ),
            "obs_dim": OBS_DIM,
            "actions": list(ACTIONS),
            "config": config,
        },
        model_path,
    )

    return str(
        model_path.relative_to(ROOT_DIR)
    )


def load_policy(
    path: str | os.PathLike,
) -> tuple[nn.Module, dict]:
    """
    Load a previously saved policy network.
    """

    checkpoint = torch.load(
        Path(path)
        if Path(path).is_absolute()
        else ROOT_DIR / path,
        map_location="cpu",
        weights_only=False,
    )

    net = mlp(
        checkpoint["obs_dim"],
        len(checkpoint["actions"]),
        tuple(checkpoint["hidden"]),
    )

    net.load_state_dict(
        checkpoint["state_dict"]
    )

    net.eval()

    return net, checkpoint