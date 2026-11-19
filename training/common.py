"""Shared helpers for all four agents: TorchRL env wrapping, networks, evaluation, logging."""

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
from torchrl.envs import GymWrapper, RewardSum, SerialEnv, TransformedEnv

from environment import ACTIONS, OBS_DIM, IbabaDroneEnv

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"
N_ACTIONS = len(ACTIONS)

# Fixed evaluation missions so every run is scored on exactly the same cases.
EVAL_SEEDS_TRAIN = list(range(10_000, 10_030))   # 30 missions to training hospitals
EVAL_SEEDS_TEST = list(range(20_000, 20_030))    # 30 missions to held-out hospitals
PROGRESS_SEEDS = list(range(30_000, 30_010))     # quick checks during training


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(num_envs: int = 4, seed: int = 0, split: str = "train"):
    """TorchRL view of the Gymnasium environment (integer actions, episode reward tracking)."""

    def _one():
        return TransformedEnv(
            GymWrapper(IbabaDroneEnv(hospital_split=split), categorical_action_encoding=True),
            RewardSum(),
        )

    env = SerialEnv(num_envs, _one) if num_envs > 1 else _one()
    env.set_seed(seed)
    return env


def mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...], act=nn.Tanh) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def logits_module(hidden=(128, 128)) -> TensorDictModule:
    return TensorDictModule(mlp(OBS_DIM, N_ACTIONS, hidden), in_keys=["observation"], out_keys=["logits"])


@torch.no_grad()
def evaluate(net: nn.Module, seeds: list[int], split: str, greedy: bool = True) -> dict:
    """Run full missions with the raw Gymnasium env. `net` maps observation -> logits/Q-values."""
    env = IbabaDroneEnv(hospital_split=split)
    rows = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            out = net(torch.as_tensor(obs).unsqueeze(0))
            if greedy:
                a = int(out.argmax(-1))
            else:
                a = int(torch.distributions.Categorical(logits=out).sample())
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        rows.append(info)
    returns = [r["total_reward"] for r in rows]
    delivered = [r for r in rows if r["delivered"]]
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean([r["outcome"] == "home" for r in rows])),
        "delivery_rate": float(np.mean([r["delivered"] for r in rows])),
        "on_target_rate": float(np.mean([r["drop_result"] == "on_target" for r in rows])),
        "crash_rate": float(np.mean([r["outcome"] in ("crash", "no_fly", "out_of_bounds") for r in rows])),
        "battery_fail_rate": float(np.mean([r["outcome"] == "battery" for r in rows])),
        "timeout_rate": float(np.mean([r["outcome"] == "timeout" for r in rows])),
        "mean_delivery_min": float(np.mean([r["delivery_minutes"] for r in delivered])) if delivered else None,
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "returns": [round(x, 2) for x in returns],
    }


class RunLogger:
    """Collects everything a run produces and writes one JSON file per run."""

    def __init__(self, algo: str, run_name: str, config: dict):
        self.algo, self.run_name, self.config = algo, run_name, config
        self.t0 = time.time()
        self.episodes: list[list[float]] = []   # [frames, episode_return]
        self.updates: list[dict] = []           # loss / entropy per update
        self.progress: list[dict] = []          # periodic greedy evaluation

    def add_episodes(self, frames: int, returns) -> None:
        for r in returns:
            self.episodes.append([int(frames), round(float(r), 3)])

    def add_update(self, frames: int, **values) -> None:
        row = {"frames": int(frames)}
        row.update({k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 5))
                    for k, v in values.items()})
        self.updates.append(row)

    def add_progress(self, frames: int, result: dict) -> None:
        self.progress.append({"frames": int(frames), "episodes": len(self.episodes),
                              "mean_return": round(result["mean_return"], 2),
                              "success_rate": result["success_rate"]})

    def save(self, final_train: dict, final_test: dict, model_path: str) -> Path:
        out = {
            "algo": self.algo, "run": self.run_name, "config": self.config,
            "wall_seconds": round(time.time() - self.t0, 1),
            "episodes": self.episodes, "updates": self.updates, "progress": self.progress,
            "final_eval_train": final_train, "final_eval_test": final_test,
            "model_path": model_path,
        }
        path = LOG_DIR / self.algo / f"{self.run_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        return path


def finished_episode_returns(data) -> list[float]:
    """Episode returns that ended inside a collected batch (via RewardSum)."""
    done = data["next", "done"].squeeze(-1)
    if done.any():
        return data["next", "episode_reward"].squeeze(-1)[done].tolist()
    return []


def save_policy(net: nn.Module, algo: str, run_name: str, config: dict, sub: str) -> str:
    """Serialize the policy network: weights + JSON config (the integration path)."""
    d = MODEL_DIR / sub
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{run_name}.pt"
    torch.save({"algo": algo, "state_dict": net.state_dict(), "hidden": config.get("hidden", [128, 128]),
                "obs_dim": OBS_DIM, "actions": list(ACTIONS), "config": config}, path)
    return str(path.relative_to(ROOT))


def load_policy(path: str | os.PathLike) -> tuple[nn.Module, dict]:
    ckpt = torch.load(Path(path) if Path(path).is_absolute() else ROOT / path, map_location="cpu", weights_only=False)
    net = mlp(ckpt["obs_dim"], len(ckpt["actions"]), tuple(ckpt["hidden"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
