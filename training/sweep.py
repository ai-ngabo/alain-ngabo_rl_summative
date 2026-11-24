"""Hyperparameter sweep: 12 configurations per algorithm, same 250k-frame budget, seed 0.

Each run is a fresh process (one CPU thread). Finished runs are skipped, so the sweep can resume.

run:
    uv run python -m training.sweep                 # everything, 2 runs in parallel
    uv run python -m training.sweep --algo ppo      # one algorithm
    uv run python -m training.sweep --workers 1
    uv run python -m training.sweep --stage seeds   # best configs on 2 more seeds
    uv run python -m training.sweep --stage final   # best configs trained for 1M frames
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SWEEPS: dict[str, list[tuple[str, dict]]] = {
    "dqn": [
        ("dqn_01_baseline", {}),
        ("dqn_02_lr1e-3", {"lr": 1e-3}),
        ("dqn_03_lr1e-4", {"lr": 1e-4}),
        ("dqn_04_gamma0.95", {"gamma": 0.95}),
        ("dqn_05_batch32", {"batch_size": 32}),
        ("dqn_06_buffer10k", {"buffer_size": 10_000}),
        ("dqn_07_eps_fast30k", {"eps_decay_frames": 30_000}),
        ("dqn_08_eps_slow200k", {"eps_decay_frames": 200_000, "eps_end": 0.1}),
        ("dqn_09_target_hard200", {"target_interval": 200}),
        ("dqn_10_target_soft", {"target_update": "soft", "tau": 0.005}),
        ("dqn_11_no_double", {"double_dqn": False}),
        ("dqn_12_net256", {"hidden": [256, 256], "lr": 3e-4}),
    ],
    "reinforce": [
        ("reinforce_01_baseline", {}),
        ("reinforce_02_no_baseline", {"baseline": False, "adv_norm": "std"}),
        ("reinforce_03_no_baseline_raw", {"baseline": False, "adv_norm": "none"}),
        ("reinforce_04_lr1e-3", {"lr": 1e-3}),
        ("reinforce_05_lr1e-2", {"lr": 1e-2}),
        ("reinforce_06_gamma0.95", {"gamma": 0.95}),
        ("reinforce_07_gamma0.999", {"gamma": 0.999}),
        ("reinforce_08_entropy0", {"entropy": 0.0}),
        ("reinforce_09_entropy0.05", {"entropy": 0.05}),
        ("reinforce_10_8eps_batch", {"num_envs": 8}),
        ("reinforce_11_2eps_batch", {"num_envs": 2}),
        ("reinforce_12_net64", {"hidden": [64, 64]}),
    ],
    "ppo": [
        ("ppo_01_baseline", {}),
        ("ppo_02_lr1e-3", {"lr": 1e-3}),
        ("ppo_03_lr1e-4", {"lr": 1e-4}),
        ("ppo_04_clip0.1", {"clip": 0.1}),
        ("ppo_05_clip0.3", {"clip": 0.3}),
        ("ppo_06_gae0.8", {"lmbda": 0.8}),
        ("ppo_07_gae0.99", {"lmbda": 0.99}),
        ("ppo_08_epochs10", {"epochs": 10}),
        ("ppo_09_batch512", {"frames_per_batch": 512, "minibatch": 128}),
        ("ppo_10_batch4096", {"frames_per_batch": 4096, "minibatch": 512}),
        ("ppo_11_entropy0", {"entropy": 0.0}),
        ("ppo_12_entropy0.05", {"entropy": 0.05}),
    ],
    "a2c": [
        ("a2c_01_baseline_nstep", {}),
        ("a2c_02_gae0.95", {"lmbda": 0.95}),
        ("a2c_03_td0", {"lmbda": 0.0}),
        ("a2c_04_rollout5", {"rollout": 5}),
        ("a2c_05_rollout64", {"rollout": 64}),
        ("a2c_06_lr3e-4", {"lr": 3e-4}),
        ("a2c_07_lr2e-3", {"lr": 2e-3}),
        ("a2c_08_entropy0", {"entropy": 0.0}),
        ("a2c_09_entropy0.05", {"entropy": 0.05}),
        ("a2c_10_critic0.25", {"critic_coeff": 0.25}),
        ("a2c_11_critic1.0", {"critic_coeff": 1.0}),
        ("a2c_12_gamma0.95", {"gamma": 0.95}),
    ],
}


def _command(algo: str, run: str, cfg: dict, frames: int) -> list[str]:
    cfg = dict(cfg)
    seed = cfg.pop("_seed", 0)
    if algo == "dqn":
        cmd = [sys.executable, "-m", "training.dqn_training", "--run", run]
    else:
        cmd = [sys.executable, "-m", "training.pg_training", "--algo", algo, "--run", run]
    return cmd + ["--frames", str(frames), "--seed", str(seed), "--config", json.dumps(cfg)]


def _run(job) -> str:
    algo, run, cfg, frames = job
    if (ROOT / "logs" / algo / f"{run}.json").exists():
        return f"skip {run}"
    console = ROOT / "logs" / "console" / f"{run}.txt"
    console.parent.mkdir(parents=True, exist_ok=True)
    with console.open("w") as fh:
        rc = subprocess.call(_command(algo, run, cfg, frames), cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    return f"{'done' if rc == 0 else 'FAILED'} {run}"


def _best_configs() -> dict[str, tuple[str, dict]]:
    """Best sweep run per algorithm (by mean evaluation reward) and its overrides."""
    out = {}
    for algo, runs in SWEEPS.items():
        scored = []
        for run, cfg in runs:
            f = ROOT / "logs" / algo / f"{run}.json"
            if f.exists():
                scored.append((json.loads(f.read_text())["final_eval_train"]["mean_return"], run, cfg))
        if scored:
            _, run, cfg = max(scored)
            out[algo] = (run, cfg)
    return out


def extra_jobs(stage: str, frames: int) -> list:
    """stage 'seeds': best config of each algorithm on seeds 1 and 2 (same 250k budget).
    stage 'final': best config of each algorithm trained 4x longer (1M frames) as the final model."""
    jobs = []
    for algo, (run, cfg) in _best_configs().items():
        if stage == "seeds":
            for seed in (1, 2):
                jobs.append((algo, f"seed{seed}_{run}", {**cfg, "_seed": seed}, frames))
        else:
            jobs.append((algo, f"final_{run}", cfg, 1_000_000))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=list(SWEEPS), default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--frames", type=int, default=250_000)
    p.add_argument("--stage", choices=["sweep", "seeds", "final"], default="sweep")
    a = p.parse_args()
    if a.stage != "sweep":
        jobs = extra_jobs(a.stage, a.frames)
        with ThreadPoolExecutor(a.workers) as pool:
            for msg in pool.map(_run, jobs):
                print(msg, flush=True)
        return
    algos = [a.algo] if a.algo else list(SWEEPS)
    # interleave algorithms so partial results cover all four early
    jobs, i = [], 0
    while any(i < len(SWEEPS[al]) for al in algos):
        for al in algos:
            if i < len(SWEEPS[al]):
                run, cfg = SWEEPS[al][i]
                jobs.append((al, run, cfg, a.frames))
        i += 1
    with ThreadPoolExecutor(a.workers) as pool:
        for msg in pool.map(_run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
