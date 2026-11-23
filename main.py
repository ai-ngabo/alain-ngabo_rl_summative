"""
Ibaba Life-Saver

Command-line entry point for training, evaluating, and deploying
reinforcement learning agents that control blood-delivery drones
serving hospitals in Rwanda.

Examples
--------
Watch the best trained agent:
    uv run main.py

Watch a specific algorithm:
    uv run main.py play --algo dqn

Train an agent:
    uv run main.py train --algo ppo

Run all hyperparameter experiments:
    uv run main.py sweep

Analyze training results:
    uv run main.py analyze

Evaluate the best-performing models:
    uv run main.py evaluate

Launch the policy inference API:
    uv run main.py api
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_best_models() -> dict:
    """Load metadata for the best saved models."""
    models_file = Path(__file__).parent / "models" / "best_models.json"
    return json.loads(models_file.read_text())


def cmd_play(rest):
    """Launch the simulation viewer."""
    import play

    play.run(play.build_parser().parse_args(rest))


def cmd_train(rest):
    """Train a reinforcement learning agent."""

    parser = argparse.ArgumentParser(prog="main.py train")

    parser.add_argument(
        "--algo",
        required=True,
        choices=["dqn", "reinforce", "ppo", "a2c"],
        help="Training algorithm."
    )

    parser.add_argument(
        "--run",
        default=None,
        help="Custom run name."
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=250_000,
        help="Number of environment interaction frames."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed."
    )

    parser.add_argument(
        "--config",
        default=None,
        help='JSON hyperparameter overrides, e.g. \'{"lr": 1e-3}\''
    )

    args = parser.parse_args(rest)

    # Optional hyperparameter overrides
    overrides = json.loads(args.config) if args.config else {}

    if args.algo == "dqn":
        from training.dqn_training import train

        train(
            args.run or "dqn_manual",
            args.frames,
            args.seed,
            **overrides,
        )
    else:
        from training.pg_training import train

        train(
            args.algo,
            args.run or f"{args.algo}_manual",
            args.frames,
            args.seed,
            **overrides,
        )


def cmd_sweep(rest):
    """Run the complete hyperparameter sweep."""
    from training import sweep

    sys.argv = ["sweep"] + rest
    sweep.main()


def cmd_analyze(rest):
    """Analyze experiment logs and generate reports."""
    from training import analysis

    analysis.main()


def cmd_evaluate(rest):
    """Evaluate the best models on training and unseen hospitals."""

    from training.common import (
        EVAL_SEEDS_TEST,
        EVAL_SEEDS_TRAIN,
        evaluate,
        load_policy,
    )

    best_models = load_best_models()

    print(
        f"{'algorithm':<10s} "
        f"{'run':<28s} "
        f"{'train reward':>12s} "
        f"{'success':>8s} "
        f"{'unseen reward':>14s} "
        f"{'success':>8s}"
    )

    for algo, entry in best_models["algorithms"].items():

        policy, _ = load_policy(entry["model_path"])

        train_metrics = evaluate(
            policy,
            EVAL_SEEDS_TRAIN,
            "train",
        )

        test_metrics = evaluate(
            policy,
            EVAL_SEEDS_TEST,
            "test",
        )

        print(
            f"{algo:<10s} "
            f"{entry['run']:<28s} "
            f"{train_metrics['mean_return']:>12.1f} "
            f"{train_metrics['success_rate']:>8.2f} "
            f"{test_metrics['mean_return']:>14.1f} "
            f"{test_metrics['success_rate']:>8.2f}"
        )


def cmd_api(rest):
    """Serve a trained policy through a simple API."""

    import time

    from environment.rendering import get_viewer
    from training.common import load_policy

    parser = argparse.ArgumentParser(prog="main.py api")

    parser.add_argument(
        "--algo",
        default="best",
        help="Algorithm to serve or 'best'."
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="API server port."
    )

    args = parser.parse_args(rest)

    best_models = load_best_models()

    selected_algo = (
        best_models["overall"]
        if args.algo == "best"
        else args.algo
    )

    viewer = get_viewer(port=args.port)

    policy, _ = load_policy(
        best_models["algorithms"][selected_algo]["model_path"]
    )

    viewer.policy = policy

    print(
        f"Serving {selected_algo} policy at "
        f"{viewer.url}api/act "
        f"(health: {viewer.url}api/health). "
        f"Ctrl+C to stop."
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


COMMANDS = {
    "play": cmd_play,
    "train": cmd_train,
    "sweep": cmd_sweep,
    "analyze": cmd_analyze,
    "evaluate": cmd_evaluate,
    "api": cmd_api,
}


def main():
    """Dispatch command-line commands."""

    cli_args = sys.argv[1:]

    if cli_args and cli_args[0] in {"-h", "--help"}:
        print(__doc__)
        return

    # Default behavior: showcase the best trained agent.
    if not cli_args or cli_args[0].startswith("-"):
        cmd_play(cli_args)
        return

    command = cli_args[0]

    if command not in COMMANDS:
        sys.exit(f"unknown command {command!r}\n{__doc__}")

    COMMANDS[command](cli_args[1:])


if __name__ == "__main__":
    main()