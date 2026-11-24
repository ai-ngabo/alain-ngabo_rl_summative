"""Deep Q-Network agent with TorchRL on the Ibaba drone environment.

run:
    uv run python -m training.dqn_training --run dqn_default
    uv run python -m training.dqn_training --run dqn_lr1e-3 --config '{"lr": 1e-3}'
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from environment import OBS_DIM
from training.common import (
    EVAL_SEEDS_TEST, EVAL_SEEDS_TRAIN, PROGRESS_SEEDS, RunLogger, evaluate,
    finished_episode_returns, make_env, mlp, save_policy, set_seed,
)

TOTAL_FRAMES = 250_000   # same interaction budget as the policy-gradient agents
REWARD_SCALE = 0.1
EVAL_EVERY = 25_000

DEFAULTS = dict(
    lr=5e-4, gamma=0.99, batch_size=128, buffer_size=100_000,
    eps_start=1.0, eps_end=0.05, eps_decay_frames=100_000,
    target_update="hard", target_interval=1000, tau=0.005,
    double_dqn=True, num_envs=8, frames_per_batch=64, updates_per_batch=8,
    warmup_frames=5_000, hidden=[128, 128], max_grad_norm=10.0,
)


def exploration_label(cfg: dict) -> str:
    return f"e-greedy {cfg['eps_start']}->{cfg['eps_end']} over {cfg['eps_decay_frames'] // 1000}k"


def train(run_name: str, total_frames: int = TOTAL_FRAMES, seed: int = 0, **overrides) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg.update(algo="dqn", seed=seed, total_frames=total_frames, exploration=exploration_label(cfg))
    set_seed(seed)
    torch.set_num_threads(1)

    env = make_env(cfg["num_envs"], seed=seed)
    spec = env.action_spec_unbatched
    qnet = mlp(OBS_DIM, spec.n, tuple(cfg["hidden"]))
    actor = QValueActor(qnet, in_keys=["observation"], spec=spec)
    egreedy = EGreedyModule(spec, eps_init=cfg["eps_start"], eps_end=cfg["eps_end"],
                            annealing_num_steps=cfg["eps_decay_frames"])
    policy = TensorDictSequential(actor, egreedy)
    collector = Collector(env, policy, frames_per_batch=cfg["frames_per_batch"], total_frames=total_frames,
                          init_random_frames=cfg["warmup_frames"])
    rb = ReplayBuffer(storage=LazyTensorStorage(cfg["buffer_size"]), batch_size=cfg["batch_size"])
    loss_mod = DQNLoss(actor, action_space=spec, delay_value=True,
                       double_dqn=cfg["double_dqn"], loss_function="smooth_l1")
    loss_mod.make_value_estimator(gamma=cfg["gamma"])
    if cfg["target_update"] == "hard":
        updater = HardUpdate(loss_mod, value_network_update_interval=cfg["target_interval"])
    else:
        updater = SoftUpdate(loss_mod, tau=cfg["tau"])
    optim = torch.optim.Adam(loss_mod.parameters(), lr=cfg["lr"])

    log = RunLogger("dqn", run_name, cfg)
    frames, next_eval = 0, EVAL_EVERY
    best = (-1e9, copy.deepcopy(qnet.state_dict()))
    loss_acc, q_acc = [], []

    for data in collector:
        frames += data.numel()
        log.add_episodes(frames, finished_episode_returns(data))
        data["next", "reward"] = data["next", "reward"] * REWARD_SCALE
        rb.extend(data.reshape(-1))
        egreedy.step(data.numel())

        if frames >= cfg["warmup_frames"]:
            for _ in range(cfg["updates_per_batch"]):
                batch = rb.sample()
                out = loss_mod(batch)
                optim.zero_grad()
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), cfg["max_grad_norm"])
                optim.step()
                updater.step()
                loss_acc.append(out["loss"].item())
            with torch.no_grad():
                q_acc.append(qnet(batch["observation"]).max(-1).values.mean().item())
        if loss_acc and (not log.updates or frames - log.updates[-1]["frames"] >= 2048):
            log.add_update(frames, loss=sum(loss_acc) / len(loss_acc), mean_max_q=sum(q_acc) / max(len(q_acc), 1),
                           epsilon=egreedy.eps.item())
            loss_acc, q_acc = [], []

        if frames >= next_eval or frames >= total_frames:
            next_eval += EVAL_EVERY
            res = evaluate(qnet, PROGRESS_SEEDS, "train")
            log.add_progress(frames, res)
            if res["mean_return"] > best[0]:
                best = (res["mean_return"], copy.deepcopy(qnet.state_dict()))
            print(f"[dqn:{run_name}] frames={frames:>7d} eps={len(log.episodes):>5d} "
                  f"epsilon={egreedy.eps.item():.2f} eval_return={res['mean_return']:7.1f} "
                  f"success={res['success_rate']:.2f}", flush=True)

    collector.shutdown()
    qnet.load_state_dict(best[1])
    final_train = evaluate(qnet, EVAL_SEEDS_TRAIN, "train")
    final_test = evaluate(qnet, EVAL_SEEDS_TEST, "test")
    model_path = save_policy(qnet, "dqn", run_name, cfg, "dqn")
    path = log.save(final_train, final_test, model_path)
    print(f"[dqn:{run_name}] FINAL train={final_train['mean_return']:.1f} "
          f"(success {final_train['success_rate']:.2f})  held-out={final_test['mean_return']:.1f} "
          f"(success {final_test['success_rate']:.2f})  -> {path.name}", flush=True)
    return {"train": final_train, "test": final_test, "log": str(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="dqn_default")
    p.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=str, default=None, help='JSON overrides, e.g. \'{"lr": 1e-3}\'')
    a = p.parse_args()
    train(a.run, a.frames, a.seed, **(json.loads(a.config) if a.config else {}))


if __name__ == "__main__":
    main()
