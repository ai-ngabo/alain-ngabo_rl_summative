"""Policy-gradient agents (REINFORCE, PPO, A2C) with TorchRL on the Ibaba drone environment.

run:
    uv run python -m training.pg_training --algo ppo --run ppo_default
    uv run python -m training.pg_training --algo a2c --run a2c_lr1e-3 --lr 1e-3
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import Collector
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

from training.common import (
    EVAL_SEEDS_TEST, EVAL_SEEDS_TRAIN, PROGRESS_SEEDS, RunLogger, evaluate,
    finished_episode_returns, make_env, mlp, save_policy, set_seed,
)
from environment import OBS_DIM

TOTAL_FRAMES = 250_000      # same interaction budget for every algorithm and run
REWARD_SCALE = 0.1          # rewards are scaled for the losses only; logs keep real rewards
EVAL_EVERY = 25_000

DEFAULTS = {
    "reinforce": dict(lr=3e-3, gamma=0.99, baseline=True, adv_norm="meanstd", entropy=0.01,
                      critic_coeff=0.5, num_envs=4, hidden=[128, 128], max_grad_norm=1.0),
    "ppo": dict(lr=3e-4, gamma=0.99, lmbda=0.95, clip=0.2, epochs=4, frames_per_batch=2048,
                minibatch=256, entropy=0.01, critic_coeff=0.5, num_envs=8, hidden=[128, 128],
                max_grad_norm=0.5),
    "a2c": dict(lr=7e-4, gamma=0.99, lmbda=1.0, rollout=16, entropy=0.01, critic_coeff=0.5,
                num_envs=8, hidden=[128, 128], max_grad_norm=0.5),
}


def _discounted_returns(reward: torch.Tensor, done: torch.Tensor, gamma: float):
    """Monte-Carlo returns per env row; also flags steps whose episode finished in the batch."""
    n_env, T = reward.shape
    G = torch.zeros_like(reward)
    complete = torch.zeros_like(done)
    running = torch.zeros(n_env)
    seen = torch.zeros(n_env, dtype=torch.bool)
    for t in range(T - 1, -1, -1):
        running = reward[:, t] + gamma * running * (~done[:, t])
        seen = seen | done[:, t]
        G[:, t] = running
        complete[:, t] = seen
    return G, complete


def _entropy_term(out) -> torch.Tensor:
    """TorchRL omits loss_entropy when the entropy coefficient is 0."""
    return out["loss_entropy"] if "loss_entropy" in out.keys() else torch.zeros(())


def _entropy(out, actor, td) -> float:
    if "entropy" in out.keys():
        return out["entropy"].item()
    with torch.no_grad():
        return actor.get_dist(td).entropy().mean().item()


def train(algo: str, run_name: str, total_frames: int = TOTAL_FRAMES, seed: int = 0, **overrides) -> dict:
    cfg = copy.deepcopy(DEFAULTS[algo])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg.update(algo=algo, seed=seed, total_frames=total_frames)
    set_seed(seed)
    torch.set_num_threads(1)

    env = make_env(cfg["num_envs"], seed=seed)
    net = mlp(OBS_DIM, env.action_spec.n, tuple(cfg["hidden"]))
    policy_module = TensorDictModule(net, in_keys=["observation"], out_keys=["logits"])
    actor = ProbabilisticActor(policy_module, in_keys=["logits"], distribution_class=Categorical,
                               return_log_prob=True, spec=env.action_spec)
    critic = ValueOperator(mlp(OBS_DIM, 1, tuple(cfg["hidden"])), in_keys=["observation"])

    if algo == "ppo":
        fpb = cfg["frames_per_batch"]
        loss_mod = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip"], entropy_bonus=True,
                               entropy_coeff=cfg["entropy"], critic_coeff=cfg["critic_coeff"],
                               loss_critic_type="smooth_l1", normalize_advantage=True)
        adv_mod = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic, average_gae=False)
        collector = Collector(env, actor, frames_per_batch=fpb, total_frames=total_frames)
    elif algo == "a2c":
        fpb = cfg["num_envs"] * cfg["rollout"]
        loss_mod = A2CLoss(actor, critic, entropy_bonus=True, entropy_coeff=cfg["entropy"],
                           critic_coeff=cfg["critic_coeff"], loss_critic_type="smooth_l1")
        adv_mod = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic, average_gae=False)
        collector = Collector(env, actor, frames_per_batch=fpb, total_frames=total_frames)
    else:  # reinforce: whole episodes, each batch starts from fresh resets
        fpb = cfg["num_envs"] * 300
        loss_mod = ReinforceLoss(actor, critic, functional=False, loss_critic_type="smooth_l1")
        adv_mod = None
        collector = Collector(env, actor, frames_per_batch=fpb, total_frames=total_frames,
                              reset_at_each_iter=True)

    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=cfg["lr"])
    log = RunLogger(algo, run_name, cfg)
    frames, next_eval = 0, EVAL_EVERY
    best = (-1e9, copy.deepcopy(net.state_dict()))

    for data in collector:
        frames += data.numel()
        log.add_episodes(frames, finished_episode_returns(data))
        data["next", "reward"] = data["next", "reward"] * REWARD_SCALE
        params = list(actor.parameters()) + list(critic.parameters())

        if algo == "ppo":
            stats = []
            for _ in range(cfg["epochs"]):
                with torch.no_grad():
                    adv_mod(data)
                flat = data.reshape(-1)
                for idx in torch.randperm(flat.numel()).split(cfg["minibatch"]):
                    out = loss_mod(flat[idx])
                    loss = out["loss_objective"] + out["loss_critic"] + _entropy_term(out)
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                    optim.step()
                    stats.append([out["loss_objective"].item(), out["loss_critic"].item(),
                                  _entropy(out, actor, flat[idx]), out.get("clip_fraction", torch.tensor(float("nan"))).item()])
            s = torch.tensor(stats).nanmean(0) if stats else torch.full((4,), float("nan"))
            log.add_update(frames, loss_policy=s[0].item(), loss_value=s[1].item(), entropy=s[2].item(),
                           clip_fraction=s[3].item())
        elif algo == "a2c":
            with torch.no_grad():
                adv_mod(data)
            out = loss_mod(data.reshape(-1))
            loss = out["loss_objective"] + out["loss_critic"] + _entropy_term(out)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            optim.step()
            if len(log.updates) == 0 or frames - log.updates[-1]["frames"] >= 2048:
                log.add_update(frames, loss_policy=out["loss_objective"].item(),
                               loss_value=out["loss_critic"].item(), entropy=_entropy(out, actor, data.reshape(-1)))
        else:
            reward = data["next", "reward"].squeeze(-1)
            done = data["next", "done"].squeeze(-1)
            G, complete = _discounted_returns(reward, done, cfg["gamma"])
            if complete.any():
                sel = data[complete]
                g = G[complete].unsqueeze(-1)
                sel.set("value_target", g)
                if cfg["baseline"]:
                    with torch.no_grad():
                        adv = g - critic(sel.select("observation"))["state_value"]
                else:
                    adv = g.clone()
                if cfg["adv_norm"] == "meanstd":
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                elif cfg["adv_norm"] == "std":
                    adv = adv / (adv.std() + 1e-8)
                sel.set("advantage", adv)
                out = loss_mod(sel)
                ent = actor.get_dist(sel).entropy().mean()
                loss = out["loss_actor"] - cfg["entropy"] * ent
                if cfg["baseline"]:
                    loss = loss + cfg["critic_coeff"] * out["loss_value"]
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                optim.step()
                log.add_update(frames, loss_policy=out["loss_actor"].item(),
                               loss_value=out["loss_value"].item() if cfg["baseline"] else None,
                               entropy=ent.item())

        if frames >= next_eval or frames >= total_frames:
            next_eval += EVAL_EVERY
            res = evaluate(net, PROGRESS_SEEDS, "train")
            log.add_progress(frames, res)
            if res["mean_return"] > best[0]:
                best = (res["mean_return"], copy.deepcopy(net.state_dict()))
            print(f"[{algo}:{run_name}] frames={frames:>7d} eps={len(log.episodes):>5d} "
                  f"eval_return={res['mean_return']:7.1f} success={res['success_rate']:.2f}", flush=True)

    collector.shutdown()
    net.load_state_dict(best[1])
    final_train = evaluate(net, EVAL_SEEDS_TRAIN, "train")
    final_test = evaluate(net, EVAL_SEEDS_TEST, "test")
    model_path = save_policy(net, algo, run_name, cfg, f"pg/{algo}")
    path = log.save(final_train, final_test, model_path)
    print(f"[{algo}:{run_name}] FINAL train={final_train['mean_return']:.1f} "
          f"(success {final_train['success_rate']:.2f})  held-out={final_test['mean_return']:.1f} "
          f"(success {final_test['success_rate']:.2f})  -> {path.name}", flush=True)
    return {"train": final_train, "test": final_test, "log": str(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=list(DEFAULTS), required=True)
    p.add_argument("--run", default=None, help="run name (default: <algo>_default)")
    p.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", type=str, default=None, help='JSON overrides, e.g. \'{"lr": 1e-3}\'')
    a = p.parse_args()
    overrides = json.loads(a.config) if a.config else {}
    train(a.algo, a.run or f"{a.algo}_default", a.frames, a.seed, **overrides)


if __name__ == "__main__":
    main()
