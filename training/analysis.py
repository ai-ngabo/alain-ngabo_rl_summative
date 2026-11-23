"""Turn the sweep logs into report tables, figures and the best-model index.

    uv run python -m training.analysis

Outputs
    logs/summary_<algo>.csv          one row per run (hyperparameters + results)
    logs/baselines.json              non-learning baselines on the same evaluation missions
    models/best_models.json          best run per algorithm (used by play.py / main.py)
    assets/figures/*.png             figures used in the report
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
FIG = ROOT / "assets" / "figures"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLORS = {"dqn": "#2a78b5", "reinforce": "#d9822b", "ppo": "#2e9d5b", "a2c": "#8e5bb5", "heuristic": "#7a7a7a"}
SMOOTH = 50


def load_runs(algo: str, prefix: str | None = None) -> list[dict]:
    """Sweep runs (prefix=None), or extra runs whose name starts with `prefix` ("seed", "final")."""
    runs = [json.loads(p.read_text()) for p in sorted((LOGS / algo).glob("*.json"))]
    if prefix is None:
        return [r for r in runs if not r["run"].startswith(("smoke", "seed", "final"))]
    return [r for r in runs if r["run"].startswith(prefix)]


def smooth(x, w=SMOOTH):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return x
    w = min(w, len(x))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / w
    head = np.array([x[: i + 1].mean() for i in range(w - 1)])
    return np.concatenate([head, out])


def episodes_to_converge(returns, hold: int = 100) -> int | None:
    """First episode where the 50-episode average reaches its final plateau (within 10% of
    the curve's range) and stays there for `hold` episodes."""
    sm = smooth(returns)
    if len(sm) < 2 * hold:
        return None
    plateau = sm[-max(hold, len(sm) // 5):].mean()
    thr = plateau - 0.1 * (sm.max() - sm.min())
    ok = sm >= thr
    for i in range(len(sm) - hold):
        if ok[i:i + hold].all():
            return int(i)
    return None


# ---------------------------------------------------------------- table columns per algorithm
def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return "x".join(str(i) for i in v)
    return str(v)


COLUMNS = {
    "dqn": [("Learning rate", "lr"), ("Gamma", "gamma"), ("Replay buffer", "buffer_size"),
            ("Batch size", "batch_size"), ("Exploration", "exploration"), ("Target update", None),
            ("Double DQN", "double_dqn"), ("Network", "hidden")],
    "reinforce": [("Learning rate", "lr"), ("Gamma", "gamma"), ("Baseline", "baseline"),
                  ("Advantage norm", "adv_norm"), ("Entropy coef", "entropy"),
                  ("Episodes/update", "num_envs"), ("Network", "hidden")],
    "ppo": [("Learning rate", "lr"), ("Gamma", "gamma"), ("Clip", "clip"), ("GAE lambda", "lmbda"),
            ("Epochs", "epochs"), ("Steps/batch", "frames_per_batch"), ("Minibatch", "minibatch"),
            ("Entropy coef", "entropy")],
    "a2c": [("Learning rate", "lr"), ("Gamma", "gamma"), ("Advantage (lambda)", "lmbda"),
            ("Rollout steps", "rollout"), ("Entropy coef", "entropy"), ("Critic coef", "critic_coeff")],
}


def run_row(algo: str, r: dict) -> dict:
    cfg = r["config"]
    row = {"Run": r["run"]}
    for name, key in COLUMNS[algo]:
        if key is None:
            row[name] = (f"hard every {cfg['target_interval']}" if cfg["target_update"] == "hard"
                         else f"soft tau={cfg['tau']}")
        elif algo == "a2c" and key == "lmbda":
            lam = cfg["lmbda"]
            row[name] = "n-step (1.0)" if lam == 1.0 else ("TD(0)" if lam == 0 else f"GAE {lam}")
        else:
            row[name] = _fmt(cfg[key])
    tr, te = r["final_eval_train"], r["final_eval_test"]
    rets = [e[1] for e in r["episodes"]]
    conv = episodes_to_converge(rets)
    row.update({
        "Mean reward": f"{tr['mean_return']:.1f}",
        "Success %": f"{100 * tr['success_rate']:.0f}",
        "On-target %": f"{100 * tr['on_target_rate']:.0f}",
        "Unseen reward": f"{te['mean_return']:.1f}",
        "Unseen success %": f"{100 * te['success_rate']:.0f}",
        "Last-100 train reward": f"{np.mean(rets[-100:]):.1f}",
        "Episodes to converge": str(conv) if conv is not None else f">{len(rets)}",
        "Train episodes": str(len(rets)),
        "Minutes": f"{r['wall_seconds'] / 60:.1f}",
    })
    return row


# ---------------------------------------------------------------- baselines
def baselines() -> dict:
    from environment import IbabaDroneEnv, heuristic_action
    from training.common import EVAL_SEEDS_TEST, EVAL_SEEDS_TRAIN

    def run(policy, seeds, split):
        env = IbabaDroneEnv(split)
        rows = []
        for s in seeds:
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                obs, _, te, tr, info = env.step(policy(env))
                done = te or tr
            rows.append(info)
        return {"mean_return": float(np.mean([i["total_reward"] for i in rows])),
                "success_rate": float(np.mean([i["outcome"] == "home" for i in rows])),
                "on_target_rate": float(np.mean([i["drop_result"] == "on_target" for i in rows])),
                "battery_fail_rate": float(np.mean([i["outcome"] == "battery" for i in rows]))}

    def fixed_speed(out_speed, back_speed):
        """Autopilot that always flies at one speed outbound and one speed home."""
        def pol(env):
            a = heuristic_action(env)
            if a in (0, 5, 6):
                wp = env.waypoints[env.wp_i]
                want = out_speed if env.payload_on else back_speed
                if wp.kind == "drop" and np.linalg.norm(wp.xy - env.pos) < 3.0:
                    want = 0
                if env.speed_i > want:
                    return 6
                if env.speed_i < want:
                    return 5
                return 0
            return a
        return pol

    rng = np.random.default_rng(0)
    out = {
        "heuristic_cruise": {"train": run(heuristic_action, EVAL_SEEDS_TRAIN, "train"),
                             "test": run(heuristic_action, EVAL_SEEDS_TEST, "test")},
        "random": {"train": run(lambda e: int(rng.integers(8)), EVAL_SEEDS_TRAIN, "train"),
                   "test": run(lambda e: int(rng.integers(8)), EVAL_SEEDS_TEST, "test")},
        "fixed_speed": {name: run(fixed_speed(o, b), EVAL_SEEDS_TRAIN, "train") for name, (o, b) in {
            "slow out / slow back": (0, 0), "cruise out / cruise back": (1, 1), "fast out / fast back": (2, 2),
            "fast out / slow back": (2, 0), "cruise out / slow back": (1, 0)}.items()},
    }
    (LOGS / "baselines.json").write_text(json.dumps(out, indent=2))
    return out


def behaviour(best: dict) -> dict:
    """How each best agent actually flies on the 30 training-hospital evaluation missions."""
    import torch
    from environment import IbabaDroneEnv, heuristic_action
    from training.common import EVAL_SEEDS_TRAIN, load_policy

    def stats(policy):
        env = IbabaDroneEnv("train")
        speed = {True: [0, 0, 0], False: [0, 0, 0]}
        rain_steps = steps = 0
        mins, batt = [], []
        for s in EVAL_SEEDS_TRAIN:
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                a = policy(env, obs)
                obs, _, te, tr, info = env.step(a)
                done = te or tr
                speed[env.payload_on][env.speed_i] += 1
                rain_steps += env._rain_edge_distance()[0] < 0
                steps += 1
            if info["delivery_minutes"]:
                mins.append(info["delivery_minutes"])
            if info["outcome"] == "home":
                batt.append(info["battery"])
        frac = {k: [round(v / max(sum(c), 1), 3) for v in c] for k, c in speed.items()}
        return {"speed_share_with_blood": frac[True], "speed_share_returning": frac[False],
                "rain_step_share": round(rain_steps / steps, 4),
                "mean_delivery_min": round(float(np.mean(mins)), 1) if mins else None,
                "mean_battery_left_home": round(float(np.mean(batt)), 3) if batt else None}

    out = {"heuristic": stats(lambda env, obs: heuristic_action(env))}
    for algo, r in best.items():
        net, _ = load_policy(r["model_path"])
        out[algo] = stats(lambda env, obs, net=net: int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(-1)))
    (LOGS / "behaviour.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------- figures
def figures(all_runs: dict, best: dict, base: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})
    FIG.mkdir(parents=True, exist_ok=True)

    # 1) cumulative reward per episode, best run of each algorithm (all runs faint)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=True)
    for ax, algo in zip(axes.ravel(), ALGOS):
        b = best[algo]
        rets = [e[1] for e in b["episodes"]]
        ax.plot(rets, color=COLORS[algo], alpha=0.18, lw=0.5)
        ax.plot(smooth(rets), color=COLORS[algo], lw=2, label=f"best: {b['run']}")
        ax.axhline(base["heuristic_cruise"]["train"]["mean_return"], color="#555", ls="--", lw=1,
                   label="hand-written autopilot")
        ax.set_title(NAMES[algo])
        ax.set_xlabel("training episode")
        ax.set_ylabel("episode reward")
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Cumulative reward per training episode, best model of each algorithm (faint = raw, bold = 50-episode average)")
    fig.tight_layout()
    fig.savefig(FIG / "cumulative_rewards.png")
    plt.close(fig)

    # 2) training stability: DQN objective, policy entropy
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    d = best["dqn"]["updates"]
    axes[0].plot([u["frames"] / 1000 for u in d], [u["loss"] for u in d], color=COLORS["dqn"])
    axes[0].set_title("DQN objective (TD loss), best run")
    axes[0].set_xlabel("frames (thousands)")
    axes[0].set_ylabel("smooth-L1 TD loss")
    ax2 = axes[1]
    ax2.plot([u["frames"] / 1000 for u in d], [u["mean_max_q"] for u in d], color=COLORS["dqn"])
    ax2.set_title("DQN mean max Q-value")
    ax2.set_xlabel("frames (thousands)")
    for algo in ("reinforce", "ppo", "a2c"):
        u = best[algo]["updates"]
        axes[2].plot([x["frames"] / 1000 for x in u], smooth([x["entropy"] for x in u], 5),
                     color=COLORS[algo], label=NAMES[algo])
    axes[2].axhline(np.log(8), color="#999", ls=":", lw=1, label="uniform (ln 8)")
    axes[2].set_title("Policy entropy, best runs")
    axes[2].set_xlabel("frames (thousands)")
    axes[2].set_ylabel("entropy (nats)")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "training_stability.png")
    plt.close(fig)

    # 3) convergence: smoothed curves on one axis + episodes-to-converge markers
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    convs = {}
    for algo in ALGOS:
        rets = [e[1] for e in best[algo]["episodes"]]
        sm = smooth(rets)
        c = episodes_to_converge(rets)
        convs[algo] = c
        axes[0].plot(sm, color=COLORS[algo], label=NAMES[algo])
        if c is not None:
            axes[0].plot(c, sm[c], "o", color=COLORS[algo], ms=6, mec="black")
    axes[0].set_xlabel("training episode")
    axes[0].set_ylabel("50-episode average reward")
    axes[0].set_title("Convergence of the best run per algorithm (dot = converged)")
    axes[0].legend(fontsize=7)
    for algo in ALGOS:
        frames = [p["frames"] / 1000 for p in best[algo]["progress"]]
        axes[1].plot(frames, [p["success_rate"] for p in best[algo]["progress"]], "-o", ms=3,
                     color=COLORS[algo], label=NAMES[algo])
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("frames (thousands)")
    axes[1].set_ylabel("success rate (10 check missions)")
    axes[1].set_title("Greedy policy checks during training")
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png")
    plt.close(fig)

    # 4) generalization: training hospitals vs unseen hospitals
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    labels = [NAMES[a] for a in ALGOS] + ["Autopilot"]
    tr = [best[a]["final_eval_train"] for a in ALGOS] + [base["heuristic_cruise"]["train"]]
    te = [best[a]["final_eval_test"] for a in ALGOS] + [base["heuristic_cruise"]["test"]]
    x = np.arange(len(labels))
    for ax, key, title in ((axes[0], "mean_return", "Mean reward"), (axes[1], "success_rate", "Mission success rate")):
        ax.bar(x - 0.2, [t[key] for t in tr], 0.4, label="training hospitals (30 missions)", color="#4c78a8")
        ax.bar(x + 0.2, [t[key] for t in te], 0.4, label="unseen hospitals (30 missions)", color="#f58518")
        ax.set_xticks(x, labels)
        ax.set_title(title)
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Generalization: 30 unseen start states each, including two hospitals never seen in training")
    fig.tight_layout()
    fig.savefig(FIG / "generalization.png")
    plt.close(fig)

    # 5) sweep overview
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, algo in zip(axes, ALGOS):
        runs = all_runs[algo]
        vals = [r["final_eval_train"]["mean_return"] for r in runs]
        names = [r["run"].split("_", 2)[-1] for r in runs]
        top = max(vals)
        ax.barh(range(len(runs)), vals, color=[COLORS[algo] if v == top else "#c7c7c7" for v in vals])
        ax.set_yticks(range(len(runs)), names, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(base["heuristic_cruise"]["train"]["mean_return"], color="#555", ls="--", lw=1)
        ax.set_title(f"{NAMES[algo]} sweep")
        ax.set_xlabel("mean eval reward")
    fig.tight_layout()
    fig.savefig(FIG / "sweep_overview.png")
    plt.close(fig)
    (LOGS / "convergence.json").write_text(json.dumps(convs))


def main():
    all_runs = {a: load_runs(a) for a in ALGOS}
    sweep_best, best, robust = {}, {}, {}
    for algo in ALGOS:
        runs = all_runs[algo]
        if not runs:
            raise SystemExit(f"no runs for {algo} yet")
        rows = [run_row(algo, r) for r in runs]
        with (LOGS / f"summary_{algo}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        sb = max(runs, key=lambda r: r["final_eval_train"]["mean_return"])
        sweep_best[algo] = sb
        finals = [r for r in load_runs(algo, "final") if r["run"] == f"final_{sb['run']}"]
        best[algo] = finals[0] if finals else sb
        seeds = [sb] + [r for r in load_runs(algo, "seed") if r["run"].endswith(sb["run"])]
        robust[algo] = {
            "config_run": sb["run"], "seeds": [r["config"]["seed"] for r in seeds],
            "train_returns": [round(r["final_eval_train"]["mean_return"], 1) for r in seeds],
            "train_success": [r["final_eval_train"]["success_rate"] for r in seeds],
            "unseen_returns": [round(r["final_eval_test"]["mean_return"], 1) for r in seeds],
            "episodes_to_converge": [episodes_to_converge([e[1] for e in r["episodes"]]) for r in seeds],
        }
        b = best[algo]
        print(f"{NAMES[algo]:<10s} sweep best={sb['run']:<26s} {sb['final_eval_train']['mean_return']:.1f} | "
              f"model={b['run']:<32s} train={b['final_eval_train']['mean_return']:.1f} "
              f"success={b['final_eval_train']['success_rate']:.2f} unseen={b['final_eval_test']['mean_return']:.1f} "
              f"seeds={robust[algo]['train_returns']}")
    (LOGS / "robustness.json").write_text(json.dumps(robust, indent=2))
    # overall best: average of training-hospital and unseen-mission reward (the demo includes an unseen hospital)
    overall = max(ALGOS, key=lambda a: best[a]["final_eval_train"]["mean_return"] + best[a]["final_eval_test"]["mean_return"])
    index = {"overall": overall, "algorithms": {
        a: {"run": best[a]["run"], "model_path": best[a]["model_path"],
            "frames": best[a]["config"]["total_frames"],
            "train_mean_reward": best[a]["final_eval_train"]["mean_return"],
            "train_success": best[a]["final_eval_train"]["success_rate"],
            "unseen_mean_reward": best[a]["final_eval_test"]["mean_return"],
            "unseen_success": best[a]["final_eval_test"]["success_rate"]} for a in ALGOS}}
    (ROOT / "models" / "best_models.json").write_text(json.dumps(index, indent=2))
    final_rows = []
    for a in ALGOS:
        b = best[a]
        rets = [e[1] for e in b["episodes"]]
        final_rows.append({"algo": NAMES[a], "run": b["run"], "frames": b["config"]["total_frames"],
                           "episodes": len(rets), "episodes_to_converge": episodes_to_converge(rets),
                           "train": {k: v for k, v in b["final_eval_train"].items() if k != "returns"},
                           "unseen": {k: v for k, v in b["final_eval_test"].items() if k != "returns"},
                           "wall_minutes": round(b["wall_seconds"] / 60, 1)})
    (LOGS / "final_models.json").write_text(json.dumps(final_rows, indent=2))
    base = baselines()
    print("autopilot baseline:", {k: round(v["mean_return"], 1) for k, v in base["heuristic_cruise"].items()})
    beh = behaviour(best)
    print("behaviour:", json.dumps(beh))
    figures(all_runs, best, base)
    print(f"overall best: {NAMES[overall]} -> models/best_models.json; figures in {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
