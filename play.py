"""Run a trained agent in the 3D Rwanda simulation (GUI in the browser + verbose terminal log).

    uv run play.py                          # best agent overall, 3-mission tour (incl. an unseen hospital)
    uv run play.py --algo dqn               # best DQN run
    uv run play.py --hospital "Byumba District Hospital"
    uv run play.py --split test             # hospitals the agent never trained on
    uv run play.py --record assets/replays/demo.json   # also save a replay file
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
import torch

from environment import ACTIONS, IbabaDroneEnv, heuristic_action
from environment.custom_env import DT_S
from environment.world import HOSPITALS, scene_description, get_terrain, settlements

ROOT = Path(__file__).resolve().parent
BEST_FILE = ROOT / "models" / "best_models.json"
TOUR = ["Rwamagana Provincial Hospital", "Byumba District Hospital", "Gahini District Hospital"]
ALGO_NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C", "heuristic": "Hand-written autopilot"}

GREEN, RED, CYAN, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[36m", "\033[2m", "\033[1m", "\033[0m"
if sys.platform == "win32":
    import os
    os.system("")            # enable ANSI colours in the Windows console


def resolve_agent(algo: str):
    """Return (policy_fn, label). Uses models/best_models.json written by the analysis step."""
    if algo == "heuristic":
        return (lambda env, obs: heuristic_action(env)), ALGO_NAMES["heuristic"]
    from training.common import load_policy
    if not BEST_FILE.exists():
        sys.exit("No trained models found. Train first: uv run main.py sweep  (or uv run main.py train --algo ppo)")
    best = json.loads(BEST_FILE.read_text())
    key = best["overall"] if algo == "best" else algo
    entry = best["algorithms"][key]
    net, _ = load_policy(entry["model_path"])

    def policy(env, obs):
        with torch.no_grad():
            return int(net(torch.as_tensor(obs).unsqueeze(0)).argmax(-1))

    return policy, f"{ALGO_NAMES[key]} ({entry['run']})"


def phase(env) -> str:
    w = env.waypoints[env.wp_i]
    if w.kind == "ring":
        return f"{'inbound' if w.inbound else 'outbound'} ring {w.index + 1}/3"
    return "to hospital" if w.kind == "drop" else "returning home"


def compass(vx: float, vy: float) -> str:
    # direction the wind blows FROM
    ang = (math.degrees(math.atan2(-vx, -vy)) + 360) % 360
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((ang + 22.5) // 45) % 8]


def run(args):
    policy, label = resolve_agent(args.algo)
    viewer = None
    if not args.no_gui:
        from environment.rendering import get_viewer
        viewer = get_viewer(port=args.port)
        if args.algo != "heuristic":
            from training.common import load_policy
            best = json.loads(BEST_FILE.read_text())
            key = best["overall"] if args.algo == "best" else args.algo
            viewer.policy, _ = load_policy(best["algorithms"][key]["model_path"])
        print(f"{BOLD}3D viewer:{RESET} {viewer.url}")
        print(f"{BOLD}API docs:{RESET}  {viewer.url}docs   |   agent endpoint: POST {viewer.url}api/act")
        if not args.no_browser:
            webbrowser.open(viewer.url)
        print(f"{DIM}waiting for the browser to connect…{RESET}")
        viewer.wait_for_client(timeout=args.wait)
        time.sleep(1.5)

    record = {"scene": None, "messages": [], "step_ms": int(args.delay * 1000)} if args.record else None
    if record:
        sc = scene_description(get_terrain())
        sc.update(settlements())
        record["scene"] = sc

    def emit(msg):
        if viewer:
            viewer.send(msg)
        if record:
            record["messages"].append(json.loads(json.dumps(msg)))

    env = IbabaDroneEnv(hospital_split=args.split)
    names = [h.name for h in HOSPITALS if args.split == "all" or h.split == args.split]
    if args.hospital:
        names = [args.hospital]
    elif not args.random and args.split == "all":
        names = TOUR          # default demo: east (airport + lake), north (hills, tea), unseen hospital
    rows = []
    rng = np.random.default_rng(args.seed)
    for m in range(args.missions):
        hosp = names[int(rng.integers(len(names)))] if args.random else names[m % len(names)]
        obs, info = env.reset(seed=args.seed + m, options={"hospital": hosp})
        h = env.hospital
        dist = float(np.linalg.norm(h.xy))
        ws = float(np.linalg.norm(env.wind_base))
        emit({"type": "mission", "index": m + 1, "total": args.missions, "agent": label, "hospital": h.name,
              "district": h.district, "distance_km": dist, "unseen": h.split == "test"})
        print(f"\n{BOLD}{CYAN}=== Mission {m + 1}/{args.missions} · Kigali Drone Hub → {h.name} "
              f"({h.district}, {dist:.1f} km){' · UNSEEN hospital' if h.split == 'test' else ''} ==={RESET}")
        print(f"agent: {label} · battery {env.battery * 100:.0f}% · wind {ws * 3.6:.0f} km/h from "
              f"{compass(*env.wind_base)} · {len(env.rain)} rain cell(s) · blood box {env.temp:.1f} °C")
        print(f"{DIM}step  min   action            speed    AGL   batt  phase                 reward    total  events{RESET}")
        emit({"type": "frame", **env.frame()})
        time.sleep(args.delay + (args.intro if viewer else 0.0))   # launch shot over the hub before take-off
        done = False
        while not done:
            a = policy(env, obs)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            f = env.frame()
            emit({"type": "frame", **f})
            col = GREEN if r >= 0 else RED
            ev = "; ".join(f"{e} ({v:+.0f})" for e, v in env.last_events)
            if args.verbose or env.last_events or env.t % 3 == 0 or done:
                print(f"{env.t:>4d} {env.t * DT_S / 60:>5.1f}   {ACTIONS[a]:<17s} {f['speed']:<7s} {f['agl']:>4.0f}m "
                      f"{f['battery'] * 100:>4.0f}%  {phase(env):<20s} {col}{r:>+7.2f}{RESET} {info['total_reward']:>+8.1f}  "
                      f"{(GREEN if sum(v for _, v in env.last_events) >= 0 else RED) + ev + RESET if ev else ''}")
            time.sleep(args.delay)
        out = info["outcome"]
        mins = f"{info['delivery_minutes']:.0f}" if info["delivery_minutes"] else None
        rows.append({"hospital": h.name, "outcome": out, "delivered": info["drop_result"] == "on_target",
                     "minutes": mins, "battery": f"{max(info['battery'], 0) * 100:.0f}%",
                     "reward": f"{info['total_reward']:+.1f}"})
        print(f"{BOLD}{GREEN if out == 'home' else RED}→ outcome: {out} · drop: {info['drop_result']} · "
              f"delivery {mins or '–'} min · battery left {max(info['battery'], 0) * 100:.0f}% · "
              f"total reward {info['total_reward']:+.1f}{RESET}")
        time.sleep(args.pause)

    emit({"type": "summary", "title": f"{label}: {sum(r['outcome'] == 'home' for r in rows)}/{len(rows)} missions completed",
          "rows": rows})
    print(f"\n{BOLD}Summary{RESET}")
    for r in rows:
        print(f"  {r['hospital']:<34s} {r['outcome']:<9s} delivered={str(r['delivered']):<5s} "
              f"minutes={r['minutes'] or '–':<4s} battery={r['battery']:<5s} reward={r['reward']}")
    if record:
        p = ROOT / args.record
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record))
        print(f"replay saved to {p.relative_to(ROOT)}")
    if viewer and not args.exit:
        print(f"\n{DIM}Viewer still open at {viewer.url} — press Ctrl+C to quit.{RESET}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default="best", choices=["best", "dqn", "reinforce", "ppo", "a2c", "heuristic"])
    p.add_argument("--missions", type=int, default=3)
    p.add_argument("--hospital", default=None, help="exact hospital name")
    p.add_argument("--split", default="all", choices=["train", "test", "all"])
    p.add_argument("--random", action="store_true", help="random hospitals instead of the demo tour")
    p.add_argument("--seed", type=int, default=16)
    p.add_argument("--delay", type=float, default=0.3, help="seconds between steps (default tour ≈ 2 min 50 s)")
    p.add_argument("--pause", type=float, default=3.0, help="seconds between missions")
    p.add_argument("--intro", type=float, default=3.0, help="seconds of launch shot over the hub (3D viewer only)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--wait", type=float, default=25.0, help="seconds to wait for the browser")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--no-gui", action="store_true", help="terminal only")
    p.add_argument("--verbose", action="store_true", help="print every step")
    p.add_argument("--record", default=None, help="save a replay JSON (relative to the repo)")
    p.add_argument("--exit", action="store_true", help="exit when the missions finish")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
