"""Basic checks for the drone environment. 
   usage: uv run python -m pytest tests -q
"""

import numpy as np
from gymnasium.utils.env_checker import check_env

from environment import ACTIONS, OBS_DIM, IbabaDroneEnv, heuristic_action
from environment.world import AIRPORT_NO_FLY_KM, AIRPORT_XY


def test_gymnasium_api():
    env = IbabaDroneEnv()
    check_env(env, skip_render_check=True)
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.n == len(ACTIONS) == 8


def test_reset_is_seeded():
    a, _ = IbabaDroneEnv().reset(seed=3)
    b, _ = IbabaDroneEnv().reset(seed=3)
    assert np.allclose(a, b)


def test_corridor_avoids_airport():
    env = IbabaDroneEnv("all")
    for s in range(40):
        env.reset(seed=s)
        for w in env.waypoints:
            if w.kind == "ring":
                assert np.linalg.norm(w.xy - np.array(AIRPORT_XY)) > AIRPORT_NO_FLY_KM


def test_release_interlock_blocks_far_drops():
    env = IbabaDroneEnv()
    env.reset(seed=0)
    _, r, *_ = env.step(7) # at the hub, far from any hospital
    assert env.payload_on and r < 0


def test_autopilot_completes_missions():
    env = IbabaDroneEnv("all")
    ok = 0
    for s in range(20):
        env.reset(seed=500 + s)
        done = False
        while not done:
            _, _, te, tr, info = env.step(heuristic_action(env))
            done = te or tr
        ok += info["outcome"] == "home" and info["delivered"]
    assert ok >= 12


def test_battery_and_payload_persist():
    env = IbabaDroneEnv()
    env.reset(seed=1)
    b0, t0 = env.battery, env.temp
    for _ in range(10):
        env.step(5) # speed up - drains faster
    assert env.battery < b0 and env.temp > t0
