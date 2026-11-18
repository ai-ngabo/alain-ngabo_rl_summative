"""Ibaba Life-saver: a blood-delivery drone environment for Rwanda (Gymnasium).

One episode = one mission. The drone launches from the Kigali hub, leaves the city
through three air-corridor rings (away from the airport no-fly zone), flies freely
over the hills to a real district hospital, drops the blood by parachute, then flies
home through the same corridor. Battery, wind, rain, terrain and payload temperature
carry over from step to step, so early choices (speed, detours, climbs) decide
whether the drone delivers in time and still makes it home.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .world import AIRPORT_NO_FLY_KM, AIRPORT_XY, BOUNDS, HOSPITALS, Hospital, get_terrain

# constants
ACTIONS = (
    "hold course", "turn left", "turn right", "climb",
    "descend", "speed up", "slow down", "release payload",
)
SPEED_NAMES = ("slow", "cruise", "fast")
AIRSPEEDS_MS = (16.7, 27.8, 36.1)  # 60, 100, 130 km/h (cruise ~101 km/h like real blood drones)
DT_S = 20.0  # one step = 20 seconds of flight
TURN_RAD = math.radians(30.0)
SETPOINT_STEP_M = 30.0 # climb/descend change the height-above-ground setpoint
SETPOINT_RANGE_M = (40.0, 200.0)
MAX_CLIMB_M = 60.0 # autopilot climb limit per step (3 m/s)
MAX_SINK_M = 80.0  # autopilot descent limit per step (4 m/s)
RING_OFFSETS_KM = (2.5, 5.0, 7.5)  # corridor rings, distance from hub
RING_RADIUS_KM = 0.45
RING_MISS_KM = 0.8
DROP_ZONE_KM = 0.30 # on-target landing radius
RELEASE_RANGE_KM = 1.0 # interlock: release only this close to the hospital
RECOVER_KM = 1.0  # off-target parachutes within this range are recovered late
DROP_MAX_AGL_M = 130.0  # higher release = more parachute drift
CHUTE_DESCENT_MS = 5.0
HOME_RADIUS_KM = 1.0
FINAL_APPROACH_KM = 2.0
MAX_STEPS = 300
BATTERY_PER_STEP = 0.0042 # cruise drain per step (fraction of full battery)
CLIMB_BATTERY_PER_M = 0.00002 # extra drain per metre climbed
RAIN_DRAIN_FACTOR = 1.8
TEMP_START_C = 4.0
TEMP_RISE_C_PER_MIN = 0.10
TEMP_LIMIT_C = 10.0 # red cells must stay within 2-10 C in transport (UK JPAC guidelines)
BENCHMARK_MIN = 42.0 # real average order-to-delivery time in Rwanda
OBS_DIM = 27


# reward table
@dataclass(frozen=True)
class Rewards:
    progress_per_km: float = 1.0
    ring_passed: float = 10.0
    ring_missed: float = -10.0
    delivered: float = 100.0
    delivered_spoiled: float = 20.0
    speed_bonus_max: float = 40.0
    off_target: float = 30.0 # blood recovered late by hospital staff
    failed_drop: float = -100.0 # blood lost, patient not served
    release_blocked: float = -1.0
    invalid_drop: float = -1.0
    home: float = 50.0
    home_without_delivery: float = 10.0
    home_battery_bonus: float = 20.0
    time_with_payload: float = -0.15
    time_empty: float = -0.05
    rain: float = -0.8
    altitude_band: float = -0.3
    ground_proximity: float = -1.0
    crash: float = -100.0
    battery_empty: float = -100.0
    no_fly: float = -100.0
    out_of_bounds: float = -100.0


REWARDS = Rewards()


@dataclass
class Waypoint:
    kind: str # "ring", "drop" or "home"
    xy: np.ndarray
    index: int = -1 # ring number 0..2
    inbound: bool = False
    status: str = "pending" # pending / passed / missed


@dataclass
class RainCell:
    xy: np.ndarray
    radius: float


@dataclass
class EpisodeLog:
    events: list = field(default_factory=list)
    path: list = field(default_factory=list)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class IbabaDroneEnv(gym.Env):
    """Blood-delivery drone mission from Kigali to a district hospital and back."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, hospital_split: str = "train", render_mode: str | None = None,
                 max_steps: int = MAX_STEPS, fixed_hospital: str | None = None):
        super().__init__()
        assert hospital_split in ("train", "test", "all")
        self.terrain = get_terrain()
        self.hospital_split = hospital_split
        self.fixed_hospital = fixed_hospital
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(-3.0, 3.0, shape=(OBS_DIM,), dtype=np.float32)
        self._viewer = None
        self._hospitals = [h for h in HOSPITALS if hospital_split == "all" or h.split == hospital_split]

    # setup helpers
    def _corridor(self, target: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """Pick the corridor bearing closest to the target that keeps clear of the airport."""
        base = math.atan2(target[1], target[0])
        ap = np.array(AIRPORT_XY)
        for deg in [0] + [s * d for d in range(5, 181, 5) for s in (1, -1)]:
            b = base + math.radians(deg)
            u = np.array([math.cos(b), math.sin(b)])
            rings = [u * d for d in RING_OFFSETS_KM]
            t = np.clip(ap @ u, 0.0, RING_OFFSETS_KM[-1])
            if np.linalg.norm(ap - u * t) > AIRPORT_NO_FLY_KM + 0.7:
                return b, rings
        raise RuntimeError("no clear corridor")

    def _spawn_rain(self, target: np.ndarray) -> list[RainCell]:
        cells = []
        n = int(self.np_random.integers(1, 4))
        for i in range(n):
            for _ in range(20):
                if i == 0 and self.np_random.random() < 0.7:
                    f = self.np_random.uniform(0.35, 0.75)
                    normal = np.array([-target[1], target[0]]) / (np.linalg.norm(target) + 1e-9)
                    xy = target * f + normal * self.np_random.normal(0.0, 1.5)
                else:
                    xy = np.array([self.np_random.uniform(BOUNDS[0], BOUNDS[1]),
                                   self.np_random.uniform(BOUNDS[2], BOUNDS[3])])
                r = float(self.np_random.uniform(2.5, 5.0))
                if np.linalg.norm(xy) > r + 3.0 and np.linalg.norm(xy - target) > r + 1.5:
                    cells.append(RainCell(xy=xy, radius=r))
                    break
        return cells

    # gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        name = options.get("hospital", self.fixed_hospital)
        if name is not None:
            self.hospital: Hospital = next(h for h in HOSPITALS if h.name == name)
        else:
            self.hospital = self._hospitals[int(self.np_random.integers(len(self._hospitals)))]
        target = np.array(self.hospital.xy)
        self.corridor_bearing, rings = self._corridor(target)
        self.corridor_u = np.array([math.cos(self.corridor_bearing), math.sin(self.corridor_bearing)])
        self.waypoints = (
            [Waypoint("ring", r.copy(), i) for i, r in enumerate(rings)]
            + [Waypoint("drop", target.copy())]
            + [Waypoint("ring", rings[i].copy(), i, inbound=True) for i in (2, 1, 0)]
            + [Waypoint("home", np.zeros(2))]
        )
        self.wp_i = 0

        self.pos = np.zeros(2)
        self.alt = float(self.terrain.height(0.0, 0.0)) + 60.0  # just after catapult launch
        self.setpoint = 100.0
        b0 = math.atan2(rings[0][1], rings[0][0])
        self.heading = _wrap(b0 + math.radians(self.np_random.uniform(-25, 25)))
        self.speed_i = 1
        self.battery = float(self.np_random.uniform(0.65, 1.0))  # busy hubs launch partly charged packs
        self.battery_start = self.battery
        self.payload_on = True
        self.temp = TEMP_START_C + float(self.np_random.uniform(0.0, 0.5))
        ws = float(self.np_random.uniform(0.0, 11.0))
        wd = float(self.np_random.uniform(-math.pi, math.pi))
        self.wind_base = np.array([ws * math.cos(wd), ws * math.sin(wd)])
        self.wind = self.wind_base.copy()
        self.rain = self._spawn_rain(target)
        self.t = 0
        self.delivered = False
        self.delivery_minutes = None
        self.drop_point = None
        self.drop_result = None
        self.outcome = None
        self.total_reward = 0.0
        self.last_action = None
        self.last_reward = 0.0
        self.last_events: list = []
        self.log = EpisodeLog()
        self.log.path.append(self._snapshot_point())
        return self._obs(), self._info()

    def step(self, action):
        a = int(action)
        self.t += 1
        self.last_action = a
        r = 0.0
        events: list[tuple[str, float]] = []

        r += REWARDS.time_with_payload if self.payload_on else REWARDS.time_empty

        # controls
        if a == 1:
            self.heading = _wrap(self.heading + TURN_RAD)
        elif a == 2:
            self.heading = _wrap(self.heading - TURN_RAD)
        elif a == 3:
            self.setpoint = min(SETPOINT_RANGE_M[1], self.setpoint + SETPOINT_STEP_M)
        elif a == 4:
            self.setpoint = max(SETPOINT_RANGE_M[0], self.setpoint - SETPOINT_STEP_M)
        elif a == 5:
            self.speed_i = min(2, self.speed_i + 1)
        elif a == 6:
            self.speed_i = max(0, self.speed_i - 1)
        elif a == 7:
            r += self._release(events)

        # motion (progress is measured to the waypoint active during the move)
        wp = self.waypoints[self.wp_i]
        d_prev = float(np.linalg.norm(wp.xy - self.pos))
        gust = self.np_random.normal(0.0, 0.7, 2)
        self.wind = 0.8 * self.wind + 0.2 * self.wind_base + 0.3 * gust
        air = AIRSPEEDS_MS[self.speed_i]
        vel = air * np.array([math.cos(self.heading), math.sin(self.heading)]) + self.wind
        prev = self.pos.copy()
        self.pos = self.pos + vel * DT_S / 1000.0
        d_new = float(np.linalg.norm(wp.xy - self.pos))
        r += REWARDS.progress_per_km * (d_prev - d_new)
        # terrain-following autopilot: chase ground + setpoint within climb/sink limits
        target_alt = float(self.terrain.height(self.pos[0], self.pos[1])) + self.setpoint
        climb = float(np.clip(target_alt - self.alt, -MAX_SINK_M, MAX_CLIMB_M))
        self.alt += climb

        # weather, battery, payload
        for c in self.rain:
            c.xy = c.xy + self.wind_base * 0.6 * DT_S / 1000.0
        in_rain = self._rain_edge_distance()[0] < 0.0
        rel = air / AIRSPEEDS_MS[1]
        drain = BATTERY_PER_STEP * (0.35 + 0.65 * rel ** 3) + CLIMB_BATTERY_PER_M * max(climb, 0.0)
        if in_rain:
            drain *= RAIN_DRAIN_FACTOR
            r += REWARDS.rain
        self.battery -= drain
        if self.payload_on:
            self.temp += TEMP_RISE_C_PER_MIN * DT_S / 60.0

        # waypoints (rings, home)
        r += self._update_waypoints(prev, events)

        # safety and altitude rules
        agl = self._agl()
        terminated = False
        wp = self.waypoints[self.wp_i]
        near_end = wp.kind in ("drop", "home") and np.linalg.norm(wp.xy - self.pos) < FINAL_APPROACH_KM
        if agl > 150.0 or (agl < 60.0 and not near_end):
            r += REWARDS.altitude_band
        if agl < 25.0:
            r += REWARDS.ground_proximity

        if self.outcome == "home":
            base = REWARDS.home if self.delivered else REWARDS.home_without_delivery
            bonus = base + REWARDS.home_battery_bonus * max(self.battery, 0.0)
            r += bonus
            events.append(("landed home", bonus))
            terminated = True
        elif agl <= 0.0:
            self.outcome = "crash"
            r += REWARDS.crash
            events.append(("crashed into hill", REWARDS.crash))
            terminated = True
        elif np.linalg.norm(self.pos - np.array(AIRPORT_XY)) < AIRPORT_NO_FLY_KM:
            self.outcome = "no_fly"
            r += REWARDS.no_fly
            events.append(("entered airport no-fly zone", REWARDS.no_fly))
            terminated = True
        elif self.battery <= 0.0:
            self.outcome = "battery"
            r += REWARDS.battery_empty
            events.append(("battery empty", REWARDS.battery_empty))
            terminated = True
        elif not (BOUNDS[0] <= self.pos[0] <= BOUNDS[1] and BOUNDS[2] <= self.pos[1] <= BOUNDS[3]):
            self.outcome = "out_of_bounds"
            r += REWARDS.out_of_bounds
            events.append(("left the map", REWARDS.out_of_bounds))
            terminated = True

        truncated = (not terminated) and self.t >= self.max_steps
        if truncated:
            self.outcome = "timeout"

        self.total_reward += r
        self.last_reward = r
        self.last_events = events
        self.log.events.extend((self.t, e, v) for e, v in events)
        self.log.path.append(self._snapshot_point())
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(r), terminated, truncated, self._info()

    # mechanics
    def _agl(self) -> float:
        return self.alt - float(self.terrain.height(self.pos[0], self.pos[1]))

    def _landing_point(self) -> np.ndarray:
        fall_s = max(self._agl(), 1.0) / CHUTE_DESCENT_MS
        return self.pos + self.wind * fall_s / 1000.0

    def _release(self, events) -> float:
        if not self.payload_on:
            events.append(("no payload to release", REWARDS.invalid_drop))
            return REWARDS.invalid_drop
        wp = self.waypoints[self.wp_i]
        hospital = np.array(self.hospital.xy)
        # safety interlock: only over the hospital area and below the drop ceiling
        if (wp.kind != "drop" or np.linalg.norm(self.pos - hospital) > RELEASE_RANGE_KM
                or self._agl() > DROP_MAX_AGL_M):
            events.append(("release blocked by interlock", REWARDS.release_blocked))
            return REWARDS.release_blocked
        self.payload_on = False
        land = self._landing_point()
        self.drop_point = land.copy()
        miss = float(np.linalg.norm(land - hospital))
        drop_idx = len(RING_OFFSETS_KM)
        self.wp_i = drop_idx + 1
        self.delivery_minutes = self.t * DT_S / 60.0
        if miss <= DROP_ZONE_KM:
            self.waypoints[drop_idx].status = "passed"
            self.delivered = True
            self.drop_result = "on_target"
            base = REWARDS.delivered if self.temp <= TEMP_LIMIT_C else REWARDS.delivered_spoiled
            bonus = REWARDS.speed_bonus_max * float(np.clip((BENCHMARK_MIN - self.delivery_minutes) / BENCHMARK_MIN, 0, 1))
            events.append((f"blood delivered in {self.delivery_minutes:.0f} min", base + bonus))
            return base + bonus
        self.waypoints[drop_idx].status = "missed"
        if miss <= RECOVER_KM:
            self.delivered = True
            self.drop_result = "off_target"
            events.append((f"landed {miss * 1000:.0f} m off target", REWARDS.off_target))
            return REWARDS.off_target
        self.drop_result = "lost"
        events.append(("parachute drifted away", REWARDS.failed_drop))
        return REWARDS.failed_drop

    def _update_waypoints(self, prev: np.ndarray, events) -> float:
        r = 0.0
        while True:
            wp = self.waypoints[self.wp_i]
            if wp.kind == "ring":
                seg = self.pos - prev
                t = float(np.clip(((wp.xy - prev) @ seg) / (seg @ seg + 1e-12), 0.0, 1.0))
                if np.linalg.norm(prev + t * seg - wp.xy) <= RING_RADIUS_KM:
                    wp.status = "passed"
                    r += REWARDS.ring_passed
                    events.append((f"ring {wp.index + 1} {'in' if wp.inbound else 'out'}", REWARDS.ring_passed))
                    self.wp_i += 1
                    continue
                along = float(self.pos @ self.corridor_u)
                d = RING_OFFSETS_KM[wp.index]
                missed = along < d - RING_MISS_KM if wp.inbound else along > d + RING_MISS_KM
                if missed:
                    wp.status = "missed"
                    r += REWARDS.ring_missed
                    events.append((f"missed ring {wp.index + 1}", REWARDS.ring_missed))
                    self.wp_i += 1
                    continue
            elif wp.kind == "home":
                seg = self.pos - prev
                t = float(np.clip((-prev @ seg) / (seg @ seg + 1e-12), 0.0, 1.0))
                if np.linalg.norm(prev + t * seg) <= HOME_RADIUS_KM:
                    wp.status = "passed"
                    self.outcome = "home"
            break
        return r

    def _rain_edge_distance(self) -> tuple[float, RainCell | None]:
        best, cell = 99.0, None
        for c in self.rain:
            d = float(np.linalg.norm(c.xy - self.pos)) - c.radius
            if d < best:
                best, cell = d, c
        return best, cell

    # observation
    def _rel(self, xy: np.ndarray) -> tuple[float, float, float]:
        v = xy - self.pos
        rel = _wrap(math.atan2(v[1], v[0]) - self.heading)
        return math.sin(rel), math.cos(rel), float(np.linalg.norm(v))

    def _obs(self) -> np.ndarray:
        wp = self.waypoints[self.wp_i]
        s, c, d = self._rel(wp.xy)
        phase = 0 if (wp.kind == "ring" and not wp.inbound) else (1 if wp.kind == "drop" else 2)
        hx, hy = math.cos(self.heading), math.sin(self.heading)
        ground = float(self.terrain.height(self.pos[0], self.pos[1]))
        ahead1 = float(self.terrain.height(self.pos[0] + hx * 1.0, self.pos[1] + hy * 1.0))
        ahead2 = float(self.terrain.height(self.pos[0] + hx * 2.5, self.pos[1] + hy * 2.5))
        tail = self.wind[0] * hx + self.wind[1] * hy
        cross = -self.wind[0] * hy + self.wind[1] * hx
        rd, cell = self._rain_edge_distance()
        rs, rc = (self._rel(cell.xy)[:2]) if cell is not None else (0.0, 0.0)
        a_s, a_c, a_d = self._rel(np.array(AIRPORT_XY))
        if self.payload_on and wp.kind == "drop":
            land_err = float(np.linalg.norm(self._landing_point() - wp.xy))
        else:
            land_err = 3.0
        o = np.array([
            s, c, min(d / 50.0, 1.5), float(wp.kind == "ring"),
            float(phase == 0), float(phase == 1), float(phase == 2),
            np.clip((self.alt - ground) / 200.0, -0.5, 2.0),
            np.clip((self.alt - ahead1) / 200.0, -2.0, 2.0),
            np.clip((self.alt - ahead2) / 200.0, -2.0, 2.0),
            self.setpoint / 200.0,
            self.speed_i / 2.0, max(self.battery, 0.0),
            tail / 12.0, cross / 12.0,
            float(rd < 0.0), rs, rc, np.clip(rd / 10.0, -0.5, 1.5),
            a_s, a_c, np.clip((a_d - AIRPORT_NO_FLY_KM) / 10.0, 0.0, 1.5),
            float(self.payload_on), (self.temp - TEMP_START_C) / 6.0,
            self.t / self.max_steps, min(float(np.linalg.norm(self.pos)) / 50.0, 1.5),
            min(land_err, 3.0) / 3.0,
        ], dtype=np.float32)
        return o

    def _info(self) -> dict:
        return {
            "hospital": self.hospital.name, "outcome": self.outcome, "delivered": self.delivered,
            "drop_result": self.drop_result,
            "delivery_minutes": self.delivery_minutes, "battery": self.battery,
            "rings_passed": sum(w.status == "passed" for w in self.waypoints if w.kind == "ring"),
            "rings_missed": sum(w.status == "missed" for w in self.waypoints if w.kind == "ring"),
            "steps": self.t, "total_reward": self.total_reward,
        }

    # rendering
    def _snapshot_point(self) -> list:
        return [round(float(self.pos[0]), 4), round(float(self.pos[1]), 4), round(float(self.alt), 1)]

    def frame(self) -> dict:
        """JSON-friendly snapshot of the current state for the 3D viewer."""
        wp = self.waypoints[self.wp_i]
        return {
            "t": self.t, "minutes": round(self.t * DT_S / 60.0, 2),
            "pos": [float(self.pos[0]), float(self.pos[1])], "alt": float(self.alt),
            "agl": round(self._agl(), 1), "setpoint": self.setpoint, "heading": float(self.heading),
            "speed": SPEED_NAMES[self.speed_i], "speed_kmh": round(AIRSPEEDS_MS[self.speed_i] * 3.6),
            "battery": round(max(self.battery, 0.0), 4), "payload_on": self.payload_on,
            "temp": round(self.temp, 2), "wind": [float(self.wind[0]), float(self.wind[1])],
            "hospital": self.hospital.name, "hospital_xy": list(self.hospital.xy),
            "waypoint": {"kind": wp.kind, "xy": wp.xy.tolist(), "index": wp.index, "inbound": wp.inbound},
            "rings": [{"xy": w.xy.tolist(), "index": w.index, "inbound": w.inbound, "status": w.status}
                      for w in self.waypoints if w.kind == "ring"],
            "corridor_bearing": float(self.corridor_bearing),
            "rain": [{"xy": c.xy.tolist(), "radius": c.radius} for c in self.rain],
            "action": None if self.last_action is None else ACTIONS[self.last_action],
            "reward": round(self.last_reward, 3), "total_reward": round(self.total_reward, 2),
            "events": [[e, round(v, 1)] for e, v in self.last_events],
            "delivered": self.delivered, "drop_result": self.drop_result, "drop_point": None if self.drop_point is None else self.drop_point.tolist(),
            "outcome": self.outcome,
        }

    def render(self):
        if self.render_mode == "human":
            from .rendering import get_viewer
            if self._viewer is None:
                self._viewer = get_viewer(self.terrain)
            self._viewer.push(self.frame())
            return None
        if self.render_mode == "rgb_array":
            from .rendering import topdown_rgb
            return topdown_rgb(self)
        return None

    def close(self):
        self._viewer = None


def heuristic_action(env: IbabaDroneEnv) -> int:
    """Hand-written autopilot used as a non-learning baseline.

    Flies straight at the next waypoint at cruise speed, keeps the default height,
    descends and slows near the hospital and drops when the predicted landing point
    is close. It ignores rain, wind strategy, battery and the airport zone, which is
    exactly what the RL agents have to learn.
    """
    o = env._obs()
    s, c = o[0], o[1]
    agl = o[7] * 200.0
    wp = env.waypoints[env.wp_i]
    dist = float(np.linalg.norm(wp.xy - env.pos))
    approach = wp.kind == "drop" and dist < FINAL_APPROACH_KM
    if env.payload_on and wp.kind == "drop" and o[26] * 3.0 <= DROP_ZONE_KM * 0.8 and agl <= DROP_MAX_AGL_M:
        return 7
    want_sp = 70.0 if approach else 100.0
    if env.setpoint > want_sp + 1:
        return 4
    if env.setpoint < want_sp - 1:
        return 3
    if abs(math.atan2(s, c)) > math.radians(15):
        return 1 if s > 0 else 2
    want = 0 if (wp.kind == "drop" and dist < 3.0) else 1
    if env.speed_i > want:
        return 6
    if env.speed_i < want:
        return 5
    return 0
