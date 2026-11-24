"""
Ibaba Life-Saver Environment

A Gymnasium environment that simulates a blood-delivery drone operating in Rwanda.
Each episode represents a complete mission from the Kigali hub to a real district
hospital and back.

Mission sequence:
1. Launch from the Kigali hub carrying a refrigerated blood pack with a partially
   charged battery.
2. Fly outbound through three corridor rings (2.5 km, 5 km, and 7.5 km from the hub)
   that safely avoid the Kigali Airport no-fly zone.
3. Cross Rwanda's hilly terrain while the autopilot follows the terrain profile.
4. Deliver the blood pack by parachute; the landing position depends on release
   altitude and wind conditions.
5. Return through the same corridor rings in reverse order.
6. Land safely at the Kigali hub before the battery is depleted.

Key challenges:
- Battery management:
  Higher speeds, climbing, and rain increase energy consumption, making early
  flight decisions critical for completing the return journey.

- Blood temperature:
  The blood pack gradually warms during transport and becomes unsuitable for use
  if its temperature exceeds 10°C.

- Wind and rain:
  Wind affects ground speed and parachute drift, while rain cells move across
  the map and increase battery consumption.

- Terrain and airspace:
  Flying into terrain, entering the airport no-fly zone, or leaving the map
  immediately ends the mission.

- Release timing:
  The payload can only be released near the hospital and below a specified
  altitude limit. The final landing location depends on both release altitude
  and wind conditions.

Battery level, weather conditions, terrain, and blood temperature evolve
continuously throughout the mission. Consequently, the agent must learn to plan
and optimize an entire mission rather than simply react to individual steps.

This environment is designed as a reinforcement learning benchmark for
long-horizon decision-making under energy, weather, navigation, and delivery
constraints inspired by real-world medical drone operations in Rwanda.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .world import AIRPORT_NO_FLY_KM, AIRPORT_XY, BOUNDS, HOSPITALS, Hospital, get_terrain

# constants
# ------------------------------------------------------------------------------

# Actions and episode
ACTIONS = (
    "hold course", "turn left", "turn right", "climb",
    "descend", "speed up", "slow down", "release payload",
)
MAX_STEPS = 300   # 300 steps x 20 s = 100 minutes

# Flight dynamics
SPEED_NAMES = ("slow", "cruise", "fast")
AIRSPEEDS_MS = (16.7, 27.8, 36.1)  # 60, 100, 130 km/h (cruise ~101 km/h like real blood drones)
DT_S = 20.0   # one step = 20 seconds of flight
TURN_RAD = math.radians(30.0)
SETPOINT_STEP_M = 30.0  # climb/descend change the height-above-ground setpoint
SETPOINT_RANGE_M = (40.0, 200.0)
MAX_CLIMB_M = 60.0  # autopilot climb limit per step (3 m/s)
MAX_SINK_M = 80.0   # autopilot descent limit per step (4 m/s)
ALTITUDE_BAND_M = (60.0, 150.0)    # preferred height above ground while cruising
GROUND_WARNING_M = 25.0  # below this height above ground the drone is penalised
RAIN_DRIFT_FACTOR = 0.6  # rain cells move at 60% of the mean wind

# Corridor navigation
RING_OFFSETS_KM = (2.5, 5.0, 7.5)  # corridor rings, distance from hub
RING_RADIUS_KM = 0.45
RING_MISS_KM = 0.8  # a ring counts as missed once the drone is this far past it
HOME_RADIUS_KM = 1.0
FINAL_APPROACH_KM = 2.0  # low flying is allowed this close to the hospital or hub

# Delivery system
DROP_ZONE_KM = 0.30   # on-target landing radius
RELEASE_RANGE_KM = 1.0  # interlock: release only this close to the hospital
RECOVER_KM = 1.0  # off-target parachutes within this range are recovered late
DROP_MAX_AGL_M = 130.0 # higher release = more parachute drift
CHUTE_DESCENT_MS = 5.0
BENCHMARK_MIN = 42.0   # real average order-to-delivery time in Rwanda

# Battery model
BATTERY_PER_STEP = 0.0042    # cruise drain per step (fraction of full battery)
CLIMB_BATTERY_PER_M = 0.00002   # extra drain per metre climbed
RAIN_DRAIN_FACTOR = 1.8

# Blood-temperature model
TEMP_START_C = 4.0
TEMP_RISE_C_PER_MIN = 0.10
TEMP_LIMIT_C = 10.0   # red cells must stay within 2-10 C in transport (UK JPAC guidelines)

# Observation space
"""Observation vector: 27 float32 values (observation_space is Box(-3, 3)).

Index  Constant                      Meaning                                  Encoding
-----  ----------------------------  ---------------------------------------  -------------------------
Navigation
  0    OBS_REL_BEARING_SIN           bearing to active waypoint vs heading    sin
  1    OBS_REL_BEARING_COS           bearing to active waypoint vs heading    cos
  2    OBS_WAYPOINT_DISTANCE         distance to active waypoint              km / 50, max 1.5
  3    OBS_WAYPOINT_IS_RING          active waypoint is a corridor ring       0 or 1
  4    OBS_PHASE_OUTBOUND            phase: outbound rings                    one-hot
  5    OBS_PHASE_DELIVERY            phase: flying to the hospital            one-hot
  6    OBS_PHASE_RETURN              phase: inbound rings and home            one-hot
Height and terrain
  7    OBS_AGL                       height above ground                      m / 200, [-0.5, 2]
  8    OBS_CLEARANCE_1KM             height above terrain 1 km ahead          m / 200, [-2, 2]
  9    OBS_CLEARANCE_2_5KM           height above terrain 2.5 km ahead        m / 200, [-2, 2]
 10    OBS_SETPOINT                  autopilot height setpoint                m / 200
Flight state
 11    OBS_SPEED                     speed level slow / cruise / fast         0, 0.5, 1
 12    OBS_BATTERY                   battery left                             0 to 1
Weather
 13    OBS_TAILWIND                  wind along the heading                   m/s / 12
 14    OBS_CROSSWIND                 wind across the heading (+ = to left)    m/s / 12
 15    OBS_IN_RAIN                   inside a rain cell                       0 or 1
 16    OBS_RAIN_BEARING_SIN          bearing to nearest rain cell             sin (0 if no rain)
 17    OBS_RAIN_BEARING_COS          bearing to nearest rain cell             cos (0 if no rain)
 18    OBS_RAIN_DISTANCE             distance to nearest rain edge            km / 10, [-0.5, 1.5]
Airspace
 19    OBS_AIRPORT_BEARING_SIN       bearing to the airport                   sin
 20    OBS_AIRPORT_BEARING_COS       bearing to the airport                   cos
 21    OBS_AIRPORT_DISTANCE          distance to the no-fly edge              km / 10, [0, 1.5]
Payload
 22    OBS_PAYLOAD                   blood still on board                     0 or 1
 23    OBS_BLOOD_TEMP                warming since launch                     (C - 4) / 6
Mission progress
 24    OBS_ELAPSED_TIME              steps used                               t / max_steps
 25    OBS_HOME_DISTANCE             distance from the hub                    km / 50, max 1.5
 26    OBS_LANDING_ERROR             predicted parachute miss (drop phase)    km / 3, 1 otherwise
"""
OBS_DIM = 27

# Navigation
OBS_REL_BEARING_SIN = 0
OBS_REL_BEARING_COS = 1
OBS_WAYPOINT_DISTANCE = 2
OBS_WAYPOINT_IS_RING = 3
OBS_PHASE_OUTBOUND = 4
OBS_PHASE_DELIVERY = 5
OBS_PHASE_RETURN = 6
# Height and terrain
OBS_AGL = 7
OBS_CLEARANCE_1KM = 8
OBS_CLEARANCE_2_5KM = 9
OBS_SETPOINT = 10
# Flight state
OBS_SPEED = 11
OBS_BATTERY = 12
# Weather
OBS_TAILWIND = 13
OBS_CROSSWIND = 14
OBS_IN_RAIN = 15
OBS_RAIN_BEARING_SIN = 16
OBS_RAIN_BEARING_COS = 17
OBS_RAIN_DISTANCE = 18
# Airspace
OBS_AIRPORT_BEARING_SIN = 19
OBS_AIRPORT_BEARING_COS = 20
OBS_AIRPORT_DISTANCE = 21
# Payload
OBS_PAYLOAD = 22
OBS_BLOOD_TEMP = 23
# Mission progress
OBS_ELAPSED_TIME = 24
OBS_HOME_DISTANCE = 25
OBS_LANDING_ERROR = 26


# reward table
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Rewards:
    """Reward values, paid per event or per step.

    Reward philosophy:
    - The mission outcome dominates. An on-time delivery (+100, plus up to +40 for
      beating the 42-minute benchmark) and a safe landing (+50, plus up to +20 for
      battery left) are worth far more than any shaping term. Every failure
      (crash, empty battery, no-fly zone, leaving the map, losing the blood) costs -100.
    - Small dense terms guide learning: +1 per km of progress towards the active
      waypoint, +10 / -10 for passing / missing a corridor ring, and mild penalties
      for rain, leaving the 60-150 m height band and flying close to the ground.
    - Time costs more while carrying blood (-0.15) than when empty (-0.05), so the
      agent hurries to the patient first, then flies home economically.
    - Shortcuts are closed: a pack that drifts away (-100) is worse than keeping
      it, landing home without delivering earns only +10, warm blood (above 10 C)
      earns only +20, and a blocked or useless release costs -1.
    - A pack that lands off target but within 1 km earns +30 (staff recover it
      late), so a near miss is still better than never releasing.
    """

    progress_per_km: float = 1.0
    ring_passed: float = 10.0
    ring_missed: float = -10.0
    delivered: float = 100.0
    delivered_spoiled: float = 20.0
    speed_bonus_max: float = 40.0
    off_target: float = 30.0     # blood recovered late by hospital staff
    failed_drop: float = -100.0  # blood lost, patient not served
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
    kind: str      # "ring", "drop" or "home"
    xy: np.ndarray
    index: int = -1   # ring number 0..2
    inbound: bool = False
    status: str = "pending"   # pending / passed / missed


@dataclass
class RainCell:
    xy: np.ndarray
    radius: float


@dataclass
class EpisodeLog:
    events: list = field(default_factory=list)
    path: list = field(default_factory=list)


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# environment
# --------------------------------------------------------------------------------
class IbabaDroneEnv(gym.Env):
    """Blood-delivery drone mission from Kigali to a district hospital and back.

    Environment specification:
    - State space: 27 continuous features (Box(-3, 3), float32) covering navigation,
      height and terrain, flight state, weather, airspace, payload and mission
      progress. See the observation table above OBS_DIM.
    - Action space: 8 discrete actions (ACTIONS): hold course, turn left or right
      by 30 degrees, climb or descend the height setpoint by 30 m, speed up, slow
      down, release payload.
    - Episode: one mission. One step is 20 s of flight; at most max_steps steps
      (300 = 100 minutes).
    - Start: at the hub, 60 m above ground, cruise speed, heading roughly towards
      ring 1, battery 65-100 %, blood at 4.0-4.5 C, random wind and 1-3 rain cells.
    - Success: blood delivered (on target or recovered) and the drone lands home.
    - Failure: crash into terrain, battery depletion, airport no-fly violation or
      leaving the map (terminated), or timeout at max_steps (truncated).
    - Hospitals: hospital_split="train" for training, "test" for the held-out
      hospitals (generalisation), "all" for both; fixed_hospital or
      reset(options={"hospital": name}) picks one by name.
    """

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

    # ------------------------------------------------------------ setup helpers
    def _corridor(self, target: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """Choose the departure corridor and place its three rings.

        Starts with the straight bearing from the hub to the hospital, then tries
        bearings 5, 10, 15 ... 180 degrees to either side (+5, -5, +10, -10, ...).
        The first bearing whose corridor segment (hub to the last ring) stays at
        least 0.7 km outside the airport no-fly circle is used.

        Returns the corridor bearing in radians and the ring centres in km, at
        RING_OFFSETS_KM along that bearing. The same rings are flown in reverse
        order on the way home.
        """
        direct_bearing = math.atan2(target[1], target[0])
        airport = np.array(AIRPORT_XY)
        for offset_deg in [0] + [sign * deg for deg in range(5, 181, 5) for sign in (1, -1)]:
            bearing = direct_bearing + math.radians(offset_deg)
            direction = np.array([math.cos(bearing), math.sin(bearing)])
            rings = [direction * offset for offset in RING_OFFSETS_KM]
            # closest point of the corridor segment to the airport
            along = np.clip(airport @ direction, 0.0, RING_OFFSETS_KM[-1])
            if np.linalg.norm(airport - direction * along) > AIRPORT_NO_FLY_KM + 0.7:
                return bearing, rings
        raise RuntimeError("no clear corridor")

    def _spawn_rain(self, target: np.ndarray) -> list[RainCell]:
        """Place 1-3 rain cells; the first one often sits across the route."""
        cells = []
        n_cells = int(self.np_random.integers(1, 4))
        for i in range(n_cells):
            for _ in range(20):
                if i == 0 and self.np_random.random() < 0.7:
                    fraction = self.np_random.uniform(0.35, 0.75)
                    normal = np.array([-target[1], target[0]]) / (np.linalg.norm(target) + 1e-9)
                    xy = target * fraction + normal * self.np_random.normal(0.0, 1.5)
                else:
                    xy = np.array([self.np_random.uniform(BOUNDS[0], BOUNDS[1]),
                                   self.np_random.uniform(BOUNDS[2], BOUNDS[3])])
                radius = float(self.np_random.uniform(2.5, 5.0))
                # keep the hub and the hospital dry at the start
                if np.linalg.norm(xy) > radius + 3.0 and np.linalg.norm(xy - target) > radius + 1.5:
                    cells.append(RainCell(xy=xy, radius=radius))
                    break
        return cells

    # gym API
    # -----------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
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
            [Waypoint("ring", ring.copy(), i) for i, ring in enumerate(rings)]
            + [Waypoint("drop", target.copy())]
            + [Waypoint("ring", rings[i].copy(), i, inbound=True) for i in (2, 1, 0)]
            + [Waypoint("home", np.zeros(2))]
        )
        self.wp_i = 0

        self.pos = np.zeros(2)
        self.alt = float(self.terrain.height(0.0, 0.0)) + 60.0  # just after catapult launch
        self.setpoint = 100.0
        first_ring_bearing = math.atan2(rings[0][1], rings[0][0])
        self.heading = _wrap(first_ring_bearing + math.radians(self.np_random.uniform(-25, 25)))
        self.speed_i = 1
        self.battery = float(self.np_random.uniform(0.65, 1.0))  # busy hubs launch partly charged packs
        self.battery_start = self.battery
        self.payload_on = True
        self.temp = TEMP_START_C + float(self.np_random.uniform(0.0, 0.5))
        wind_speed = float(self.np_random.uniform(0.0, 11.0))
        wind_direction = float(self.np_random.uniform(-math.pi, math.pi))
        self.wind_base = np.array([wind_speed * math.cos(wind_direction),
                                   wind_speed * math.sin(wind_direction)])
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

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the mission by one 20-second step.

        The order is fixed: time cost, pilot command (including release), motion,
        weather and battery, waypoint checks, altitude rules, terminal checks.
        """
        action_id = int(action)
        self.t += 1
        self.last_action = action_id
        events: list[tuple[str, float]] = []

        # charged before the action, so the step that releases the blood still
        # pays the "carrying blood" rate
        reward = REWARDS.time_with_payload if self.payload_on else REWARDS.time_empty
        reward += self._apply_action(action_id, events)
        progress_reward, previous_position, climb_m, airspeed_ms = self._update_motion()
        reward += progress_reward
        reward += self._update_weather_and_battery(airspeed_ms, climb_m)
        reward += self._update_waypoints(previous_position, events)
        reward += self._altitude_reward()
        terminal_reward, terminated = self._check_terminal_conditions(events)
        reward += terminal_reward

        truncated = (not terminated) and self.t >= self.max_steps
        if truncated:
            self.outcome = "timeout"

        self.total_reward += reward
        self.last_reward = reward
        self.last_events = events
        self.log.events.extend((self.t, event, value) for event, value in events)
        self.log.path.append(self._snapshot_point())
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, self._info()

    # step parts
    # -------------------------------------------------------------------------------
    def _apply_action(self, action_id: int, events: list[tuple[str, float]]) -> float:
        """Apply the pilot command. Only "release payload" pays a reward here."""
        if action_id == 1:      # turn left
            self.heading = _wrap(self.heading + TURN_RAD)
        elif action_id == 2:    # turn right
            self.heading = _wrap(self.heading - TURN_RAD)
        elif action_id == 3:    # climb
            self.setpoint = min(SETPOINT_RANGE_M[1], self.setpoint + SETPOINT_STEP_M)
        elif action_id == 4:    # descend
            self.setpoint = max(SETPOINT_RANGE_M[0], self.setpoint - SETPOINT_STEP_M)
        elif action_id == 5:    # speed up
            self.speed_i = min(2, self.speed_i + 1)
        elif action_id == 6:    # slow down
            self.speed_i = max(0, self.speed_i - 1)
        elif action_id == 7:    # release payload
            return self._release(events)
        return 0.0

    def _update_motion(self) -> tuple[float, np.ndarray, float, float]:
        """Fly for one step and let the terrain-following autopilot adjust height.

        Returns (progress reward, position before the move, metres climbed,
        airspeed in m/s). Progress is measured to the waypoint that was active
        during the move, so a ring passed mid-step still pays for the distance.
        """
        waypoint = self.waypoints[self.wp_i]
        distance_before_move = float(np.linalg.norm(waypoint.xy - self.pos))
        gust = self.np_random.normal(0.0, 0.7, 2)
        self.wind = 0.8 * self.wind + 0.2 * self.wind_base + 0.3 * gust
        airspeed_ms = AIRSPEEDS_MS[self.speed_i]
        velocity = airspeed_ms * np.array([math.cos(self.heading), math.sin(self.heading)]) + self.wind
        previous_position = self.pos.copy()
        self.pos = self.pos + velocity * DT_S / 1000.0
        distance_after_move = float(np.linalg.norm(waypoint.xy - self.pos))
        progress_reward = REWARDS.progress_per_km * (distance_before_move - distance_after_move)

        # chase ground + setpoint within the climb / sink limits
        target_alt = float(self.terrain.height(self.pos[0], self.pos[1])) + self.setpoint
        climb_m = float(np.clip(target_alt - self.alt, -MAX_SINK_M, MAX_CLIMB_M))
        self.alt += climb_m
        return progress_reward, previous_position, climb_m, airspeed_ms

    def _update_weather_and_battery(self, airspeed_ms: float, climb_m: float) -> float:
        """Move the rain, drain the battery and warm the blood. Returns the rain penalty."""
        for cell in self.rain:
            cell.xy = cell.xy + self.wind_base * RAIN_DRIFT_FACTOR * DT_S / 1000.0
        in_rain = self._rain_edge_distance()[0] < 0.0
        relative_speed = airspeed_ms / AIRSPEEDS_MS[1]
        # drain grows with speed cubed (drag power) plus the energy to climb
        drain = (BATTERY_PER_STEP * (0.35 + 0.65 * relative_speed ** 3)
                 + CLIMB_BATTERY_PER_M * max(climb_m, 0.0))
        rain_reward = 0.0
        if in_rain:
            drain *= RAIN_DRAIN_FACTOR
            rain_reward = REWARDS.rain
        self.battery -= drain
        if self.payload_on:
            self.temp += TEMP_RISE_C_PER_MIN * DT_S / 60.0
        return rain_reward

    def _update_waypoints(self, previous_position: np.ndarray,
                          events: list[tuple[str, float]]) -> float:
        """Check the active waypoint against this step's flight segment.

        Rings: the straight segment from previous_position to the current position
        is tested, so a fast drone cannot jump over a ring between two steps.
        - Passed: the segment comes within RING_RADIUS_KM of the ring centre (+10).
        - Missed: the drone is more than RING_MISS_KM past the ring along the
          corridor (outbound: further out; inbound: closer to the hub) (-10).
        After a pass or a miss the next waypoint becomes active and is checked
        straight away, so one step can clear several rings.

        Home: if the segment comes within HOME_RADIUS_KM of the hub, the outcome
        is set to "home"; the landing reward is paid in _check_terminal_conditions.

        The drop waypoint is never handled here; only _release() completes it.
        Returns the ring rewards earned this step.
        """
        reward = 0.0
        while True:
            waypoint = self.waypoints[self.wp_i]
            if waypoint.kind == "ring":
                segment = self.pos - previous_position
                fraction = float(np.clip(((waypoint.xy - previous_position) @ segment)
                                         / (segment @ segment + 1e-12), 0.0, 1.0))
                if np.linalg.norm(previous_position + fraction * segment - waypoint.xy) <= RING_RADIUS_KM:
                    waypoint.status = "passed"
                    reward += REWARDS.ring_passed
                    direction = "in" if waypoint.inbound else "out"
                    events.append((f"ring {waypoint.index + 1} {direction}", REWARDS.ring_passed))
                    self.wp_i += 1
                    continue
                distance_along_corridor = float(self.pos @ self.corridor_u)
                ring_offset_km = RING_OFFSETS_KM[waypoint.index]
                if waypoint.inbound:
                    missed = distance_along_corridor < ring_offset_km - RING_MISS_KM
                else:
                    missed = distance_along_corridor > ring_offset_km + RING_MISS_KM
                if missed:
                    waypoint.status = "missed"
                    reward += REWARDS.ring_missed
                    events.append((f"missed ring {waypoint.index + 1}", REWARDS.ring_missed))
                    self.wp_i += 1
                    continue
            elif waypoint.kind == "home":
                segment = self.pos - previous_position
                fraction = float(np.clip((-previous_position @ segment)
                                         / (segment @ segment + 1e-12), 0.0, 1.0))
                if np.linalg.norm(previous_position + fraction * segment) <= HOME_RADIUS_KM:
                    waypoint.status = "passed"
                    self.outcome = "home"
            break
        return reward

    def _altitude_reward(self) -> float:
        """Penalty for leaving the cruise height band or flying close to the ground."""
        agl = self._agl()
        waypoint = self.waypoints[self.wp_i]
        near_end = (waypoint.kind in ("drop", "home")
                    and np.linalg.norm(waypoint.xy - self.pos) < FINAL_APPROACH_KM)
        reward = 0.0
        if agl > ALTITUDE_BAND_M[1] or (agl < ALTITUDE_BAND_M[0] and not near_end):
            reward += REWARDS.altitude_band
        if agl < GROUND_WARNING_M:
            reward += REWARDS.ground_proximity
        return reward

    def _check_terminal_conditions(self, events: list[tuple[str, float]]) -> tuple[float, bool]:
        """End the mission on landing or failure. Returns (reward, terminated).

        Checked in this order: landed home, crash, airport no-fly zone, empty
        battery, out of the map. The time limit is handled in step().
        """
        if self.outcome == "home":
            base = REWARDS.home if self.delivered else REWARDS.home_without_delivery
            landing_reward = base + REWARDS.home_battery_bonus * max(self.battery, 0.0)
            events.append(("landed home", landing_reward))
            return landing_reward, True
        if self._agl() <= 0.0:
            self.outcome = "crash"
            events.append(("crashed into hill", REWARDS.crash))
            return REWARDS.crash, True
        if np.linalg.norm(self.pos - np.array(AIRPORT_XY)) < AIRPORT_NO_FLY_KM:
            self.outcome = "no_fly"
            events.append(("entered airport no-fly zone", REWARDS.no_fly))
            return REWARDS.no_fly, True
        if self.battery <= 0.0:
            self.outcome = "battery"
            events.append(("battery empty", REWARDS.battery_empty))
            return REWARDS.battery_empty, True
        if not (BOUNDS[0] <= self.pos[0] <= BOUNDS[1] and BOUNDS[2] <= self.pos[1] <= BOUNDS[3]):
            self.outcome = "out_of_bounds"
            events.append(("left the map", REWARDS.out_of_bounds))
            return REWARDS.out_of_bounds, True
        return 0.0, False

    # mechanics
    #-------------------------------------------------------------------------------
    def _agl(self) -> float:
        """Height above the ground directly below the drone, in metres."""
        return self.alt - float(self.terrain.height(self.pos[0], self.pos[1]))

    def _landing_point(self) -> np.ndarray:
        """Where a parachute released now would land, drifting with the current wind."""
        fall_s = max(self._agl(), 1.0) / CHUTE_DESCENT_MS
        return self.pos + self.wind * fall_s / 1000.0

    def _release(self, events: list[tuple[str, float]]) -> float:
        """Try to release the blood pack by parachute and score the landing.

        1. No payload on board: -1, nothing happens.
        2. Safety interlock: release is blocked (-1) unless the hospital is the
           active waypoint, the drone is within RELEASE_RANGE_KM of it and below
           DROP_MAX_AGL_M above ground.
        3. Otherwise the pack falls at CHUTE_DESCENT_MS and drifts with the wind.
           The miss distance from the hospital decides the result:
           - within DROP_ZONE_KM: delivered on target, +100 (+20 if the blood is
             above TEMP_LIMIT_C) plus a speed bonus of up to +40 for beating the
             BENCHMARK_MIN delivery time;
           - within RECOVER_KM: off target but recovered late by staff, +30;
           - further: the pack is lost, -100.
        In all three cases the payload is gone and the return leg starts.
        """
        if not self.payload_on:
            events.append(("no payload to release", REWARDS.invalid_drop))
            return REWARDS.invalid_drop
        waypoint = self.waypoints[self.wp_i]
        hospital = np.array(self.hospital.xy)
        if (waypoint.kind != "drop" or np.linalg.norm(self.pos - hospital) > RELEASE_RANGE_KM
                or self._agl() > DROP_MAX_AGL_M):
            events.append(("release blocked by interlock", REWARDS.release_blocked))
            return REWARDS.release_blocked
        self.payload_on = False
        landing_point = self._landing_point()
        self.drop_point = landing_point.copy()
        miss_km = float(np.linalg.norm(landing_point - hospital))
        drop_index = len(RING_OFFSETS_KM)
        self.wp_i = drop_index + 1
        self.delivery_minutes = self.t * DT_S / 60.0
        if miss_km <= DROP_ZONE_KM:
            self.waypoints[drop_index].status = "passed"
            self.delivered = True
            self.drop_result = "on_target"
            base = REWARDS.delivered if self.temp <= TEMP_LIMIT_C else REWARDS.delivered_spoiled
            time_saved = (BENCHMARK_MIN - self.delivery_minutes) / BENCHMARK_MIN
            bonus = REWARDS.speed_bonus_max * float(np.clip(time_saved, 0, 1))
            events.append((f"blood delivered in {self.delivery_minutes:.0f} min", base + bonus))
            return base + bonus
        self.waypoints[drop_index].status = "missed"
        if miss_km <= RECOVER_KM:
            self.delivered = True
            self.drop_result = "off_target"
            events.append((f"landed {miss_km * 1000:.0f} m off target", REWARDS.off_target))
            return REWARDS.off_target
        self.drop_result = "lost"
        events.append(("parachute drifted away", REWARDS.failed_drop))
        return REWARDS.failed_drop

    def _rain_edge_distance(self) -> tuple[float, RainCell | None]:
        """Distance to the edge of the nearest rain cell (negative = inside it)."""
        best_km, nearest_cell = 99.0, None
        for cell in self.rain:
            edge_km = float(np.linalg.norm(cell.xy - self.pos)) - cell.radius
            if edge_km < best_km:
                best_km, nearest_cell = edge_km, cell
        return best_km, nearest_cell

    # observation
    # ---------------------------------------------------------------------------------
    def _rel(self, xy: np.ndarray) -> tuple[float, float, float]:
        """ Bearing of a point relative to the heading (sin, cos) and its distance in km.
        """
        offset = xy - self.pos
        relative_bearing = _wrap(math.atan2(offset[1], offset[0]) - self.heading)
        return math.sin(relative_bearing), math.cos(relative_bearing), float(np.linalg.norm(offset))

    def _obs(self) -> np.ndarray:
        """Build the 27-value observation (see the table above OBS_DIM)."""
        waypoint = self.waypoints[self.wp_i]
        bearing_sin, bearing_cos, distance_km = self._rel(waypoint.xy)
        if waypoint.kind == "ring" and not waypoint.inbound:
            phase = 0
        elif waypoint.kind == "drop":
            phase = 1
        else:
            phase = 2
        heading_x, heading_y = math.cos(self.heading), math.sin(self.heading)
        ground = float(self.terrain.height(self.pos[0], self.pos[1]))
        ground_1km = float(self.terrain.height(self.pos[0] + heading_x * 1.0, self.pos[1] + heading_y * 1.0))
        ground_2_5km = float(self.terrain.height(self.pos[0] + heading_x * 2.5, self.pos[1] + heading_y * 2.5))
        tailwind = self.wind[0] * heading_x + self.wind[1] * heading_y
        crosswind = -self.wind[0] * heading_y + self.wind[1] * heading_x
        rain_edge_km, nearest_cell = self._rain_edge_distance()
        if nearest_cell is not None:
            rain_sin, rain_cos = self._rel(nearest_cell.xy)[:2]
        else:
            rain_sin, rain_cos = 0.0, 0.0
        airport_sin, airport_cos, airport_km = self._rel(np.array(AIRPORT_XY))
        if self.payload_on and waypoint.kind == "drop":
            landing_error_km = float(np.linalg.norm(self._landing_point() - waypoint.xy))
        else:
            landing_error_km = 3.0

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[OBS_REL_BEARING_SIN] = bearing_sin
        obs[OBS_REL_BEARING_COS] = bearing_cos
        obs[OBS_WAYPOINT_DISTANCE] = min(distance_km / 50.0, 1.5)
        obs[OBS_WAYPOINT_IS_RING] = float(waypoint.kind == "ring")
        obs[OBS_PHASE_OUTBOUND] = float(phase == 0)
        obs[OBS_PHASE_DELIVERY] = float(phase == 1)
        obs[OBS_PHASE_RETURN] = float(phase == 2)
        obs[OBS_AGL] = np.clip((self.alt - ground) / 200.0, -0.5, 2.0)
        obs[OBS_CLEARANCE_1KM] = np.clip((self.alt - ground_1km) / 200.0, -2.0, 2.0)
        obs[OBS_CLEARANCE_2_5KM] = np.clip((self.alt - ground_2_5km) / 200.0, -2.0, 2.0)
        obs[OBS_SETPOINT] = self.setpoint / 200.0
        obs[OBS_SPEED] = self.speed_i / 2.0
        obs[OBS_BATTERY] = max(self.battery, 0.0)
        obs[OBS_TAILWIND] = tailwind / 12.0
        obs[OBS_CROSSWIND] = crosswind / 12.0
        obs[OBS_IN_RAIN] = float(rain_edge_km < 0.0)
        obs[OBS_RAIN_BEARING_SIN] = rain_sin
        obs[OBS_RAIN_BEARING_COS] = rain_cos
        obs[OBS_RAIN_DISTANCE] = np.clip(rain_edge_km / 10.0, -0.5, 1.5)
        obs[OBS_AIRPORT_BEARING_SIN] = airport_sin
        obs[OBS_AIRPORT_BEARING_COS] = airport_cos
        obs[OBS_AIRPORT_DISTANCE] = np.clip((airport_km - AIRPORT_NO_FLY_KM) / 10.0, 0.0, 1.5)
        obs[OBS_PAYLOAD] = float(self.payload_on)
        obs[OBS_BLOOD_TEMP] = (self.temp - TEMP_START_C) / 6.0
        obs[OBS_ELAPSED_TIME] = self.t / self.max_steps
        obs[OBS_HOME_DISTANCE] = min(float(np.linalg.norm(self.pos)) / 50.0, 1.5)
        obs[OBS_LANDING_ERROR] = min(landing_error_km, 3.0) / 3.0
        return obs

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
        waypoint = self.waypoints[self.wp_i]
        return {
            "t": self.t, "minutes": round(self.t * DT_S / 60.0, 2),
            "pos": [float(self.pos[0]), float(self.pos[1])], "alt": float(self.alt),
            "agl": round(self._agl(), 1), "setpoint": self.setpoint, "heading": float(self.heading),
            "speed": SPEED_NAMES[self.speed_i], "speed_kmh": round(AIRSPEEDS_MS[self.speed_i] * 3.6),
            "battery": round(max(self.battery, 0.0), 4), "payload_on": self.payload_on,
            "temp": round(self.temp, 2), "wind": [float(self.wind[0]), float(self.wind[1])],
            "hospital": self.hospital.name, "hospital_xy": list(self.hospital.xy),
            "waypoint": {"kind": waypoint.kind, "xy": waypoint.xy.tolist(),
                         "index": waypoint.index, "inbound": waypoint.inbound},
            "rings": [{"xy": w.xy.tolist(), "index": w.index, "inbound": w.inbound, "status": w.status}
                      for w in self.waypoints if w.kind == "ring"],
            "corridor_bearing": float(self.corridor_bearing),
            "rain": [{"xy": c.xy.tolist(), "radius": c.radius} for c in self.rain],
            "action": None if self.last_action is None else ACTIONS[self.last_action],
            "reward": round(self.last_reward, 3), "total_reward": round(self.total_reward, 2),
            "events": [[event, round(value, 1)] for event, value in self.last_events],
            "delivered": self.delivered, "drop_result": self.drop_result,
            "drop_point": None if self.drop_point is None else self.drop_point.tolist(),
            "outcome": self.outcome,
        }

    def render(self) -> np.ndarray | None:
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

    def close(self) -> None:
        self._viewer = None


def heuristic_action(env: IbabaDroneEnv) -> int:
    """Hand-written autopilot used as a non-learning baseline.

    Flies straight at the next waypoint at cruise speed, keeps the default height,
    descends and slows near the hospital and drops when the predicted landing point
    is close. It ignores rain, wind strategy, battery and the airport zone, which is
    exactly what the RL agents have to learn.
    """
    obs = env._obs()
    bearing_sin, bearing_cos = obs[OBS_REL_BEARING_SIN], obs[OBS_REL_BEARING_COS]
    agl = obs[OBS_AGL] * 200.0
    waypoint = env.waypoints[env.wp_i]
    distance_km = float(np.linalg.norm(waypoint.xy - env.pos))
    approach = waypoint.kind == "drop" and distance_km < FINAL_APPROACH_KM
    landing_error_km = obs[OBS_LANDING_ERROR] * 3.0
    if (env.payload_on and waypoint.kind == "drop"
            and landing_error_km <= DROP_ZONE_KM * 0.8 and agl <= DROP_MAX_AGL_M):
        return 7                                    # release payload
    wanted_setpoint = 70.0 if approach else 100.0
    if env.setpoint > wanted_setpoint + 1:
        return 4                                    # descend
    if env.setpoint < wanted_setpoint - 1:
        return 3                                    # climb
    if abs(math.atan2(bearing_sin, bearing_cos)) > math.radians(15):
        return 1 if bearing_sin > 0 else 2          # turn towards the waypoint
    wanted_speed_i = 0 if (waypoint.kind == "drop" and distance_km < 3.0) else 1
    if env.speed_i > wanted_speed_i:
        return 6     # slow down
    if env.speed_i < wanted_speed_i:
        return 5     # speed up
    return 0   # hold course
