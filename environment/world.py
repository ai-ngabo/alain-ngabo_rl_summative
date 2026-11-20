"""
Static world definition for the Ibaba Life-Saver Environment.

This module defines the geographic and environmental features used by the
simulation, including:

- The Kigali drone hub (mission origin).
- District and provincial hospitals used as delivery destinations.
- Kigali International Airport and its no-fly zone.
- Lake Muhazi, the Nyabarongo River, and major terrain landmarks.
- A deterministic terrain model inspired by Rwanda's hilly landscape.
- Settlements, roads, and visual landmarks used for rendering.

Coordinates are expressed in kilometres on a local Cartesian grid centred
on the Kigali hub, where:

- x increases towards the east.
- y increases towards the north.

Hospital locations are based on approximate public geographic coordinates
(latitude and longitude) converted to local kilometre coordinates using an
equirectangular projection.

The world is deterministic and shared across episodes, ensuring that agents
encounter the same geography while still experiencing varying weather,
battery, and mission conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Kigali drone hub (simulation origin and mission base).
# The location was chosen as a modelling design decision.
HUB_LAT, HUB_LON = -1.9540, 30.0870
KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON = 111.32 * math.cos(math.radians(HUB_LAT))

# Map bounds in km (x_min, x_max, y_min, y_max).
BOUNDS = (-48.0, 56.0, -32.0, 50.0)


def latlon_to_km(lat: float, lon: float) -> tuple[float, float]:
    """Convert latitude/longitude to (x, y) km relative to the hub."""
    return ((lon - HUB_LON) * KM_PER_DEG_LON, (lat - HUB_LAT) * KM_PER_DEG_LAT)


@dataclass(frozen=True)
class Hospital:
    name: str
    district: str
    lat: float
    lon: float
    split: str  # "train" or "test"; test hospitals are held out for generalization evaluation.

    @property
    def xy(self) -> tuple[float, float]:
        return latlon_to_km(self.lat, self.lon)


# District and provincial hospitals used as delivery destinations.
# Coordinates are approximate real-world locations.
HOSPITALS: tuple[Hospital, ...] = (
    Hospital("Rutongo District Hospital", "Rulindo", -1.8270, 30.0530, "train"),
    Hospital("Nyamata District Hospital", "Bugesera", -2.1490, 30.0930, "train"),
    Hospital("Remera-Rukoma District Hospital", "Kamonyi", -2.0330, 29.9100, "train"),
    Hospital("Kabgayi District Hospital", "Muhanga", -2.0960, 29.7540, "train"),
    Hospital("Byumba District Hospital", "Gicumbi", -1.5790, 30.0650, "train"),
    Hospital("Rwamagana Provincial Hospital", "Rwamagana", -1.9490, 30.4350, "train"),
    Hospital("Kinihira Provincial Hospital", "Rulindo", -1.7120, 29.9900, "test"),
    Hospital("Gahini District Hospital", "Kayonza", -1.8430, 30.4770, "test"),
)

AIRPORT_NAME = "Kigali International Airport"
AIRPORT_XY = latlon_to_km(-1.9686, 30.1395)
AIRPORT_NO_FLY_KM = 4.0

# Lake Muhazi (long, thin lake north-east of Kigali), as a polyline with a width.
LAKE_MUHAZI = [latlon_to_km(lat, lon) for lat, lon in (
    (-1.877, 30.195), (-1.868, 30.250), (-1.862, 30.305),
    (-1.853, 30.360), (-1.846, 30.415), (-1.849, 30.462),
)]
LAKE_HALF_WIDTH_KM = 0.6
LAKE_LEVEL_M = 1443.0
SHORE_WIDTH_KM = 6.0
AIRPORT_LEVEL_M = 1490.0

# Nyabarongo River (visual landmark used for rendering).
NYABARONGO = [latlon_to_km(lat, lon) for lat, lon in (
    (-2.060, 29.780), (-2.010, 29.900), (-1.985, 29.990), (-2.000, 30.040),
    (-2.030, 30.090), (-2.070, 30.160), (-2.110, 30.230),
)]

# Prominent hills around Kigali used as visual landmarks and terrain features.
NAMED_HILLS = (
    ("Mount Kigali", latlon_to_km(-1.9770, 30.0330), 330.0, 2.2),
    ("Mount Jali", latlon_to_km(-1.8920, 30.0870), 380.0, 2.6),
    ("Rebero", latlon_to_km(-2.0050, 30.0750), 220.0, 1.8),
)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Terrain:
    """
    Procedurally generated terrain model (metres above sea level).

    The terrain combines a large-scale elevation trend with a collection of
    Gaussian hills generated from a fixed random seed, producing a deterministic
    representation of Rwanda's rolling landscape. Additional shaping is applied
    around Lake Muhazi and Kigali International Airport to create realistic
    water and plateau regions.

    Because the same seed is used for every episode, the terrain remains
    identical across runs, allowing consistent benchmarking and evaluation.

    Slopes are intentionally constrained so that the drone can safely traverse
    the environment while still requiring terrain-aware navigation.
    """

    def __init__(self, seed: int = 2026, n_hills: int = 150):
        rng = np.random.default_rng(seed)
        x0, x1, y0, y1 = BOUNDS
        cx = rng.uniform(x0 - 5, x1 + 5, n_hills)
        cy = rng.uniform(y0 - 5, y1 + 5, n_hills)
        amp = rng.uniform(50.0, 260.0, n_hills)
        sig = rng.uniform(2.6, 6.0, n_hills)
        # Bugesera (south-east) is flatter; the north is more rugged.
        flat = (cy < -10) & (cx > -5)
        amp[flat] *= 0.45
        # No big random hill on top of the airport plateau.
        keep = np.hypot(cx - AIRPORT_XY[0], cy - AIRPORT_XY[1]) > 7.0
        cx, cy, amp, sig = cx[keep], cy[keep], amp[keep], sig[keep]
        names = np.array([h[1] for h in NAMED_HILLS])
        self.cx = np.concatenate([cx, names[:, 0]])
        self.cy = np.concatenate([cy, names[:, 1]])
        self.amp = np.concatenate([amp, [h[2] for h in NAMED_HILLS]])
        self.sig = np.concatenate([sig, [h[3] for h in NAMED_HILLS]])
        self.inv2s2 = 1.0 / (2.0 * self.sig**2)
        self._lake = np.array(LAKE_MUHAZI)
        self._build_grid()

    def _lake_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = np.full(x.shape, np.inf)
        for (ax, ay), (bx, by) in zip(self._lake[:-1], self._lake[1:]):
            vx, vy = bx - ax, by - ay
            t = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0.0, 1.0)
            d = np.minimum(d, np.hypot(x - (ax + t * vx), y - (ay + t * vy)))
        return d

    # fast lookup
    GRID_RES_KM = 0.2

    def _build_grid(self) -> None:
        x0, x1, y0, y1 = BOUNDS
        self.gx = np.arange(x0, x1 + 1e-9, self.GRID_RES_KM)
        self.gy = np.arange(y0, y1 + 1e-9, self.GRID_RES_KM)
        rows = [self.height_exact(self.gx, np.full_like(self.gx, yy)) for yy in self.gy]
        self.grid = np.stack(rows)  # shape (len(gy), len(gx))
        self._flat = self.grid.ravel().tolist() # python floats: fastest scalar access
        self._nx, self._ny = len(self.gx), len(self.gy)

    def height(self, x, y):
        """Terrain height (m) by bilinear lookup on a 200 m grid. Scalars or arrays (km)."""
        if np.ndim(x) == 0 and np.ndim(y) == 0:
            fx = (float(x) - BOUNDS[0]) / self.GRID_RES_KM
            fy = (float(y) - BOUNDS[2]) / self.GRID_RES_KM
            fx = min(max(fx, 0.0), self._nx - 1.000001)
            fy = min(max(fy, 0.0), self._ny - 1.000001)
            j, i = int(fx), int(fy)
            tx, ty = fx - j, fy - i
            k = i * self._nx + j
            g = self._flat
            a = g[k] + (g[k + 1] - g[k]) * tx
            b = g[k + self._nx] + (g[k + self._nx + 1] - g[k + self._nx]) * tx
            return a + (b - a) * ty
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        fx = np.clip((x - BOUNDS[0]) / self.GRID_RES_KM, 0.0, self._nx - 1.000001)
        fy = np.clip((y - BOUNDS[2]) / self.GRID_RES_KM, 0.0, self._ny - 1.000001)
        j, i = fx.astype(int), fy.astype(int)
        tx, ty = fx - j, fy - i
        g = self.grid
        a = g[i, j] + (g[i, j + 1] - g[i, j]) * tx
        b = g[i + 1, j] + (g[i + 1, j + 1] - g[i + 1, j]) * tx
        return a + (b - a) * ty

    def height_exact(self, x, y):
        """Analytic terrain height in metres for scalar or array inputs (km coordinates)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        xs, ys = x[..., None], y[..., None]
        bumps = (self.amp * np.exp(-((xs - self.cx) ** 2 + (ys - self.cy) ** 2) * self.inv2s2)).sum(-1)
        h = 1380.0 + 6.0 * y - 2.5 * x + bumps
        # Kigali International Airport sits on a flat plateau.
        da = np.hypot(x - AIRPORT_XY[0], y - AIRPORT_XY[1])
        wa = _smoothstep((da - 2.0) / 3.5)
        h = AIRPORT_LEVEL_M + (h - AIRPORT_LEVEL_M) * wa
        # Lake Muhazi lies in a valley: flat water, wide sloping shores.
        d = self._lake_distance(x, y)
        w = _smoothstep((d - LAKE_HALF_WIDTH_KM) / SHORE_WIDTH_KM)
        h = LAKE_LEVEL_M + (h - LAKE_LEVEL_M) * w
        return h

    def params(self, step: int = 2) -> dict:
        """Height grid (every `step` cells) so the 3D viewer draws the same terrain the agent flies over."""
        g = self.grid[::step, ::step]
        return {
            "x0": BOUNDS[0], "y0": BOUNDS[2], "res": self.GRID_RES_KM * step,
            "nx": g.shape[1], "ny": g.shape[0], "heights": g.round(1).ravel().tolist(),
            "lake_level": LAKE_LEVEL_M, "lake_half_width": LAKE_HALF_WIDTH_KM,
        }


_TERRAIN_CACHE: dict[int, Terrain] = {}


def get_terrain(seed: int = 2026) -> Terrain:
    """
    Return a shared terrain instance.

    The terrain grid is cached because its construction is relatively
    expensive and the same terrain is used throughout all episodes.
    """
    if seed not in _TERRAIN_CACHE:
        _TERRAIN_CACHE[seed] = Terrain(seed)
    return _TERRAIN_CACHE[seed]


def scene_description(terrain: Terrain) -> dict:
    """
    Return all static world information required by the viewer.

    Includes terrain, hospitals, landmarks, airspace constraints,
    water bodies, and map boundaries.
    """
    return {
        "bounds": BOUNDS,
        "terrain": terrain.params(),
        "hub": {"name": "Kigali Drone Hub", "xy": [0.0, 0.0], "ground": float(terrain.height(0.0, 0.0))},
        "airport": {"name": AIRPORT_NAME, "xy": list(AIRPORT_XY), "radius": AIRPORT_NO_FLY_KM,
                    "ground": float(terrain.height(*AIRPORT_XY))},
        "hospitals": [
            {"name": h.name, "district": h.district, "xy": list(h.xy), "split": h.split,
             "ground": float(terrain.height(*h.xy))}
            for h in HOSPITALS
        ],
        "lake_muhazi": [list(p) for p in LAKE_MUHAZI],
        "river": [list(p) for p in NYABARONGO],
        "hills": [{"name": n, "xy": list(xy)} for n, xy, _, _ in NAMED_HILLS],
    }


# Settlements, roads, and landmarks used for visual rendering.
# Built-up areas of Kigali (centre, radius km): Kimihurura/KCC, CBD (Nyarugenge), Nyamirambo,
# Remera, Kicukiro, Gisozi, Kacyiru, Kimironko, Gikondo.
KIGALI_AREAS = [
    ((0.0, 0.0), 2.6), (latlon_to_km(-1.9460, 30.0600), 2.4), (latlon_to_km(-1.9790, 30.0450), 2.2),
    (latlon_to_km(-1.9560, 30.1100), 2.3), (latlon_to_km(-1.9900, 30.0900), 2.4),
    (latlon_to_km(-1.9200, 30.0600), 2.0), (latlon_to_km(-1.9380, 30.0820), 1.8),
    (latlon_to_km(-1.9350, 30.1250), 2.0), (latlon_to_km(-1.9750, 30.0700), 1.8),
]
KIGALI_LANDMARKS = {
    "kcc": latlon_to_km(-1.9545, 30.0930),  # Kigali Convention Centre building
    "city_tower": latlon_to_km(-1.9445, 30.0600),   # CBD tower
    "roundabout": latlon_to_km(-1.9470, 30.0620),
    "arena": latlon_to_km(-1.9540, 30.1160), # BK arena Remera
}
TEA_ESTATES = [  # (centre, radius km) tea-growing hills north of Kigali (Gicumbi / Mulindi area)
    (latlon_to_km(-1.6750, 30.0400), 3.2), (latlon_to_km(-1.6300, 30.1000), 2.6),
    (latlon_to_km(-1.7200, 30.0950), 2.2),
]


def settlements(seed: int = 7) -> dict:
    """Deterministic villages, hospital towns and roads (shared by texture and 3D viewer)."""
    rng = np.random.default_rng(seed)
    terrain = get_terrain()
    lake = np.array(LAKE_MUHAZI)
    villages = []
    while len(villages) < 170:
        x = rng.uniform(BOUNDS[0] + 2, BOUNDS[1] - 2)
        y = rng.uniform(BOUNDS[2] + 2, BOUNDS[3] - 2)
        if np.hypot(x, y) < 9 or np.hypot(x - AIRPORT_XY[0], y - AIRPORT_XY[1]) < 5:
            continue
        if terrain._lake_distance(np.array([x]), np.array([y]))[0] < 1.5:
            continue
        villages.append({"xy": [round(x, 3), round(y, 3)], "r": round(float(rng.uniform(0.4, 0.9)), 3),
                         "n": int(rng.integers(8, 26))})
    towns = [{"xy": list(h.xy), "r": 1.6, "n": 70} for h in HOSPITALS]
    # Main roads connecting the hub to hospital towns.
    # Slight perturbations are added to avoid perfectly straight paths.

    roads = []
    for h in HOSPITALS:
        p0, p1 = np.zeros(2), np.array(h.xy)
        n = max(8, int(np.linalg.norm(p1) / 1.5))
        t = np.linspace(0, 1, n)
        normal = np.array([-p1[1], p1[0]]) / np.linalg.norm(p1)
        wig = np.cumsum(rng.normal(0, 0.35, n))
        wig -= np.linspace(wig[0], wig[-1], n)
        pts = p0[None] + t[:, None] * (p1 - p0)[None] + wig[:, None] * normal[None]
        roads.append({"kind": "highway", "pts": pts.round(3).tolist()})
    # Secondary tracks connecting neighbouring villages.
    vxy = np.array([v["xy"] for v in villages])
    for i, p in enumerate(vxy):
        d = np.hypot(*(vxy - p).T)
        for j in np.argsort(d)[1:3]:
            if j > i and d[j] < 9:
                q = vxy[j]
                mid = (p + q) / 2 + rng.normal(0, 0.6, 2)
                roads.append({"kind": "track", "pts": [p.round(3).tolist(), mid.round(3).tolist(), q.round(3).tolist()]})
    return {"villages": villages, "towns": towns, "roads": roads,
            "kigali": [{"xy": list(c), "r": r} for c, r in KIGALI_AREAS],
            "landmarks": {k: list(v) for k, v in KIGALI_LANDMARKS.items()},
            "tea": [{"xy": list(c), "r": r} for c, r in TEA_ESTATES]}
