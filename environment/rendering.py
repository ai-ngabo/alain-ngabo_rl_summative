"""Rendering for Ibaba Life-saver.

* `LiveViewer` - a small FastAPI server that serves the three.js 3D viewer (assets/viewer)
  and streams every environment step to the browser over a WebSocket. The same server
  exposes `POST /api/act` (observation in, action out): the integration path a dispatch
  dashboard or mobile app would use.
* `build_terrain_texture` - paints the Rwandan landscape (fields, terraces, tea, roads,
  Lake Muhazi, Kigali, the airport) into one image used by the 3D terrain.
* `topdown_rgb` - a 2D top-down picture of an episode (Gymnasium "rgb_array" mode).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .world import (
    AIRPORT_NO_FLY_KM, AIRPORT_XY, BOUNDS, HOSPITALS, LAKE_HALF_WIDTH_KM, NYABARONGO,
    get_terrain, scene_description, settlements,
)

ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = ROOT / "assets" / "viewer"
TEXTURE_PATH = VIEWER_DIR / "terrain_texture.jpg"


# terrain texture
def _hex(c: str) -> np.ndarray:
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)


def _value_noise(shape, cells, rng):
    """Smooth random field in [0, 1] (bilinear upsampled noise)."""
    h, w = shape
    g = rng.random((cells[0] + 2, cells[1] + 2)).astype(np.float32)
    ys = np.linspace(0, cells[0], h, dtype=np.float32)
    xs = np.linspace(0, cells[1], w, dtype=np.float32)
    y0, x0 = ys.astype(int), xs.astype(int)
    ty, tx = (ys - y0)[:, None], (xs - x0)[None, :]
    ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty


def build_terrain_texture(path: Path = TEXTURE_PATH, width: int = 4096, seed: int = 11) -> Path:
    from PIL import Image, ImageDraw, ImageFilter

    rng = np.random.default_rng(seed)
    terrain = get_terrain()
    x0, x1, y0, y1 = BOUNDS
    height = int(round(width * (y1 - y0) / (x1 - x0)))
    km_per_px = (x1 - x0) / width
    xs = x0 + (np.arange(width, dtype=np.float32) + 0.5) * km_per_px
    ys = y1 - (np.arange(height, dtype=np.float32) + 0.5) * km_per_px
    X, Y = np.meshgrid(xs, ys)
    H = terrain.height(X, Y).astype(np.float32)
    gy, gx = np.gradient(H, km_per_px * 1000.0)
    slope = np.hypot(gx, gy)

    n_big = _value_noise((height, width), (14, 18), rng)
    n_mid = _value_noise((height, width), (90, 115), rng)
    n_fine = rng.random((height, width), dtype=np.float32)

    # base vegetation by altitude
    low, mid, high = _hex("#6f9e3d"), _hex("#5c8f34"), _hex("#44752b")
    t = np.clip((H - 1300) / 700, 0, 1)[..., None]
    img = np.where(t < 0.5, low + (mid - low) * (t * 2), mid + (high - mid) * ((t - 0.5) * 2))

    # patchwork of farm plots: jittered cells with darker hedge lines
    cell = 0.22  # km
    pts_rng = np.random.default_rng(seed + 1)
    ncx = int((x1 - x0) / cell) + 3
    ncy = int((y1 - y0) / cell) + 3
    jit = pts_rng.random((ncy, ncx, 2)).astype(np.float32)
    colour_id = pts_rng.integers(0, 12, (ncy, ncx))
    palette = np.stack([_hex(c) for c in (
        "#6e9d3c", "#7ba944", "#5f8e35", "#8fb050", "#a5b25b", "#8f7049", "#577f30",
        "#84a64a", "#6a8d3a", "#9aa856", "#7b9b40", "#a8835a")])
    plots = np.empty((height, width, 3), dtype=np.float32)
    edge = np.empty((height, width), dtype=np.float32)
    for r0 in range(0, height, 256):
        r1 = min(height, r0 + 256)
        px_ = (X[r0:r1] - x0) / cell + 1
        py_ = (Y[r0:r1] - y0) / cell + 1
        ix, iy = px_.astype(int), py_.astype(int)
        best = np.full(px_.shape, 9.0, np.float32)
        second = np.full(px_.shape, 9.0, np.float32)
        bid = np.zeros(px_.shape, dtype=np.int64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cx_ = np.clip(ix + dx, 0, ncx - 1)
                cy_ = np.clip(iy + dy, 0, ncy - 1)
                fx = cx_ + jit[cy_, cx_, 0]
                fy = cy_ + jit[cy_, cx_, 1] * 0.8
                d = (px_ - fx) ** 2 + ((py_ - fy) * 1.25) ** 2
                closer = d < best
                second = np.where(closer, best, np.minimum(second, d))
                bid = np.where(closer, colour_id[cy_, cx_], bid)
                best = np.minimum(best, d)
        plots[r0:r1] = palette[bid]
        edge[r0:r1] = np.clip((np.sqrt(second) - np.sqrt(best)) * 9, 0, 1)
    plots *= (0.86 + 0.14 * edge)[..., None]
    farmland = np.clip(1.5 - slope * 5, 0.55, 1.0)[..., None] * np.clip((n_big[..., None] - 0.08) * 6, 0, 1)
    img = img * (1 - 0.7 * farmland) + plots * (0.7 * farmland)

    # terraces: contour stripes on hillsides
    stripes = (np.mod(H / 7.0 + n_mid * 0.5, 1.0) < 0.5).astype(np.float32)
    terr = np.clip((slope - 0.03) * 12, 0, 1) * (n_big > 0.25)
    img *= (1.0 - 0.12 * terr * stripes)[..., None]

    # eucalyptus woods and banana groves on some hill tops
    n_wood = _value_noise((height, width), (34, 44), rng)
    hilltop = np.clip((n_wood - 0.74) * 6, 0, 1) * np.clip((H - 1550) / 250, 0, 1)
    woods = _hex("#3d6a2e") * (0.8 + 0.4 * (n_fine[..., None] > 0.5))
    img = img * (1 - hilltop[..., None] * 0.7) + woods * (hilltop[..., None] * 0.7)

    # tea estates (bright green rows)
    s = settlements()
    for t_ in s["tea"]:
        d = np.hypot(X - t_["xy"][0], Y - t_["xy"][1])
        w = np.clip((t_["r"] - d) / 0.6, 0, 1) * (n_mid > 0.35)
        rows = (np.mod(H / 3.0, 1.0) < 0.55).astype(np.float32)
        tea = _hex("#3fae3a") * (0.85 + 0.15 * rows)[..., None]
        img = img * (1 - w[..., None]) + tea * w[..., None]

    # valleys along the Nyabarongo
    river = np.array(NYABARONGO)
    dr = np.full(X.shape, np.inf, dtype=np.float32)
    for (ax, ay), (bx, by) in zip(river[:-1], river[1:]):
        vx, vy = bx - ax, by - ay
        tt = np.clip(((X - ax) * vx + (Y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        dr = np.minimum(dr, np.hypot(X - (ax + tt * vx), Y - (ay + tt * vy)))
    marsh = np.clip((1.6 - dr) / 1.2, 0, 1)
    img = img * (1 - marsh[..., None] * 0.6) + _hex("#93c25c") * (marsh[..., None] * 0.6)
    riverline = np.clip((0.12 - dr) / 0.06, 0, 1)
    img = img * (1 - riverline[..., None]) + _hex("#5a7f86") * riverline[..., None]

    # Kigali: light earthy urban ground with trees (3D buildings on top)
    urban = np.zeros(X.shape, dtype=np.float32)
    for k in s["kigali"]:
        d = np.hypot(X - k["xy"][0], Y - k["xy"][1])
        urban = np.maximum(urban, np.clip((k["r"] * 1.3 - d) / (k["r"] * 0.8), 0, 1))
    urban *= np.clip(1.25 - n_mid * 0.7, 0, 1)
    ground = _hex("#b8a58a") * (0.9 + 0.2 * n_fine[..., None])
    trees = (n_fine > 0.82).astype(np.float32)[..., None]
    ground = ground * (1 - trees) + _hex("#4a7a33") * trees
    img = img * (1 - urban[..., None] * 0.8) + ground * (urban[..., None] * 0.8)

    # Lake Muhazi
    dl = terrain._lake_distance(X.astype(np.float64), Y.astype(np.float64)).astype(np.float32)
    water = np.clip((LAKE_HALF_WIDTH_KM + 0.05 - dl) / 0.1, 0, 1)
    shore = np.clip((LAKE_HALF_WIDTH_KM + 0.35 - dl) / 0.3, 0, 1) * (1 - water)
    img = img * (1 - shore[..., None] * 0.5) + _hex("#8db36a") * (shore[..., None] * 0.5)
    img = img * (1 - water[..., None]) + (_hex("#2f6283") + n_mid[..., None] * 18) * water[..., None]

    # soft hill shading (sun from the north-east) and grain
    shade = np.clip(1.0 + (-gx * 0.6 - gy * 0.8) * 1.6, 0.8, 1.15)
    img *= (shade * (0.94 + 0.1 * n_big + 0.04 * (n_fine - 0.5)))[..., None]
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    # roads and the airport (vector drawing)
    draw = ImageDraw.Draw(pil)

    def px(p):
        return ((p[0] - x0) / km_per_px, (y1 - p[1]) / km_per_px)

    for r in s["roads"]:
        pts = [px(p) for p in r["pts"]]
        if r["kind"] == "track":
            draw.line(pts, fill=(184, 104, 62), width=2, joint="curve")
    street_rng = np.random.default_rng(seed + 2)
    for k in s["kigali"]:
        for _ in range(26):
            p = np.array(k["xy"]) + street_rng.normal(0, k["r"] * 0.6, 2)
            ang = street_rng.uniform(0, 2 * np.pi)
            pts = []
            for _ in range(8):
                pts.append(px(p))
                ang += street_rng.normal(0, 0.45)
                p = p + 0.28 * np.array([np.cos(ang), np.sin(ang)])
            draw.line(pts, fill=(196, 170, 140), width=2, joint="curve")
    for r in s["roads"]:
        if r["kind"] == "highway":
            pts = [px(p) for p in r["pts"]]
            draw.line(pts, fill=(168, 104, 70), width=3, joint="curve")
            draw.line(pts, fill=(78, 78, 76), width=2, joint="curve")
    ax, ay = AIRPORT_XY
    rw = [px((ax - 1.75, ay + 0.03)), px((ax + 1.75, ay - 0.03))]
    draw.rectangle([rw[0][0], rw[0][1] - 3, rw[1][0], rw[1][1] + 3], fill=(60, 62, 64))
    for i in range(14):
        xa = rw[0][0] + 10 + i * (rw[1][0] - rw[0][0] - 20) / 14
        draw.line([(xa, (rw[0][1] + rw[1][1]) / 2), (xa + 6, (rw[0][1] + rw[1][1]) / 2)], fill=(235, 235, 235), width=1)
    ap = px((ax - 0.6, ay + 0.35)), px((ax + 0.5, ay + 0.15))
    draw.rectangle([ap[0][0], ap[0][1], ap[1][0], ap[1][1]], fill=(110, 112, 114))
    pil = pil.filter(ImageFilter.SMOOTH)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(path, quality=86, optimize=True)
    pil.resize((1024, int(1024 * height / width))).save(path.with_name("minimap.jpg"), quality=85)
    return path


# 2D top-down image
def topdown_rgb(env, size: int = 900) -> np.ndarray:
    """Top-down map of the current episode as an RGB array."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    x0, x1, y0, y1 = BOUNDS
    fig, ax = plt.subplots(figsize=(size / 100, size / 100 * (y1 - y0) / (x1 - x0)), dpi=100)
    if TEXTURE_PATH.with_name("minimap.jpg").exists():
        ax.imshow(plt.imread(TEXTURE_PATH.with_name("minimap.jpg")), extent=BOUNDS)
    else:
        ax.imshow(env.terrain.grid, origin="lower", extent=BOUNDS, cmap="terrain")
    ax.add_patch(Circle(AIRPORT_XY, AIRPORT_NO_FLY_KM, color="#e24b4a", alpha=0.3))
    ax.annotate("Airport no-fly zone", AIRPORT_XY, color="#7a1f1f", fontsize=7, ha="center")
    for c in env.rain:
        ax.add_patch(Circle(tuple(c.xy), c.radius, color="#5f6f80", alpha=0.35))
    for h in HOSPITALS:
        on = h.name == env.hospital.name
        ax.plot(*h.xy, "s", color="#e24b4a" if on else "white", mec="black", ms=7 if on else 5)
        ax.annotate(h.name.replace(" Hospital", ""), h.xy, xytext=(4, 4), textcoords="offset points", fontsize=6)
    for w in env.waypoints:
        if w.kind == "ring":
            col = {"passed": "#1d9e75", "missed": "#e24b4a"}.get(w.status, "#0f6e56")
            ax.add_patch(Circle(tuple(w.xy), 0.45, fill=False, color=col, lw=1.5))
    path = np.array(env.log.path)
    ax.plot(path[:, 0], path[:, 1], "-", color="#0b3d91", lw=1.6)
    ax.plot(*env.pos, "o", color="#0b3d91", ms=6)
    ax.plot(0, 0, "^", color="black", ms=8)
    ax.annotate("Kigali hub", (0, 0), xytext=(4, -10), textcoords="offset points", fontsize=7)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("km east of hub")
    ax.set_ylabel("km north of hub")
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


# live 3D viewer
class ActRequest(BaseModel):
    """Body of POST /api/act: the 27 observation values in environment order."""
    observation: list[float]


class LiveViewer:
    """Serves the three.js viewer and streams frames to every connected browser."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, policy=None):
        import uvicorn

        from .custom_env import ACTIONS, OBS_DIM

        if not TEXTURE_PATH.exists():
            build_terrain_texture()
        self.host, self.port = host, port
        self.policy = policy
        self.clients: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.last: dict | None = None
        self.history: list[dict] = []
        scene = scene_description(get_terrain())
        scene.update(settlements())
        self.scene = scene

        app = FastAPI(title="Ibaba Life-saver drone API")

        @app.get("/api/health")
        def health():
            return {"status": "ok", "policy_loaded": self.policy is not None}

        @app.get("/api/scene")
        def get_scene():
            return JSONResponse(self.scene)

        @app.post("/api/act")
        def act(req: ActRequest):
            import torch
            if self.policy is None:
                return JSONResponse({"error": "no policy loaded"}, status_code=503)
            if len(req.observation) != OBS_DIM:
                return JSONResponse({"error": f"observation must have {OBS_DIM} values"}, status_code=422)
            with torch.no_grad():
                out = self.policy(torch.tensor(req.observation, dtype=torch.float32).unsqueeze(0))[0]
            a = int(out.argmax())
            return {"action": a, "action_name": ACTIONS[a], "scores": [round(float(v), 4) for v in out]}

        @app.websocket("/ws")
        async def ws(socket: WebSocket):
            await socket.accept()
            self.clients.add(socket)
            try:
                for msg in self.history[-400:]:
                    await socket.send_text(msg)
                while True:
                    await socket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                self.clients.discard(socket)

        app.mount("/", StaticFiles(directory=VIEWER_DIR, html=True), name="viewer")
        self.server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)

    def _run(self):
        async def serve():
            self.loop = asyncio.get_running_loop()
            await self.server.serve()
        asyncio.run(serve())

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def send(self, message: dict) -> None:
        text = json.dumps(message)
        if message.get("type") == "mission":
            self.history = []
        self.history.append(text)
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._broadcast(text), self.loop)

    def push(self, frame: dict) -> None:
        self.last = frame
        self.send({"type": "frame", **frame})

    async def _broadcast(self, text: str):
        for c in list(self.clients):
            try:
                await c.send_text(text)
            except Exception:
                self.clients.discard(c)

    def wait_for_client(self, timeout: float = 30.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.clients:
                return True
            time.sleep(0.2)
        return False

    def stop(self):
        self.server.should_exit = True


_VIEWER: LiveViewer | None = None


def get_viewer(terrain=None, **kwargs) -> LiveViewer:
    global _VIEWER
    if _VIEWER is None:
        _VIEWER = LiveViewer(**kwargs)
    return _VIEWER
