// Ibaba Life-saver 3D viewer (three.js).
// Live mode: frames stream from Python over ws://<host>/ws (see environment/rendering.py).
// Replay mode: viewer.html?replay=replays/demo.json plays a recorded episode without Python.
import * as THREE from "./vendor/three.bundle.min.js";
import { OrbitControls, Sky, mergeGeometries } from "./vendor/three.bundle.min.js";

// ---------------------------------------------------------------- units
// 1 scene unit = 100 m horizontally; heights are exaggerated 2.5x so Rwanda's hills read well.
const S = 10, VEX = 2.5, BASE_ALT = 1100;   // hills drawn 2.5x taller than real
const wx = (x) => x * S;
const wz = (y) => -y * S;
const wy = (alt) => ((alt - BASE_ALT) / 100) * VEX;

const params = new URLSearchParams(location.search);
const REPLAY = params.get("replay");
const SPEED = parseFloat(params.get("speed") || "1");
const LOW = params.get("quality") === "low";   // ?quality=low for slow laptops: no shadows, fewer trees
if (params.get("clean")) for (const id of ["hud", "events", "actionbar", "camera-buttons", "status"]) document.getElementById(id).style.display = "none";

// ---------------------------------------------------------------- renderer, scene, camera
const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = !LOW;
renderer.shadowMap.type = THREE.PCFShadowMap;

const scene = new THREE.Scene();
const HAZE = new THREE.Color("#c4d3e0");
scene.fog = new THREE.Fog(HAZE, 300, 2600);
scene.background = HAZE;

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 8000);
camera.position.set(0, 80, 120);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enabled = false;
controls.maxPolarAngle = Math.PI * 0.49;

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// physical sky (Rayleigh/Mie scattering) with soft clouds; it also lights the scene (image-based lighting)
const SUN_DIR = new THREE.Vector3(-0.5, 0.74, 0.42).normalize();   // afternoon sun from the west-south-west
function makeSky(scale) {
  const sky = new Sky(); sky.scale.setScalar(scale);
  sky.material.fragmentShader = sky.material.fragmentShader.replace(   // the shader is tuned for exposure 0.5
    "gl_FragColor = vec4( texColor, 1.0 );", "gl_FragColor = vec4( texColor * 0.55, 1.0 );");
  const u = sky.material.uniforms;
  u.turbidity.value = 3.2; u.rayleigh.value = 1.6; u.mieCoefficient.value = 0.004; u.mieDirectionalG.value = 0.82;
  u.sunPosition.value.copy(SUN_DIR); u.showSunDisc.value = 0;
  u.cloudCoverage.value = 0.42; u.cloudDensity.value = 0.55; u.cloudScale.value = 0.00035; u.cloudElevation.value = 0.55;
  return sky;
}
const sky = makeSky(4400); scene.add(sky);
let ENV = null;
{
  const pmrem = new THREE.PMREMGenerator(renderer), envScene = new THREE.Scene();
  const envSky = makeSky(100); envSky.material.uniforms.cloudCoverage.value = 0; envScene.add(envSky);
  ENV = pmrem.fromScene(envScene).texture;   // used only by glass and metal (reflections); diffuse fill comes from the hemisphere light
}
const hemi = new THREE.HemisphereLight("#d6e6ff", "#5d6b3c", 0.95);
scene.add(hemi);
const sun = new THREE.DirectionalLight("#fff1d6", 2.6);
sun.castShadow = !LOW;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.04;
scene.add(sun, sun.target);
let shadowSpan = 0;
function updateSun(focus, span) {   // the shadow box follows whatever the camera is looking at
  if (span !== shadowSpan) {
    shadowSpan = span; const c = sun.shadow.camera;
    c.left = c.bottom = -span; c.right = c.top = span; c.near = 1; c.far = 1200; c.updateProjectionMatrix();
  }
  sun.target.position.copy(focus);
  sun.position.copy(focus).addScaledVector(SUN_DIR, 500);
}
updateSun(new THREE.Vector3(), 70);

// ---------------------------------------------------------------- helpers
let SCENE = null, GRID = null;
function heightAt(x, y) {           // terrain height (m) at km coordinates, bilinear on the shared grid
  const g = GRID;
  let fx = (x - g.x0) / g.res, fy = (y - g.y0) / g.res;
  fx = Math.min(Math.max(fx, 0), g.nx - 1.0001); fy = Math.min(Math.max(fy, 0), g.ny - 1.0001);
  const i = Math.floor(fx), j = Math.floor(fy), tx = fx - i, ty = fy - j, k = j * g.nx + i, h = g.heights;
  const a = h[k] + (h[k + 1] - h[k]) * tx, b = h[k + g.nx] + (h[k + g.nx + 1] - h[k + g.nx]) * tx;
  return a + (b - a) * ty;
}
const groundY = (x, y) => wy(heightAt(x, y));
function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

function label(text, opts = {}) {
  const { size = 26, color = "#0f172a", bg = "rgba(255,255,255,0.88)", scale = 1, accent = null } = opts;
  const c = document.createElement("canvas"), ctx = c.getContext("2d");
  ctx.font = `700 ${size}px Segoe UI, system-ui, sans-serif`;
  const w = Math.ceil(ctx.measureText(text).width) + 28, h = size + 18;
  c.width = w; c.height = h;
  ctx.font = `700 ${size}px Segoe UI, system-ui, sans-serif`;
  ctx.fillStyle = bg; ctx.beginPath(); ctx.roundRect(0, 0, w, h, 10); ctx.fill();
  if (accent) { ctx.fillStyle = accent; ctx.fillRect(0, 0, 7, h); }
  ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.fillText(text, accent ? 17 : 14, h / 2 + 1);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true, sizeAttenuation: false }));
  const hh = 0.032 * scale;   // fixed on-screen size, readable at any distance
  spr.scale.set((hh * w) / h, hh, 1);
  spr.renderOrder = 10;
  return spr;
}

function groundDisc(x, y, radiusKm, color, opacity = 0.9, inner = 0) {   // a ring/disc that hugs the terrain
  const geo = new THREE.RingGeometry(inner * S, radiusKm * S, 64, 6);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const px = pos.getX(i), py = pos.getY(i);
    pos.setXYZ(i, wx(x) + px, groundY(x + px / S, y + py / S) + 0.25, wz(y) - py);
  }
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }));
  m.renderOrder = 2;
  return m;
}

const mat = (c, o = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.75, metalness: 0.05, ...o });

// ---------------------------------------------------------------- static world
const world = new THREE.Group();
scene.add(world);
let minimapImg = new Image();

async function buildWorld(sc) {
  SCENE = sc; GRID = sc.terrain;
  const { nx, ny, x0, y0, res, heights } = GRID;
  // terrain mesh
  const pos = new Float32Array(nx * ny * 3), uv = new Float32Array(nx * ny * 2);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const k = j * nx + i, x = x0 + i * res, y = y0 + j * res;
    pos[3 * k] = wx(x); pos[3 * k + 1] = wy(heights[k]) - (inLakeBed(x, y) ? 1.2 : 0); pos[3 * k + 2] = wz(y);
    uv[2 * k] = i / (nx - 1); uv[2 * k + 1] = j / (ny - 1);
  }
  const idx = new Uint32Array((nx - 1) * (ny - 1) * 6);
  let n = 0;
  for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
    idx[n++] = a; idx[n++] = b; idx[n++] = c; idx[n++] = b; idx[n++] = d; idx[n++] = c;
  }
  const tgeo = new THREE.BufferGeometry();
  tgeo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  tgeo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  tgeo.setIndex(new THREE.BufferAttribute(idx, 1));
  tgeo.computeVertexNormals();
  const tex = await new THREE.TextureLoader().loadAsync("terrain_texture.jpg");
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const tmat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95, metalness: 0 });
  const detail = detailTexture();
  tmat.onBeforeCompile = (sh) => {   // aerial-photo detail near the camera: small farm plots, crop rows, terraces, tree crowns
    sh.uniforms.detailMap = { value: detail };
    sh.vertexShader = "varying vec3 vWPos;\nvarying vec3 vWNorm;\n" + sh.vertexShader.replace("#include <project_vertex>", `#include <project_vertex>
      vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz; vWNorm = normalize(mat3(modelMatrix) * objectNormal);`);
    sh.fragmentShader = GROUND_GLSL + sh.fragmentShader.replace("#include <map_fragment>", `#include <map_fragment>
      diffuseColor.rgb = groundDetail(diffuseColor.rgb, vWPos, vWNorm, length(vWPos - cameraPosition));
      float dA = texture2D(detailMap, vMapUv * 900.0).r, dB = texture2D(detailMap, vMapUv * 131.0).g;
      diffuseColor.rgb *= (0.86 + 0.28 * dA) * (0.9 + 0.2 * dB);`);
  };
  const terrain = new THREE.Mesh(tgeo, tmat);
  terrain.receiveShadow = true; terrain.userData.terrain = true;
  world.add(terrain);
  minimapImg.onload = () => drawMinimap(lastFrame);
  minimapImg.src = "minimap.jpg";

  // land beyond the map edge (fades into haze)
  const farm = canvasTex(512, 512, (c, w, h) => {
    const rr = rng(77), cols = ["#6d8f3f", "#7f9a4a", "#5f8238", "#8c8a52", "#738f45", "#9a8a5a", "#58773a"];
    c.fillStyle = "#6c8c40"; c.fillRect(0, 0, w, h);
    c.globalAlpha = 0.55;
    for (let i = 0; i < 900; i++) { c.fillStyle = cols[Math.floor(rr() * cols.length)]; c.fillRect(rr() * w, rr() * h, 8 + rr() * 40, 6 + rr() * 30); }
  });
  farm.repeat.set(140, 140);
  const skirt = new THREE.Mesh(new THREE.PlaneGeometry(9000, 9000), new THREE.MeshStandardMaterial({ map: farm, roughness: 1 }));
  skirt.rotation.x = -Math.PI / 2; skirt.position.y = wy(1250); world.add(skirt);

  buildWater(sc);
  buildVolcanoes();
  planRoads(sc);
  buildKigali(sc);
  buildAirport(sc);
  buildHub();
  buildHospitals(sc);
  buildVillages(sc);
  buildRoads();
  buildSettlementMeshes();
  buildTrees(sc);
  console.log(`viewer: ${HOUSES.length} houses, ${Object.values(BLOCKS).reduce((a, b) => a + b.length, 0)} blocks, ${TREES.reduce((a, b) => a + b.length, 0)} trees`);
  world.traverse((o) => {   // everything solid casts and receives sun shadows
    if (!(o.isMesh || o.isInstancedMesh) || o.userData.terrain || o.userData.noShadow) return;
    const m = Array.isArray(o.material) ? o.material[0] : o.material;
    if (m.transparent || m.isMeshBasicMaterial) return;
    o.castShadow = true; o.receiveShadow = true;
  });
}

// ---------------------------------------------------------------- ground shader: what a low aerial photo of rural Rwanda shows
const GROUND_GLSL = /* glsl */ `
uniform sampler2D detailMap;
varying vec3 vWPos;
varying vec3 vWNorm;
float gh21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
vec2 gh22(vec2 p) { float n = gh21(p); return vec2(n, gh21(p + n + 17.7)); }
float gnoise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(gh21(i), gh21(i + vec2(1, 0)), f.x), mix(gh21(i + vec2(0, 1)), gh21(i + vec2(1, 1)), f.x), f.y); }
vec3 groundDetail(vec3 base, vec3 wp, vec3 wn, float dist) {
  float near = 1.0 - smoothstep(30.0, 95.0, dist);
  if (near <= 0.0) return base;
  vec3 sb = pow(max(base, vec3(0.0)), vec3(1.0 / 2.2)) * 255.0;            // back to 0-255 sRGB to read the land class
  float green = sb.g - max(sb.r, sb.b);
  float farm = smoothstep(4.0, 18.0, green) * (1.0 - smoothstep(0.0, 25.0, sb.b - sb.g));
  float forest = farm * (1.0 - smoothstep(70.0, 95.0, (sb.r + sb.g + sb.b) / 3.0));
  // small strip-shaped plots (20-60 m) whose direction changes from hill to hill: beans, sorghum, maize, cassava, fallow, dug soil
  vec2 bc = floor(wp.xz / 3.0), lp = wp.xz - bc * 3.0;   // each 300 m patch of land has its own strip direction and size
  float th = gh21(bc + 3.1) * 3.1416, cs = cos(th), sn = sin(th);
  vec2 q = vec2(cs * lp.x - sn * lp.y, sn * lp.x + cs * lp.y);
  q.y += 0.04 * sin(q.x * 3.1 + bc.x) + 0.02 * sin(q.x * 7.3 + bc.y);   // strips follow the slope, not a ruler
  float rowH = 0.16 + 0.16 * gh21(bc + 9.7), row = floor(q.y / rowH);
  float rh = gh21(vec2(row, bc.x * 7.0 + bc.y * 13.0)), len = 0.3 + 0.7 * rh, xo = q.x + rh * 3.7, colI = floor(xo / len);
  vec2 cid = vec2(row, colI) + bc * 31.0;
  float fx = fract(xo / len), fy = fract(q.y / rowH);
  float edgeDist = min(min(fx, 1.0 - fx) * len, min(fy, 1.0 - fy) * rowH);   // distance to the plot edge (scene units)
  float border = 1.0 - smoothstep(0.004, 0.016, edgeDist);
  float h = gh21(cid), h2 = gh21(cid + 7.13);
  vec3 c;   // typical field colours seen from the air (sRGB), tinted by the region
  if (h < 0.11) c = vec3(128.0, 88.0, 60.0);           // freshly dug red-brown soil
  else if (h < 0.22) c = vec3(150.0, 142.0, 86.0);     // ripening sorghum, dry grass
  else if (h < 0.42) c = vec3(62.0, 96.0, 42.0);       // dark beans, cassava
  else if (h < 0.6) c = vec3(118.0, 150.0, 60.0);      // young maize
  else if (h < 0.76) c = vec3(92.0, 132.0, 52.0);      // fresh green
  else if (h < 0.88) c = vec3(52.0, 86.0, 38.0);       // banana grove
  else c = vec3(112.0, 128.0, 72.0);                    // fallow grass
  vec3 crop = mix(pow(c * (0.9 + 0.2 * h2) / 255.0, vec3(2.2)), base, 0.35);
  float ang = h2 * 3.1416, rows = 0.5 + 0.5 * sin(dot(wp.xz, vec2(cos(ang), sin(ang))) * 110.0);
  crop *= 1.0 + (rows - 0.5) * 0.16 * (1.0 - smoothstep(6.0, 18.0, dist));
  vec3 fields = mix(crop, pow(vec3(52.0, 80.0, 38.0) / 255.0, vec3(2.2)), border * 0.42 * (1.0 - smoothstep(15.0, 45.0, dist)));   // hedgerows and paths
  // radical terraces on the slopes: contour lines every ~2.5 m of height
  float slope = 1.0 - clamp(wn.y, 0.0, 1.0);
  float band = fract(wp.y / 0.0625 + gh21(floor(wp.xz / 2.0)) * 0.3);
  fields *= 1.0 - smoothstep(0.02, 0.1, slope) * 0.2 * smoothstep(0.7, 0.95, band) * (1.0 - smoothstep(10.0, 35.0, dist));
  // woodland: individual tree crowns (~8 m)
  vec2 tc = floor(wp.xz / 0.08); float crown = gh21(tc);
  vec3 woods = base * (0.65 + 0.7 * crown * crown);
  vec3 col = mix(fields, woods, forest);
  return mix(base, col, near * farm * 0.92);
}
`;

function detailTexture() {
  const N = 256, c = document.createElement("canvas"); c.width = c.height = N; const g = c.getContext("2d");
  const small = document.createElement("canvas"); small.width = small.height = 32; const sg = small.getContext("2d"), r = rng(5);
  const img = sg.createImageData(32, 32);
  for (let i = 0; i < 32 * 32; i++) { img.data[i * 4] = 60 + r() * 136; img.data[i * 4 + 1] = 60 + r() * 136; img.data[i * 4 + 2] = 128; img.data[i * 4 + 3] = 255; }
  sg.putImageData(img, 0, 0);
  g.imageSmoothingEnabled = true; g.drawImage(small, 0, 0, N, N);   // soft blotches
  const fine = g.getImageData(0, 0, N, N);
  for (let i = 0; i < N * N; i++) { const k = (r() - 0.5) * 70; fine.data[i * 4] = Math.max(0, Math.min(255, fine.data[i * 4] + k)); }   // grain
  g.putImageData(fine, 0, 0);
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8;
  return t;
}

function ribbon(points, halfWidthKm, y, material) {
  const verts = [], index = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i], q = points[Math.min(i + 1, points.length - 1)], o = points[Math.max(i - 1, 0)];
    let dx = q[0] - o[0], dy = q[1] - o[1]; const L = Math.hypot(dx, dy) || 1; dx /= L; dy /= L;
    const nxk = -dy * halfWidthKm, nyk = dx * halfWidthKm;
    verts.push(wx(p[0] + nxk), y, wz(p[1] + nyk), wx(p[0] - nxk), y, wz(p[1] - nyk));
    if (i < points.length - 1) { const a = 2 * i; index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3)); g.setIndex(index); g.computeVertexNormals();
  return new THREE.Mesh(g, material);
}

function buildWater(sc) {
  const water = new THREE.MeshStandardMaterial({ color: "#24576a", roughness: 0.14, metalness: 0.05, envMap: ENV, envMapIntensity: 0.45 });
  const lvl = wy(GRID.lake_level + 4);
  const lake = sc.lake_muhazi;
  world.add(ribbon(lake, GRID.lake_half_width + 0.12, lvl, water));
  for (const p of [lake[0], lake[lake.length - 1]]) {
    const cap = new THREE.Mesh(new THREE.CircleGeometry((GRID.lake_half_width + 0.12) * S, 24), water);
    cap.rotation.x = -Math.PI / 2; cap.position.set(wx(p[0]), lvl, wz(p[1])); world.add(cap);
  }
  const mid = lake[Math.floor(lake.length / 2)];
  const l = label("Lake Muhazi", { color: "#0c4a6e", scale: 0.9 }); l.position.set(wx(mid[0]), lvl + 9, wz(mid[1])); world.add(l);
}

function buildVolcanoes() {   // Virunga chain on the north-west horizon
  const vols = [["Karisimbi", -98, 58, 4507, 120], ["Bisoke", -91, 64, 3711, 80], ["Sabyinyo", -82, 70, 3634, 85],
                ["Gahinga", -75, 73, 3474, 70], ["Muhabura", -69, 77, 4127, 100]];
  const m = new THREE.MeshStandardMaterial({ color: "#4d6655", roughness: 1 });
  for (const [name, x, y, h, r] of vols) {
    const H = wy(h) - wy(1300), prof = [], rr = rng(Math.round(h));
    for (let i = 0; i <= 24; i++) { const t = i / 24; prof.push(new THREE.Vector2(r * (0.06 + 0.94 * Math.pow(1 - t, 1.9)), t * H)); }
    prof.push(new THREE.Vector2(r * 0.03, H * 0.97));   // crater
    const geo = new THREE.LatheGeometry(prof, 64), p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) {   // gullies and ridges
      const px = p.getX(i), pz = p.getZ(i), py = p.getY(i), a = Math.atan2(pz, px);
      const k = 1 + 0.06 * Math.sin(a * 11 + rr() * 0.2) + 0.04 * Math.sin(a * 23 + py * 0.05);
      p.setX(i, px * k); p.setZ(i, pz * k);
    }
    geo.computeVertexNormals();
    const cone = new THREE.Mesh(geo, m); cone.userData.noShadow = true;
    cone.position.set(wx(x), wy(1300), wz(y)); world.add(cone);
  }
  const l = label("Virunga volcanoes", { scale: 1.0, color: "#14532d" });
  l.position.set(wx(-82), wy(4300), wz(70)); world.add(l);
}

// ---------------------------------------------------------------- building materials and textures
const ll = (lat, lon) => [(lon - 30.0870) * 111.32 * Math.cos((-1.954 * Math.PI) / 180), (lat + 1.954) * 110.57];   // same projection as world.py

function canvasTex(w, h, draw, repeat = true) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

// windows: one tile = 4 bays x 4 floors
function windowsTex(wall, glass, frame, seed = 1) {
  const r = rng(seed);
  return canvasTex(256, 256, (g, w, h) => {
    g.fillStyle = wall; g.fillRect(0, 0, w, h);
    const bw = w / 4, fh = h / 4;
    for (let f = 0; f < 4; f++) {
      g.fillStyle = "rgba(0,0,0,0.10)"; g.fillRect(0, f * fh + fh - 5, w, 5);   // floor slab line
      for (let b = 0; b < 4; b++) {
        const x = b * bw, y = f * fh;
        g.fillStyle = frame; g.fillRect(x + bw * 0.12, y + fh * 0.2, bw * 0.76, fh * 0.6);
        const l = 0.85 + r() * 0.3;
        g.fillStyle = glass; g.globalAlpha = 1; g.fillRect(x + bw * 0.16, y + fh * 0.24, bw * 0.68, fh * 0.52);
        const gr = g.createLinearGradient(0, y + fh * 0.24, 0, y + fh * 0.76);
        gr.addColorStop(0, `rgba(255,255,255,${0.22 * l})`); gr.addColorStop(0.5, "rgba(255,255,255,0.04)"); gr.addColorStop(1, "rgba(0,0,0,0.12)");
        g.fillStyle = gr; g.fillRect(x + bw * 0.16, y + fh * 0.24, bw * 0.68, fh * 0.52);
        g.fillStyle = frame; g.fillRect(x + bw * 0.49, y + fh * 0.24, 3, fh * 0.52);
      }
    }
  });
}

// glass curtain wall: mullions, floor slabs and slightly different panes
function curtainTex(base, seed = 2) {
  const r = rng(seed), c0 = new THREE.Color(base);
  return canvasTex(256, 256, (g, w, h) => {
    for (let f = 0; f < 8; f++) for (let b = 0; b < 8; b++) {
      const c = c0.clone().offsetHSL(0, 0, (r() - 0.5) * 0.07 + (f % 2 ? 0.02 : 0));
      g.fillStyle = `#${c.getHexString()}`; g.fillRect(b * 32, f * 32, 32, 32);
    }
    const gr = g.createLinearGradient(0, 0, w, h);
    gr.addColorStop(0, "rgba(255,255,255,0.18)"); gr.addColorStop(0.5, "rgba(255,255,255,0)"); gr.addColorStop(1, "rgba(255,255,255,0.1)");
    g.fillStyle = gr; g.fillRect(0, 0, w, h);
    g.fillStyle = "rgba(210,220,228,0.9)";
    for (let i = 0; i < 8; i++) { g.fillRect(i * 32, 0, 2, h); g.fillRect(0, i * 32, w, 3); }
  });
}

// facade material for instanced boxes: the texture repeats with the building's real width and height
function facadeMat(tex, axis, tileW, tileH, o = {}) {
  const m = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.55, metalness: 0.1, envMap: ENV, envMapIntensity: 0.6, ...o });
  m.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader.replace("#include <uv_vertex>", `#include <uv_vertex>
      #ifdef USE_INSTANCING
        mat4 wm = modelMatrix * instanceMatrix;
        vMapUv *= vec2(length(wm[${axis}].xyz) / ${tileW.toFixed(4)}, length(wm[1].xyz) / ${tileH.toFixed(4)});
      #endif`);
  };
  m.customProgramCacheKey = () => `facade-${axis}-${tileW}-${tileH}-${tex.uuid}`;
  return m;
}
const ROOF_MAT = new THREE.MeshStandardMaterial({ color: "#a9a59c", roughness: 0.92 });
const FACADES = {};
function facadeSet(kind) {   // [+x, -x, +y, -y, +z, -z] faces of a BoxGeometry
  if (!FACADES[kind]) {
    const spec = {
      office: [windowsTex("#f1efe9", "#2c4a63", "#d9d6cf", 3), 14 * M, 4 * 3.4 * MV, {}],          // 4 bays x 4 floors
      apartment: [windowsTex("#efe4d2", "#3b4a55", "#c9b89c", 4), 13 * M, 4 * 3.2 * MV, {}],
      glass: [curtainTex("#35607f", 5), 14 * M, 8 * 3.4 * MV, { metalness: 0.35, roughness: 0.2, envMapIntensity: 0.55 }],   // 8 x 8 panes
      greenglass: [curtainTex("#2f6866", 6), 14 * M, 8 * 3.4 * MV, { metalness: 0.35, roughness: 0.2, envMapIntensity: 0.55 }],
    }[kind];
    const [tex, tw, th, o] = spec;
    const fx = facadeMat(tex, 2, tw, th, o), fz = facadeMat(tex, 0, tw, th, o);
    FACADES[kind] = [fx, fx, ROOF_MAT, ROOF_MAT, fz, fz];
  }
  return FACADES[kind];
}
function facadeBoxes(list, kind, parent = world) {   // list of {x, y, z, w, d, hgt, rot, color?}
  if (!list.length) return null;
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), facadeSet(kind), list.length);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), c = new THREE.Color();
  list.forEach((b, i) => {
    q.setFromEuler(new THREE.Euler(0, b.rot || 0, 0));
    m4.compose(new THREE.Vector3(b.x, b.y + b.hgt / 2, b.z), q, new THREE.Vector3(b.w, b.hgt, b.d));
    mesh.setMatrixAt(i, m4); mesh.setColorAt(i, c.set(b.color || "#ffffff"));
  });
  parent.add(mesh);
  return mesh;
}

function chunked(list, getXZ, cell = 100) {   // 10 km tiles: the camera and the shadow pass skip tiles out of view
  const tiles = new Map();
  for (const item of list) { const [x, z] = getXZ(item), k = `${Math.floor(x / cell)},${Math.floor(z / cell)}`; if (!tiles.has(k)) tiles.set(k, []); tiles.get(k).push(item); }
  return [...tiles.values()];
}

// hip roof over a 1 x 1 footprint (ridge along x), the most common roof shape in Rwanda
function hipRoofGeometry() {
  const A = [-0.5, 0, -0.5], B = [0.5, 0, -0.5], C = [0.5, 0, 0.5], D = [-0.5, 0, 0.5], E = [-0.22, 1, 0], F = [0.22, 1, 0];
  const tri = [A, E, F, A, F, B, D, C, F, D, F, E, A, D, E, B, F, C].flat();
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(tri, 3));
  g.computeVertexNormals();
  return g;
}

// instanced houses: walls + hip roof (iron sheet or clay tile), colours per instance

const HOUSE_GEO = { body: new THREE.BoxGeometry(1, 1, 1), roof: hipRoofGeometry() };
const HOUSE_MAT = { body: new THREE.MeshStandardMaterial({ roughness: 0.9 }), roof: new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.35, envMap: ENV, envMapIntensity: 0.5 }) };
function houseTile(list, roofPalette, wallPalette) {
  const body = new THREE.InstancedMesh(HOUSE_GEO.body, HOUSE_MAT.body, list.length);
  const roof = new THREE.InstancedMesh(HOUSE_GEO.roof, HOUSE_MAT.roof, list.length);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), c = new THREE.Color();
  list.forEach((h, i) => {
    q.setFromEuler(new THREE.Euler(0, h.rot, 0));
    m4.compose(new THREE.Vector3(h.x, h.y + h.hgt / 2 - 0.03, h.z), q, new THREE.Vector3(h.w, h.hgt + 0.06, h.d));
    body.setMatrixAt(i, m4); body.setColorAt(i, c.set(wallPalette[h.wall % wallPalette.length]));
    m4.compose(new THREE.Vector3(h.x, h.y + h.hgt, h.z), q, new THREE.Vector3(h.w * 1.14, h.roofH, h.d * 1.18));
    roof.setMatrixAt(i, m4); roof.setColorAt(i, c.set(roofPalette[h.roof % roofPalette.length]));
  });
  body.computeBoundingSphere(); roof.computeBoundingSphere();
  world.add(body, roof);
  return [body, roof];
}
const ROOFS = ["#9aa1a7", "#b2b8bc", "#8b4a32", "#a9432a", "#6e7479", "#c0582f", "#9aa1a7", "#7a3f2e", "#8a9096", "#4f7f78"];   // iron sheet, rusty iron, clay tile, painted iron
const WALLS = ["#efe6d6", "#e6d3b5", "#f5f1ea", "#d9b98f", "#b9694a", "#e8cfbf"];                        // plaster, ochre, burnt brick

// a flat pad that follows the ground (plazas, aprons, pitches)
function groundPad(x, y, geo, material, lift = 0.06) {
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const px = pos.getX(i), pz = pos.getZ(i);
    pos.setY(i, groundY(x + px / S, y - pz / S) + lift);
  }
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, material);
  m.position.set(wx(x), 0, wz(y));
  m.receiveShadow = true;
  material.polygonOffset = true; material.polygonOffsetFactor = -2;
  return m;
}
function flatShape(shape, segments = 48) { const g = new THREE.ShapeGeometry(shape, segments); g.rotateX(-Math.PI / 2); return g; }
function ellipseShape(rx, rz) { const s = new THREE.Shape(); s.absellipse(0, 0, rx, rz, 0, Math.PI * 2, false, 0); return s; }
function roundedRectShape(w, d, r) {
  const s = new THREE.Shape(), x = -w / 2, y = -d / 2;
  s.moveTo(x + r, y); s.lineTo(x + w - r, y); s.quadraticCurveTo(x + w, y, x + w, y + r); s.lineTo(x + w, y + d - r);
  s.quadraticCurveTo(x + w, y + d, x + w - r, y + d); s.lineTo(x + r, y + d); s.quadraticCurveTo(x, y + d, x, y + d - r);
  s.lineTo(x, y + r); s.quadraticCurveTo(x, y, x + r, y);
  return s;
}

function inLakeBed(x, y) {
  const L = SCENE.lake_muhazi;
  for (let i = 0; i < L.length - 1; i++) {
    const [ax, ay] = L[i], [bx, by] = L[i + 1], vx = bx - ax, vy = by - ay;
    const t = Math.max(0, Math.min(1, ((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy)));
    if (Math.hypot(x - ax - t * vx, y - ay - t * vy) < GRID.lake_half_width + 0.05) return true;
  }
  return false;
}
function inLake(x, y) {
  const L = SCENE.lake_muhazi;
  for (let i = 0; i < L.length - 1; i++) {
    const [ax, ay] = L[i], [bx, by] = L[i + 1], vx = bx - ax, vy = by - ay;
    const t = Math.max(0, Math.min(1, ((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy)));
    if (Math.hypot(x - ax - t * vx, y - ay - t * vy) < GRID.lake_half_width + 0.4) return true;
  }
  return false;
}
function nearAirport(x, y, m = 0) { const a = SCENE.airport.xy; return Math.hypot(x - a[0], y - a[1]) < 2.2 + m; }

// ---------------------------------------------------------------- scale: real sizes
// Horizontal: 1 scene unit = 100 m. Buildings are drawn 1.6x taller than real (the hills 2.5x), so
// everything keeps believable proportions from the drone's camera.
const M = 0.01;      // metres -> scene units (horizontal)
const MV = 0.016;    // metres -> scene units (vertical, buildings)

// tiles that are only drawn near the camera (houses, trees): far away they are sub-pixel anyway
const LOD_TILES = [];
function lodAdd(mesh, range) {
  mesh.computeBoundingSphere();
  LOD_TILES.push({ mesh, range: range + mesh.boundingSphere.radius, c: mesh.boundingSphere.center.clone() });
}
function updateLod() {
  const p = camera.position;
  for (const t of LOD_TILES) t.mesh.visible = Math.hypot(p.x - t.c.x, p.z - t.c.z) < t.range * (LOW ? 0.7 : 1);
}

// ---------------------------------------------------------------- Kigali city and landmarks
const LANDMARKS = {
  kcc: ll(-1.9545, 30.0930),          // Kigali Convention Centre (Kimihurura)
  cityTower: ll(-1.9445, 30.0600),    // Kigali City Tower (CBD)
  amahoro: ll(-1.9557, 30.1188),      // Amahoro Stadium (Remera)
  bkArena: ll(-1.9527, 30.1163),      // BK Arena (Remera)
  cbdRoundabout: ll(-1.9470, 30.0620),
};
const CLEAR = [[0, 0, 0.16], [...LANDMARKS.kcc, 0.3], [...LANDMARKS.cityTower, 0.1], [...LANDMARKS.amahoro, 0.19], [...LANDMARKS.bkArena, 0.12],
               [...LANDMARKS.cbdRoundabout, 0.07]];
const isClear = (x, y) => CLEAR.every(([cx, cy, r]) => Math.hypot(x - cx, y - cy) > r);

// spatial hash of road points, so houses keep off the streets
class RoadIndex {
  constructor(cell = 0.05) { this.cell = cell; this.map = new Map(); }
  key(x, y) { return (Math.floor(x / this.cell) + 5000) * 10000 + Math.floor(y / this.cell) + 5000; }
  add(pts, id, half) { for (const [x, y] of pts) { const k = this.key(x, y); if (!this.map.has(k)) this.map.set(k, []); this.map.get(k).push([x, y, id, half]); } }
  clear(x, y, margin, ignore = -1) {   // true if no road (other than `ignore`) is closer than its half-width + margin
    const cx = Math.floor(x / this.cell), cy = Math.floor(y / this.cell);
    for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
      const list = this.map.get((cx + i + 5000) * 10000 + cy + j + 5000); if (!list) continue;
      for (const [px, py, id, half] of list) if (id !== ignore && Math.hypot(x - px, y - py) < half + margin) return false;
    }
    return true;
  }
}

function resample(pts, step = 0.04) {   // km polyline -> points every `step` km
  const out = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    const [ax, ay] = pts[i - 1], [bx, by] = pts[i], L = Math.hypot(bx - ax, by - ay), n = Math.max(1, Math.ceil(L / step));
    for (let k = 1; k <= n; k++) out.push([ax + ((bx - ax) * k) / n, ay + ((by - ay) * k) / n]);
  }
  return out;
}
function curve(a, b, bend, seed) {   // gently curved avenue between two points
  const r = rng(seed), mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
  const c = [mx - (dy / L) * bend * L, my + (dx / L) * bend * L], pts = [];
  for (let i = 0; i <= 40; i++) {
    const t = i / 40, u = 1 - t;
    const w = 0.04 * Math.sin(t * 9 + r() * 0.3) * Math.sin(t * Math.PI);
    pts.push([u * u * a[0] + 2 * u * t * c[0] + t * t * b[0] - (dy / L) * w, u * u * a[1] + 2 * u * t * c[1] + t * t * b[1] + (dx / L) * w]);
  }
  return resample(pts);
}
function streetGrid(cx, cy, radius, spacing, theta, seed, wiggle = 0.03) {   // two families of winding streets, with gaps
  const r = rng(seed), lines = [];
  for (const fam of [0, 1]) {
    const n = Math.floor(radius / spacing);
    for (let k = -n; k <= n; k++) {
      const th = theta + fam * Math.PI / 2 + (r() - 0.5) * 0.35, u = [Math.cos(th), Math.sin(th)], v = [-u[1], u[0]];
      const off = k * spacing + (r() - 0.5) * spacing * 0.4;
      const half = Math.sqrt(Math.max(0, radius * radius - off * off)) * (0.7 + r() * 0.3);
      if (half < spacing * 0.8) continue;
      const ph = r() * 6.28, f1 = 1.6 + r() * 1.6, f2 = 4 + r() * 4;
      let pts = [];
      for (let s = -half; s <= half + 1e-9; s += 0.035) {
        const w = wiggle * 1.6 * Math.sin(s * f1 + ph) + wiggle * 0.6 * Math.sin(s * f2 + ph * 2);
        pts.push([cx + u[0] * s + v[0] * (off + w), cy + u[1] * s + v[1] * (off + w)]);
        if (pts.length > 8 && r() < 0.012) { lines.push(pts); pts = []; s += 0.05 + r() * 0.2; }   // the street stops, restarts further on
      }
      if (pts.length > 3) lines.push(pts);
    }
  }
  return lines;
}

// ---------------------------------------------------------------- road surfaces
const ROAD_TEX = {
  asphalt: () => canvasTex(64, 256, (c, w, h) => {   // u across the road, v along it
    c.fillStyle = "#43464a"; c.fillRect(0, 0, w, h);
    const r = rng(3); for (let i = 0; i < 900; i++) { c.fillStyle = `rgba(${r() < 0.5 ? 0 : 255},${r() < 0.5 ? 0 : 255},255,0.05)`; c.fillRect(r() * w, r() * h, 2, 2); }
    c.fillStyle = "#c9c6bd"; c.fillRect(0, 0, 5, h); c.fillRect(w - 5, 0, 5, h);   // kerbs
    c.fillStyle = "rgba(240,240,235,0.9)"; c.fillRect(8, 0, 2, h); c.fillRect(w - 10, 0, 2, h);   // edge lines
    for (let y = 0; y < h; y += 64) c.fillRect(w / 2 - 1.5, y, 3, 34);                       // centre dashes
  }),
  cobble: () => canvasTex(64, 128, (c, w, h) => {
    c.fillStyle = "#77726a"; c.fillRect(0, 0, w, h);
    const r = rng(4); for (let y = 0; y < h; y += 6) for (let x = (y / 6) % 2 ? 0 : 3; x < w; x += 6) { c.fillStyle = `rgba(${r() < 0.5 ? 60 : 200},${r() < 0.5 ? 55 : 190},50,0.18)`; c.fillRect(x, y, 5, 5); }
    c.fillStyle = "#b5b0a5"; c.fillRect(0, 0, 4, h); c.fillRect(w - 4, 0, 4, h);
  }),
  murram: () => canvasTex(64, 128, (c, w, h) => {
    const g = c.createLinearGradient(0, 0, w, 0);
    g.addColorStop(0, "rgba(150,98,62,0)"); g.addColorStop(0.18, "rgba(150,98,62,1)"); g.addColorStop(0.82, "rgba(150,98,62,1)"); g.addColorStop(1, "rgba(150,98,62,0)");
    c.fillStyle = g; c.fillRect(0, 0, w, h);
    const r = rng(5); for (let i = 0; i < 500; i++) { c.fillStyle = `rgba(${90 + r() * 80},${60 + r() * 40},${40 + r() * 20},0.35)`; c.fillRect(12 + r() * 40, r() * h, 2, 3); }
    c.fillStyle = "rgba(110,70,45,0.35)"; c.fillRect(w * 0.3, 0, 4, h); c.fillRect(w * 0.66, 0, 4, h);   // wheel ruts
  }),
};
const ROAD_KIND = {   // full width in metres (drawn a little wider than real so they read from the air), surface, lift
  avenue: { width: 24, tex: "asphalt", lift: 0.05, along: 0.6 },
  highway: { width: 20, tex: "asphalt", lift: 0.045, along: 0.6 },
  street: { width: 8, tex: "cobble", lift: 0.04, along: 0.3 },
  murram: { width: 7, tex: "murram", lift: 0.035, along: 0.3 },
  track: { width: 6, tex: "murram", lift: 0.03, along: 0.3 },
};
const ROAD_LINES = { avenue: [], highway: [], street: [], murram: [], track: [] };
function buildRoads() {   // one draped ribbon mesh per surface type
  for (const [kind, lines] of Object.entries(ROAD_LINES)) {
    if (!lines.length) continue;
    const spec = ROAD_KIND[kind], half = (spec.width / 2) * M, pos = [], uv = [], idx = [];
    for (const raw of lines) {
      const pts = resample(raw, 0.04); let along = 0;
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i], q = pts[Math.min(i + 1, pts.length - 1)], o = pts[Math.max(i - 1, 0)];
        let dx = q[0] - o[0], dy = q[1] - o[1]; const L = Math.hypot(dx, dy) || 1; dx /= L; dy /= L;
        if (i) along += Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) * S;
        const nx = -dy * half / S, ny = dx * half / S, base = pos.length / 3;
        for (const s of [1, -1]) {
          const x = p[0] + nx * s, y = p[1] + ny * s;
          pos.push(wx(x), groundY(x, y) + spec.lift, wz(y)); uv.push(s > 0 ? 0 : 1, along / spec.along);
        }
        if (i < pts.length - 1) idx.push(base, base + 1, base + 2, base + 1, base + 3, base + 2);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx); g.computeVertexNormals();
    const m = new THREE.MeshStandardMaterial({ map: ROAD_TEX[spec.tex](), roughness: 0.92, transparent: spec.tex === "murram", depthWrite: spec.tex !== "murram",
                                               polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 });
    const mesh = new THREE.Mesh(g, m); mesh.receiveShadow = true; mesh.userData.noShadow = true; mesh.renderOrder = 1;
    world.add(mesh);
  }
}
function roundabout(x, y, rOuter) {   // asphalt ring around a planted island
  const gy = groundY(x, y), ring = new THREE.Mesh(new THREE.RingGeometry(rOuter * 0.55, rOuter, 48), new THREE.MeshStandardMaterial({ color: "#45484c", roughness: 0.9, polygonOffset: true, polygonOffsetFactor: -5 }));
  ring.rotation.x = -Math.PI / 2; ring.position.set(wx(x), gy + 0.06, wz(y)); ring.userData.noShadow = true; world.add(ring);
  const island = new THREE.Mesh(new THREE.CylinderGeometry(rOuter * 0.55, rOuter * 0.58, 0.05, 40), mat("#4e8a35", { roughness: 1 }));
  island.position.set(wx(x), gy + 0.05, wz(y)); world.add(island);
  const kerb = new THREE.Mesh(new THREE.TorusGeometry(rOuter * 0.56, 0.008, 6, 48), mat("#e5e2da")); kerb.rotation.x = -Math.PI / 2; kerb.position.set(wx(x), gy + 0.08, wz(y)); world.add(kerb);
}

// ---------------------------------------------------------------- neighbourhoods: streets lined with houses
const HOUSES = [], BLOCKS = { office: [], apartment: [], glass: [], greenglass: [] }, TREES = [[], [], []];   // trees: eucalyptus, broadleaf, banana
const ROADIDX = new RoadIndex();
let streetId = 0;
function okSite(x, y) { return isClear(x, y) && !inLake(x, y) && !nearAirport(x, y, -0.2); }
function addHouse(r, x, y, ang, style) {
  const big = style === "villa", w = (big ? 13 : 8) + r() * (big ? 8 : 6), d = (big ? 10 : 7) + r() * (big ? 5 : 4);
  HOUSES.push({ x: wx(x), y: groundY(x, y), z: wz(y), w: w * M, d: d * M, hgt: (big ? 4.2 : 3.1 + r() * 0.6) * MV,
                roofH: (big ? 2.6 : 1.8 + r() * 0.6) * MV, rot: ang + (r() - 0.5) * 0.12, roof: Math.floor(r() * 10), wall: Math.floor(r() * 6) });
}
function lineStreet(pts, kind) {
  const id = streetId++; ROAD_LINES[kind].push(pts); ROADIDX.add(resample(pts, 0.03), id, (ROAD_KIND[kind].width / 2) / 1000); return id;
}
// walk along a street and put buildings on both sides, facing it
function lineBuildings(pts, id, step, offset, prob, r, place) {
  let carry = 0;
  for (let i = 1; i < pts.length; i++) {
    const [ax, ay] = pts[i - 1], [bx, by] = pts[i], L = Math.hypot(bx - ax, by - ay); if (!L) continue;
    const tx = (bx - ax) / L, ty = (by - ay) / L, ang = Math.atan2(ty, tx);
    for (let s = carry; s < L; s += step * (0.85 + r() * 0.3)) {
      for (const side of [1, -1]) {
        if (r() > prob) continue;
        const off = offset * (0.9 + r() * 0.35), x = ax + tx * s - ty * off * side, y = ay + ty * s + tx * off * side;
        if (okSite(x, y) && ROADIDX.clear(x, y, offset * 0.55, id)) place(x, y, ang);
      }
      carry = s + step - L;
    }
  }
}
function neighbourhood(cx, cy, radius, style, seed) {
  const r = rng(seed), dense = style === "dense", spacing = style === "cbd" ? 0.17 : style === "villa" ? 0.22 : style === "village" ? 0.15 : 0.17;
  const lines = streetGrid(cx, cy, radius, spacing, r() * Math.PI, seed, style === "village" ? 0.04 : 0.03);
  const kind = style === "cbd" || style === "villa" ? "street" : "murram";
  const ids = lines.map((pts) => lineStreet(pts, style === "dense" && r() < 0.3 ? "street" : kind));
  lines.forEach((pts, i) => {
    if (style === "cbd") {
      lineBuildings(pts, ids[i], 0.055, 0.04, 0.8, r, (x, y, a) => {
        const d = Math.hypot(x - LANDMARKS.cityTower[0], y - LANDMARKS.cityTower[1]), tall = Math.max(0, 1.1 - d / 1.4);
        const storeys = 3 + Math.floor(r() * (4 + 14 * tall)), kindB = storeys > 8 ? (r() < 0.6 ? "glass" : "greenglass") : r() < 0.5 ? "office" : "apartment";
        BLOCKS[kindB].push({ x: wx(x), y: groundY(x, y), z: wz(y), w: (22 + r() * 22) * M, d: (16 + r() * 16) * M, hgt: storeys * 3.4 * MV, rot: a,
                             color: ["#ffffff", "#f3efe6", "#eef2f4", "#fbf3e4"][Math.floor(r() * 4)] });
      });
    } else {
      const step = dense ? 0.022 : style === "villa" ? 0.042 : style === "village" ? 0.034 : 0.03;
      const prob = dense ? 0.85 : style === "village" ? 0.42 : 0.7, off = style === "villa" ? 0.026 : 0.017;
      if (dense) lineBuildings(pts, ids[i], 0.026, 0.043, 0.5, r, (x, y, a) => addHouse(r, x, y, a, style));   // back row
      lineBuildings(pts, ids[i], step, off, prob, r, (x, y, a) => {
        if (style !== "village" && r() < 0.035) {   // a few apartment blocks and shops among the houses
          BLOCKS[r() < 0.6 ? "apartment" : "office"].push({ x: wx(x), y: groundY(x, y), z: wz(y), w: (18 + r() * 10) * M, d: (12 + r() * 6) * M,
            hgt: (2 + Math.floor(r() * 3)) * 3.3 * MV, rot: a, color: ["#ffffff", "#fbf3e4", "#f3e3d3", "#fde7cf"][Math.floor(r() * 4)] });
        } else addHouse(r, x, y, a, style);
        if (r() < (style === "village" ? 0.5 : 0.18)) {   // garden: banana plants or a tree behind the house
          const bx = x + (r() - 0.5) * 0.03, by = y + (r() - 0.5) * 0.03;
          if (ROADIDX.clear(bx, by, 0.004)) TREES[style === "village" || r() < 0.5 ? 2 : 1].push([bx, by]);
        }
      });
    }
  });
  // trees inside the blocks
  const nT = Math.round(radius * radius * (style === "cbd" ? 30 : style === "village" ? 110 : 60));
  for (let i = 0; i < nT; i++) {
    const a = r() * 6.283, d = radius * Math.sqrt(r()), x = cx + Math.cos(a) * d, y = cy + Math.sin(a) * d;
    if (okSite(x, y) && ROADIDX.clear(x, y, 0.006)) TREES[r() < 0.45 ? 0 : r() < 0.7 ? 1 : 2].push([x, y]);
  }
}

function buildKigali(sc) {
  const lm = LANDMARKS, A = sc.kigali.map((k) => k.xy);
  // avenues joining the neighbourhood centres (asphalt, lined with trees)
  const hub = [0, 0], cbd = lm.cbdRoundabout, airport = SCENE.airport.xy;
  const links = [[hub, lm.kcc], [lm.kcc, cbd], [cbd, A[2]], [lm.kcc, A[3]], [A[3], [airport[0] - 0.9, airport[1] + 0.3]], [hub, A[4]], [cbd, A[5]],
                 [hub, A[6]], [A[3], A[7]], [cbd, A[8]], [A[8], A[4]], [A[6], A[5]]];
  links.forEach(([a, b], i) => {
    const pts = curve(a, b, (i % 2 ? 0.08 : -0.06), 50 + i), id = lineStreet(pts, "avenue"), r = rng(90 + i);
    lineBuildings(pts, id, 0.045, 0.02, 0.85, r, (x, y) => TREES[r() < 0.5 ? 1 : 0].push([x, y]));   // street trees
  });
  roundabout(...lm.kcc.map((v, i) => v + (i ? 0.17 : -0.24)), 0.6);
  roundabout(...cbd, 0.55);
  // neighbourhoods: CBD towers, Kimihurura/Kacyiru villas, dense hillside housing elsewhere
  sc.kigali.forEach((k, i) => {
    const style = i === 1 ? "cbd" : i === 0 || i === 6 ? "villa" : "dense";
    neighbourhood(k.xy[0], k.xy[1], k.r * (style === "cbd" ? 0.75 : 1.05), style, 300 + i);
  });

  buildKCC(lm.kcc);
  buildCityTower(lm.cityTower);
  buildAmahoro(lm.amahoro);
  buildBKArena(lm.bkArena);
  const lbl = label("KIGALI", { scale: 1.6, color: "#0b3b4a", accent: "#00a1de" });
  lbl.position.set(wx(-0.8), wy(heightAt(-0.8, 0.5)) + 9, wz(0.5)); world.add(lbl);
}

function planRoads(sc) {   // national roads and village tracks first, so no house is built on them
  for (const road of sc.roads) {
    if (road.kind !== "highway") { lineStreet(road.pts, "track"); continue; }
    const pts = resample(road.pts, 0.1).filter(([x, y]) => Math.hypot(x, y) > 0.25);   // national roads start at the hub junction
    if (pts.length > 1) lineStreet(pts, "highway");
  }
}

function buildVillages(sc) {
  const r = rng(321), [x0, x1, y0, y1] = SCENE.bounds;
  for (let i = 0; i < (LOW ? 12000 : 40000); i++) {   // about 5 homesteads per km2 on the farmland
    const x = x0 + r() * (x1 - x0), y = y0 + r() * (y1 - y0);
    if (Math.hypot(x, y) < 7 || !okSite(x, y) || !ROADIDX.clear(x, y, 0.02)) continue;
    addHouse(r, x, y, r() * Math.PI, "village");
    for (let k = 0; k < 2 + r() * 5; k++) TREES[2].push([x + (r() - 0.5) * 0.06, y + (r() - 0.5) * 0.06]);
    if (r() < 0.5) TREES[1].push([x + (r() - 0.5) * 0.05, y + (r() - 0.5) * 0.05]);
  }
  sc.villages.forEach((v, i) => neighbourhood(v.xy[0], v.xy[1], v.r * 0.6, "village", 600 + i));
  sc.towns.forEach((t, i) => {
    const hz = sc.hospitals[i];   // keep the hospital grounds clear
    if (hz) CLEAR.push([hz.xy[0], hz.xy[1], 0.28]);
    neighbourhood(t.xy[0] + 0.35, t.xy[1] + 0.2, 0.7, "dense", 800 + i);
  });
}

// ---------------------------------------------------------------- everything instanced, in tiles near the camera
function buildSettlementMeshes() {
  for (const list of chunked(HOUSES, (h) => [h.x, h.z], 15)) { const [b, r] = houseTile(list, ROOFS, WALLS); lodAdd(b, 38); lodAdd(r, 38); }
  for (const [kind, list] of Object.entries(BLOCKS)) for (const part of chunked(list, (b) => [b.x, b.z], 30)) { const m = facadeBoxes(part, kind); if (m) lodAdd(m, 110); }
}

// Kigali Convention Centre: the dome shaped like the traditional royal palace roof, with spiral glazing,
// on a glass podium, between the curved Radisson Blu hotel and the curved office block
function buildKCC([x, y]) {
  const K = 0.3, KV = 0.38, g = new THREE.Group(), gy = groundY(x, y);
  g.position.set(wx(x), gy, wz(y)); g.scale.set(K, KV, K); world.add(g);
  world.add(groundPad(x, y, flatShape(ellipseShape(2.9 * K, 2.6 * K)), new THREE.MeshStandardMaterial({ color: "#d9d3c4", roughness: 0.85 }), 0.04));
  const lawn = new THREE.MeshStandardMaterial({ color: "#5d8f36", roughness: 1 });
  for (const [ox, oz, rr] of [[-1.7, 1.4, 0.7], [1.9, 1.2, 0.6], [0.2, 2.1, 0.55]]) {
    world.add(groundPad(x + (ox * K) / S, y - (oz * K) / S, flatShape(ellipseShape(rr * K, rr * K * 0.8)), lawn, 0.05));
  }
  // dome
  const R = 1.12, H = 1.62, prof = [];
  for (let i = 0; i <= 48; i++) { const t = i / 48; prof.push(new THREE.Vector2(Math.max(R * Math.pow(1 - Math.pow(t, 1.9), 0.5), 0.003), t * H)); }
  const domeTex = canvasTex(1024, 512, (c, w, h) => {
    c.fillStyle = "#e9ecef"; c.fillRect(0, 0, w, h);
    for (let k = 0; k < 28; k++) for (const shift of [0, -w, w]) {       // spiral glazing bands (half a turn from base to top)
      const u0 = (k / 28) * w + shift;
      c.fillStyle = k % 2 ? "#26435d" : "#345a78";
      c.beginPath(); c.moveTo(u0, h); c.lineTo(u0 + 16, h); c.lineTo(u0 + 16 + w * 0.5, 0); c.lineTo(u0 + w * 0.5, 0); c.closePath(); c.fill();
    }
    c.strokeStyle = "rgba(255,255,255,0.75)"; c.lineWidth = 2;               // counter-spiral frame: the woven look
    for (let k = 0; k < 28; k++) for (const shift of [0, -w, w]) {
      const u0 = (k / 28) * w + shift; c.beginPath(); c.moveTo(u0, h); c.lineTo(u0 - w * 0.5, 0); c.stroke();
    }
    c.fillStyle = "rgba(90,100,110,0.55)"; for (let i = 1; i < 14; i++) c.fillRect(0, (i * h) / 14, w, 2);   // rings
  }, true);
  const dome = new THREE.Mesh(new THREE.LatheGeometry(prof, 96),
    new THREE.MeshStandardMaterial({ map: domeTex, metalness: 0.35, roughness: 0.3, envMap: ENV, envMapIntensity: 0.9 }));
  dome.position.y = 0.42; g.add(dome);
  const tip = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.06, 0.35, 12), mat("#d1d5db", { metalness: 0.7, roughness: 0.3 }));
  tip.position.y = 0.42 + H + 0.15; g.add(tip);
  const podiumTex = curtainTex("#3f5f78", 9); podiumTex.repeat.set(10, 1);
  const podium = new THREE.Mesh(new THREE.CylinderGeometry(1.3, 1.34, 0.42, 96, 1, true),
    new THREE.MeshStandardMaterial({ map: podiumTex, metalness: 0.5, roughness: 0.2, envMap: ENV, envMapIntensity: 1 }));
  podium.position.y = 0.21; g.add(podium);
  const podiumRoof = new THREE.Mesh(new THREE.CircleGeometry(1.32, 96), mat("#cfd4d8")); podiumRoof.rotation.x = -Math.PI / 2; podiumRoof.position.y = 0.42; g.add(podiumRoof);
  // curved hotel (7 floors) and office block (5 floors)
  const hotelTex = canvasTex(512, 256, (c, w, h) => {
    c.fillStyle = "#e9edf0"; c.fillRect(0, 0, w, h);
    const fins = ["#2f80b7", "#43a3a0", "#7cc0d8", "#e6b84a", "#d9643a", "#a3c95b"];
    for (let f = 0; f < 7; f++) {
      c.fillStyle = "#2b465b"; c.fillRect(0, (f * h) / 7 + 7, w, h / 7 - 12);
      c.fillStyle = "rgba(255,255,255,0.16)"; c.fillRect(0, (f * h) / 7 + 7, w, 6);
    }
    for (let i = 0; i < 64; i++) { c.fillStyle = i % 5 === 0 ? fins[(i / 5) % fins.length] : "rgba(225,232,236,0.85)"; c.fillRect(i * 8, 0, i % 5 === 0 ? 4 : 2, h); }
  });
  const crescent = (rIn, rOut, hgt, a0, a1, tex, reps) => {
    const len = a1 - a0, grp = new THREE.Group();
    const t = tex.clone(); t.needsUpdate = true; t.repeat.set(reps, 1);
    const wallMat = new THREE.MeshStandardMaterial({ map: t, metalness: 0.3, roughness: 0.35, envMap: ENV, envMapIntensity: 0.8, side: THREE.DoubleSide });
    for (const rr of [rIn, rOut]) {
      const wall = new THREE.Mesh(new THREE.CylinderGeometry(rr, rr, hgt, 64, 1, true, a0, len), wallMat); wall.position.y = hgt / 2; grp.add(wall);
    }
    const roof = new THREE.Mesh(new THREE.RingGeometry(rIn, rOut, 64, 1, a0 - Math.PI / 2, len), mat("#c9cdd1", { side: THREE.DoubleSide }));
    roof.rotation.x = -Math.PI / 2; roof.position.y = hgt; grp.add(roof);
    for (const a of [a0, a1]) {
      const cap = new THREE.Mesh(new THREE.BoxGeometry(0.03, hgt, rOut - rIn), mat("#dfe3e6"));
      const rm = (rIn + rOut) / 2; cap.position.set(Math.sin(a) * rm, hgt / 2, Math.cos(a) * rm); cap.rotation.y = a; grp.add(cap);
    }
    return grp;
  };
  g.add(crescent(1.85, 2.3, 1.05, Math.PI * 0.55, Math.PI * 1.3, hotelTex, 6));
  const officeTex = windowsTex("#eef1f3", "#2d4b62", "#d3d8dc", 12);
  g.add(crescent(1.8, 2.15, 0.75, Math.PI * 1.55, Math.PI * 2.2, officeTex, 7));
  const lk = label("Kigali Convention Centre", { scale: 0.9 }); lk.position.set(wx(x), gy + 1.6, wz(y)); world.add(lk);
}

// Kigali City Tower: round glass tower on a rectangular shopping podium, next to the CBD roundabout
function buildCityTower([x, y]) {
  const K = 0.33, KV = 0.285, g = new THREE.Group(), gy = groundY(x, y);
  g.position.set(wx(x), gy, wz(y)); g.scale.set(K, KV, K); world.add(g);
  world.add(groundPad(x, y, flatShape(roundedRectShape(3.0 * K, 2.4 * K, 0.3 * K)), new THREE.MeshStandardMaterial({ color: "#c8c3b8", roughness: 0.9 }), 0.04));
  facadeBoxes([{ x: 0.35, y: 0, z: 0.25, w: 2.2, d: 1.5, hgt: 0.55, rot: 0, color: "#ffffff" }], "office", g);
  const tTex = curtainTex("#2e5e80", 21); tTex.repeat.set(9, 10);
  const glassMat = new THREE.MeshStandardMaterial({ map: tTex, metalness: 0.4, roughness: 0.16, envMap: ENV, envMapIntensity: 0.6 });
  const tower = new THREE.Mesh(new THREE.CylinderGeometry(0.46, 0.46, 3.6, 48, 1, true), glassMat);
  tower.position.set(-0.3, 0.55 + 1.8, -0.1); g.add(tower);
  const ribMat = mat("#e5e7eb", { metalness: 0.6, roughness: 0.35 });
  for (let i = 0; i < 4; i++) {   // white bands that wrap the tower
    const band = new THREE.Mesh(new THREE.CylinderGeometry(0.475, 0.475, 0.06, 48, 1, true), ribMat);
    band.position.set(-0.3, 0.55 + 0.75 + i * 0.88, -0.1); g.add(band);
  }
  const crown = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.48, 0.45, 48, 1, false), mat("#dfe3e8", { metalness: 0.5, roughness: 0.3 }));
  crown.position.set(-0.3, 0.55 + 3.6 + 0.22, -0.1); g.add(crown);
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.04, 1.1, 6), mat("#f8fafc"));
  mast.position.set(-0.3, 0.55 + 4.05 + 0.55, -0.1); g.add(mast);
  const lt = label("Kigali City Tower", { scale: 0.9 }); lt.position.set(wx(x), gy + 2.2, wz(y)); world.add(lt);
}

// Amahoro Stadium: oval bowl under a white roof ring, football pitch in the middle
function buildAmahoro([x, y]) {
  const K = 0.86, KV = 0.65, g = new THREE.Group(), gy = groundY(x, y);
  g.position.set(wx(x), gy, wz(y)); g.rotation.y = 0.25; g.scale.set(K, KV, K); world.add(g);
  world.add(groundPad(x, y, flatShape(ellipseShape(1.75 * K, 1.45 * K)), new THREE.MeshStandardMaterial({ color: "#bdb7ab", roughness: 0.9 }), 0.04));
  const oval = new THREE.Group(); oval.scale.set(1.28, 1, 1); g.add(oval);
  const wallTex = canvasTex(256, 64, (c, w, h) => {
    c.fillStyle = "#e4e4e0"; c.fillRect(0, 0, w, h);
    for (let i = 0; i < 16; i++) { c.fillStyle = "#9aa3ab"; c.fillRect(i * 16 + 6, 10, 8, h - 20); }
  }); wallTex.repeat.set(10, 1);
  const outer = new THREE.Mesh(new THREE.CylinderGeometry(1.0, 1.02, 0.62, 96, 1, true), new THREE.MeshStandardMaterial({ map: wallTex, roughness: 0.7 }));
  outer.position.y = 0.31; oval.add(outer);
  const seatTex = canvasTex(64, 256, (c, w, h) => {
    for (let i = 0; i < 32; i++) { c.fillStyle = i % 2 ? "#2f5d8a" : "#3a6c9b"; c.fillRect(0, i * 8, w, 8); c.fillStyle = "#c9d1d8"; c.fillRect(0, i * 8, w, 1); }
  });
  seatTex.repeat.set(24, 1);
  const bowlProf = [new THREE.Vector2(0.6, 0.06), new THREE.Vector2(0.66, 0.12), new THREE.Vector2(0.98, 0.6)];
  const bowl = new THREE.Mesh(new THREE.LatheGeometry(bowlProf, 96), new THREE.MeshStandardMaterial({ map: seatTex, roughness: 0.8, side: THREE.DoubleSide }));
  oval.add(bowl);
  const roof = new THREE.Mesh(new THREE.RingGeometry(0.74, 1.06, 96, 1), new THREE.MeshStandardMaterial({ color: "#f4f5f2", roughness: 0.45, side: THREE.DoubleSide }));
  roof.rotation.x = -Math.PI / 2; roof.position.y = 0.74; oval.add(roof);
  for (let i = 0; i < 24; i++) {   // roof supports
    const a = (i / 24) * Math.PI * 2, post = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.14, 5), mat("#cbd5e1"));
    post.position.set(Math.sin(a) * 1.0, 0.67, Math.cos(a) * 1.0); oval.add(post);
  }
  const apron = new THREE.Mesh(new THREE.CircleGeometry(0.62, 64), mat("#3f7a34", { roughness: 1 })); apron.rotation.x = -Math.PI / 2; apron.position.y = 0.05; oval.add(apron);
  const pitchTex = canvasTex(512, 330, (c, w, h) => {
    for (let i = 0; i < 12; i++) { c.fillStyle = i % 2 ? "#4c9a3c" : "#58a846"; c.fillRect((i * w) / 12, 0, w / 12 + 1, h); }
    c.strokeStyle = "#f8fafc"; c.lineWidth = 4; c.strokeRect(12, 12, w - 24, h - 24);
    c.beginPath(); c.moveTo(w / 2, 12); c.lineTo(w / 2, h - 12); c.stroke();
    c.beginPath(); c.arc(w / 2, h / 2, 46, 0, Math.PI * 2); c.stroke();
    c.strokeRect(12, h / 2 - 80, 70, 160); c.strokeRect(w - 82, h / 2 - 80, 70, 160);
  }, false);
  const pitch = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 0.65), new THREE.MeshStandardMaterial({ map: pitchTex, roughness: 1 }));
  pitch.rotation.x = -Math.PI / 2; pitch.position.y = 0.07; pitch.scale.set(1 / 1.28, 1, 1); oval.add(pitch);
  const l = label("Amahoro Stadium", { scale: 0.85 }); l.position.set(wx(x), gy + 1.2, wz(y)); world.add(l);
}

// BK Arena: rounded indoor arena with a silver roof and a glass band
function buildBKArena([x, y]) {
  const g = new THREE.Group(), gy = groundY(x, y);
  g.position.set(wx(x), gy, wz(y)); g.rotation.y = 0.25; g.scale.set(0.75, 0.67, 0.75); world.add(g);
  const ex = (shape, depth, m, y0) => {
    const geo = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelSize: 0.05, bevelThickness: 0.05, bevelSegments: 3, curveSegments: 24 });
    geo.rotateX(-Math.PI / 2); const mesh = new THREE.Mesh(geo, m); mesh.position.y = y0; g.add(mesh); return mesh;
  };
  ex(roundedRectShape(1.45, 1.1, 0.35), 0.18, new THREE.MeshStandardMaterial({ color: "#2e4a5e", metalness: 0.6, roughness: 0.15, envMap: ENV, envMapIntensity: 1.1 }), 0.0);
  ex(roundedRectShape(1.5, 1.15, 0.38), 0.32, new THREE.MeshStandardMaterial({ color: "#cfd5da", metalness: 0.55, roughness: 0.35, envMap: ENV, envMapIntensity: 0.7 }), 0.18);
  const roofGeo = new THREE.SphereGeometry(1, 48, 12, 0, Math.PI * 2, 0, Math.PI / 2);
  const roof = new THREE.Mesh(roofGeo, mat("#e3e7ea", { metalness: 0.5, roughness: 0.3 })); roof.scale.set(0.7, 0.12, 0.52); roof.position.y = 0.52; g.add(roof);
  const l = label("BK Arena", { scale: 0.8 }); l.position.set(wx(x), gy + 1.0, wz(y)); world.add(l);
}

function buildAirport(sc) {
  const [ax, ay] = sc.airport.xy, gy = groundY(ax, ay);
  const grp = new THREE.Group(); grp.position.set(wx(ax), gy, wz(ay)); grp.rotation.y = -0.17; world.add(grp);   // runway 10/28
  const rwTex = canvasTex(2048, 64, (c, w, h) => {
    c.fillStyle = "#3b3d40"; c.fillRect(0, 0, w, h);
    c.fillStyle = "#f8fafc";
    c.fillRect(0, 3, w, 2); c.fillRect(0, h - 5, w, 2);                                            // edge lines
    for (let x = 140; x < w - 140; x += 36) c.fillRect(x, h / 2 - 1, 20, 3);                        // centre line
    for (const x0 of [8, w - 40]) for (let i = 0; i < 8; i++) c.fillRect(x0, 8 + i * 6.2, 32, 3.5);  // threshold bars
    c.save(); c.font = "bold 34px sans-serif"; c.translate(78, h / 2); c.rotate(Math.PI / 2); c.textAlign = "center"; c.textBaseline = "middle"; c.fillText("10", 0, 0); c.restore();
    c.save(); c.font = "bold 34px sans-serif"; c.translate(w - 78, h / 2); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.textBaseline = "middle"; c.fillText("28", 0, 0); c.restore();
    for (const x0 of [300, w - 360]) { c.fillRect(x0, 14, 60, 8); c.fillRect(x0, h - 22, 60, 8); }   // touchdown zone
  }, false);
  const runway = new THREE.Mesh(new THREE.PlaneGeometry(35, 0.6), new THREE.MeshStandardMaterial({ map: rwTex, roughness: 0.9 }));
  runway.rotation.x = -Math.PI / 2; runway.position.y = 0.08; grp.add(runway);
  const asphalt = mat("#55585c", { roughness: 0.95 });
  const taxi = new THREE.Mesh(new THREE.PlaneGeometry(26, 0.22), asphalt); taxi.rotation.x = -Math.PI / 2; taxi.position.set(-1, 0.07, 1.4); grp.add(taxi);
  for (const ox of [-8, 0, 6]) { const link = new THREE.Mesh(new THREE.PlaneGeometry(0.2, 1.4), asphalt); link.rotation.x = -Math.PI / 2; link.position.set(ox, 0.07, 0.7); grp.add(link); }
  const apron = new THREE.Mesh(new THREE.PlaneGeometry(3.2, 0.9), mat("#8d9196", { roughness: 0.95 })); apron.rotation.x = -Math.PI / 2; apron.position.set(-1.2, 0.075, 2.75); grp.add(apron);
  facadeBoxes([{ x: -1.2, y: 0.05, z: 3.5, w: 2.4, d: 0.5, hgt: 0.24, rot: 0 }], "glass", grp);
  const troof = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.04, 0.62), mat("#e5e7eb", { metalness: 0.4 })); troof.position.set(-1.2, 0.3, 3.5); grp.add(troof);
  const tower = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.06, 0.5, 12), mat("#f1f5f9")); tower.position.set(1.0, 0.25, 3.4); grp.add(tower);
  const cab = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.065, 0.08, 8), new THREE.MeshStandardMaterial({ color: "#3c6382", metalness: 0.6, roughness: 0.15, envMap: ENV }));
  cab.position.set(1.0, 0.54, 3.4); grp.add(cab);
  for (let i = 0; i < 3; i++) {   // parked airliners
    const p = new THREE.Group(), white = mat("#ffffff", { roughness: 0.4 });
    const fus = new THREE.Mesh(new THREE.CapsuleGeometry(0.09, 1.0, 4, 10), white); fus.rotation.z = Math.PI / 2; p.add(fus);
    const wing = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.025, 1.25), mat("#e2e8f0")); wing.position.x = 0.05; p.add(wing);
    const tail = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.3, 0.025), mat("#1e3a8a")); tail.position.set(-0.5, 0.17, 0); p.add(tail);
    const stab = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.02, 0.4), white); stab.position.set(-0.5, 0.05, 0); p.add(stab);
    p.scale.setScalar(0.36); p.position.set(-2.0 + i * 0.6, 0.08, 2.75); p.rotation.y = Math.PI / 2; grp.add(p);
  }
  // no-fly zone dome + ground ring
  const R = sc.airport.radius * S;
  const dome = new THREE.Mesh(new THREE.SphereGeometry(R, 48, 16, 0, Math.PI * 2, 0, Math.PI / 2),
    new THREE.MeshBasicMaterial({ color: "#ef4444", transparent: true, opacity: 0.1, depthWrite: false, side: THREE.DoubleSide }));
  dome.scale.y = 0.45; dome.position.set(wx(ax), gy, wz(ay)); world.add(dome);
  const wire = new THREE.Mesh(dome.geometry, new THREE.MeshBasicMaterial({ color: "#ef4444", wireframe: true, transparent: true, opacity: 0.18 }));
  wire.scale.copy(dome.scale); wire.position.copy(dome.position); world.add(wire);
  world.add(groundDisc(ax, ay, sc.airport.radius, "#ef4444", 0.75, sc.airport.radius - 0.08));
  const l = label("NO-FLY ZONE · Kigali International Airport", { color: "#7f1d1d", accent: "#ef4444", scale: 1.0 });
  l.position.set(wx(ax), gy + R * 0.45 + 3, wz(ay)); world.add(l);
}

function buildHub() {
  const gy = groundY(0, 0);
  const hangar = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.45, 20, 1, false, 0, Math.PI), mat("#f8fafc"));
  hangar.rotation.z = Math.PI / 2; hangar.rotation.y = Math.PI / 2; hangar.position.set(-0.25, gy, 0.25); world.add(hangar);
  const rail = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.03, 0.06), mat("#dc2626"));
  rail.position.set(0.12, gy + 0.08, 0); rail.rotation.z = 0.2; world.add(rail);
  world.add(groundDisc(0, 0, 0.09, "#0e7490", 0.55));
  const l = label("Kigali Drone Hub", { color: "#0b3b4a", accent: "#0e7490", scale: 1.1 });
  l.position.set(0, gy + 2.4, 0); world.add(l);
}

const hospitalObjs = {};
function buildHospitals(sc) {
  for (const h of sc.hospitals) {
    const [x, y] = h.xy, gy = groundY(x, y), g = new THREE.Group();
    const blue = mat("#2b6cb0", { roughness: 0.55, metalness: 0.3 });
    const wings = [[0, 0, 1.6, 0.5, 0.7], [-1.1, 0.5, 0.6, 0.4, 1.3], [1.0, -0.6, 0.8, 0.4, 0.6]];
    facadeBoxes(wings.map(([ox, oz, w, hh, d]) => ({ x: ox, y: 0, z: oz, w, d, hgt: hh, rot: 0, color: "#ffffff" })), "apartment", g);
    wings.forEach(([ox, oz, w, hh, d]) => {   // blue iron-sheet roofs, typical of Rwandan district hospitals
      const r = new THREE.Mesh(hipRoofGeometry(), blue); r.scale.set(w * 1.08, 0.16, d * 1.1); r.position.set(ox, hh, oz); g.add(r);
    });
    const cross = new THREE.Group();
    cross.add(new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.15, 0.05), mat("#dc2626")));
    cross.add(new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.5, 0.05), mat("#dc2626")));
    cross.position.set(0, 0.95, 0.36); g.add(cross);
    g.position.set(wx(x) + 0.9, gy, wz(y) - 0.9); g.scale.set(0.4, 0.32, 0.4); world.add(g);
    const zone = groundDisc(x, y, 0.3, "#ffffff", 0.85, 0.24); world.add(zone);
    const pad = groundDisc(x, y, 0.08, "#dc2626", 0.9); world.add(pad);
    const lbl = label(`${h.name}${h.split === "test" ? " (unseen)" : ""}`, { accent: "#dc2626", scale: 1.0 });
    lbl.position.set(wx(x), gy + 4, wz(y)); world.add(lbl);
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 60, 16, 1, true),
      new THREE.MeshBasicMaterial({ color: "#22d3ee", transparent: true, opacity: 0.0, depthWrite: false, side: THREE.DoubleSide }));
    beam.position.set(wx(x), gy + 30, wz(y)); world.add(beam);
    hospitalObjs[h.name] = { beam, lbl, xy: h.xy };
  }
}

// ---------------------------------------------------------------- trees: eucalyptus, broadleaf (avocado, mango) and banana
function paint(geo, hex) {
  const g = geo.index ? geo.toNonIndexed() : geo, c = new THREE.Color(hex), n = g.attributes.position.count, a = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) a.set([c.r, c.g, c.b], i * 3);
  g.setAttribute("color", new THREE.BufferAttribute(a, 3)); g.deleteAttribute("uv");
  return g;
}
function blob(r, sx, sy, sz, x, y, z, hex, seed) {   // lumpy canopy
  const g = new THREE.IcosahedronGeometry(r, 0), p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const px = p.getX(i), py = p.getY(i), pz = p.getZ(i);
    const k = 1 + 0.18 * Math.sin(px * 37.1 + py * 19.7 + pz * 23.3 + seed);
    p.setXYZ(i, px * k * sx + x, py * k * sy + y, pz * k * sz + z);
  }
  g.computeVertexNormals();
  return paint(g, hex);
}
function trunk(r0, r1, h, hex) { const g = new THREE.CylinderGeometry(r0, r1, h, 5); g.translate(0, h / 2, 0); return paint(g, hex); }
function treeGeometries() {
  const euc = mergeGeometries([trunk(0.022, 0.04, 0.78, "#9b8f80"),
    blob(0.13, 1, 1.6, 1, 0.02, 0.62, 0.0, "#6f8a5c", 1), blob(0.1, 1, 1.7, 1, -0.04, 0.9, 0.02, "#7d9869", 2)]);
  const broad = mergeGeometries([trunk(0.03, 0.045, 0.26, "#6b5444"),
    blob(0.2, 1.25, 0.8, 1.2, 0, 0.36, 0, "#3e6b2b", 4), blob(0.13, 1, 0.8, 1, 0.1, 0.44, 0.04, "#4a7a33", 5)]);
  const leaves = [trunk(0.022, 0.03, 0.17, "#6f7f3c")];
  for (let i = 0; i < 7; i++) {
    const leaf = new THREE.PlaneGeometry(0.06, 0.26); leaf.translate(0, 0.13, 0);
    leaf.rotateX(-0.75 - (i % 2) * 0.35); leaf.rotateY((i / 7) * Math.PI * 2); leaf.translate(0, 0.17, 0);
    leaves.push(paint(leaf, i % 3 ? "#7fb23c" : "#6aa232"));
  }
  return { euc, broad, banana: mergeGeometries(leaves) };
}

function buildTrees(sc) {
  const r = rng(99), [x0, x1, y0, y1] = SCENE.bounds;
  const free = (x, y) => okSite(x, y) && !nearAirport(x, y, 0.3) && ROADIDX.clear(x, y, 0.006);
  for (let i = 0; i < (LOW ? 300 : 700); i++) {   // eucalyptus woodlots on the hills
    const gx = x0 + r() * (x1 - x0), gyk = y0 + r() * (y1 - y0), n = 5 + Math.floor(r() * 14), spread = 0.04 + r() * 0.1;
    for (let k = 0; k < n; k++) { const x = gx + (r() - 0.5) * spread * 2, y = gyk + (r() - 0.5) * spread * 2; if (free(x, y)) TREES[0].push([x, y]); }
  }
  for (let i = 0; i < (LOW ? 2000 : 5000); i++) {   // single trees in the fields
    const x = x0 + r() * (x1 - x0), y = y0 + r() * (y1 - y0); if (free(x, y)) TREES[r() < 0.75 ? 1 : 0].push([x, y]);
  }
  // real sizes: eucalyptus ~25 m, fruit trees ~10 m, banana ~4 m (heights drawn 1.6x like the buildings)
  const geos = treeGeometries(), names = ["euc", "broad", "banana"];
  const size = [[0.32, 0.36, 0.14], [0.2, 0.27, 0.08], [0.11, 0.14, 0.05]];   // horizontal, vertical, random extra
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), c = new THREE.Color(), up = new THREE.Vector3(0, 1, 0);
  names.forEach((name, type) => {
    const treeMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85, side: type === 2 ? THREE.DoubleSide : THREE.FrontSide });
    const pts = LOW ? TREES[type].filter((_, i) => i % 2 === 0) : TREES[type];
    for (const list of chunked(pts, ([x, y]) => [wx(x), wz(y)], 20)) {
      const mesh = new THREE.InstancedMesh(geos[name], treeMat, list.length);
      list.forEach(([x, y], i) => {
        const e = 1 + (r() - 0.3) * size[type][2] / size[type][0] * 2;
        q.setFromAxisAngle(up, r() * Math.PI * 2);
        m4.compose(new THREE.Vector3(wx(x), groundY(x, y) - 0.01, wz(y)), q, new THREE.Vector3(size[type][0] * e, size[type][1] * e * (0.85 + r() * 0.3), size[type][0] * e));
        mesh.setMatrixAt(i, m4); const k = 0.8 + r() * 0.28; mesh.setColorAt(i, c.setRGB(k, k, k));
      });
      world.add(mesh); lodAdd(mesh, type === 2 ? 18 : 32);
    }
  });
}

// ---------------------------------------------------------------- drone
function buildDrone() {
  const g = new THREE.Group(), white = mat("#f8fafc", { roughness: 0.35 }), red = mat("#dc2626", { roughness: 0.4 });
  const fus = new THREE.Mesh(new THREE.CapsuleGeometry(0.09, 0.62, 6, 14), white); fus.rotation.z = Math.PI / 2; g.add(fus);
  const wing = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.025, 1.5), white); wing.position.set(0.04, 0.03, 0); g.add(wing);
  for (const z of [-0.74, 0.74]) { const tip = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.03, 0.06), red); tip.position.set(0.04, 0.03, z); g.add(tip); }
  for (const z of [-0.28, 0.28]) {
    const boom = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.03, 0.03), white); boom.position.set(-0.22, 0.02, z); g.add(boom);
    const fin = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.16, 0.02), red); fin.position.set(-0.62, 0.1, z); g.add(fin);
  }
  const stab = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.02, 0.6), white); stab.position.set(-0.64, 0.17, 0); g.add(stab);
  const prop = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.36, 0.04), mat("#334155")); prop.position.set(-0.43, 0, 0); g.add(prop);
  const pod = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.1, 0.12), red); pod.position.set(0.05, -0.1, 0); g.add(pod);
  const navL = new THREE.Mesh(new THREE.SphereGeometry(0.035), new THREE.MeshBasicMaterial({ color: "#ef4444" })); navL.position.set(0.04, 0.03, 0.78); g.add(navL);
  const navR = new THREE.Mesh(new THREE.SphereGeometry(0.035), new THREE.MeshBasicMaterial({ color: "#22c55e" })); navR.position.set(0.04, 0.03, -0.78); g.add(navR);
  g.scale.setScalar(1.0);
  g.userData = { prop, pod };
  return g;
}
const drone = buildDrone();
const droneRoot = new THREE.Group(); droneRoot.add(drone); scene.add(droneRoot);
drone.traverse((o) => { if (o.isMesh && !o.material.isMeshBasicMaterial) o.castShadow = true; });
const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.5, 20), new THREE.MeshBasicMaterial({ color: "#000", transparent: true, opacity: 0.22, depthWrite: false }));
shadow.rotation.x = -Math.PI / 2; scene.add(shadow);

// ---------------------------------------------------------------- mission objects
const missionGroup = new THREE.Group(); scene.add(missionGroup);
let rings = [], rainObjs = [], trail = null, trailPts = [], chute = null, lastDropShown = false, mission = null;

function clearMission() {
  while (missionGroup.children.length) missionGroup.remove(missionGroup.children[0]);
  rings = []; rainObjs = []; trailPts = []; chute = null; lastDropShown = false;
  drone.userData.pod.visible = true;
  for (const k in hospitalObjs) hospitalObjs[k].beam.material.opacity = 0;
}

let introUntil = 0;
function setupMission(f) {
  if (f.t === 0 && camMode === "chase" && !params.get("nointro")) {
    introUntil = performance.now() + 3200;
    const k = LANDMARKS.kcc;
    camera.position.set(wx(-0.8), groundY(-0.8, -1.05) + 6.5, wz(-1.05));
    camLook.set(wx(k[0] * 0.5), groundY(0.33, 0) + 0.4, wz(k[1] * 0.5));
  }
  clearMission();
  const byIndex = {};
  for (const r of f.rings) if (!r.inbound) byIndex[r.index] = r;
  for (let i = 0; i < 3; i++) {
    const r = byIndex[i]; if (!r) continue;
    const [x, y] = r.xy, gy = wy(heightAt(x, y) + 100);
    const torus = new THREE.Mesh(new THREE.TorusGeometry(4.5, 0.12, 12, 72),
      new THREE.MeshStandardMaterial({ color: "#22d3ee", emissive: "#0891b2", emissiveIntensity: 0.8, roughness: 0.3 }));
    torus.position.set(wx(x), gy, wz(y)); torus.rotation.y = f.corridor_bearing + Math.PI / 2;
    missionGroup.add(torus); rings.push(torus);
  }
  for (const c of f.rain) rainObjs.push(makeRain(c));
  const tg = new THREE.BufferGeometry(); tg.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(3 * 1000), 3)); tg.setDrawRange(0, 0);
  trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: "#0e7490", transparent: true, opacity: 0.85 }));
  missionGroup.add(trail);
  const target = hospitalObjs[f.hospital]; if (target) target.beam.material.opacity = 0.22;
}

const CLOUD_TEX = (() => {   // soft cumulus puff, used for the rain cells
  const c = document.createElement("canvas"); c.width = c.height = 256; const g = c.getContext("2d"), r = rng(3);
  for (let i = 0; i < 28; i++) {
    const x = 50 + r() * 156, y = 90 + r() * 90, rad = 30 + r() * 55;
    const gr = g.createRadialGradient(x, y, 0, x, y, rad);
    gr.addColorStop(0, "rgba(255,255,255,0.55)"); gr.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = gr; g.fillRect(0, 0, 256, 256);
  }
  const shade = g.createLinearGradient(0, 60, 0, 230);   // darker base
  shade.addColorStop(0, "rgba(0,0,0,0)"); shade.addColorStop(1, "rgba(40,48,58,0.55)");
  g.globalCompositeOperation = "source-atop"; g.fillStyle = shade; g.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
})();

function makeRain(c) {
  const g = new THREE.Group(), r = rng(Math.floor(c.radius * 1000));
  const R = c.radius * S;
  for (let i = 0; i < 26; i++) {
    const a = r() * Math.PI * 2, d = R * 0.8 * Math.sqrt(r()), size = R * (0.55 + r() * 0.5);
    const puff = new THREE.Sprite(new THREE.SpriteMaterial({ map: CLOUD_TEX, color: i % 3 ? "#7d8792" : "#9aa3ad", transparent: true, opacity: 0.9, depthWrite: false }));
    puff.position.set(Math.cos(a) * d, 1 + r() * 4, Math.sin(a) * d); puff.scale.set(size, size * 0.5, 1); g.add(puff);
  }
  const N = 700, pos = new Float32Array(N * 6), speeds = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const a = r() * Math.PI * 2, d = R * 0.85 * Math.sqrt(r()), h = -r() * 15;
    pos.set([Math.cos(a) * d, h, Math.sin(a) * d, Math.cos(a) * d - 0.2, h - 1.3, Math.sin(a) * d], i * 6);
    speeds[i] = 25 + r() * 15;
  }
  const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: "#9fb4c8", transparent: true, opacity: 0.55 }));
  g.add(lines);
  g.userData = { c, lines, speeds };
  placeRain(g, c);
  missionGroup.add(g);
  return g;
}
function placeRain(g, c) { g.position.set(wx(c.xy[0]), groundY(c.xy[0], c.xy[1]) + 15, wz(c.xy[1])); g.userData.c = c; }

function makeChute(from, to) {
  const g = new THREE.Group();
  const canopyGeo = new THREE.SphereGeometry(0.28, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2);
  const cols = []; const cc = new THREE.Color();
  for (let i = 0; i < canopyGeo.attributes.position.count; i++) {
    const x = canopyGeo.attributes.position.getX(i), z = canopyGeo.attributes.position.getZ(i);
    cc.set(Math.floor(((Math.atan2(z, x) + Math.PI) / (Math.PI * 2)) * 8) % 2 ? "#dc2626" : "#ffffff"); cols.push(cc.r, cc.g, cc.b);
  }
  canopyGeo.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
  const canopy = new THREE.Mesh(canopyGeo, new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  canopy.position.y = 0.5; g.add(canopy);
  const box = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.07, 0.1), mat("#dc2626")); g.add(box);
  const lp = []; for (let i = 0; i < 6; i++) { const a = (i / 6) * Math.PI * 2; lp.push(0, 0.03, 0, Math.cos(a) * 0.27, 0.5, Math.sin(a) * 0.27); }
  const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.Float32BufferAttribute(lp, 3));
  g.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: "#475569" })));
  g.position.copy(from); g.userData = { from: from.clone(), to: to.clone(), t0: performance.now() };
  missionGroup.add(g);
  return g;
}

// ---------------------------------------------------------------- HUD
const $ = (id) => document.getElementById(id);
const PHASES = { ring: "Outbound corridor", drop: "Flying to hospital", home: "Returning to hub" };
function phaseText(f) {
  const w = f.waypoint;
  if (f.outcome && f.outcome !== "timeout") return { home: "Landed at hub ✓", crash: "Crashed", no_fly: "Airspace violation", battery: "Battery empty", out_of_bounds: "Left the map" }[f.outcome] || f.outcome;
  if (w.kind === "ring") return `${w.inbound ? "Inbound" : "Outbound"} corridor · ring ${w.index + 1}/3`;
  if (w.kind === "drop") return "Flying to hospital";
  return "Returning to hub";
}
function fmt(v, d = 1) { return (v >= 0 ? "+" : "") + v.toFixed(d); }
let toastTimer = null;
function toast(text, good) {
  const t = $("toast"); t.textContent = text; t.className = `show ${good ? "good" : "bad"}`;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.className = good ? "good" : "bad"), 1700);
}
function flash(good) { const f = $("flash"); f.className = good ? "good" : "bad"; requestAnimationFrame(() => setTimeout(() => (f.className = ""), 90)); }

function updateHud(f) {
  $("m-hospital").textContent = f.hospital;
  if (mission) $("m-meta").textContent = `${mission.district} District · ${mission.distance_km.toFixed(1)} km away · mission ${mission.index}/${mission.total}`;
  $("m-agent").textContent = mission ? mission.agent : "–";
  $("m-phase").textContent = phaseText(f);
  $("m-time").textContent = `${f.minutes.toFixed(1)} min`;
  const b = f.battery * 100;
  $("g-batt").style.width = `${Math.max(0, b)}%`;
  $("g-batt").style.background = b > 40 ? "#16a34a" : b > 20 ? "#f59e0b" : "#dc2626";
  $("g-batt-v").textContent = `${b.toFixed(0)}%`;
  const tp = Math.min(1, (f.temp - 2) / 8);
  $("g-temp").style.width = `${tp * 100}%`;
  $("g-temp").style.background = f.temp < 6 ? "#0ea5e9" : f.temp < 9 ? "#f59e0b" : "#dc2626";
  $("g-temp-v").textContent = `${f.temp.toFixed(1)} °C`;
  $("t-speed").textContent = `${f.speed} · ${f.speed_kmh} km/h`;
  $("t-agl").textContent = `${Math.round(f.agl)} m (set ${Math.round(f.setpoint)})`;
  const ws = Math.hypot(f.wind[0], f.wind[1]);
  $("t-wind").textContent = `${(ws * 3.6).toFixed(0)} km/h`;
  $("t-wind-arrow").style.transform = `rotate(${90 - (Math.atan2(f.wind[1], f.wind[0]) * 180) / Math.PI}deg)`;
  $("t-payload").textContent = f.payload_on ? "Blood on board" : f.drop_result === "on_target" ? "Delivered ✓" : f.drop_result ? `Dropped (${f.drop_result.replace("_", " ")})` : "–";
  $("a-action").textContent = f.action || "launch";
  const ar = $("a-reward"); ar.textContent = fmt(f.reward, 2); ar.className = `ab-num ${f.reward >= 0 ? "pos" : "neg"}`;
  const at = $("a-total"); at.textContent = fmt(f.total_reward, 1); at.className = `ab-num ${f.total_reward >= 0 ? "pos" : "neg"}`;
  $("a-step").textContent = `${f.t}`;
  for (const [text, val] of f.events) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${text}</span><b class="${val >= 0 ? "pos" : "neg"}">${fmt(val, 1)}</b>`;
    $("ev-list").prepend(li);
    while ($("ev-list").children.length > 7) $("ev-list").lastChild.remove();
    if (Math.abs(val) >= 5) toast(`${fmt(val, 0)}  ${text}`, val >= 0);
  }
  if (f.events.length) flash(f.events.reduce((s, e) => s + e[1], 0) >= 0);
  else if (Math.abs(f.reward) > 0.35) flash(f.reward >= 0);
}

const mm = $("minimap"), mmx = mm.getContext("2d");
function drawMinimap(f) {
  if (!SCENE) return;
  const [x0, x1, y0, y1] = SCENE.bounds, W = mm.width, H = mm.height;
  const px = (x) => ((x - x0) / (x1 - x0)) * W, py = (y) => ((y1 - y) / (y1 - y0)) * H, pr = (r) => (r / (x1 - x0)) * W;
  mmx.clearRect(0, 0, W, H);
  if (minimapImg.complete && minimapImg.naturalWidth) mmx.drawImage(minimapImg, 0, 0, W, H);
  const a = SCENE.airport; mmx.fillStyle = "rgba(220,38,38,0.35)"; mmx.beginPath(); mmx.arc(px(a.xy[0]), py(a.xy[1]), pr(a.radius), 0, 7); mmx.fill();
  if (f) {
    mmx.fillStyle = "rgba(71,85,105,0.45)";
    for (const c of f.rain) { mmx.beginPath(); mmx.arc(px(c.xy[0]), py(c.xy[1]), pr(c.radius), 0, 7); mmx.fill(); }
    for (const r of f.rings) { if (r.inbound) continue; mmx.strokeStyle = "#0891b2"; mmx.lineWidth = 1.5; mmx.beginPath(); mmx.arc(px(r.xy[0]), py(r.xy[1]), 3, 0, 7); mmx.stroke(); }
  }
  for (const h of SCENE.hospitals) {
    const on = f && h.name === f.hospital;
    mmx.fillStyle = on ? "#dc2626" : "#ffffff"; mmx.strokeStyle = "#0f172a"; mmx.lineWidth = 1;
    mmx.fillRect(px(h.xy[0]) - (on ? 4 : 3), py(h.xy[1]) - (on ? 4 : 3), on ? 8 : 6, on ? 8 : 6);
    mmx.strokeRect(px(h.xy[0]) - (on ? 4 : 3), py(h.xy[1]) - (on ? 4 : 3), on ? 8 : 6, on ? 8 : 6);
  }
  mmx.fillStyle = "#0f172a"; mmx.beginPath(); mmx.moveTo(px(0), py(0) - 5); mmx.lineTo(px(0) - 4, py(0) + 3); mmx.lineTo(px(0) + 4, py(0) + 3); mmx.fill();
  if (trailPts.length > 1) {
    mmx.strokeStyle = "#0b3d91"; mmx.lineWidth = 2; mmx.beginPath();
    trailPts.forEach((p, i) => (i ? mmx.lineTo(px(p[0]), py(p[1])) : mmx.moveTo(px(p[0]), py(p[1])))); mmx.stroke();
  }
  if (f) {
    const x = px(f.pos[0]), y = py(f.pos[1]), h = f.heading;
    mmx.save(); mmx.translate(x, y); mmx.rotate(-h); mmx.fillStyle = "#facc15"; mmx.strokeStyle = "#0f172a";
    mmx.beginPath(); mmx.moveTo(7, 0); mmx.lineTo(-5, 4); mmx.lineTo(-3, 0); mmx.lineTo(-5, -4); mmx.closePath(); mmx.fill(); mmx.stroke(); mmx.restore();
  }
}

// ---------------------------------------------------------------- frame handling and animation
let prev = null, next = null, tArrive = 0, interval = 400, lastArrival = 0, lastFrame = null;
const RING_COL = { pending: "#22d3ee", passed: "#22c55e", missed: "#ef4444" };

function onMessage(msg) {
  if (msg.type === "mission") {
    mission = msg; $("summary").classList.add("hidden"); $("ev-list").innerHTML = "";
    const t = $("toast"); t.textContent = `Mission ${msg.index}/${msg.total} · ${msg.hospital}${msg.unseen ? " (never seen in training)" : ""}`;
    t.className = "show info"; clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.className = "info"), 2600);
    return;
  }
  if (msg.type === "summary") { showSummary(msg); return; }
  if (msg.type !== "frame") return;
  const f = msg;
  const now = performance.now();
  if (f.t === 0 || !lastFrame || lastFrame.hospital !== f.hospital || f.t < lastFrame.t) {
    setupMission(f); prev = f; next = f; trailPts = [];
  } else {
    if (lastArrival) interval = Math.min(1500, Math.max(80, 0.7 * interval + 0.3 * (now - lastArrival)));
    prev = currentState(); next = f;
  }
  lastArrival = now; tArrive = now; lastFrame = f;
  trailPts.push([f.pos[0], f.pos[1], f.agl]);
  const arr = trail.geometry.attributes.position;
  const n = Math.min(trailPts.length, 1000);
  for (let i = 0; i < n; i++) { const p = trailPts[trailPts.length - n + i]; arr.setXYZ(i, wx(p[0]), wy(heightAt(p[0], p[1]) + p[2]), wz(p[1])); }
  arr.needsUpdate = true; trail.geometry.setDrawRange(0, n);
  // ring colours: show inbound status once the payload is gone
  const out = f.rings.filter((r) => !r.inbound), inb = f.rings.filter((r) => r.inbound);
  rings.forEach((torus, i) => {
    const o = out.find((r) => r.index === i), b = inb.find((r) => r.index === i);
    const st = !f.payload_on && b ? (b.status === "pending" ? "pending" : b.status) : o ? o.status : "pending";
    torus.material.color.set(RING_COL[st]); torus.material.emissive.set(RING_COL[st]);
    torus.userData.active = f.waypoint.kind === "ring" && f.waypoint.index === i;
  });
  f.rain.forEach((c, i) => rainObjs[i] && placeRain(rainObjs[i], c));
  if (f.drop_point && !lastDropShown) {
    lastDropShown = true; drone.userData.pod.visible = false;
    const to = new THREE.Vector3(wx(f.drop_point[0]), groundY(f.drop_point[0], f.drop_point[1]) + 0.15, wz(f.drop_point[1]));
    chute = makeChute(droneRoot.position.clone(), to);
  }
  updateHud(f);
  drawMinimap(f);
  $("status").textContent = REPLAY ? "replay" : "live";
}

function currentState() {
  if (!prev || !next) return next;
  const k = Math.min(1, (performance.now() - tArrive) / interval);
  let dh = next.heading - prev.heading; while (dh > Math.PI) dh -= 2 * Math.PI; while (dh < -Math.PI) dh += 2 * Math.PI;
  return { ...next, pos: [prev.pos[0] + (next.pos[0] - prev.pos[0]) * k, prev.pos[1] + (next.pos[1] - prev.pos[1]) * k],
           alt: prev.alt + (next.alt - prev.alt) * k, agl: prev.agl + (next.agl - prev.agl) * k, heading: prev.heading + dh * k, _turn: dh };
}

function showSummary(msg) {
  const rows = msg.rows.map((r) => `<tr><td>${r.hospital}</td><td>${r.outcome}</td><td>${r.delivered ? "yes" : "no"}</td><td>${r.minutes ?? "–"}</td><td>${r.battery}</td><td>${r.reward}</td></tr>`).join("");
  $("summary").innerHTML = `<h3>${msg.title}</h3><table><tr><th>Hospital</th><th>Outcome</th><th>Delivered</th><th>Delivery (min)</th><th>Battery left</th><th>Reward</th></tr>${rows}</table>`;
  $("summary").classList.remove("hidden");
}

let camMode = "chase", fixedView = null;
function setView(v) {   // v = [eyeX, eyeY, eyeH, lookX, lookY, lookH]: km on the map, metres above the ground
  const [ex, ey, eh, lx, ly, lh] = v.map(Number);
  fixedView = { eye: new THREE.Vector3(wx(ex), groundY(ex, ey) + (eh / 100) * VEX, wz(ey)),
                look: new THREE.Vector3(wx(lx), groundY(lx, ly) + (lh / 100) * VEX, wz(ly)) };
  camMode = "fixed"; camera.position.copy(fixedView.eye); camLook.copy(fixedView.look); camera.lookAt(camLook);
}
document.querySelectorAll("#camera-buttons button").forEach((b) => b.addEventListener("click", () => setCam(b.dataset.cam)));
window.addEventListener("keydown", (e) => { const m = { c: "chase", o: "orbit", m: "overview" }[e.key.toLowerCase()]; if (m) setCam(m); });
function setCam(m) {
  camMode = m; controls.enabled = m === "orbit";
  document.querySelectorAll("#camera-buttons button").forEach((b) => b.classList.toggle("on", b.dataset.cam === m));
  if (m === "orbit") controls.target.copy(droneRoot.position);
}

let lastT = performance.now(), perf = { frames: 0, t0: 0, done: LOW }, lodFrame = 0;
function checkPerformance(t) {   // a slow laptop gets a smooth picture without shadows instead of a choppy one
  if (perf.done) return;
  if (!perf.t0) { perf.t0 = t; return; }
  perf.frames++;
  if (t - perf.t0 < 5000) return;
  perf.done = true;
  const fps = (perf.frames * 1000) / (t - perf.t0);
  if (fps < 24 && !params.get("view") && params.get("quality") !== "high") {
    renderer.shadowMap.enabled = false; sun.castShadow = false;
    scene.traverse((o) => { if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => (m.needsUpdate = true)); });
    console.log(`viewer: ${fps.toFixed(0)} fps, shadows switched off (add ?quality=low to the URL to also thin out trees)`);
  }
}
const camLook = new THREE.Vector3();
function animate() {
  requestAnimationFrame(animate);
  const t = performance.now(), dt = Math.min((t - lastT) / 1000, 0.1);
  lastT = t;
  checkPerformance(t);
  const s = currentState();
  if (s && GRID) {
    const p = new THREE.Vector3(wx(s.pos[0]), wy(heightAt(s.pos[0], s.pos[1]) + s.agl), wz(s.pos[1]));   // height above the drawn ground
    droneRoot.position.copy(p);
    droneRoot.rotation.set(0, s.heading, 0);
    drone.rotation.x = THREE.MathUtils.lerp(drone.rotation.x, -(s._turn || 0) * 1.4, 0.08);
    drone.userData.prop.rotation.x += dt * 40;
    shadow.position.set(p.x, groundY(s.pos[0], s.pos[1]) + 0.2, p.z);
    const fwd = new THREE.Vector3(Math.cos(s.heading), 0, -Math.sin(s.heading));
    if (camMode === "chase" && t < introUntil) {   // slow push-in over the hub and the Convention Centre
      camera.position.lerp(new THREE.Vector3(wx(-0.45), groundY(-0.45, -0.62) + 3.4, wz(-0.62)), 1 - Math.exp(-dt * 0.5));
      camera.lookAt(camLook);
    } else if (camMode === "chase") {
      const want = p.clone().addScaledVector(fwd, -10).add(new THREE.Vector3(0, 3.6, 0));
      const gmin = groundY(s.pos[0] - Math.cos(s.heading) * 1.0, s.pos[1] - Math.sin(s.heading) * 1.0) + 1.8;
      want.y = Math.max(want.y, gmin);
      camera.position.lerp(want, 1 - Math.exp(-dt * 2.5));
      camLook.lerp(p.clone().addScaledVector(fwd, 4), 1 - Math.exp(-dt * 4));
      camera.lookAt(camLook);
    } else if (camMode === "overview" && next) {
      const hx = next.hospital_xy, cx = (hx[0] + 0) / 2, cy = (hx[1] + 0) / 2, span = Math.hypot(hx[0], hx[1]);
      const want = new THREE.Vector3(wx(cx) + span * 2, wy(1800) + span * 4.2 + 45, wz(cy) + span * 8 + 70);
      camera.position.lerp(want, 1 - Math.exp(-dt * 2));
      camLook.lerp(new THREE.Vector3(wx(cx), wy(1500), wz(cy)), 1 - Math.exp(-dt * 3));
      camera.lookAt(camLook);
    } else if (camMode === "orbit") {
      controls.target.lerp(p, 0.1); controls.update();
    }
    shadow.visible = LOW;
    if (camMode === "chase") updateSun(p, 32);
    else if (camMode === "orbit") updateSun(controls.target, 40);
    else if (camMode === "overview") updateSun(camLook, 260);
  }
  if (camMode === "fixed") { camera.position.copy(fixedView.eye); camera.lookAt(fixedView.look); updateSun(fixedView.look, 40); }
  if (t - lodFrame > 300) { lodFrame = t; updateLod(); }
  rings.forEach((r) => { const k = r.userData.active ? 1 + 0.08 * Math.sin(t / 180) : 1; r.scale.set(k, 0.5 * k, k);   // oval gate: 900 m wide, 450 m tall
    r.material.emissiveIntensity = r.userData.active ? 1.2 : 0.5; });
  for (const g of rainObjs) {
    const pos = g.userData.lines.geometry.attributes.position, sp = g.userData.speeds;
    for (let i = 0; i < sp.length; i++) {
      let y = pos.getY(i * 2) - sp[i] * dt;
      if (y < -15) y += 15;
      pos.setY(i * 2, y); pos.setY(i * 2 + 1, y - 1.3);
    }
    pos.needsUpdate = true;
  }
  if (chute) {
    const u = chute.userData, k = Math.min(1, (t - u.t0) / 3500);
    chute.position.lerpVectors(u.from, u.to, k);
    chute.rotation.z = Math.sin(t / 300) * 0.12 * (1 - k);
  }
  for (const k in hospitalObjs) { const b = hospitalObjs[k].beam; if (b.material.opacity > 0) b.material.opacity = 0.16 + 0.08 * Math.sin(t / 400); }
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------- start: live websocket or replay file
async function start() {
  let sc, replay = null;
  if (REPLAY) { replay = await (await fetch(REPLAY)).json(); sc = replay.scene; }
  else sc = await (await fetch("/api/scene")).json();
  await buildWorld(sc);
  drawMinimap(null);
  camera.position.set(wx(-4), wy(2600), wz(-14)); camLook.set(0, wy(1600), 0); camera.lookAt(camLook);
  updateLod();
  $("loading").classList.add("done");
  animate();
  if (!replay && params.get("view")) setView(params.get("view").split(","));
  if (replay) {
    const msgs = replay.messages;
    let i = 0;
    const at = parseInt(params.get("at") || "0", 10), hold = params.get("hold");
    while (i < at && i < msgs.length) onMessage(msgs[i++]);   // jump to a moment in the replay
    if (params.get("cam")) setCam(params.get("cam"));
    if (params.get("view")) setView(params.get("view").split(","));
    if (hold) { onMessage(msgs[i]); return; }
    const step = () => {
      if (i >= msgs.length) { i = 0; }
      const m = msgs[i++]; onMessage(m);
      const intro = m.type === "frame" && m.t === 0 ? 3000 : 0;   // same launch shot as live mode
      setTimeout(step, (m.type === "frame" ? (replay.step_ms || 350) + intro : 600) / SPEED);
    };
    step();
  } else {
    const connect = () => {
      const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
      ws.onopen = () => ($("status").textContent = "live · waiting for the agent…");
      ws.onmessage = (e) => onMessage(JSON.parse(e.data));
      ws.onclose = () => { $("status").textContent = "disconnected · retrying"; setTimeout(connect, 1500); };
    };
    connect();
  }
}
start();
window.__ibaba = { scene, camera, setCam, onMessage, renderer, sky, sun, hemi, THREE, LOD_TILES, world };
