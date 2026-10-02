/* ═══════════════════════════════════════════════════════════════════════
   MOFI — Interactive Preview Script
   Three.js 3D simulation + 2D dual-omics kidney viewer

   Compact browser-demo data format:
     demo_data/simu/ode_trajectories.json       → {shape:[400,200,2], times:[...], data:[...]}
     demo_data/simu/ode_trajectories_3d.json    → {shape:[400,200,3], times:[...], data:[...]}
     demo_data/simu/decorative_trajectories.json → {shape:[18,200,2], ...}
     demo_data/kidney/traj_sample_rna.json      → {shape:[...], cell_types:[...], ...}
     demo_data/kidney/traj_sample_atac.json     → {shape:[...], ...}
   ═══════════════════════════════════════════════════════════════════════ */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/* ── Constants ──────────────────────────────────────────────────────── */
const TRAJ_COLORS = ["#25346F", "#7B4CC2", "#D89B2B"];
const SIMU_TIME_LABELS = ["t = 0", "t = 1", "t = 2", "t = 3", "t = 4"];

// 肾脏细胞类型颜色 (用户指定)
const CELL_TYPE_COLORS = {
  NPC: "#C18D51",
  dev: "#5689AE",
  LOH_DN: "#64956D",
  PT: "#B35965",
  POD: "#9678AA",
};

// 轨迹点4色渐变 (早→晚)
const TRAJ_4COLORS = ["#25346F", "#5E63D6", "#B65683", "#D89B2B"];

// 真实天数: ODE t∈[0,4] → Day 7~26
const REAL_DAYS = [7, 12, 16, 19, 26];

const TRAIL_LENGTH = 8;
const KIDNEY_BACKGROUND_ALPHA = 0.04;
const KIDNEY_TRAJECTORY_ALPHA = 1;
const KIDNEY_TYPE_TRANSITION_WINDOW = 1.0;

/* ── Helpers ────────────────────────────────────────────────────────── */
function smoothstep(t) {
  t = Math.max(0, Math.min(1, t));
  return t * t * (3 - 2 * t);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function hexToRgb(hex) {
  hex = hex.replace("#", "");
  return [
    parseInt(hex.substring(0, 2), 16) / 255,
    parseInt(hex.substring(2, 4), 16) / 255,
    parseInt(hex.substring(4, 6), 16) / 255,
  ];
}

function lerpColor(c1, c2, t) {
  return [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)];
}

function timeToColor(t) {
  const colors = TRAJ_COLORS.map(hexToRgb);
  const idx = t * (colors.length - 1);
  const i = Math.min(Math.floor(idx), colors.length - 2);
  const frac = idx - i;
  return lerpColor(colors[i], colors[i + 1], frac);
}

// 4色轨迹渐变
function traj4Color(t) {
  const colors = TRAJ_4COLORS.map(hexToRgb);
  const idx = t * (colors.length - 1);
  const i = Math.min(Math.floor(idx), colors.length - 2);
  const frac = idx - i;
  return lerpColor(colors[i], colors[i + 1], frac);
}

// ODE时间 → 真实天数
function odeTimeToDay(t) {
  // t∈[0,4] → Day 7~26, 线性插值
  return 7 + t * (26 - 7) / 4;
}

function rgbStr(c, alpha) {
  if (alpha !== undefined) {
    return `rgba(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)},${alpha})`;
  }
  return `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;
}

function cellTypeRgb(type) {
  return hexToRgb(CELL_TYPE_COLORS[type] || "#888888");
}

function trajectoryTypeColor(trajData, cellIdx, timeFloat) {
  const { dynamic_types, final_types, times, shape } = trajData;
  const samples = [];

  if (dynamic_types && dynamic_types.length > 0) {
    for (const sample of dynamic_types) {
      samples.push({
        time: sample.time,
        type: sample.types ? sample.types[cellIdx] : undefined,
      });
    }
  }

  if (final_types && final_types[cellIdx]) {
    const nBins = shape ? shape[1] : times.length;
    const tMax = times[nBins - 1];
    const last = samples[samples.length - 1];
    if (!last || Math.abs(last.time - tMax) > 1e-6) {
      samples.push({ time: tMax, type: final_types[cellIdx] });
    } else {
      last.type = final_types[cellIdx];
    }
  }

  const validSamples = samples
    .filter((sample) => sample.type)
    .sort((a, b) => a.time - b.time);

  if (validSamples.length === 0) return CELL_TYPE_COLORS.dev;
  if (timeFloat <= validSamples[0].time) return CELL_TYPE_COLORS[validSamples[0].type] || "#888888";

  const halfWindow = KIDNEY_TYPE_TRANSITION_WINDOW / 2;
  let activeTransition = null;
  for (let i = 0; i < validSamples.length - 1; i++) {
    const from = validSamples[i];
    const to = validSamples[i + 1];
    if (from.type !== to.type) {
      const changeTime = to.time;
      const transitionStart = Math.max(validSamples[0].time, changeTime - halfWindow);
      const transitionEnd = Math.min(validSamples[validSamples.length - 1].time, changeTime + halfWindow);
      if (timeFloat >= transitionStart && timeFloat <= transitionEnd) {
        const distance = Math.abs(timeFloat - changeTime);
        if (!activeTransition || distance < activeTransition.distance) {
          activeTransition = { from, to, transitionStart, transitionEnd, distance };
        }
      }
    }
  }

  if (activeTransition) {
    const { from, to, transitionStart, transitionEnd } = activeTransition;
    const span = transitionEnd - transitionStart || 1;
    const mix = smoothstep((timeFloat - transitionStart) / span);
    return rgbStr(lerpColor(cellTypeRgb(from.type), cellTypeRgb(to.type), mix));
  }

  for (let i = validSamples.length - 1; i >= 0; i--) {
    if (timeFloat >= validSamples[i].time) {
      return CELL_TYPE_COLORS[validSamples[i].type] || "#888888";
    }
  }

  const last = validSamples[validSamples.length - 1];
  return CELL_TYPE_COLORS[last.type] || "#888888";
}

/* ── 从ODE轨迹数据中插值取某一时刻所有细胞的位置 ────────────────────── */
function getPositionsAtTime(trajData, timeFloat) {
  // trajData: {shape:[nCells, nBins, dim], times:[...], data:[...]}
  const { shape, times, data } = trajData;
  const nCells = shape[0];
  const nBins = shape[1];
  const dim = shape[2];
  const tMin = times[0];
  const tMax = times[nBins - 1];
  const tNorm = (timeFloat - tMin) / (tMax - tMin); // 0~1
  const binF = tNorm * (nBins - 1);
  const bin0 = Math.max(0, Math.min(Math.floor(binF), nBins - 2));
  const bin1 = bin0 + 1;
  const frac = smoothstep(binF - bin0);

  const result = new Float32Array(nCells * dim);
  for (let i = 0; i < nCells; i++) {
    for (let d = 0; d < dim; d++) {
      result[i * dim + d] = lerp(data[i][bin0][d], data[i][bin1][d], frac);
    }
  }
  return { positions: result, nCells, dim };
}

/* ── Data Cache ─────────────────────────────────────────────────────── */
const dataCache = {};

async function loadJSON(path) {
  if (dataCache[path]) return dataCache[path];
  const resp = await fetch(path);
  const data = await resp.json();
  dataCache[path] = data;
  return data;
}

/* ══════════════════════════════════════════════════════════════════════
   SIMULATION 3D VIEWER (Three.js)
   ══════════════════════════════════════════════════════════════════════ */

class SimuViewer {
  constructor(container, viewMode) {
    this.container = container;
    this.currentTime = 0;
    this.playing = false;
    this.viewMode = viewMode;  // "2d" or "3d" - fixed for each instance
    this.trajData = null;      // ode_trajectories.json (2D)
    this.trajData3d = null;    // ode_trajectories_3d.json (3D with T)
    this.decorTraj = null;     // decorative_trajectories.json
    this.tSurface = null;
    this.animStartTime = null;
    this.transitionDuration = 16000; // 16秒走完 t=0→4
    this.speedMultiplier = 1;

    this.init();
  }

  init() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf6f8fa);

    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 1000);
    this.camera.position.set(0, 0, 20);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 50;
    this.controls.zoomSpeed = 0.3;

    // 自定义平滑缩放
    this.targetDistance = this.camera.position.distanceTo(this.controls.target);
    this.currentDistance = this.targetDistance;
    this.controls.enableZoom = false;

    this.container.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.08 : 0.92;
      this.targetDistance *= factor;
      this.targetDistance = Math.max(2, Math.min(50, this.targetDistance));
    }, { passive: false });

    const ambient = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.5);
    dir.position.set(5, 10, 7);
    this.scene.add(dir);

    this.cellGroup = new THREE.Group();
    this.trailGroup = new THREE.Group();
    this.surfaceGroup = new THREE.Group();
    this.decorGroup = new THREE.Group();
    this.scene.add(this.cellGroup);
    this.scene.add(this.trailGroup);
    this.scene.add(this.surfaceGroup);
    this.scene.add(this.decorGroup);

    // Grid
    const gridHelper = new THREE.GridHelper(10, 20, 0xe0e0e0, 0xf0f0f0);
    gridHelper.rotation.x = Math.PI / 2;
    this.scene.add(gridHelper);

    // 添加坐标刻度标签
    this.addAxisTicks();

    // InstancedMesh (persistent, update per frame)
    this.instancedMesh = null;
    this.sphereGeo = new THREE.SphereGeometry(0.08, 12, 12);
    this.cellMaterial = new THREE.MeshBasicMaterial({
      transparent: true, opacity: 0.92,
    });

    window.addEventListener("resize", () => this.onResize());
    this.animate();
  }

  addAxisTicks() {
    // 数据范围
    const xMin = -0.5, xMax = 3.5;
    const yMin = -0.5, yMax = 2.8;

    // 添加 X 轴刻度 (底部)
    const xTicks = [0, 1, 2, 3];
    xTicks.forEach(val => {
      const tick = document.createElement('div');
      tick.textContent = val.toString();
      // 计算位置百分比
      const percent = ((val - xMin) / (xMax - xMin)) * 100;
      tick.style.cssText = `
        position: absolute;
        bottom: 5px;
        left: ${percent}%;
        transform: translateX(-50%);
        color: #666;
        font-size: 10px;
        font-family: Arial, sans-serif;
        pointer-events: none;
        z-index: 10;
      `;
      this.container.appendChild(tick);
    });

    // 添加 Y 轴刻度 (左侧)
    const yTicks = [0, 1, 2];
    yTicks.forEach(val => {
      const tick = document.createElement('div');
      tick.textContent = val.toString();
      // 计算位置百分比
      const percent = ((val - yMin) / (yMax - yMin)) * 100;
      tick.style.cssText = `
        position: absolute;
        left: 5px;
        bottom: ${percent}%;
        transform: translateY(50%);
        color: #666;
        font-size: 10px;
        font-family: Arial, sans-serif;
        pointer-events: none;
        z-index: 10;
      `;
      this.container.appendChild(tick);
    });

    // 轴标签
    const xLabel = document.createElement('div');
    xLabel.textContent = 'X';
    xLabel.style.cssText = `
      position: absolute;
      bottom: 5px;
      right: 10px;
      color: #333;
      font-size: 12px;
      font-weight: bold;
      font-family: Arial, sans-serif;
      pointer-events: none;
      z-index: 10;
    `;
    this.container.appendChild(xLabel);

    const yLabel = document.createElement('div');
    yLabel.textContent = 'Y';
    yLabel.style.cssText = `
      position: absolute;
      top: 10px;
      left: 5px;
      color: #333;
      font-size: 12px;
      font-weight: bold;
      font-family: Arial, sans-serif;
      pointer-events: none;
      z-index: 10;
    `;
    this.container.appendChild(yLabel);
  }

  onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  async loadData() {
    const [traj2d, traj3d, decor, tSurface] = await Promise.all([
      loadJSON("demo_data/simu/ode_trajectories.json"),
      loadJSON("demo_data/simu/ode_trajectories_3d.json"),
      loadJSON("demo_data/simu/decorative_trajectories.json"),
      loadJSON("demo_data/simu/t_surface.json"),
    ]);
    this.trajData = traj2d;
    this.trajData3d = traj3d;
    this.decorTraj = decor;
    this.tSurface = tSurface;

    const nCells = traj2d.shape[0];
    // Create persistent InstancedMesh
    this.instancedMesh = new THREE.InstancedMesh(this.sphereGeo, this.cellMaterial, nCells);
    this.cellGroup.add(this.instancedMesh);

    this.buildTSurface();
    this.buildDecorativeLines();
    this.setCameraForView();
    this.renderFrame(0);
  }

  buildTSurface() {
    const { x, y, z, z_min, z_max } = this.tSurface;
    const nx = x.length;
    const ny = y.length;

    // 创建自定义几何体，使用实际的 x, y 坐标
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    const colors = [];
    const indices = [];
    const zRange = z_max - z_min || 1;

    // 创建顶点
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        // 使用实际的 x, y 坐标
        vertices.push(x[ix], y[iy], z[iy][ix]);

        // 颜色
        const t = (z[iy][ix] - z_min) / zRange;
        colors.push(
          lerp(0.96, 0.36, t),
          lerp(0.97, 0.55, t),
          lerp(0.98, 0.74, t)
        );
      }
    }

    // 创建面索引
    for (let iy = 0; iy < ny - 1; iy++) {
      for (let ix = 0; ix < nx - 1; ix++) {
        const a = iy * nx + ix;
        const b = iy * nx + ix + 1;
        const c = (iy + 1) * nx + ix;
        const d = (iy + 1) * nx + ix + 1;
        indices.push(a, b, d);
        indices.push(a, d, c);
      }
    }

    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true, transparent: true, opacity: 0.45,
      side: THREE.DoubleSide, shininess: 30, depthWrite: false,
    });

    const mesh = new THREE.Mesh(geometry, material);
    this.surfaceGroup.add(mesh);
  }

  buildDecorativeLines() {
    if (!this.decorTraj) return;
    const { shape, times, data } = this.decorTraj;
    const nDecor = shape[0];
    const nBins = shape[1];
    const tMin = times[0];
    const tMax = times[nBins - 1];
    const tRange = tMax - tMin || 1;

    for (let j = 0; j < nDecor; j++) {
      const positions = [];
      const colors = [];
      for (let k = 0; k < nBins; k++) {
        positions.push(data[j][k][0], data[j][k][1], 0.05);
        const c = timeToColor((times[k] - tMin) / tRange);
        colors.push(c[0], c[1], c[2]);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
      const mat = new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.18, linewidth: 1,
      });
      this.decorGroup.add(new THREE.Line(geo, mat));
    }
  }

  renderFrame(timeFloat) {
    const useTraj = this.viewMode === "3d" ? this.trajData3d : this.trajData;
    if (!useTraj || !this.instancedMesh) return;

    const { positions, nCells, dim } = getPositionsAtTime(useTraj, timeFloat);
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    for (let i = 0; i < nCells; i++) {
      const px = positions[i * dim];
      const py = positions[i * dim + 1];
      const pz = dim >= 3 ? positions[i * dim + 2] + 0.1 : 0;
      dummy.position.set(px, py, pz);
      dummy.updateMatrix();
      this.instancedMesh.setMatrixAt(i, dummy.matrix);

      const tNorm = timeFloat / 4;
      const tc = timeToColor(tNorm);
      color.setRGB(tc[0], tc[1], tc[2]);
      this.instancedMesh.setColorAt(i, color);
    }

    this.instancedMesh.instanceMatrix.needsUpdate = true;
    if (this.instancedMesh.instanceColor) this.instancedMesh.instanceColor.needsUpdate = true;

    this.updateTrails(useTraj, timeFloat);
    this.updateInfo(nCells);
  }

  updateTrails(trajData, timeFloat) {
    // Clear old trails
    while (this.trailGroup.children.length > 0) {
      const child = this.trailGroup.children[0];
      this.trailGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }

    // Draw trails by sampling recent time points
    const trailSteps = 5;
    const stepSize = 0.08; // 时间步长
    const { shape, data, times } = trajData;
    const nCells = shape[0];
    const dim = shape[2];
    const nBins = shape[1];
    const tMin = times[0];
    const tMax = times[nBins - 1];

    const sphereGeo = new THREE.SphereGeometry(0.05, 8, 8);

    for (let s = trailSteps; s >= 1; s--) {
      const tTrail = timeFloat - s * stepSize;
      if (tTrail < tMin) continue;

      const alpha = ((trailSteps - s + 1) / trailSteps) * 0.3;
      const scale = 0.3 + ((trailSteps - s) / trailSteps) * 0.5;
      const tNorm = tTrail / 4;
      const tc = timeToColor(Math.max(0, Math.min(1, tNorm)));

      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(tc[0], tc[1], tc[2]),
        transparent: true, opacity: alpha,
      });

      const tNormTrail = (tTrail - tMin) / (tMax - tMin);
      const binF = tNormTrail * (nBins - 1);
      const bin0 = Math.max(0, Math.min(Math.floor(binF), nBins - 2));
      const frac = smoothstep(binF - bin0);

      for (let i = 0; i < nCells; i += 3) { // 每3个细胞画一个trail点(性能)
        const px = lerp(data[i][bin0][0], data[i][bin0 + 1][0], frac);
        const py = lerp(data[i][bin0][1], data[i][bin0 + 1][1], frac);
        const pz = dim >= 3 ? lerp(data[i][bin0][2], data[i][bin0 + 1][2], frac) + 0.1 : 0.01;
        const mesh = new THREE.Mesh(sphereGeo, mat);
        mesh.position.set(px, py, pz);
        mesh.scale.set(scale, scale, scale);
        this.trailGroup.add(mesh);
      }
    }
  }

  updateInfo(nCells) {
    const el = document.getElementById("simu-cell-count");
    if (el) el.textContent = nCells;
    const el2 = document.getElementById("simu-traj-count");
    if (el2) el2.textContent = this.decorTraj ? this.decorTraj.shape[0] : "—";
    const el3 = document.getElementById("simu-mean-growth");
    if (el3) el3.textContent = "—";
  }

  setTime(t) {
    this.currentTime = Math.max(0, Math.min(t, 4));
    this.renderFrame(this.currentTime);
    this.updateTimeLabel();
    // 如果正在播放，更新animStartTime以继续从新位置播放
    if (this.playing) {
      this.animStartTime = performance.now() - (this.currentTime / 4) * this.transitionDuration;
    }
  }

  updateTimeLabel() {
    const label = document.getElementById("simu-time-label");
    if (label) {
      // 显示当前的精确时间值
      label.textContent = `t = ${this.currentTime.toFixed(2)}`;
    }
  }

  setCameraForView() {
    if (!this.trajData) return;

    // 计算所有时间点的细胞位置边界（而不仅仅是t=0）
    const useTraj = this.viewMode === "3d" ? this.trajData3d : this.trajData;
    if (!useTraj) return;

    const { shape, times, data } = useTraj;
    const nCells = shape[0];
    const nBins = shape[1];
    const dim = shape[2];

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;

    // 遍历所有时间点计算全局边界
    for (let bin = 0; bin < nBins; bin++) {
      for (let i = 0; i < nCells; i++) {
        const px = data[i][bin][0];
        const py = data[i][bin][1];
        if (px < xMin) xMin = px;
        if (px > xMax) xMax = px;
        if (py < yMin) yMin = py;
        if (py > yMax) yMax = py;
      }
    }

    // 添加一些边距
    const padding = 0.1;
    xMin -= padding;
    xMax += padding;
    yMin -= padding;
    yMax += padding;

    const cx = (xMin + xMax) / 2;
    const cy = (yMin + yMax) / 2;
    const range = Math.max(xMax - xMin, yMax - yMin) * 0.6;

    if (this.viewMode === "2d") {
      const dist = range * 1.8;
      this.camera.position.set(cx, cy, dist);
      this.camera.lookAt(cx, cy, 0);
      this.controls.target.set(cx, cy, 0);
      this.surfaceGroup.visible = false;
      this.targetDistance = dist;
      this.currentDistance = dist;
    } else {
      const dist = range * 2.5;
      const elev = (36 * Math.PI) / 180;
      const azim = (222 * Math.PI) / 180;
      this.camera.position.set(
        cx + dist * Math.cos(elev) * Math.cos(azim),
        cy + dist * Math.cos(elev) * Math.sin(azim),
        dist * Math.sin(elev)
      );
      this.camera.lookAt(cx, cy, 0);
      this.controls.target.set(cx, cy, 0);
      this.surfaceGroup.visible = true;
      this.targetDistance = dist;
      this.currentDistance = dist;
    }
    this.controls.update();
  }

  togglePlay() {
    this.playing = !this.playing;
    const btn = document.getElementById("simu-play-btn");
    if (btn) {
      btn.textContent = this.playing ? "❚❚" : "▶";
      btn.classList.toggle("playing", this.playing);
    }
    if (this.playing) {
      this.animStartTime = performance.now() - (this.currentTime / 4) * this.transitionDuration;
    }
  }

  toggleSpeed() {
    this.speedMultiplier = this.speedMultiplier === 1 ? 2 : 1;
    this.transitionDuration = 16000 / this.speedMultiplier;
    const btn = document.getElementById("simu-speed-btn");
    if (btn) {
      btn.textContent = this.speedMultiplier === 1 ? "1×" : "2×";
      btn.classList.toggle("fast", this.speedMultiplier !== 1);
    }
    if (this.playing) {
      this.animStartTime = performance.now() - (this.currentTime / 4) * this.transitionDuration;
    }
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    if (this.playing && this.trajData) {
      const elapsed = performance.now() - this.animStartTime;
      const tNorm = (elapsed / this.transitionDuration) % 1; // 0~1 循环
      this.currentTime = tNorm * 4;
      this.renderFrame(this.currentTime);
      const slider = document.getElementById("simu-slider");
      if (slider) slider.value = this.currentTime;
      this.updateTimeLabel();
    }

    // 平滑缩放动画 - 每帧插值当前距离到目标距离
    this.currentDistance += (this.targetDistance - this.currentDistance) * 0.06;
    const direction = this.camera.position.clone().sub(this.controls.target).normalize();
    this.camera.position.copy(this.controls.target).add(direction.multiplyScalar(this.currentDistance));

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

/* ══════════════════════════════════════════════════════════════════════
   KIDNEY DUAL-OMICS 2D VIEWER
   ══════════════════════════════════════════════════════════════════════ */

class KidneyViewer {
  constructor() {
    this.currentTime = 0;
    this.playing = false;
    this.trajRna = null;
    this.trajAtac = null;
    this.decorRna = null;
    this.decorAtac = null;
    this.background = null;
    this.proportionsContinuous = null;
    this.animStartTime = null;
    this.transitionDuration = 40000;
    this.speedMultiplier = 1;
    this.loaded = false;

    // Trajectory selection
    this.fateMode = "mixed";
    this.trajCount = 40;
    this.selectedIndices = [];

    this.canvasRna = document.getElementById("kidney-rna-canvas");
    this.canvasAtac = document.getElementById("kidney-atac-canvas");
    this.ctxRna = this.canvasRna ? this.canvasRna.getContext("2d") : null;
    this.ctxAtac = this.canvasAtac ? this.canvasAtac.getContext("2d") : null;
    this.canvasProps = document.getElementById("proportions-canvas");
    this.ctxProps = this.canvasProps ? this.canvasProps.getContext("2d") : null;

    this.bounds = null;
    this.buildLegend();
  }

  async loadData() {
    if (this.loaded) return;

    const [trajRna, trajAtac, decorRna, decorAtac, bg, propsCont, selectionInfo] = await Promise.all([
      loadJSON("demo_data/kidney/traj_sample_rna.json"),
      loadJSON("demo_data/kidney/traj_sample_atac.json"),
      loadJSON("demo_data/kidney/decorative_traj_rna.json"),
      loadJSON("demo_data/kidney/decorative_traj_atac.json"),
      loadJSON("demo_data/kidney/background.json"),
      loadJSON("demo_data/kidney/proportions_continuous.json"),
      loadJSON("demo_data/kidney/traj_selection_info.json"),
    ]);

    this.trajRna = trajRna;
    this.trajAtac = trajAtac;
    this.decorRna = decorRna;
    this.decorAtac = decorAtac;
    this.background = bg;
    this.selectionInfo = selectionInfo;

    // 修复proportions数据：将50个点插值到完整时间范围0-4（Day 7-26）
    const { times: propTimes, data: propData, cell_type_colors } = propsCont;
    const nOrig = propData[Object.keys(propData)[0]].length; // 50
    const fullTimes = [];
    const fullData = {};
    for (const ct of Object.keys(propData)) {
      fullData[ct] = [];
    }

    // 生成101个时间点，覆盖完整范围0-4
    for (let i = 0; i < 101; i++) {
      const t = i * 4.0 / 100; // 0 to 4
      fullTimes.push(t);

      // 在50个原始点之间插值
      // 原始50个点对应的时间范围是0-1.96，需要映射到0-4
      const tOrig = t * (propTimes[nOrig - 1] / 4.0); // 映射到原始时间范围
      const binF = (tOrig / propTimes[nOrig - 1]) * (nOrig - 1);
      const bin0 = Math.max(0, Math.min(Math.floor(binF), nOrig - 2));
      const frac = binF - bin0;

      for (const ct of Object.keys(propData)) {
        const v0 = propData[ct][bin0] || 0;
        const v1 = propData[ct][bin0 + 1] || 0;
        fullData[ct].push(v0 + (v1 - v0) * frac);
      }
    }

    this.proportionsContinuous = {
      times: fullTimes,
      data: fullData,
      cell_type_colors: cell_type_colors,
    };

    this.computeBounds();
    this.selectTrajectories();

    this.loaded = true;
    this.renderFrame(0);
  }

  computeBounds() {
    let xMinR = Infinity, xMaxR = -Infinity, yMinR = Infinity, yMaxR = -Infinity;
    let xMinA = Infinity, xMaxA = -Infinity, yMinA = Infinity, yMaxA = -Infinity;

    for (const key of Object.keys(this.background)) {
      const bg = this.background[key];
      for (let i = 0; i < bg.positions_rna.length; i++) {
        const [x, y] = bg.positions_rna[i];
        if (x < xMinR) xMinR = x; if (x > xMaxR) xMaxR = x;
        if (y < yMinR) yMinR = y; if (y > yMaxR) yMaxR = y;
      }
      for (let i = 0; i < bg.positions_atac.length; i++) {
        const [x, y] = bg.positions_atac[i];
        if (x < xMinA) xMinA = x; if (x > xMaxA) xMaxA = x;
        if (y < yMinA) yMinA = y; if (y > yMaxA) yMaxA = y;
      }
    }

    this.bounds = {
      rna: { xMin: xMinR - 1, xMax: xMaxR + 1, yMin: yMinR - 1, yMax: yMaxR + 1 },
      atac: { xMin: xMinA - 1, xMax: xMaxA + 1, yMin: yMinA - 1, yMax: yMaxA + 1 },
    };
  }

  /* ── Trajectory selection ────────────────────────────────────────── */
  selectTrajectories() {
    if (!this.trajRna || !this.trajRna.final_types || !this.selectionInfo) return;

    const count = Math.min(this.trajCount, 100);
    if (this.fateMode === "mixed") {
      this.selectedIndices = this.selectProportionalRandomTrajectories(count);
      const countEl = document.getElementById("kidney-traj-count");
      if (countEl) countEl.textContent = this.selectedIndices.length;
      return;
    }

    // 使用预选的轨迹索引
    const selection = this.selectionInfo[this.fateMode];

    if (selection && selection.new_indices) {
      this.selectedIndices = selection.new_indices.slice(0, count);
    } else {
      // 回退：如果没有预选数据，从所有轨迹中选择
      const finalTypes = this.trajRna.final_types;
      const nTotal = finalTypes.length;
      const matching = [];
      for (let i = 0; i < nTotal; i++) {
        if (finalTypes[i] === this.fateMode) matching.push(i);
      }
      this.selectedIndices = this._shuffle(matching).slice(0, count);
    }

    // Update count display
    const countEl = document.getElementById("kidney-traj-count");
    if (countEl) countEl.textContent = this.selectedIndices.length;
  }

  getFinalFateQuotas(count) {
    const types = Object.keys(CELL_TYPE_COLORS);
    const data = this.proportionsContinuous ? this.proportionsContinuous.data : null;
    const proportions = {};

    for (const type of types) {
      const values = data && data[type];
      proportions[type] = values && values.length ? values[values.length - 1] : 1 / types.length;
    }

    const total = Object.values(proportions).reduce((sum, value) => sum + value, 0) || 1;
    const quotas = {};
    const remainders = [];
    let assigned = 0;

    for (const type of types) {
      const raw = (proportions[type] / total) * count;
      const whole = Math.floor(raw);
      quotas[type] = whole;
      assigned += whole;
      remainders.push({ type, remainder: raw - whole });
    }

    remainders.sort((a, b) => b.remainder - a.remainder);
    for (let i = 0; assigned < count; i++, assigned++) {
      quotas[remainders[i % remainders.length].type]++;
    }

    return quotas;
  }

  selectProportionalRandomTrajectories(count) {
    const quotas = this.getFinalFateQuotas(count);
    const finalTypes = this.trajRna.final_types;
    const picked = [];
    const used = new Set();
    const randomSource = (this.selectionInfo.Random && this.selectionInfo.Random.new_indices)
      ? this.selectionInfo.Random.new_indices
      : finalTypes.map((_, idx) => idx);

    const tryPick = (idx) => {
      if (used.has(idx)) return false;
      const fate = finalTypes[idx];
      if (!quotas[fate] || quotas[fate] <= 0) return false;
      picked.push(idx);
      used.add(idx);
      quotas[fate]--;
      return true;
    };

    for (const idx of randomSource) {
      if (picked.length >= count) break;
      tryPick(idx);
    }

    for (const fate of Object.keys(CELL_TYPE_COLORS)) {
      if (!quotas[fate] || quotas[fate] <= 0) continue;
      const source = (this.selectionInfo[fate] && this.selectionInfo[fate].new_indices)
        ? this.selectionInfo[fate].new_indices
        : finalTypes.map((type, idx) => type === fate ? idx : -1).filter((idx) => idx >= 0);
      for (const idx of source) {
        if (picked.length >= count || quotas[fate] <= 0) break;
        tryPick(idx);
      }
    }

    if (picked.length < count) {
      for (const idx of randomSource) {
        if (picked.length >= count) break;
        if (!used.has(idx)) {
          picked.push(idx);
          used.add(idx);
        }
      }
    }

    return picked;
  }

  _shuffle(arr) {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  setFateMode(mode) {
    this.fateMode = mode;
    this.selectTrajectories();
    if (this.loaded) this.renderFrame(this.currentTime);
  }

  setTrajCount(count) {
    this.trajCount = count;
    this.selectTrajectories();
    if (this.loaded) this.renderFrame(this.currentTime);
  }

  buildLegend() {
    const container = document.getElementById("kidney-legend-items");
    if (!container) return;
    container.innerHTML = "";
    for (const [name, color] of Object.entries(CELL_TYPE_COLORS)) {
      const item = document.createElement("div");
      item.className = "legend-item";
      item.innerHTML = `<span class="legend-dot" style="background:${color}"></span>${name.replace(/_/g, " ")}`;
      container.appendChild(item);
    }
  }

  renderFrame(timeFloat) {
    if (!this.loaded) return;

    this.drawCanvas(this.ctxRna, this.canvasRna, this.trajRna, this.decorRna, timeFloat, "rna");
    this.drawCanvas(this.ctxAtac, this.canvasAtac, this.trajAtac, this.decorAtac, timeFloat, "atac");
    this.drawProportions(timeFloat);

    this.updateInfo(timeFloat);
  }

  drawCanvas(ctx, canvas, trajData, decorData, timeFloat, omics) {
    if (!ctx || !canvas || !trajData) return;

    const w = canvas.width;
    const h = canvas.height;
    const padding = 40;
    const bnd = this.bounds[omics];
    const xRange = bnd.xMax - bnd.xMin || 1;
    const yRange = bnd.yMax - bnd.yMin || 1;

    const toScreen = (x, y) => [
      padding + ((x - bnd.xMin) / xRange) * (w - 2 * padding),
      h - padding - ((y - bnd.yMin) / yRange) * (h - 2 * padding),
    ];

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#f6f8fa";
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = "rgba(16,32,51,.06)";
    ctx.lineWidth = 1;
    for (let gx = padding; gx <= w - padding; gx += 50) {
      ctx.beginPath(); ctx.moveTo(gx, padding); ctx.lineTo(gx, h - padding); ctx.stroke();
    }
    for (let gy = padding; gy <= h - padding; gy += 50) {
      ctx.beginPath(); ctx.moveTo(padding, gy); ctx.lineTo(w - padding, gy); ctx.stroke();
    }

    // ── 1. Background: all real cells as small crosses ───────────────
    if (this.background) {
      const bgKeys = Object.keys(this.background).sort((a, b) => {
        const ai = parseInt(a.replace(/\D/g, ""), 10);
        const bi = parseInt(b.replace(/\D/g, ""), 10);
        return ai - bi;
      });
      ctx.lineWidth = 1;
      ctx.globalAlpha = KIDNEY_BACKGROUND_ALPHA;
      for (const bgKey of bgKeys) {
        const bg = this.background[bgKey];
        if (!bg) continue;
        const positions = omics === "rna" ? bg.positions_rna : bg.positions_atac;
        const types = bg.types;
        for (let i = 0; i < positions.length; i++) {
          const [px, py] = positions[i];
          const [sx, sy] = toScreen(px, py);
          const color = CELL_TYPE_COLORS[types[i]] || "#ccc";
          ctx.strokeStyle = color;
          const sz = 3;
          ctx.beginPath(); ctx.moveTo(sx - sz, sy - sz); ctx.lineTo(sx + sz, sy + sz); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(sx + sz, sy - sz); ctx.lineTo(sx - sz, sy + sz); ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
    }

    // ── 2. Decorative trajectories (up to current time) ──────────────
    if (decorData) {
      const { shape, times, data } = decorData;
      const nDecor = shape[0];
      const nBins = shape[1];
      const tMin = times[0];
      const tMax = times[nBins - 1];
      const tNorm = (timeFloat - tMin) / (tMax - tMin);
      const endBin = Math.min(Math.floor(tNorm * (nBins - 1)), nBins - 1);

      for (let j = 0; j < nDecor; j++) {
        ctx.lineWidth = 1.2;
        ctx.globalAlpha = 0.15;
        ctx.strokeStyle = "rgba(100,100,100,.4)";
        ctx.beginPath();
        const [sx0, sy0] = toScreen(data[j][0][0], data[j][0][1]);
        ctx.moveTo(sx0, sy0);
        for (let k = 1; k <= endBin; k += 2) {
          const [sx, sy] = toScreen(data[j][k][0], data[j][k][1]);
          ctx.lineTo(sx, sy);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // ── 3. Selected trajectories: gray departure lines + colored dots ─
    const { positions, dim } = getPositionsAtTime(trajData, timeFloat);
    const { shape: trajShape, times: trajTimes, data: trajRaw } = trajData;
    const nBins = trajShape[1];
    const tMin = trajTimes[0];
    const tMax = trajTimes[nBins - 1];
    const tNormTraj = (timeFloat - tMin) / (tMax - tMin);
    const endBinTraj = Math.min(Math.floor(tNormTraj * (nBins - 1)), nBins - 1);

    for (const idx of this.selectedIndices) {
      if (idx >= trajShape[0]) continue;

      // Gray departure line: full path from start to current time
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.3;
      ctx.strokeStyle = "rgba(120,120,120,.5)";
      ctx.beginPath();
      const [sx0, sy0] = toScreen(trajRaw[idx][0][0], trajRaw[idx][0][1]);
      ctx.moveTo(sx0, sy0);
      for (let k = 1; k <= endBinTraj; k += 1) {
        const [sx, sy] = toScreen(trajRaw[idx][k][0], trajRaw[idx][k][1]);
        ctx.lineTo(sx, sy);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Colored dot at current position, with smooth cell-type transition colors.
      const px = positions[idx * dim];
      const py = positions[idx * dim + 1];
      const [sx, sy] = toScreen(px, py);
      const tc = trajectoryTypeColor(trajData, idx, timeFloat);
      const radius = 4;

      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fillStyle = tc;
      ctx.globalAlpha = KIDNEY_TRAJECTORY_ALPHA;
      ctx.fill();

      // Highlight
      ctx.beginPath();
      ctx.arc(sx - 1, sy - 1, radius * 0.3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,.5)";
      ctx.fill();

      // Border
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,.7)";
      ctx.lineWidth = 0.8;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Axis labels
    ctx.fillStyle = "#66758a";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("UMAP 1", w / 2, h - 8);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("UMAP 2", 0, 0);
    ctx.restore();
  }

  /* ── Bar chart for cell type proportions at current time ─────── */
  drawProportions(timeFloat) {
    const ctx = this.ctxProps;
    const canvas = this.canvasProps;
    if (!ctx || !canvas || !this.proportionsContinuous) return;

    const w = canvas.width;
    const h = canvas.height;
    const legendW = 100;
    const padL = legendW + 10, padR = 16, padT = 12, padB = 16;
    const chartW = w - padL - padR;
    const chartH = h - padT - padB;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#f6f8fa";
    ctx.fillRect(0, 0, w, h);

    const { times, data } = this.proportionsContinuous;
    const types = Object.keys(CELL_TYPE_COLORS);
    const nBins = times.length;
    const tMin = times[0];
    const tMax = times[nBins - 1];

    // Interpolate proportions at current time
    const tNorm = Math.max(0, Math.min(1, (timeFloat - tMin) / (tMax - tMin)));
    const binF = tNorm * (nBins - 1);
    const bin0 = Math.max(0, Math.min(Math.floor(binF), nBins - 2));
    const frac = smoothstep(binF - bin0);

    const currentProps = {};
    const bin1 = bin0 + 1;
    for (const ct of types) {
      const v0 = (data[ct] && data[ct][bin0]) || 0;
      const v1 = (data[ct] && data[ct][bin1]) || 0;
      currentProps[ct] = lerp(v0, v1, frac);
    }

    const maxProp = Math.max(...Object.values(currentProps), 0.01);
    const barW = chartW / types.length;
    const gap = 6;

    // 左侧图例（一列展示）
    const legendX = 8;
    let legendY = padT;
    const legendItemH = chartH / types.length;

    for (let i = 0; i < types.length; i++) {
      const ct = types[i];
      const prop = currentProps[ct] || 0;
      const color = CELL_TYPE_COLORS[ct];

      // 颜色圆点
      ctx.beginPath();
      ctx.arc(legendX + 8, legendY + legendItemH / 2, 6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // 类型名称
      ctx.fillStyle = "#1a2332";
      ctx.font = "bold 12px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(ct, legendX + 20, legendY + legendItemH / 2 - 7);

      // 百分比
      ctx.fillStyle = "#66758a";
      ctx.font = "11px Inter, sans-serif";
      ctx.fillText(`${Math.round(prop * 100)}%`, legendX + 20, legendY + legendItemH / 2 + 7);

      legendY += legendItemH;
    }

    // Draw bars
    for (let i = 0; i < types.length; i++) {
      const ct = types[i];
      const prop = currentProps[ct] || 0;
      const barH = (prop / maxProp) * chartH;
      const x = padL + i * barW + gap / 2;
      const y = padT + chartH - barH;
      const bw = barW - gap;

      // Bar
      ctx.fillStyle = CELL_TYPE_COLORS[ct];
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      const r = 4;
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + bw - r, y);
      ctx.quadraticCurveTo(x + bw, y, x + bw, y + r);
      ctx.lineTo(x + bw, padT + chartH);
      ctx.lineTo(x, padT + chartH);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.fill();
      ctx.globalAlpha = 1;

      // Percentage label on top
      ctx.fillStyle = "#4a5568";
      ctx.font = "bold 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${Math.round(prop * 100)}%`, x + bw / 2, y - 3);
    }

    // Y axis
    ctx.strokeStyle = "rgba(16,32,51,.1)";
    ctx.lineWidth = 0.5;
    for (let p = 0; p <= 1; p += 0.25) {
      const y = padT + chartH * (1 - p / maxProp);
      if (y < padT) continue;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + chartW, y);
      ctx.stroke();
      ctx.fillStyle = "#999";
      ctx.font = "9px Inter, sans-serif";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(`${Math.round(p * 100)}%`, padL - 4, y);
    }
  }

  updateInfo(timeFloat) {
    const day = odeTimeToDay(timeFloat);
    const dayStr = `Day ${Math.round(day)}`;

    // Phase: use day boundaries
    // REAL_DAYS = [7, 12, 16, 19, 26]
    // Phase 0: Day 7-12, Phase 1: Day 12-16, Phase 2: Day 16-19, Phase 3: Day 19-26
    const phaseBounds = REAL_DAYS; // [7, 12, 16, 19, 26]
    let phaseIdx = 0;
    for (let i = 0; i < phaseBounds.length - 1; i++) {
      if (day >= phaseBounds[i]) phaseIdx = i;
    }
    const phaseStart = phaseBounds[phaseIdx];
    const phaseEnd = phaseBounds[Math.min(phaseIdx + 1, phaseBounds.length - 1)];
    const phaseStr = phaseIdx < phaseBounds.length - 1
      ? `Day ${phaseStart} → Day ${phaseEnd}`
      : `Day ${phaseEnd}`;

    const el1 = document.getElementById("kidney-time-info");
    if (el1) el1.textContent = dayStr;

    const phaseChip = document.getElementById("kidney-time-phase");
    if (phaseChip) phaseChip.textContent = phaseStr;

    // Update fate btn active states
    const countEl = document.getElementById("kidney-traj-count");
    if (countEl) countEl.textContent = this.selectedIndices.length;
  }

  setTime(t) {
    this.currentTime = Math.max(0, Math.min(t, 4));
    if (this.loaded) {
      this.renderFrame(this.currentTime);
      this.updateTimeLabel();
    }
    // 如果正在播放，更新animStartTime以继续从新位置播放
    if (this.playing) {
      this.animStartTime = performance.now() - (this.currentTime / 4) * this.transitionDuration;
    }
  }

  updateTimeLabel() {
    const label = document.getElementById("kidney-time-label");
    if (label) {
      const day = odeTimeToDay(this.currentTime);
      label.textContent = `Day ${Math.round(day)}`;
    }
  }

  togglePlay() {
    this.playing = !this.playing;
    const btn = document.getElementById("kidney-play-btn");
    if (btn) {
      btn.textContent = this.playing ? "❚❚" : "▶";
      btn.classList.toggle("playing", this.playing);
    }
    if (this.playing) {
      this.animStartTime = performance.now() - (this.currentTime / 4) * this.transitionDuration;
    }
  }

  toggleSpeed() {
    this.speedMultiplier = this.speedMultiplier === 1 ? 2 : 1;
    this.transitionDuration = 40000 / this.speedMultiplier;
    const btn = document.getElementById("kidney-speed-btn");
    if (btn) {
      btn.textContent = this.speedMultiplier === 1 ? "1×" : "2×";
      btn.classList.toggle("fast", this.speedMultiplier !== 1);
    }
    if (this.playing) {
      this.animStartTime = performance.now() - (this.currentTime / 4) * this.transitionDuration;
    }
  }

  tick() {
    if (this.playing && this.loaded) {
      const elapsed = performance.now() - this.animStartTime;
      const tNorm = (elapsed / this.transitionDuration) % 1;
      this.currentTime = tNorm * 4;
      this.renderFrame(this.currentTime);
      const slider = document.getElementById("kidney-slider");
      if (slider) slider.value = this.currentTime;
      this.updateTimeLabel();
    }
  }
}

/* ══════════════════════════════════════════════════════════════════════
   TF UMAP INTERACTIVE VIEWER
   Two side-by-side scatter plots with hover showing variable names
   ══════════════════════════════════════════════════════════════════════ */

class TfUmapViewer {
  constructor() {
    this.data = null;
    this.canvasRna = document.getElementById("tf-rna-canvas");
    this.canvasSub = document.getElementById("tf-sub-canvas");
    this.ctxRna = this.canvasRna ? this.canvasRna.getContext("2d") : null;
    this.ctxSub = this.canvasSub ? this.canvasSub.getContext("2d") : null;
    this.labelRna = document.getElementById("tf-label-rna");
    this.labelSub = document.getElementById("tf-label-sub");
    this.hoverRna = -1;
    this.hoverSub = -1;
    this.defaultRna = "RNASE3";
    this.defaultSub = "CD49d";

    this._initMouse();
  }

  async loadData() {
    if (this.data) return;
    this.data = await loadJSON("demo_data/tf_umap.json");
    // Set default labels with class 0 color
    const defaultColor = this.data.colors[0];
    if (this.labelRna) {
      this.labelRna.textContent = this.defaultRna;
      this.labelRna.style.borderColor = defaultColor;
      this.labelRna.style.color = defaultColor;
    }
    if (this.labelSub) {
      this.labelSub.textContent = this.defaultSub;
      this.labelSub.style.borderColor = defaultColor;
      this.labelSub.style.color = defaultColor;
    }
    this.draw();
  }

  _initMouse() {
    const handleMove = (canvas, side) => (e) => {
      if (!this.data) return;
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
      const my = (e.clientY - rect.top) * (canvas.height / rect.height);
      const d = side === "rna" ? this.data.main : this.data.sub;
      const bounds = this._getBounds(d);
      const pad = 30;
      const w = canvas.width, h = canvas.height;
      let closest = -1, minDist = 15;
      for (let i = 0; i < d.variables.length; i++) {
        const sx = pad + ((d.umap1[i] - bounds.xMin) / (bounds.xRange || 1)) * (w - 2 * pad);
        const sy = h - pad - ((d.umap2[i] - bounds.yMin) / (bounds.yRange || 1)) * (h - 2 * pad);
        const dist = Math.sqrt((mx - sx) ** 2 + (my - sy) ** 2);
        if (dist < minDist) { minDist = dist; closest = i; }
      }
      if (side === "rna") {
        this.hoverRna = closest;
        if (this.labelRna) {
          this.labelRna.textContent = closest >= 0 ? d.variables[closest] : this.defaultRna;
          if (closest >= 0) {
            const cls = d.class_id[closest];
            this.labelRna.style.borderColor = this.data.colors[cls % 20];
            this.labelRna.style.color = this.data.colors[cls % 20];
          } else {
            this.labelRna.style.borderColor = "";
            this.labelRna.style.color = "";
          }
        }
      } else {
        this.hoverSub = closest;
        if (this.labelSub) {
          this.labelSub.textContent = closest >= 0 ? d.variables[closest] : this.defaultSub;
          if (closest >= 0) {
            const cls = d.class_id[closest];
            this.labelSub.style.borderColor = this.data.colors[cls % 20];
            this.labelSub.style.color = this.data.colors[cls % 20];
          } else {
            this.labelSub.style.borderColor = "";
            this.labelSub.style.color = "";
          }
        }
      }
      this.draw();
    };

    if (this.canvasRna) {
      this.canvasRna.addEventListener("mousemove", handleMove(this.canvasRna, "rna"));
      this.canvasRna.addEventListener("mouseleave", () => { this.hoverRna = -1; this.draw(); });
    }
    if (this.canvasSub) {
      this.canvasSub.addEventListener("mousemove", handleMove(this.canvasSub, "sub"));
      this.canvasSub.addEventListener("mouseleave", () => { this.hoverSub = -1; this.draw(); });
    }
  }

  _getBounds(d) {
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (let i = 0; i < d.umap1.length; i++) {
      if (d.umap1[i] < xMin) xMin = d.umap1[i];
      if (d.umap1[i] > xMax) xMax = d.umap1[i];
      if (d.umap2[i] < yMin) yMin = d.umap2[i];
      if (d.umap2[i] > yMax) yMax = d.umap2[i];
    }
    return { xMin, xMax, yMin, yMax, xRange: xMax - xMin || 1, yRange: yMax - yMin || 1 };
  }

  _drawPanel(ctx, canvas, d, hoverIdx) {
    if (!ctx || !canvas || !d) return;
    const w = canvas.width, h = canvas.height;
    const pad = 30;
    const bounds = this._getBounds(d);
    const colors = this.data.colors;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#f6f8fa";
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = "rgba(16,32,51,.06)";
    ctx.lineWidth = 0.5;
    for (let gx = pad; gx <= w - pad; gx += 60) {
      ctx.beginPath(); ctx.moveTo(gx, pad); ctx.lineTo(gx, h - pad); ctx.stroke();
    }
    for (let gy = pad; gy <= h - pad; gy += 60) {
      ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w - pad, gy); ctx.stroke();
    }

    // Points
    const radius = d.variables.length > 200 ? 3 : 5;
    for (let i = 0; i < d.variables.length; i++) {
      const sx = pad + ((d.umap1[i] - bounds.xMin) / bounds.xRange) * (w - 2 * pad);
      const sy = h - pad - ((d.umap2[i] - bounds.yMin) / bounds.yRange) * (h - 2 * pad);
      const cls = d.class_id[i];
      const color = colors[cls % 20];
      const isHover = i === hoverIdx;

      ctx.beginPath();
      ctx.arc(sx, sy, isHover ? radius + 3 : radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = isHover ? 1 : 0.7;
      ctx.fill();

      if (isHover) {
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 2;
        ctx.stroke();
        // Label
        ctx.fillStyle = color;
        ctx.font = "bold 12px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.globalAlpha = 1;
        ctx.fillText(d.variables[i], sx + radius + 4, sy - 2);
      }
    }
    ctx.globalAlpha = 1;

    // Axes
    ctx.fillStyle = "#999";
    ctx.font = "9px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("UMAP 1", w / 2, h - 4);
    ctx.save();
    ctx.translate(10, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("UMAP 2", 0, 0);
    ctx.restore();
  }

  draw() {
    if (!this.data) return;
    this._drawPanel(this.ctxRna, this.canvasRna, this.data.main, this.hoverRna);
    this._drawPanel(this.ctxSub, this.canvasSub, this.data.sub, this.hoverSub);
  }
}

/* ══════════════════════════════════════════════════════════════════════
   APP INITIALIZATION
   ══════════════════════════════════════════════════════════════════════ */

let simuViewer2d = null;
let simuViewer3d = null;
let kidneyViewer = null;
let tfUmapViewer = null;

function initSimu() {
  const container2d = document.getElementById("simu-canvas-1");
  const container3d = document.getElementById("simu-canvas-2");

  if (container2d) {
    simuViewer2d = new SimuViewer(container2d, "2d");
    simuViewer2d.loadData();
  }

  if (container3d) {
    simuViewer3d = new SimuViewer(container3d, "3d");
    simuViewer3d.loadData();
  }

  const slider = document.getElementById("simu-slider");
  if (slider) slider.addEventListener("input", (e) => {
    const t = parseFloat(e.target.value);
    if (simuViewer2d) simuViewer2d.setTime(t);
    if (simuViewer3d) simuViewer3d.setTime(t);
  });

  const playBtn = document.getElementById("simu-play-btn");
  if (playBtn) playBtn.addEventListener("click", () => {
    if (simuViewer2d) simuViewer2d.togglePlay();
    if (simuViewer3d) simuViewer3d.togglePlay();
  });

  const speedBtn = document.getElementById("simu-speed-btn");
  if (speedBtn) speedBtn.addEventListener("click", () => {
    if (simuViewer2d) simuViewer2d.toggleSpeed();
    if (simuViewer3d) simuViewer3d.toggleSpeed();
  });
}

function initKidney() {
  kidneyViewer = new KidneyViewer();

  // Time slider
  const slider = document.getElementById("kidney-slider");
  if (slider) slider.addEventListener("input", (e) => kidneyViewer.setTime(parseFloat(e.target.value)));

  // Play button
  const playBtn = document.getElementById("kidney-play-btn");
  if (playBtn) playBtn.addEventListener("click", () => kidneyViewer.togglePlay());

  // Speed button
  const speedBtn = document.getElementById("kidney-speed-btn");
  if (speedBtn) speedBtn.addEventListener("click", () => kidneyViewer.toggleSpeed());

  // Fate selector buttons - 设置对应细胞类型颜色
  document.querySelectorAll(".fate-btn").forEach((btn) => {
    const fate = btn.dataset.fate;
    // 设置文字颜色为对应细胞类型颜色（Random保持默认）
    if (fate !== "mixed" && CELL_TYPE_COLORS[fate]) {
      btn.style.color = CELL_TYPE_COLORS[fate];
    }
    btn.addEventListener("click", () => {
      document.querySelectorAll(".fate-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      kidneyViewer.setFateMode(fate);
    });
  });

  // Count slider
  const countSlider = document.getElementById("kidney-count-slider");
  const countValue = document.getElementById("kidney-count-value");
  if (countSlider) {
    countSlider.addEventListener("input", (e) => {
      const val = parseInt(e.target.value);
      if (countValue) countValue.textContent = val;
      kidneyViewer.setTrajCount(val);
    });
  }
}

function switchTab(tab) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });
  const simuDemo = document.getElementById("demo-simu");
  const kidneyDemo = document.getElementById("demo-kidney");
  if (tab === "simu") {
    simuDemo.style.display = "";
    kidneyDemo.style.display = "none";
    if (simuViewer2d) simuViewer2d.onResize();
    if (simuViewer3d) simuViewer3d.onResize();
  } else {
    simuDemo.style.display = "none";
    kidneyDemo.style.display = "";
    if (kidneyViewer && !kidneyViewer.loaded) kidneyViewer.loadData();
  }
  if (window.mofiCueManager) {
    window.mofiCueManager.running = false;
    window.setTimeout(() => window.mofiCueManager.runIfDemosInView(), 450);
    window.setTimeout(() => window.mofiCueManager.runIfDemosInView(), 2800);
  }
}

function animationLoop() {
  requestAnimationFrame(animationLoop);
  if (kidneyViewer) kidneyViewer.tick();
}

class InteractionCueManager {
  constructor() {
    this.storagePrefix = "mofi-interactive-cue-v4-used:";
    this.periodMs = 22000;
    this.cueMs = 4200;
    this.stepMs = 700;
    this.inView = false;
    this.running = false;
    this.lastCueAt = 0;
    this.usedKeys = new Set();
    this.cues = [
      { key: "simu-play", selector: "#simu-play-btn", events: ["click"] },
      { key: "simu-slider", selector: "#simu-slider", events: ["input", "change", "pointerdown"] },
      { key: "kidney-tab", selector: '.tab-btn[data-tab="kidney"]', events: ["click"] },
      { key: "kidney-play", selector: "#kidney-play-btn", events: ["click"] },
      { key: "kidney-slider", selector: "#kidney-slider", events: ["input", "change", "pointerdown"] },
    ];
  }

  init() {
    this.cues.forEach((cue) => {
      const el = document.querySelector(cue.selector);
      if (!el) return;
      cue.events.forEach((eventName) => {
        el.addEventListener(eventName, () => this.markUsed(cue.key));
      });
    });

    const demos = document.getElementById("demos");
    if (!demos) return;

    const checkInView = () => {
      const rect = demos.getBoundingClientRect();
      this.inView = rect.top < window.innerHeight * 0.82 && rect.bottom > window.innerHeight * 0.18;
      if (this.inView) this.runCurrentSequence();
    };

    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver((entries) => {
        const entry = entries[0];
        this.inView = entry.isIntersecting;
        if (this.inView) this.runCurrentSequence();
      }, { rootMargin: "-18% 0px -22% 0px", threshold: [0, 0.01] });
      observer.observe(demos);
    } else {
    }

    window.addEventListener("scroll", checkInView, { passive: true });
    window.addEventListener("resize", checkInView);
    window.setTimeout(checkInView, 700);

    window.setInterval(() => {
      if (!this.inView || Date.now() - this.lastCueAt < this.periodMs) return;
      this.runCurrentSequence();
    }, 5000);
  }

  runIfDemosInView() {
    const demos = document.getElementById("demos");
    if (!demos) return;
    const rect = demos.getBoundingClientRect();
    this.inView = rect.top < window.innerHeight * 0.82 && rect.bottom > window.innerHeight * 0.18;
    if (this.inView) this.runCurrentSequence();
  }

  isUsed(key) {
    if (this.usedKeys.has(key)) return true;
    try {
      return window.localStorage.getItem(this.storagePrefix + key) === "1";
    } catch (_) {
      return false;
    }
  }

  markUsed(key) {
    this.usedKeys.add(key);
    try {
      window.localStorage.setItem(this.storagePrefix + key, "1");
    } catch (_) {
      // localStorage can be unavailable in privacy-restricted contexts.
    }
    const cue = this.cues.find((item) => item.key === key);
    const el = cue ? document.querySelector(cue.selector) : null;
    if (el) el.classList.remove("use-cue");
    this.updateHint();
  }

  activePanelKeys() {
    const kidneyDemo = document.getElementById("demo-kidney");
    const kidneyVisible = kidneyDemo && kidneyDemo.style.display !== "none";
    return kidneyVisible
      ? ["kidney-play", "kidney-slider"]
      : ["simu-play", "simu-slider", "kidney-tab"];
  }

  isAvailable(cue) {
    const el = document.querySelector(cue.selector);
    if (!el || this.isUsed(cue.key)) return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  runCurrentSequence() {
    if (this.running) return;
    const cueMap = new Map(this.cues.map((cue) => [cue.key, cue]));
    const targets = this.activePanelKeys()
      .map((key) => cueMap.get(key))
      .filter((cue) => cue && this.isAvailable(cue));

    if (targets.length === 0) return;

    this.running = true;
    this.lastCueAt = Date.now();
    this.updateHint();

    targets.forEach((cue, idx) => {
      window.setTimeout(() => this.flash(cue), idx * this.stepMs);
    });

    window.setTimeout(() => {
      this.running = false;
    }, targets.length * this.stepMs + this.cueMs);
  }

  flash(cue) {
    if (this.isUsed(cue.key)) return;
    const el = document.querySelector(cue.selector);
    if (!el) return;
    el.classList.remove("use-cue");
    void el.offsetWidth;
    el.classList.add("use-cue");
    this.placeHint(this.hintContainerFor(cue.key), this.hintTextFor(cue.key));
    window.setTimeout(() => el.classList.remove("use-cue"), this.cueMs);
  }

  hintContainerFor(key) {
    if (key === "kidney-tab") return document.querySelector(".tab-switcher");
    if (key.startsWith("kidney-")) return document.querySelector("#demo-kidney .demo-controls");
    return document.querySelector("#demo-simu .demo-controls");
  }

  hintTextFor(key) {
    if (key === "kidney-tab") return "Switch to Kidney Organoid";
    return "Try the play button or drag the timeline";
  }

  updateHint() {
    this.removeHints();
    const activeKeys = this.activePanelKeys().filter((key) => !this.isUsed(key));
    if (activeKeys.length === 0) return;

    const kidneyDemo = document.getElementById("demo-kidney");
    const kidneyVisible = kidneyDemo && kidneyDemo.style.display !== "none";
    if (kidneyVisible) {
      this.placeHint(
        document.querySelector("#demo-kidney .demo-controls"),
        "Try the play button or drag the timeline"
      );
      return;
    }

    const needsSimControls = activeKeys.includes("simu-play") || activeKeys.includes("simu-slider");
    if (needsSimControls) {
      this.placeHint(
        document.querySelector("#demo-simu .demo-controls"),
        "Try the play button or drag the timeline"
      );
    }

    if (activeKeys.includes("kidney-tab")) {
      this.placeHint(
        document.querySelector(".tab-switcher"),
        "Switch to Kidney Organoid"
      );
    }
  }

  placeHint(container, text) {
    if (!container || container.querySelector(".cue-hint")) return;
    const hint = document.createElement("span");
    hint.className = "cue-hint";
    hint.textContent = text;
    container.appendChild(hint);

    if (container.classList.contains("tab-switcher")) {
      const kidneyTab = container.querySelector('.tab-btn[data-tab="kidney"]');
      if (kidneyTab) {
        const containerRect = container.getBoundingClientRect();
        const tabRect = kidneyTab.getBoundingClientRect();
        hint.style.left = `${tabRect.right - containerRect.left + 12}px`;
        hint.style.top = `${tabRect.top - containerRect.top + tabRect.height / 2}px`;
      }
    }

    window.setTimeout(() => hint.remove(), this.cueMs);
  }

  removeHints() {
    document.querySelectorAll(".cue-hint").forEach((hint) => hint.remove());
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  initSimu();
  initKidney();
  tfUmapViewer = new TfUmapViewer();
  tfUmapViewer.loadData();
  window.mofiCueManager = new InteractionCueManager();
  window.mofiCueManager.init();
  window.setTimeout(() => window.mofiCueManager.runIfDemosInView(), 1200);
  animationLoop();
  switchTab("simu");
});
