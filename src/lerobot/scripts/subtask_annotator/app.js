"use strict";

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
const S = {
  meta: null,
  cfg: null,              // {feature_name, default_value, subtasks:[{name,color}], palette}
  episodes: [],
  epIndex: null,
  ep: null,               // {length, action, state, annotation:{keyframes,labels}, ...}
  cursor: 0,
  mode: "region",         // 'region' | 'single'
  selectedSubtask: null,  // index into cfg.subtasks, or 'eraser', or null
  selectedRegion: null,   // region index in region-click mode
  series: [],             // [{joint, kind, idx}]  kind: 'action'|'obs'
  playing: false,
  playTimer: null,
  saveTimer: null,
  dragKf: null,           // index in keyframes currently being dragged
  scrubbing: false,
};

// Layout constants (CSS px)
const L = {
  marginLeft: 122, marginRight: 16, marginTop: 6,
  kfStripH: 16, subtaskLaneH: 30, sectionGap: 10,
  jointLaneH: 64, laneGap: 8, marginBottom: 10,
};

const ACTION_COLOR = "#5b9dd9";
const OBS_COLOR = "#e15759";
const GREY = "#41454f";

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  return r.json();
}

function toast(msg, kind) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast show" + (kind ? " " + kind : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.className = "toast"), 2800);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  S.meta = await apiGet("/api/meta");
  S.cfg = S.meta.config;
  S.episodes = S.meta.episodes;
  if (!S.cfg.subtasks) S.cfg.subtasks = [];

  $("featureName").value = S.cfg.feature_name || "subtask";
  $("taskLabel").textContent = S.meta.task ? "任务: " + S.meta.task : "";

  buildEpisodeSelect();
  buildJointList();
  restoreUiState();
  renderSubtasks();
  updateModeHelp();

  wireControls();

  const first = S.episodes.length ? S.episodes[0].index : null;
  if (first === null) {
    toast("该 run 没有任何 episode", "err");
    return;
  }
  await loadEpisode(first);
}

function buildEpisodeSelect() {
  const sel = $("episodeSelect");
  sel.innerHTML = "";
  for (const ep of S.episodes) {
    const o = document.createElement("option");
    o.value = ep.index;
    o.textContent = `ep_${String(ep.index).padStart(6, "0")} (${ep.length ?? "?"}帧)`;
    sel.appendChild(o);
  }
}

// ---------------------------------------------------------------------------
// Episode loading
// ---------------------------------------------------------------------------
async function loadEpisode(idx) {
  stopPlay();
  S.epIndex = idx;
  $("episodeSelect").value = idx;
  S.ep = await apiGet(`/api/episode/${idx}`);
  if (!S.ep.annotation.labels || S.ep.annotation.labels.length !== S.ep.length) {
    const labels = (S.ep.annotation.labels || []).slice(0, S.ep.length);
    while (labels.length < S.ep.length) labels.push(null);
    S.ep.annotation.labels = labels;
  }
  if (!S.ep.annotation.keyframes || !S.ep.annotation.keyframes.length) {
    S.ep.annotation.keyframes = [0, Math.max(S.ep.length - 1, 0)];
  }
  S.cursor = 0;
  S.selectedRegion = null;
  $("frameSlider").max = Math.max(S.ep.length - 1, 0);
  // refresh series indices against this episode (names are run-wide though)
  buildCameras();
  layoutCanvas();
  draw();
  updateFrameUi();
}

function normKeyframes() {
  const last = Math.max(S.ep.length - 1, 0);
  let kf = (S.ep.annotation.keyframes || []).map((v) => clamp(v | 0, 0, last));
  kf.push(0); kf.push(last);
  kf = Array.from(new Set(kf)).sort((a, b) => a - b);
  S.ep.annotation.keyframes = kf;
  return kf;
}

// ---------------------------------------------------------------------------
// Cameras
// ---------------------------------------------------------------------------
function buildCameras() {
  const wrap = $("cameras");
  wrap.innerHTML = "";
  const cams = S.meta.cameras || [];
  const w = cams.length ? Math.max(220, Math.min(340, Math.floor(960 / cams.length))) : 300;
  for (const cam of cams) {
    const box = document.createElement("div");
    box.className = "cam-box";
    const title = document.createElement("div");
    title.className = "cam-title";
    title.textContent = cam.label;
    const img = document.createElement("img");
    img.width = w;
    img.dataset.subdir = cam.subdir;
    img.alt = cam.label;
    box.appendChild(title);
    box.appendChild(img);
    wrap.appendChild(box);
  }
  updateCameras();
}

function updateCameras() {
  const imgs = $("cameras").querySelectorAll("img");
  imgs.forEach((img) => {
    img.src = `/api/episode/${S.epIndex}/img/${img.dataset.subdir}/${S.cursor}`;
  });
}

// ---------------------------------------------------------------------------
// Subtasks palette
// ---------------------------------------------------------------------------
function subtaskByName(name) {
  return (S.cfg.subtasks || []).find((s) => s.name === name);
}
function colorForLabel(name) {
  if (name == null || name === "") return GREY;
  const s = subtaskByName(name);
  return s ? s.color : "#888";
}

function renderSubtasks() {
  const list = $("subtaskList");
  list.innerHTML = "";
  S.cfg.subtasks.forEach((sub, i) => {
    const chip = document.createElement("div");
    chip.className = "subtask-chip" + (S.selectedSubtask === i ? " selected" : "");
    chip.draggable = true;

    const sw = document.createElement("input");
    sw.type = "color";
    sw.className = "swatch";
    sw.value = sub.color;
    sw.title = "点击改颜色";
    sw.addEventListener("input", () => { sub.color = sw.value; saveConfig(); draw(); });
    sw.addEventListener("click", (e) => e.stopPropagation());

    const idx = document.createElement("span");
    idx.className = "idx";
    idx.textContent = (i + 1) + ".";

    const name = document.createElement("span");
    name.className = "name";
    name.textContent = sub.name;
    name.title = "双击重命名";
    name.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      const nv = prompt("重命名 subtask:", sub.name);
      if (nv && nv.trim()) { renameSubtask(sub.name, nv.trim()); }
    });

    const del = document.createElement("span");
    del.className = "del";
    del.textContent = "✕";
    del.title = "删除";
    del.addEventListener("click", (e) => { e.stopPropagation(); deleteSubtask(i); });

    chip.appendChild(sw);
    chip.appendChild(idx);
    chip.appendChild(name);
    chip.appendChild(del);

    chip.addEventListener("click", () => selectSubtask(i, true));
    chip.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", String(i));
      e.dataTransfer.effectAllowed = "copy";
    });
    list.appendChild(chip);
  });

  // Eraser chip
  const er = document.createElement("div");
  er.className = "subtask-chip eraser" + (S.selectedSubtask === "eraser" ? " selected" : "");
  er.draggable = true;
  er.innerHTML = '<span class="swatch" style="background:repeating-linear-gradient(45deg,#555,#555 4px,#333 4px,#333 8px)"></span><span class="name">清除 / 无标注</span>';
  er.addEventListener("click", () => selectSubtask("eraser", true));
  er.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData("text/plain", "eraser");
    e.dataTransfer.effectAllowed = "copy";
  });
  list.appendChild(er);
}

function selectSubtask(i, apply) {
  S.selectedSubtask = i;
  renderSubtasks();
  if (apply) applySelectedToTarget();
}

function applySelectedToTarget() {
  if (S.selectedSubtask == null) return;
  const sub = S.selectedSubtask === "eraser" ? null : S.cfg.subtasks[S.selectedSubtask];
  const name = sub ? sub.name : null;
  if (S.mode === "region") {
    let region = S.selectedRegion;
    if (region == null) region = regionContaining(S.cursor);
    fillRegion(region, name);
  } else {
    setFrameLabel(S.cursor, name);
  }
}

function addSubtask(name) {
  if (!name) return;
  if (subtaskByName(name)) { toast("已存在同名 subtask", "err"); return; }
  const palette = S.cfg.palette || [];
  const color = palette[S.cfg.subtasks.length % palette.length] || "#4e79a7";
  S.cfg.subtasks.push({ name, color });
  saveConfig();
  renderSubtasks();
}

function deleteSubtask(i) {
  const sub = S.cfg.subtasks[i];
  if (!confirm(`删除 subtask "${sub.name}"？已标注的帧会保留文字但失去颜色。`)) return;
  S.cfg.subtasks.splice(i, 1);
  if (S.selectedSubtask === i) S.selectedSubtask = null;
  saveConfig();
  renderSubtasks();
  draw();
}

function renameSubtask(oldName, newName) {
  if (subtaskByName(newName)) { toast("已存在同名 subtask", "err"); return; }
  const sub = subtaskByName(oldName);
  if (sub) sub.name = newName;
  saveConfig();
  // Note: existing labels in OTHER episodes keep old name; current episode updated below.
  if (S.ep) {
    S.ep.annotation.labels = S.ep.annotation.labels.map((v) => (v === oldName ? newName : v));
    scheduleSave();
  }
  renderSubtasks();
  draw();
}

// ---------------------------------------------------------------------------
// Joint selection
// ---------------------------------------------------------------------------
function buildJointList() {
  const list = $("jointList");
  list.innerHTML = "";
  const actNames = S.meta.action_names || [];
  const stNames = S.meta.state_names || [];
  const joints = actNames.length ? actNames : stNames;
  joints.forEach((jname) => {
    const actIdx = actNames.indexOf(jname);
    const obsIdx = stNames.indexOf(jname);
    const row = document.createElement("div");
    row.className = "joint-row";
    const nm = document.createElement("span");
    nm.className = "jname";
    nm.textContent = jname;

    const aBtn = document.createElement("span");
    aBtn.className = "toggle act" + (actIdx < 0 ? " disabled" : "");
    aBtn.textContent = "A";
    aBtn.title = "action";
    if (actIdx >= 0) aBtn.addEventListener("click", () => toggleSeries(jname, "action", actIdx, aBtn));

    const oBtn = document.createElement("span");
    oBtn.className = "toggle obs" + (obsIdx < 0 ? " disabled" : "");
    oBtn.textContent = "O";
    oBtn.title = "observation";
    if (obsIdx >= 0) oBtn.addEventListener("click", () => toggleSeries(jname, "obs", obsIdx, oBtn));

    row.appendChild(nm);
    row.appendChild(aBtn);
    row.appendChild(oBtn);
    row.dataset.joint = jname;
    list.appendChild(row);
  });
}

function seriesIdx(joint, kind) {
  return S.series.findIndex((s) => s.joint === joint && s.kind === kind);
}

function toggleSeries(joint, kind, idx, btn) {
  const k = seriesIdx(joint, kind);
  if (k >= 0) {
    S.series.splice(k, 1);
    if (btn) btn.classList.remove("on");
  } else {
    S.series.push({ joint, kind, idx });
    if (btn) btn.classList.add("on");
  }
  saveUiState();
  layoutCanvas();
  draw();
}

function syncJointButtons() {
  $("jointList").querySelectorAll(".joint-row").forEach((row) => {
    const joint = row.dataset.joint;
    const a = row.querySelector(".toggle.act");
    const o = row.querySelector(".toggle.obs");
    a.classList.toggle("on", seriesIdx(joint, "action") >= 0);
    o.classList.toggle("on", seriesIdx(joint, "obs") >= 0);
  });
}

function pickGrippers() {
  S.series = [];
  const actNames = S.meta.action_names || [];
  const stNames = S.meta.state_names || [];
  (actNames.length ? actNames : stNames).forEach((jname) => {
    if (!/gripper/i.test(jname)) return;
    const a = actNames.indexOf(jname);
    const o = stNames.indexOf(jname);
    if (a >= 0) S.series.push({ joint: jname, kind: "action", idx: a });
    if (o >= 0) S.series.push({ joint: jname, kind: "obs", idx: o });
  });
  saveUiState();
  syncJointButtons();
  layoutCanvas();
  draw();
}

function clearSeries() {
  S.series = [];
  saveUiState();
  syncJointButtons();
  layoutCanvas();
  draw();
}

// ---------------------------------------------------------------------------
// Annotation editing
// ---------------------------------------------------------------------------
// Regions are half-open [kf[k], kf[k+1]): the right keyframe frame belongs to
// the next region. This makes the colored block for a region span exactly
// frameX(kf[k])..frameX(kf[k+1]) — both edges sit on the keyframes. The very
// last region additionally owns the final frame so nothing is left uncovered.
function regionContaining(frame) {
  const kf = normKeyframes();
  for (let k = 0; k < kf.length - 1; k++) {
    if (frame >= kf[k] && frame < kf[k + 1]) return k;
  }
  return Math.max(kf.length - 2, 0); // frame == last keyframe -> last region
}

function fillRegion(regionIdx, name) {
  const kf = normKeyframes();
  if (regionIdx < 0 || regionIdx >= kf.length - 1) return;
  const a = kf[regionIdx], b = kf[regionIdx + 1];
  const isLast = regionIdx === kf.length - 2;
  const hi = isLast ? b : b - 1; // half-open, except the last region includes b
  for (let f = a; f <= hi; f++) S.ep.annotation.labels[f] = name;
  scheduleSave();
  draw();
}

function setFrameLabel(frame, name) {
  S.ep.annotation.labels[frame] = name;
  scheduleSave();
  draw();
}

function toggleKeyframe(frame) {
  const last = Math.max(S.ep.length - 1, 0);
  if (frame === 0 || frame === last) return; // endpoints are fixed
  const kf = normKeyframes();
  const at = kf.indexOf(frame);
  if (at >= 0) {
    kf.splice(at, 1);
  } else {
    kf.push(frame);
    kf.sort((a, b) => a - b);
  }
  S.ep.annotation.keyframes = kf;
  S.selectedRegion = null;
  scheduleSave();
  draw();
}

function scheduleSave() {
  clearTimeout(S.saveTimer);
  $("saveStatus").textContent = "● 未保存";
  S.saveTimer = setTimeout(saveAnnotation, 450);
}

async function saveAnnotation() {
  if (!S.ep) return;
  try {
    await apiPost(`/api/episode/${S.epIndex}/annotation`, {
      keyframes: S.ep.annotation.keyframes,
      labels: S.ep.annotation.labels,
    });
    $("saveStatus").textContent = "✓ 已保存";
  } catch (e) {
    $("saveStatus").textContent = "保存失败";
    toast("保存失败: " + e.message, "err");
  }
}

let cfgSaveTimer = null;
function saveConfig() {
  S.cfg.feature_name = $("featureName").value.trim() || "subtask";
  clearTimeout(cfgSaveTimer);
  cfgSaveTimer = setTimeout(() => apiPost("/api/config", S.cfg).catch((e) => toast(e.message, "err")), 300);
}

// ---------------------------------------------------------------------------
// Canvas layout + drawing
// ---------------------------------------------------------------------------
function canvasMetrics() {
  const c = $("board");
  const cssW = c.clientWidth || 900;
  const totalH =
    L.marginTop + L.kfStripH + 4 + L.subtaskLaneH + L.sectionGap +
    S.series.length * (L.jointLaneH + L.laneGap) + L.marginBottom;
  const plotW = cssW - L.marginLeft - L.marginRight;
  const n = S.ep ? S.ep.length : 1;
  // Line-chart spacing: frame 0 sits on the left edge, frame n-1 on the right
  // edge. Every element (keyframes, cursor, plots, subtask fill) uses frameX(),
  // so they all align to the exact same per-frame x positions.
  const step = plotW / Math.max(n - 1, 1);
  const rightX = L.marginLeft + plotW;
  const subY = L.marginTop + L.kfStripH + 4;
  const jointTop = subY + L.subtaskLaneH + L.sectionGap;
  return { cssW, totalH, plotW, step, rightX, n, subY, jointTop };
}

function layoutCanvas() {
  const c = $("board");
  const m = canvasMetrics();
  const dpr = window.devicePixelRatio || 1;
  c.style.height = m.totalH + "px";
  c.width = Math.round(m.cssW * dpr);
  c.height = Math.round(m.totalH * dpr);
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// Single source of truth for the x of frame f (the per-frame position).
function frameX(f, m) { return L.marginLeft + f * m.step; }
function xToFrame(x, m) { return clamp(Math.round((x - L.marginLeft) / m.step), 0, m.n - 1); }

function draw() {
  if (!S.ep) return;
  const c = $("board");
  const ctx = c.getContext("2d");
  const m = canvasMetrics();
  ctx.clearRect(0, 0, m.cssW, m.totalH);
  ctx.font = "11px -apple-system, sans-serif";
  ctx.textBaseline = "middle";

  drawSubtaskLane(ctx, m);
  drawKeyframes(ctx, m);
  S.series.forEach((s, i) => drawJointLane(ctx, m, s, i));
  drawCursor(ctx, m);
}

function drawSubtaskLane(ctx, m) {
  const y = m.subY, h = L.subtaskLaneH;
  const labels = S.ep.annotation.labels;
  // background
  ctx.fillStyle = "#1a1c21";
  ctx.fillRect(L.marginLeft, y, m.plotW, h);

  // contiguous runs
  let start = 0;
  for (let f = 1; f <= labels.length; f++) {
    if (f === labels.length || labels[f] !== labels[start]) {
      const v = labels[start];
      if (v != null && v !== "") {
        ctx.fillStyle = colorForLabel(v);
        const x0 = frameX(start, m);
        // The run covers frames [start, f-1]; its right edge is frame f's x,
        // clamped to the right edge for the run that includes the final frame.
        const x1 = Math.min(frameX(f, m), m.rightX);
        ctx.fillRect(x0, y, x1 - x0, h);
        // label text if wide enough
        if (x1 - x0 > 30) {
          ctx.fillStyle = "rgba(0,0,0,0.75)";
          ctx.fillText(v, x0 + 4, y + h / 2);
        }
      }
      start = f;
    }
  }

  // selected region outline
  if (S.mode === "region" && S.selectedRegion != null) {
    const kf = normKeyframes();
    if (S.selectedRegion < kf.length - 1) {
      const a = kf[S.selectedRegion], b = kf[S.selectedRegion + 1];
      // Outline spans exactly from keyframe a to keyframe b — same frameX() the
      // markers, cursor and fill boundary use, so all four coincide.
      const x0 = frameX(a, m), x1 = frameX(b, m);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.strokeRect(x0, y + 1, x1 - x0, h - 2);
    }
  }

  // border + left label
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1;
  ctx.strokeRect(L.marginLeft, y, m.plotW, h);
  ctx.fillStyle = "#9aa0aa";
  ctx.textAlign = "right";
  ctx.fillText(S.cfg.feature_name || "subtask", L.marginLeft - 8, y + h / 2);
  ctx.textAlign = "left";
}

function drawKeyframes(ctx, m) {
  const kf = normKeyframes();
  const yTop = L.marginTop, stripH = L.kfStripH;
  const yBase = m.subY; // ticks reach down to subtask lane
  const last = Math.max(S.ep.length - 1, 0);
  ctx.fillStyle = "#9aa0aa";
  ctx.textAlign = "right";
  ctx.fillText("关键帧", L.marginLeft - 8, yTop + stripH / 2);
  ctx.textAlign = "left";

  for (const f of kf) {
    const x = frameX(f, m);
    const fixed = (f === 0 || f === last);
    // tick line
    ctx.strokeStyle = fixed ? "#888" : "#d8d8d8";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, yTop);
    ctx.lineTo(x, yBase + L.subtaskLaneH);
    ctx.stroke();
    // handle triangle
    ctx.fillStyle = fixed ? "#888" : "#e6e6e6";
    ctx.beginPath();
    ctx.moveTo(x - 5, yTop);
    ctx.lineTo(x + 5, yTop);
    ctx.lineTo(x, yTop + 9);
    ctx.closePath();
    ctx.fill();
  }
}

function drawJointLane(ctx, m, s, i) {
  const top = m.jointTop + i * (L.jointLaneH + L.laneGap);
  const h = L.jointLaneH;
  const pad = 8;
  const innerTop = top + pad, innerH = h - 2 * pad;
  const data = s.kind === "action" ? S.ep.action : S.ep.state;
  const color = s.kind === "action" ? ACTION_COLOR : OBS_COLOR;

  // background + border
  ctx.fillStyle = "#1a1c21";
  ctx.fillRect(L.marginLeft, top, m.plotW, h);
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1;
  ctx.strokeRect(L.marginLeft, top, m.plotW, h);

  // gather values
  const vals = new Array(m.n);
  let mn = Infinity, mx = -Infinity;
  for (let f = 0; f < m.n; f++) {
    const row = data[f];
    const v = row && row[s.idx] != null ? row[s.idx] : null;
    vals[f] = v;
    if (v != null) { if (v < mn) mn = v; if (v > mx) mx = v; }
  }
  if (!isFinite(mn)) { mn = 0; mx = 1; }
  if (mn === mx) { mn -= 0.5; mx += 0.5; }
  const valToY = (v) => innerTop + (1 - (v - mn) / (mx - mn)) * innerH;

  // zero / mid gridline
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.beginPath();
  ctx.moveTo(L.marginLeft, innerTop + innerH / 2);
  ctx.lineTo(L.marginLeft + m.plotW, innerTop + innerH / 2);
  ctx.stroke();

  // polyline
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  let started = false;
  for (let f = 0; f < m.n; f++) {
    if (vals[f] == null) { started = false; continue; }
    const x = frameX(f, m), y = valToY(vals[f]);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // left labels: axis min/max pinned to top/bottom edges, joint name + kind
  // vertically centered between them so they never overlap the value text.
  const lx = L.marginLeft - 8;
  ctx.textAlign = "right";
  ctx.fillStyle = "#6c7280";
  ctx.fillText(mx.toFixed(3), lx, top + 9);
  ctx.fillText(mn.toFixed(3), lx, top + h - 9);
  ctx.fillStyle = color;
  ctx.fillText(s.joint, lx, top + h / 2 - 6);
  ctx.fillStyle = "#9aa0aa";
  ctx.fillText(s.kind === "action" ? "action" : "obs", lx, top + h / 2 + 8);
  ctx.textAlign = "left";

  // current value marker
  const cv = vals[S.cursor];
  if (cv != null) {
    const x = frameX(S.cursor, m), y = valToY(cv);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#e6e6e6";
    const txt = cv.toFixed(3);
    const tx = clamp(x + 6, L.marginLeft, L.marginLeft + m.plotW - 40);
    ctx.fillText(txt, tx, top + 11);
  }
}

function drawCursor(ctx, m) {
  const x = frameX(S.cursor, m);
  ctx.strokeStyle = "#ffd24a";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, L.marginTop);
  ctx.lineTo(x, m.totalH - L.marginBottom);
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Board interaction
// ---------------------------------------------------------------------------
function zoneAtY(y, m) {
  if (y < m.subY - 2) return "kf";
  if (y < m.subY + L.subtaskLaneH + 2) return "subtask";
  return "joint";
}

function nearestKeyframe(frame, m) {
  // returns {idx, frame, distPx} of nearest non-fixed keyframe within threshold
  const kf = normKeyframes();
  const last = Math.max(S.ep.length - 1, 0);
  const px = frameX(frame, m);
  let best = null;
  kf.forEach((f, idx) => {
    if (f === 0 || f === last) return;
    const d = Math.abs(frameX(f, m) - px);
    if (best == null || d < best.d) best = { idx, f, d };
  });
  return best && best.d <= 8 ? best : null;
}

function wireBoard() {
  const c = $("board");

  c.addEventListener("mousedown", (e) => {
    const m = canvasMetrics();
    const rect = c.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const frame = xToFrame(x, m);
    const zone = zoneAtY(y, m);

    if (zone === "kf") {
      const nk = nearestKeyframe(frame, m);
      if (nk) { S.dragKf = nk.idx; return; }
    }
    // scrub
    S.scrubbing = true;
    setCursor(frame);
    if (zone === "subtask" && S.mode === "region") {
      S.selectedRegion = regionContaining(frame);
      draw();
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (S.dragKf == null && !S.scrubbing) return;
    const m = canvasMetrics();
    const rect = c.getBoundingClientRect();
    const frame = xToFrame(e.clientX - rect.left, m);
    if (S.dragKf != null) {
      const kf = S.ep.annotation.keyframes;
      const last = Math.max(S.ep.length - 1, 0);
      const lo = (kf[S.dragKf - 1] ?? -1) + 1;
      const hi = (kf[S.dragKf + 1] ?? last + 1) - 1;
      kf[S.dragKf] = clamp(frame, Math.max(lo, 1), Math.min(hi, last - 1));
      draw();
    } else if (S.scrubbing) {
      setCursor(frame);
    }
  });

  window.addEventListener("mouseup", () => {
    if (S.dragKf != null) { normKeyframes(); S.selectedRegion = null; scheduleSave(); draw(); }
    S.dragKf = null;
    S.scrubbing = false;
  });

  c.addEventListener("dblclick", (e) => {
    const m = canvasMetrics();
    const rect = c.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    if (zoneAtY(y, m) === "kf") {
      const frame = xToFrame(x, m);
      const nk = nearestKeyframe(frame, m);
      toggleKeyframe(nk ? nk.f : frame);
    }
  });

  // drag-drop subtasks
  c.addEventListener("dragover", (e) => { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; });
  c.addEventListener("drop", (e) => {
    e.preventDefault();
    const raw = e.dataTransfer.getData("text/plain");
    const m = canvasMetrics();
    const rect = c.getBoundingClientRect();
    const frame = xToFrame(e.clientX - rect.left, m);
    const name = raw === "eraser" ? null : (S.cfg.subtasks[+raw] ? S.cfg.subtasks[+raw].name : null);
    if (S.mode === "region") {
      fillRegion(regionContaining(frame), name);
    } else {
      setFrameLabel(frame, name);
    }
  });
}

// ---------------------------------------------------------------------------
// Cursor / playback
// ---------------------------------------------------------------------------
function setCursor(f) {
  f = clamp(f, 0, Math.max(S.ep.length - 1, 0));
  if (f === S.cursor) return;
  S.cursor = f;
  updateCameras();
  updateFrameUi();
  draw();
}

function updateFrameUi() {
  $("frameSlider").value = S.cursor;
  const cur = S.ep.annotation.labels[S.cursor];
  const tag = cur ? `  [${cur}]` : "";
  $("frameLabel").textContent = `${S.cursor} / ${Math.max(S.ep.length - 1, 0)}${tag}`;
}

function togglePlay() { S.playing ? stopPlay() : startPlay(); }
function startPlay() {
  if (S.playing) return;
  S.playing = true;
  $("playBtn").textContent = "⏸";
  const interval = 1000 / (S.meta.fps || 30);
  S.playTimer = setInterval(() => {
    if (S.cursor >= S.ep.length - 1) { stopPlay(); return; }
    setCursor(S.cursor + 1);
  }, interval);
}
function stopPlay() {
  S.playing = false;
  $("playBtn").textContent = "▶";
  if (S.playTimer) { clearInterval(S.playTimer); S.playTimer = null; }
}

// ---------------------------------------------------------------------------
// UI state persistence (localStorage)
// ---------------------------------------------------------------------------
function uiKey() { return "subtaskUi:" + (S.meta ? S.meta.root : ""); }
function saveUiState() {
  try {
    localStorage.setItem(uiKey(), JSON.stringify({
      series: S.series.map((s) => ({ joint: s.joint, kind: s.kind })),
      mode: S.mode,
    }));
  } catch (e) {}
}
function restoreUiState() {
  let st = null;
  try { st = JSON.parse(localStorage.getItem(uiKey()) || "null"); } catch (e) {}
  if (!st) { pickGrippers(); return; }
  const actNames = S.meta.action_names || [], stNames = S.meta.state_names || [];
  S.series = (st.series || []).map((s) => ({
    joint: s.joint, kind: s.kind,
    idx: s.kind === "action" ? actNames.indexOf(s.joint) : stNames.indexOf(s.joint),
  })).filter((s) => s.idx >= 0);
  if (st.mode) {
    S.mode = st.mode;
    document.querySelector(`input[name=mode][value=${st.mode}]`).checked = true;
  }
  syncJointButtons();
}

// ---------------------------------------------------------------------------
// Controls wiring
// ---------------------------------------------------------------------------
function updateModeHelp() {
  $("modeHelp").textContent = S.mode === "region"
    ? "区间模式：拖动 subtask 到时间轴某段，或先单击某段选中再点 subtask，会填充两关键帧之间(含端点)。"
    : "单帧模式：把 subtask 拖到时间轴某处，或选中 subtask 后按数字键/点击，只标注当前播放头所在帧。";
}

function wireControls() {
  $("episodeSelect").addEventListener("change", (e) => loadEpisode(+e.target.value));
  $("featureName").addEventListener("input", saveConfig);

  $("addSubtaskBtn").addEventListener("click", () => {
    const inp = $("newSubtaskName");
    addSubtask(inp.value.trim());
    inp.value = "";
  });
  $("newSubtaskName").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { addSubtask(e.target.value.trim()); e.target.value = ""; }
  });

  document.querySelectorAll("input[name=mode]").forEach((r) =>
    r.addEventListener("change", (e) => {
      S.mode = e.target.value;
      S.selectedRegion = null;
      saveUiState();
      updateModeHelp();
      draw();
    })
  );

  $("pickGrippers").addEventListener("click", pickGrippers);
  $("clearSeries").addEventListener("click", clearSeries);

  $("firstBtn").addEventListener("click", () => setCursor(0));
  $("prevBtn").addEventListener("click", () => setCursor(S.cursor - 1));
  $("nextBtn").addEventListener("click", () => setCursor(S.cursor + 1));
  $("lastBtn").addEventListener("click", () => setCursor(S.ep.length - 1));
  $("playBtn").addEventListener("click", togglePlay);
  $("frameSlider").addEventListener("input", (e) => setCursor(+e.target.value));
  $("addKfBtn").addEventListener("click", () => toggleKeyframe(S.cursor));

  $("exportBtn").addEventListener("click", doExport);

  wireBoard();

  window.addEventListener("resize", () => { layoutCanvas(); draw(); });

  document.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    // Ctrl/Cmd + Enter (or Ctrl/Cmd + K): add/remove a keyframe at the cursor.
    // Lets you scrub with ←/→ then mark the keyframe without leaving the keyboard.
    if ((e.ctrlKey || e.metaKey) && (e.key === "Enter" || e.key.toLowerCase() === "k")) {
      toggleKeyframe(S.cursor); e.preventDefault(); return;
    }
    if (e.ctrlKey || e.metaKey) return; // leave other Ctrl/Cmd combos to the browser
    if (e.key === "ArrowRight") { setCursor(S.cursor + 1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { setCursor(S.cursor - 1); e.preventDefault(); }
    else if (e.key === " ") { togglePlay(); e.preventDefault(); }
    else if (e.key === "k") { toggleKeyframe(S.cursor); }
    else if (e.key === "0") { selectSubtask("eraser", true); }
    else if (/^[1-9]$/.test(e.key)) {
      const i = +e.key - 1;
      if (i < S.cfg.subtasks.length) selectSubtask(i, true);
    }
  });
}

async function doExport() {
  $("featureName").blur();
  saveConfig();
  await saveAnnotation();
  try {
    const res = await apiPost("/api/export", {});
    const s = res.summary;
    const nEp = s.episodes.length;
    toast(`已导出 ${nEp} 个 episode 的 extras.parquet（feature="${s.feature_name}"）。未标注帧总数: ${s.total_unlabeled}`, "ok");
  } catch (e) {
    toast("导出失败: " + e.message, "err");
  }
}

init().catch((e) => { console.error(e); toast("初始化失败: " + e.message, "err"); });
