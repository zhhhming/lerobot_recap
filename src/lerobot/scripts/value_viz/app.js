const COLORS = {
  global_remaining_gt: "#79d2ff",
  global_remaining_pred: "#ffbd66",
  global_elapsed_gt: "#8de6a4",
  subtask_remaining_gt: "#89a7ff",
  subtask_remaining_gt_head: "#ff8a75",
  subtask_remaining_smooth_head: "#d79bff",
  subtask_elapsed_gt: "#56d6b2",
};
const BAND_COLORS = ["#1d4d66", "#4c3766", "#315e48", "#66512d", "#663c43", "#334f73", "#5c3f66", "#53662f"];

const state = {
  meta: null,
  episode: null,
  frame: 0,
  unit: "norm",
  boundary: "gt",
  curves: null,
  frameData: null,
  enabled: new Set(),
  requestSerial: 0,
};
const ids = ["runMeta", "warnings", "unitSelect", "boundarySelect", "episodeCount", "episodeList", "frameTitle", "frameSubtask", "cameraSelect", "frameImage", "prevFrame", "nextFrame", "frameSlider", "chunkToggle", "seriesControls", "valueChart", "chartMeta", "subtaskLegend", "currentValues", "chunkSegments", "status"];
const els = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));

async function jsonFetch(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function formatNumber(value) {
  if (value == null) return "unavailable";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function setStatus(message, error = false) {
  els.status.textContent = message || "";
  els.status.classList.toggle("error", error);
}

function populateMeta() {
  const { meta } = state;
  els.runMeta.textContent = `${meta.root} · ${meta.episodes.length} episodes · ${meta.fps} fps · ${meta.robot_type || "unknown robot"}`;
  els.episodeCount.textContent = meta.episodes.length;

  els.boundarySelect.replaceChildren();
  for (const boundary of meta.boundaries) {
    const option = document.createElement("option");
    option.value = boundary.id;
    option.textContent = boundary.available ? boundary.label : `${boundary.label} · unavailable`;
    option.disabled = !boundary.available;
    els.boundarySelect.appendChild(option);
  }
  const firstBoundary = meta.boundaries.find((item) => item.available);
  state.boundary = firstBoundary?.id || "gt";
  els.boundarySelect.value = state.boundary;

  els.cameraSelect.replaceChildren();
  for (const camera of meta.cameras) {
    const option = document.createElement("option");
    option.value = camera.subdir;
    option.textContent = camera.label;
    els.cameraSelect.appendChild(option);
  }
  const thirdPerson = meta.cameras.find((camera) => camera.subdir === "third_person");
  if (thirdPerson) els.cameraSelect.value = thirdPerson.subdir;

  els.warnings.replaceChildren();
  for (const mode of ["global", "subtask"]) {
    const info = meta.provenance[mode];
    if (!info?.warning) continue;
    const warning = document.createElement("p");
    warning.className = `warning ${info.status}`;
    warning.textContent = `${mode}: ${info.warning}`;
    els.warnings.appendChild(warning);
  }

  els.episodeList.replaceChildren();
  for (const episode of meta.episodes) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.episode = episode.index;
    button.innerHTML = `<strong>${episode.name}</strong><span>${episode.length.toLocaleString()} frames</span>`;
    button.addEventListener("click", () => selectEpisode(episode.index));
    item.appendChild(button);
    els.episodeList.appendChild(item);
  }
  buildSeriesControls();
}

function buildSeriesControls() {
  els.seriesControls.replaceChildren();
  state.enabled.clear();
  for (const series of state.meta.series) {
    const column = series.columns[state.unit];
    const label = document.createElement("label");
    label.className = `seriesToggle ${column ? "" : "unavailable"}`;
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = series.id;
    input.disabled = !column;
    input.checked = Boolean(column && series.kind === "remaining");
    if (input.checked) state.enabled.add(series.id);
    input.addEventListener("change", () => {
      if (input.checked) state.enabled.add(series.id); else state.enabled.delete(series.id);
      renderChart();
      renderCurrentValues();
    });
    const swatch = document.createElement("i");
    swatch.style.background = COLORS[series.id];
    const text = document.createElement("span");
    text.textContent = column ? series.label : `${series.label} · unavailable`;
    label.append(input, swatch, text);
    els.seriesControls.appendChild(label);
  }
}

async function selectEpisode(index) {
  const episode = state.meta.episodes.find((item) => item.index === index);
  if (!episode) return;
  state.episode = episode;
  state.frame = 0;
  els.frameSlider.max = Math.max(episode.length - 1, 0);
  els.frameSlider.value = 0;
  for (const button of els.episodeList.querySelectorAll("button")) {
    button.classList.toggle("active", Number(button.dataset.episode) === index);
  }
  await loadCurves();
  await setFrame(0);
}

async function loadCurves() {
  if (!state.episode) return;
  setStatus("Loading episode curves…");
  const params = new URLSearchParams({ unit: state.unit, boundary: state.boundary, max_points: "2000" });
  state.curves = await jsonFetch(`/api/episode/${state.episode.index}/curves?${params}`);
  const available = state.curves.curves.filter((curve) => curve.available).length;
  els.chartMeta.textContent = `${state.curves.sampled_points.toLocaleString()} plotted points · ${available}/${state.curves.curves.length} series available`;
  renderSubtaskLegend();
  renderChart();
  setStatus(available ? "" : "This episode has no value columns; unavailable series are shown explicitly.");
}

async function setFrame(frame) {
  if (!state.episode) return;
  const clamped = Math.max(0, Math.min(Number(frame), state.episode.length - 1));
  state.frame = clamped;
  els.frameSlider.value = clamped;
  els.prevFrame.disabled = clamped <= 0;
  els.nextFrame.disabled = clamped >= state.episode.length - 1;
  els.frameTitle.textContent = `${state.episode.name} · frame_${String(clamped).padStart(6, "0")}`;
  updateImage();
  const serial = ++state.requestSerial;
  try {
    const params = new URLSearchParams({ boundary: state.boundary });
    const frameData = await jsonFetch(`/api/episode/${state.episode.index}/frame/${clamped}?${params}`);
    if (serial !== state.requestSerial) return;
    state.frameData = frameData;
    const subtask = frameData.subtask;
    els.frameSubtask.textContent = subtask ? `${subtask.name} · ${frameData.time_seconds.toFixed(2)} s` : `${frameData.time_seconds.toFixed(2)} s · subtask unavailable`;
    renderCurrentValues();
    renderChunkSegments();
    renderChart();
  } catch (error) {
    if (serial === state.requestSerial) setStatus(error.message, true);
  }
}

function updateImage() {
  if (!state.episode || !els.cameraSelect.value) {
    els.frameImage.removeAttribute("src");
    return;
  }
  els.frameImage.src = `/api/episode/${state.episode.index}/img/${encodeURIComponent(els.cameraSelect.value)}/${state.frame}`;
}

function renderCurrentValues() {
  els.currentValues.replaceChildren();
  if (!state.frameData) return;
  for (const series of state.meta.series) {
    const card = document.createElement("div");
    card.className = `valueCard ${state.enabled.has(series.id) ? "active" : ""}`;
    const value = state.frameData.values[series.id]?.[state.unit];
    card.innerHTML = `<i style="background:${COLORS[series.id]}"></i><span>${series.label}</span><strong>${formatNumber(value)}</strong>`;
    els.currentValues.appendChild(card);
  }
}

function renderChunkSegments() {
  els.chunkSegments.replaceChildren();
  if (!state.frameData) return;
  for (const [index, segment] of state.frameData.chunk_segments.entries()) {
    const card = document.createElement("div");
    card.className = "segment";
    card.style.borderColor = BAND_COLORS[(segment.id ?? index) % BAND_COLORS.length];
    card.innerHTML = `<strong>${segment.start}–${segment.end}</strong><span>${segment.name}</span><small>${segment.end - segment.start + 1} sampled states</small>`;
    els.chunkSegments.appendChild(card);
  }
  if (!state.frameData.chunk_segments.length) {
    els.chunkSegments.textContent = "Subtask boundary unavailable.";
  }
}

function renderSubtaskLegend() {
  els.subtaskLegend.replaceChildren();
  if (!state.curves) return;
  const seen = new Set();
  for (const interval of state.curves.subtask_intervals) {
    const key = `${interval.id}:${interval.name}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const item = document.createElement("span");
    const colorIndex = Math.abs(interval.id ?? seen.size - 1) % BAND_COLORS.length;
    item.innerHTML = `<i style="background:${BAND_COLORS[colorIndex]}"></i>${interval.name}`;
    els.subtaskLegend.appendChild(item);
  }
}

function renderChart() {
  const canvas = els.valueChart;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(Math.floor(rect.width), 320);
  const height = 380;
  if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0a1119"; ctx.fillRect(0, 0, width, height);
  if (!state.curves) return;

  const margin = { left: 60, right: 18, top: 18, bottom: 38 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xMax = Math.max(state.curves.frame_count - 1, 1);
  const selectedCurves = state.curves.curves.filter((curve) => curve.available && state.enabled.has(curve.id));
  const values = selectedCurves.flatMap((curve) => {
    const sampled = curve.points.map((point) => point[1]).filter((value) => value != null);
    const exact = state.frameData?.values[curve.id]?.[state.unit];
    if (exact != null) sampled.push(exact);
    return sampled;
  });
  let yMin = values.length ? Math.min(...values) : 0;
  let yMax = values.length ? Math.max(...values) : 1;
  if (state.unit === "norm") { yMin = Math.min(0, yMin); yMax = Math.max(1, yMax); }
  if (yMax <= yMin) yMax = yMin + 1;
  const x = (frame) => margin.left + (frame / xMax) * plotW;
  const y = (value) => margin.top + (1 - (value - yMin) / (yMax - yMin)) * plotH;

  for (const interval of state.curves.subtask_intervals) {
    const colorIndex = Math.abs(interval.id ?? 0) % BAND_COLORS.length;
    ctx.globalAlpha = 0.18; ctx.fillStyle = BAND_COLORS[colorIndex];
    ctx.fillRect(x(interval.start), margin.top, Math.max(x(interval.end + 1) - x(interval.start), 1), plotH);
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "#314152"; ctx.lineWidth = 1;
  ctx.fillStyle = "#93a3b8"; ctx.font = "12px system-ui";
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = yMin + ((yMax - yMin) * tick) / 4;
    const py = y(value);
    ctx.beginPath(); ctx.moveTo(margin.left, py); ctx.lineTo(width - margin.right, py); ctx.stroke();
    ctx.fillText(value.toFixed(state.unit === "norm" ? 2 : 0), 8, py + 4);
  }
  ctx.fillText("0", margin.left - 4, height - 12);
  ctx.fillText(String(xMax), width - margin.right - ctx.measureText(String(xMax)).width, height - 12);

  if (els.chunkToggle.checked && state.frameData) {
    ctx.save(); ctx.setLineDash([5, 5]); ctx.strokeStyle = "#efdc82"; ctx.globalAlpha = 0.7;
    for (const segment of state.frameData.chunk_segments.slice(1)) {
      ctx.beginPath(); ctx.moveTo(x(segment.start), margin.top); ctx.lineTo(x(segment.start), margin.top + plotH); ctx.stroke();
    }
    ctx.strokeStyle = "#ff9b67";
    ctx.beginPath(); ctx.moveTo(x(state.frameData.chunk_end), margin.top); ctx.lineTo(x(state.frameData.chunk_end), margin.top + plotH); ctx.stroke();
    ctx.restore();
  }

  for (const curve of selectedCurves) {
    ctx.beginPath(); ctx.strokeStyle = COLORS[curve.id]; ctx.lineWidth = 2;
    let penDown = false;
    for (const [frame, value] of curve.points) {
      if (value == null) { penDown = false; continue; }
      if (!penDown) { ctx.moveTo(x(frame), y(value)); penDown = true; } else ctx.lineTo(x(frame), y(value));
    }
    ctx.stroke();
    const exact = state.frameData?.values[curve.id]?.[state.unit];
    if (exact != null) {
      ctx.beginPath(); ctx.fillStyle = COLORS[curve.id]; ctx.arc(x(state.frame), y(exact), 4, 0, Math.PI * 2); ctx.fill();
    }
  }
  ctx.strokeStyle = "#ffffff"; ctx.globalAlpha = 0.8; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x(state.frame), margin.top); ctx.lineTo(x(state.frame), margin.top + plotH); ctx.stroke();
  ctx.globalAlpha = 1;
  if (!selectedCurves.length) {
    ctx.fillStyle = "#a8b7c8"; ctx.font = "16px system-ui";
    ctx.fillText("Selected value curves are unavailable.", margin.left + 20, margin.top + 36);
  }
}

function adjustFrame(delta) { if (state.episode) setFrame(state.frame + delta); }

async function changeCurveContract() {
  state.unit = els.unitSelect.value;
  state.boundary = els.boundarySelect.value;
  buildSeriesControls();
  await loadCurves();
  await setFrame(state.frame);
}

async function main() {
  state.meta = await jsonFetch("/api/meta");
  populateMeta();
  els.unitSelect.addEventListener("change", changeCurveContract);
  els.boundarySelect.addEventListener("change", changeCurveContract);
  els.cameraSelect.addEventListener("change", updateImage);
  els.frameSlider.addEventListener("input", () => setFrame(Number(els.frameSlider.value)));
  els.prevFrame.addEventListener("click", () => adjustFrame(-1));
  els.nextFrame.addEventListener("click", () => adjustFrame(1));
  els.chunkToggle.addEventListener("change", renderChart);
  window.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    event.preventDefault();
    const amount = event.shiftKey ? 10 : 1;
    adjustFrame(event.key === "ArrowLeft" ? -amount : amount);
  });
  new ResizeObserver(renderChart).observe(els.valueChart.parentElement);
  if (state.meta.episodes.length) await selectEpisode(state.meta.episodes[0].index);
}

main().catch((error) => { setStatus(error.message, true); console.error(error); });
