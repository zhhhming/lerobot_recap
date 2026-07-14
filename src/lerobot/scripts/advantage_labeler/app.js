const state = {
  meta: null,
  items: [],
  selected: null,
  selectedElement: null,
  offset: 0,
  page: 1,
  totalPages: 1,
  pageSize: 150,
  overridesByMode: { global: {}, subtask: {} },
};

const els = Object.fromEntries(
  [
    "runMeta", "syntheticWarning", "valueMode", "sortOrder", "tiePolicy", "topPercent",
    "topPercentText", "exportBtn", "counts", "chunkList", "chunkTitle", "chunkDetails",
    "cameraSelect", "frameImage", "prevFrame", "nextFrame", "frameOffset", "status",
    "clearOverride", "prevPage", "nextPage", "pageInfo",
  ].map((id) => [id, document.getElementById(id)]),
);

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function modeOverrides() {
  return state.overridesByMode[els.valueMode.value] || {};
}

function requestPayload() {
  return {
    value_mode: els.valueMode.value,
    top_percent: Number(els.topPercent.value) / 100,
    sort_order: els.sortOrder.value,
    tie_policy: els.tiePolicy.value,
    overrides: modeOverrides(),
    page: state.page,
    page_size: state.pageSize,
  };
}

function updateWarning() {
  const eligibility = state.meta.eligibility_by_mode[els.valueMode.value];
  els.syntheticWarning.hidden = !eligibility || !eligibility.warning;
  els.syntheticWarning.textContent = eligibility?.warning || "";
}

async function loadMeta() {
  state.meta = await jsonFetch("/api/meta");
  state.overridesByMode = state.meta.overrides_by_mode;
  els.runMeta.textContent = `${state.meta.episodes.length} episodes | ${state.meta.task || "untitled task"}`;
  els.valueMode.value = state.meta.value_mode;
  els.sortOrder.value = state.meta.sort_order;
  els.tiePolicy.value = state.meta.tie_policy;
  els.topPercent.value = Math.round(state.meta.top_percent * 100);
  els.topPercentText.textContent = `${els.topPercent.value}%`;
  for (const camera of state.meta.cameras) {
    const option = document.createElement("option");
    option.value = camera.subdir;
    option.textContent = camera.label;
    els.cameraSelect.appendChild(option);
  }
  updateWarning();
}

async function refresh({ resetPage = false } = {}) {
  if (resetPage) state.page = 1;
  els.topPercentText.textContent = `${els.topPercent.value}%`;
  const preview = await jsonFetch("/api/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestPayload()),
  });
  state.items = preview.items;
  state.page = preview.page;
  state.totalPages = preview.total_pages;
  renderCounts(preview.counts);
  renderList();
  renderPagination();
}

function renderCounts(counts) {
  els.counts.replaceChildren();
  for (const label of ["positive", "negative", "ignore"]) {
    const item = document.createElement("span");
    item.className = `badge ${label}`;
    item.textContent = `${label}: ${counts[label] || 0}`;
    els.counts.appendChild(item);
  }
}

function renderList() {
  els.chunkList.replaceChildren();
  for (const chunk of state.items) {
    const item = document.createElement("li");
    item.className = "chunkItem";
    item.dataset.key = chunk.key;
    item.innerHTML = `
      <div><strong>${chunk.episode_name} frame_${String(chunk.frame_index).padStart(6, "0")}</strong>
      <span>adv=${chunk.advantage.toFixed(3)} | h=${chunk.valid_horizon}</span>
      <span>stored=${chunk.stored_label || "unset"} | threshold=${chunk.threshold_label}</span>
      <span>override=${chunk.manual_override_label || "none"} | source=${chunk.label_source}</span></div>
      <span class="badge ${chunk.preview_label}">${chunk.preview_label}</span>`;
    item.addEventListener("click", () => selectChunk(chunk, 0, item));
    els.chunkList.appendChild(item);
  }
}

function renderPagination() {
  els.pageInfo.textContent = `Page ${state.page} / ${state.totalPages}`;
  els.prevPage.disabled = state.page <= 1;
  els.nextPage.disabled = state.page >= state.totalPages;
}

function selectChunk(chunk, offset, element = state.selectedElement) {
  state.selectedElement?.classList.remove("active");
  state.selected = chunk;
  state.selectedElement = element;
  state.selectedElement?.classList.add("active");
  state.offset = Math.max(0, Math.min(offset, Math.max(chunk.valid_horizon, 0)));
  els.chunkTitle.textContent = `${chunk.episode_name} frame_${String(chunk.frame_index).padStart(6, "0")}`;
  els.chunkDetails.textContent = `offset ${state.offset}/${chunk.valid_horizon} | stored ${chunk.stored_label || "unset"} | threshold ${chunk.threshold_label} | override ${chunk.manual_override_label || "none"} | preview ${chunk.preview_label}`;
  els.frameOffset.max = Math.max(chunk.valid_horizon, 0);
  els.frameOffset.value = state.offset;
  const frame = chunk.frame_index + state.offset;
  els.frameImage.src = `/api/episode/${chunk.episode_index}/img/${els.cameraSelect.value}/${frame}`;
}

function adjustOffset(delta) {
  if (state.selected) selectChunk(state.selected, state.offset + delta);
}

async function setOverride(label) {
  if (!state.selected) return;
  modeOverrides()[state.selected.key] = label;
  await refresh();
}

async function clearOverride() {
  if (!state.selected) return;
  delete modeOverrides()[state.selected.key];
  await refresh();
}

async function exportLabels() {
  els.status.textContent = "Preparing change summary...";
  const preview = await jsonFetch("/api/export-preview", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestPayload()),
  });
  const changes = preview.summary.change_summary;
  const message = `Export ${changes.total} labels? changed=${changes.changed}, unset=${changes.unset_to_labeled}, unchanged=${changes.unchanged}`;
  if (!window.confirm(message)) {
    els.status.textContent = "Export cancelled";
    return;
  }
  const body = { ...requestPayload(), confirm: true };
  const result = await jsonFetch("/api/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  state.meta.overrides_by_mode[els.valueMode.value] = result.summary.overrides;
  els.status.textContent = `Exported ${result.summary.columns_written.join(", ")}`;
  await refresh();
}

function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

async function main() {
  await loadMeta();
  await refresh();
  els.valueMode.addEventListener("change", () => { updateWarning(); refresh({ resetPage: true }); });
  els.sortOrder.addEventListener("change", () => refresh({ resetPage: true }));
  els.tiePolicy.addEventListener("change", () => refresh({ resetPage: true }));
  els.topPercent.addEventListener("input", debounce(() => refresh({ resetPage: true }), 150));
  els.prevPage.addEventListener("click", () => { state.page -= 1; refresh(); });
  els.nextPage.addEventListener("click", () => { state.page += 1; refresh(); });
  els.cameraSelect.addEventListener("change", () => state.selected && selectChunk(state.selected, state.offset));
  els.prevFrame.addEventListener("click", () => adjustOffset(-1));
  els.nextFrame.addEventListener("click", () => adjustOffset(1));
  els.frameOffset.addEventListener("input", () => state.selected && selectChunk(state.selected, Number(els.frameOffset.value)));
  els.exportBtn.addEventListener("click", exportLabels);
  els.clearOverride.addEventListener("click", clearOverride);
  document.querySelectorAll("[data-label]").forEach((button) => button.addEventListener("click", () => setOverride(button.dataset.label)));
  window.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") adjustOffset(-1);
    if (event.key === "ArrowRight") adjustOffset(1);
  });
}

main().catch((error) => {
  els.status.textContent = error.message;
  console.error(error);
});
