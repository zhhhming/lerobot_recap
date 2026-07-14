const state = { meta: null, groups: [], selectedGroup: null, chunks: [], selected: null, offset: 0, groupPage: 1, groupPages: 1, chunkPage: 1, chunkPages: 1 };
const els = Object.fromEntries(["runMeta", "valueMode", "warning", "groupStrip", "prevGroups", "nextGroups", "groupPage", "groupTitle", "sortOrder", "chunkList", "prevChunks", "nextChunks", "chunkPage", "chunkTitle", "chunkDetails", "cameraSelect", "frameImage", "prevFrame", "nextFrame", "frameOffset", "status"].map((id) => [id, document.getElementById(id)]));

async function jsonFetch(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

async function loadMeta() {
  state.meta = await jsonFetch("/api/meta");
  els.valueMode.value = state.meta.value_mode;
  els.runMeta.textContent = `${state.meta.episodes.length} episodes | ${state.meta.task || "untitled task"}`;
  for (const camera of state.meta.cameras) {
    const option = document.createElement("option"); option.value = camera.subdir; option.textContent = camera.label; els.cameraSelect.appendChild(option);
  }
  updateWarning();
}

function updateWarning() {
  const detail = state.meta.modes[els.valueMode.value];
  const warning = detail?.eligibility?.warning || "";
  els.warning.hidden = !warning; els.warning.textContent = warning;
}

async function loadGroups() {
  const payload = await jsonFetch(`/api/groups?value_mode=${els.valueMode.value}&page=${state.groupPage}&page_size=100`);
  state.groups = payload.items; state.groupPage = payload.page; state.groupPages = payload.total_pages;
  els.groupStrip.replaceChildren();
  for (const group of state.groups) {
    const button = document.createElement("button"); button.className = "groupCard";
    button.innerHTML = `<strong>${group.group_id}</strong><span>n=${group.size} | +${group.positive_count} / -${group.negative_count} / i${group.ignore_count}</span><span>w ${group.min_weight.toFixed(3)}–${group.max_weight.toFixed(3)}</span>`;
    button.addEventListener("click", () => selectGroup(group)); els.groupStrip.appendChild(button);
  }
  els.groupPage.textContent = `Bins ${state.groupPage} / ${state.groupPages}`;
  els.prevGroups.disabled = state.groupPage <= 1; els.nextGroups.disabled = state.groupPage >= state.groupPages;
}

async function selectGroup(group) { state.selectedGroup = group; state.chunkPage = 1; els.groupTitle.textContent = group.group_id; await loadChunks(); }

async function loadChunks() {
  if (!state.selectedGroup) return;
  const group = encodeURIComponent(state.selectedGroup.group_id);
  const payload = await jsonFetch(`/api/chunks?value_mode=${els.valueMode.value}&group_id=${group}&page=${state.chunkPage}&page_size=150&sort_order=${els.sortOrder.value}`);
  state.chunks = payload.items; state.chunkPage = payload.page; state.chunkPages = payload.total_pages;
  els.chunkList.replaceChildren();
  for (const chunk of state.chunks) {
    const item = document.createElement("li"); item.className = "chunkItem";
    const rank = chunk.positive_rank == null ? "—" : `${chunk.positive_rank.toFixed(1)}/${chunk.positive_group_size}`;
    item.innerHTML = `<strong>${chunk.episode_name} frame_${String(chunk.frame_index).padStart(6, "0")}</strong><span>adv=${chunk.advantage.toFixed(3)} | ${chunk.label}</span><span>rank=${rank} | weight=${chunk.weight.toFixed(4)}</span>`;
    item.addEventListener("click", () => selectChunk(chunk, 0)); els.chunkList.appendChild(item);
  }
  els.chunkPage.textContent = `Chunks ${state.chunkPage} / ${state.chunkPages}`;
  els.prevChunks.disabled = state.chunkPage <= 1; els.nextChunks.disabled = state.chunkPage >= state.chunkPages;
}

function selectChunk(chunk, offset) {
  state.selected = chunk; state.offset = Math.max(0, Math.min(offset, Math.max(chunk.valid_horizon, 0)));
  els.chunkTitle.textContent = `${chunk.episode_name} frame_${String(chunk.frame_index).padStart(6, "0")}`;
  els.chunkDetails.textContent = `offset ${state.offset}/${chunk.valid_horizon} | advantage ${chunk.advantage.toFixed(4)} | label ${chunk.label} | rank ${chunk.positive_rank ?? "—"} | weight ${chunk.weight.toFixed(4)} | ${chunk.group_id}`;
  els.frameOffset.max = Math.max(chunk.valid_horizon, 0); els.frameOffset.value = state.offset;
  els.frameImage.src = `/api/episode/${chunk.episode_index}/img/${els.cameraSelect.value}/${chunk.frame_index + state.offset}`;
}

function adjustOffset(delta) { if (state.selected) selectChunk(state.selected, state.offset + delta); }

async function changeMode() { state.groupPage = 1; state.selectedGroup = null; state.chunks = []; els.chunkList.replaceChildren(); updateWarning(); await loadGroups(); }

async function main() {
  await loadMeta(); await loadGroups();
  els.valueMode.addEventListener("change", changeMode); els.sortOrder.addEventListener("change", loadChunks);
  els.prevGroups.addEventListener("click", async () => { state.groupPage -= 1; await loadGroups(); }); els.nextGroups.addEventListener("click", async () => { state.groupPage += 1; await loadGroups(); });
  els.prevChunks.addEventListener("click", async () => { state.chunkPage -= 1; await loadChunks(); }); els.nextChunks.addEventListener("click", async () => { state.chunkPage += 1; await loadChunks(); });
  els.prevFrame.addEventListener("click", () => adjustOffset(-1)); els.nextFrame.addEventListener("click", () => adjustOffset(1));
  els.frameOffset.addEventListener("input", () => state.selected && selectChunk(state.selected, Number(els.frameOffset.value))); els.cameraSelect.addEventListener("change", () => state.selected && selectChunk(state.selected, state.offset));
  window.addEventListener("keydown", (event) => { if (event.key === "ArrowLeft") adjustOffset(-1); if (event.key === "ArrowRight") adjustOffset(1); });
}

main().catch((error) => { els.status.textContent = error.message; console.error(error); });
