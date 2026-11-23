// Client for the SAE feature viewer: latent list (virtual scroll), feature page, UMAP map, notes.

type SaeInfo = {
  key: string; run: string; n: number; hook: string; tokens: number; docs: number; alive: number;
  eval: Record<string, number> | null; labeled: number; hasUmap: boolean; notes: string;
};
type List = {
  freq: number[]; max: number[]; top: string[]; promotes: string[]; label: string[]; category: string[];
  umap: { x: number[]; y: number[] } | null;
};
type Example = { doc: number; pos: number; act: number; source: string; left: string; token: string; right: string };
type Neighbor = { id: number; cos: number; freq: number; label: string; top: string[] };
type Feature = {
  id: number; freq: number; mean_act: number; max_act: number; examples: Example[]; promotes: string[];
  suppresses: string[]; label: { label: string; category?: string; confidence?: string; explanation?: string } | null;
  neighbors: Record<string, Neighbor[]>;
};

export const CATEGORIES: Record<string, [string, string]> = {
  token: ["#4e79a7", "word / token"],
  topic: ["#59a14f", "topic / entity"],
  grammar: ["#f28e2b", "grammar / syntax"],
  discourse: ["#b07aa1", "discourse / reasoning"],
  format: ["#76b7b2", "formatting / punctuation"],
  chat: ["#e15759", "chat template / assistant"],
  code: ["#9c755f", "code / markup"],
  math: ["#edc948", "math / numbers"],
  dense: ["#bab0ac", "dense / positional"],
  unclear: ["#6b6b6b", "unclear"],
};

const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector(s) as T;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const fmtFreq = (f: number) => (f === 0 ? "dead" : f >= 0.01 ? `${(f * 100).toFixed(1)}%` : `${(f * 100).toPrecision(2)}%`);
const fmtN = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(0)}k` : `${n}`);
const showTok = (t: string) => t.replace(/\n/g, "⏎");

const state = {
  saes: [] as SaeInfo[],
  sae: "",
  list: null as List | null,
  rows: [] as number[],
  id: -1,
  tab: "feature",
  cats: new Set<string>(),
};

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { sae: p.get("sae") ?? "", id: p.has("id") ? Number(p.get("id")) : -1, tab: p.get("tab") ?? "feature" };
}
function writeHash() {
  const p = new URLSearchParams({ sae: state.sae, tab: state.tab });
  if (state.id >= 0) p.set("id", String(state.id));
  history.replaceState(null, "", `#${p}`);
}

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

// ------------------------------------------------------------------ header

function renderHeader() {
  const cur = state.saes.find((s) => s.key === state.sae);
  const runs = [...new Set(state.saes.map((s) => s.run))];
  $("#saes").innerHTML = state.saes
    .filter((s) => s.run === cur?.run)
    .map((s) => `<button data-key="${esc(s.key)}" class="${s.key === state.sae ? "on" : ""}">${s.n / 2048}x · ${fmtN(s.n)}</button>`)
    .join("") + (runs.length > 1 ? `<select id="runsel">${runs.map((r) => `<option ${r === cur?.run ? "selected" : ""}>${r}</option>`).join("")}</select>` : "");
  if (!cur) return;
  const ev = cur.eval ?? {};
  const stat = (k: string, v: string) => `<span>${k} <b>${v}</b></span>`;
  $("#stats").innerHTML = [
    stat("hook", cur.hook),
    ev.fvu != null ? stat("FVU", ev.fvu.toFixed(3)) : "",
    ev.loss_recovered != null ? stat("loss recovered", `${(ev.loss_recovered * 100).toFixed(1)}%`) : "",
    stat("alive", `${cur.alive.toLocaleString()} / ${cur.n.toLocaleString()}`),
    stat("labeled", cur.labeled.toLocaleString()),
    stat("scanned", `${fmtN(cur.tokens)} tokens`),
  ].join("");
  document.querySelectorAll<HTMLButtonElement>("#saes button").forEach((b) => (b.onclick = () => selectSae(b.dataset.key!)));
  const rs = document.getElementById("runsel") as HTMLSelectElement | null;
  if (rs) rs.onchange = () => selectSae(state.saes.find((s) => s.run === rs.value)!.key);
}

// ------------------------------------------------------------------ list

function renderCats() {
  const L = state.list!;
  const counts: Record<string, number> = {};
  L.category.forEach((c) => c && (counts[c] = (counts[c] ?? 0) + 1));
  $("#cats").innerHTML = Object.entries(CATEGORIES)
    .filter(([k]) => counts[k])
    .map(([k, [color, name]]) => `<span class="chip ${state.cats.has(k) ? "on" : ""}" data-cat="${k}" title="${name}"><span class="dot" style="background:${color}"></span>${k} ${counts[k]}</span>`)
    .join("");
  document.querySelectorAll<HTMLElement>("#cats .chip").forEach((el) => {
    el.onclick = () => {
      const c = el.dataset.cat!;
      state.cats.has(c) ? state.cats.delete(c) : state.cats.add(c);
      renderCats();
      filter();
    };
  });
}

function filter() {
  const L = state.list!;
  const q = $<HTMLInputElement>("#q").value.trim().toLowerCase();
  const only = $<HTMLInputElement>("#onlyLabeled").checked;
  const sort = $<HTMLSelectElement>("#sort").value;
  const n = L.freq.length;
  let rows: number[] = [];
  const qid = /^\d+$/.test(q) ? Number(q) : -1;
  for (let i = 0; i < n; i++) {
    if (only && !L.label[i]) continue;
    if (state.cats.size && !state.cats.has(L.category[i])) continue;
    if (q && i !== qid && !(L.label[i].toLowerCase().includes(q) || L.top[i].toLowerCase().includes(q) || L.promotes[i].toLowerCase().includes(q))) continue;
    rows.push(i);
  }
  const by: Record<string, (a: number, b: number) => number> = {
    id: (a, b) => a - b,
    "freq-desc": (a, b) => L.freq[b] - L.freq[a],
    "freq-asc": (a, b) => (L.freq[a] || 1) - (L.freq[b] || 1),
    "max-desc": (a, b) => L.max[b] - L.max[a],
    labeled: (a, b) => Number(!L.label[a]) - Number(!L.label[b]) || a - b,
  };
  rows.sort(by[sort]);
  if (qid >= 0 && qid < n) rows = [qid, ...rows.filter((r) => r !== qid)];
  state.rows = rows;
  $("#count").textContent = `${rows.length.toLocaleString()} latents`;
  $("#list").scrollTop = 0;
  renderList();
  if (state.tab === "map") drawMap();
}

const ROW = 44;
function renderList() {
  const el = $("#list");
  const L = state.list!;
  const start = Math.max(0, Math.floor(el.scrollTop / ROW) - 5);
  const end = Math.min(state.rows.length, start + Math.ceil(el.clientHeight / ROW) + 10);
  let html = `<div style="height:${state.rows.length * ROW}px"></div>`;
  for (let r = start; r < end; r++) {
    const i = state.rows[r];
    const cat = L.category[i];
    const dot = cat ? `<span class="dot" style="background:${CATEGORIES[cat]?.[0] ?? "#888"}"></span>` : "";
    const lab = L.label[i] ? `<span class="lab">${dot}${esc(L.label[i])}</span>` : `<span class="lab none">“${esc(showTok(L.top[i]))}”</span>`;
    html += `<div class="item ${i === state.id ? "sel" : ""}" data-id="${i}" style="top:${r * ROW}px">
      <div class="l1"><span class="id">#${i}</span>${lab}<span class="fq">${fmtFreq(L.freq[i])}</span></div>
      <div class="l2">→ ${esc(L.promotes[i])}</div></div>`;
  }
  el.innerHTML = html;
  el.querySelectorAll<HTMLElement>(".item").forEach((it) => (it.onclick = () => selectFeature(Number(it.dataset.id))));
}

// ------------------------------------------------------------------ feature page

function exampleHtml(e: Example, max: number) {
  const a = Math.max(0.15, Math.min(1, e.act / (max || 1)));
  const text = (s: string) => esc(s).replace(/\n/g, '<span class="nl">⏎</span>\n');
  return `<div class="ex"><div class="a">${e.act.toFixed(2)}</div><div class="s" title="${e.source}">${e.source === "chat" ? "C" : "W"}</div>
    <div class="t">${text(e.left)}<mark style="--a:${a.toFixed(2)}">${text(e.token)}</mark>${text(e.right)}</div></div>`;
}

async function renderFeature() {
  const el = $("#detail");
  if (state.id < 0) {
    el.innerHTML = `<div class="empty">Pick a latent on the left, or search. Labeled latents come first.</div>`;
    return;
  }
  const f = await getJson<Feature>(`/api/feature?sae=${encodeURIComponent(state.sae)}&id=${state.id}`);
  if (f.id !== state.id) return;
  const lab = f.label;
  const cat = lab?.category ? CATEGORIES[lab.category] : null;
  const toks = (xs: string[], cls: string) => xs.map((t) => `<span class="tok ${cls}">${esc(showTok(t))}</span>`).join("");
  const nbs = Object.entries(f.neighbors)
    .sort(([a], [b]) => Number(a) - Number(b))
    .map(([n, list]) => {
      const key = state.saes.find((s) => s.key.endsWith(`/sae_${n}`) && s.key.startsWith(state.sae.split("/")[0]))?.key;
      return `<div class="nb"><h4>${Number(n) / 2048}x SAE (${fmtN(Number(n))})</h4>${list
        .map((x) => `<a data-key="${esc(key ?? "")}" data-id="${x.id}"><span class="c">#${x.id}</span><span class="c">${x.cos.toFixed(2)}</span><span class="w">${x.label ? esc(x.label) : `<span class="muted">${esc(x.top.map(showTok).join(" · "))}</span>`}</span></a>`)
        .join("")}</div>`;
    })
    .join("");
  el.innerHTML = `
    <div class="h"><span class="idbig">#${f.id}</span><h2>${lab ? esc(lab.label) : '<span class="muted">unlabeled</span>'}</h2>
      ${cat ? `<span class="badge"><span class="dot" style="background:${cat[0]}"></span>${cat[1]}</span>` : ""}
      ${lab?.confidence ? `<span class="badge">confidence: ${esc(lab.confidence)}</span>` : ""}</div>
    ${lab?.explanation ? `<p class="expl">${esc(lab.explanation)}</p>` : ""}
    <div class="kv">
      <div><span>fires on</span><b>${fmtFreq(f.freq)}</b></div>
      <div><span>mean act</span><b>${f.mean_act.toFixed(2)}</b></div>
      <div><span>max act</span><b>${f.max_act.toFixed(2)}</b></div>
    </div>
    <h3>Direct effect on output tokens</h3>
    <div class="toks" style="margin-bottom:6px">${toks(f.promotes, "up")}</div>
    <div class="toks">${toks(f.suppresses, "down")}</div>
    <h3>Top activations (held-out, one per document)</h3>
    ${f.examples.map((e) => exampleHtml(e, f.max_act)).join("") || '<div class="muted">never fires</div>'}
    ${nbs ? `<h3>Closest decoder directions in the other SAEs (cosine)</h3><div class="nbs">${nbs}</div>` : ""}`;
  el.querySelectorAll<HTMLElement>(".nb a").forEach((a) => {
    a.onclick = async () => {
      if (a.dataset.key) await selectSae(a.dataset.key, Number(a.dataset.id));
    };
  });
}

// ------------------------------------------------------------------ map

let mapView = { scale: 1, ox: 0, oy: 0 };
let mapPts: { px: Float32Array; py: Float32Array } | null = null;

function drawMap() {
  const canvas = $<HTMLCanvasElement>("#map");
  const L = state.list!;
  const dpr = devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue("--bg");
  ctx.fillRect(0, 0, w, h);
  if (!L.umap) {
    ctx.fillStyle = css.getPropertyValue("--ink-3");
    ctx.font = "14px Inter";
    ctx.fillText("No map for this SAE yet: run layout.py.", 24, 40);
    $("#maplegend").innerHTML = "";
    return;
  }
  const { x, y } = L.umap;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < x.length; i++) {
    x0 = Math.min(x0, x[i]); x1 = Math.max(x1, x[i]); y0 = Math.min(y0, y[i]); y1 = Math.max(y1, y[i]);
  }
  const pad = 30;
  const s = Math.min((w - 2 * pad) / (x1 - x0), (h - 2 * pad) / (y1 - y0)) * mapView.scale;
  const px = new Float32Array(x.length), py = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    px[i] = w / 2 + (x[i] - (x0 + x1) / 2) * s + mapView.ox;
    py[i] = h / 2 + (y[i] - (y0 + y1) / 2) * s + mapView.oy;
  }
  mapPts = { px, py };
  const mode = $<HTMLSelectElement>("#mapcolor").value;
  const inRows = new Uint8Array(x.length);
  const filtered = state.rows.length !== x.length;
  for (const r of state.rows) inRows[r] = 1;
  const color = (i: number) => {
    if (mode === "freq") {
      const t = L.freq[i] ? Math.min(1, Math.max(0, (Math.log10(L.freq[i]) + 5) / 4)) : 0;
      return `hsl(${250 - 210 * t} 70% ${45 + 10 * t}%)`;
    }
    return L.category[i] ? CATEGORIES[L.category[i]]?.[0] ?? "#888" : css.getPropertyValue("--ink-3");
  };
  // background points first, then labeled / matching points on top
  const r0 = x.length > 40000 ? 1 : 1.4;
  for (const pass of [0, 1]) {
    for (let i = 0; i < x.length; i++) {
      const hi = filtered ? inRows[i] === 1 : !!L.label[i] || mode === "freq";
      if ((pass === 1) !== hi) continue;
      ctx.globalAlpha = hi ? 0.9 : filtered ? 0.08 : 0.25;
      ctx.fillStyle = hi || mode === "freq" ? color(i) : css.getPropertyValue("--ink-3");
      const r = hi && L.label[i] ? 2.6 : r0;
      ctx.fillRect(px[i] - r, py[i] - r, 2 * r, 2 * r);
    }
  }
  ctx.globalAlpha = 1;
  if (state.id >= 0) {
    ctx.strokeStyle = css.getPropertyValue("--ink");
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(px[state.id], py[state.id], 7, 0, Math.PI * 2);
    ctx.stroke();
  }
  $("#maplegend").innerHTML =
    mode === "freq"
      ? `<div>color: firing frequency, 10⁻⁵ (blue) → 10⁻¹ (yellow)</div>`
      : Object.entries(CATEGORIES).map(([k, [c, n]]) => `<div><span class="dot" style="background:${c}"></span>${n}</div>`).join("") +
        `<div class="muted">grey: unlabeled · scroll to zoom, drag to pan</div>`;
}

function nearestPoint(mx: number, my: number): number {
  if (!mapPts) return -1;
  let best = -1, bd = 64;
  for (let i = 0; i < mapPts.px.length; i++) {
    const d = (mapPts.px[i] - mx) ** 2 + (mapPts.py[i] - my) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function setupMap() {
  const canvas = $<HTMLCanvasElement>("#map");
  const tip = $("#maptip");
  let drag: { x: number; y: number; moved: boolean } | null = null;
  canvas.onmousedown = (e) => (drag = { x: e.clientX, y: e.clientY, moved: false });
  window.addEventListener("mouseup", (e) => {
    if (drag && !drag.moved && e.target === canvas) {
      const i = nearestPoint(e.offsetX, e.offsetY);
      if (i >= 0) selectFeature(i, false);
    }
    drag = null;
  });
  canvas.onmousemove = (e) => {
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.moved) {
        mapView.ox += dx; mapView.oy += dy;
        drag.x = e.clientX; drag.y = e.clientY;
        drawMap();
      }
      tip.style.display = "none";
      return;
    }
    const i = nearestPoint(e.offsetX, e.offsetY);
    const L = state.list!;
    if (i < 0) { tip.style.display = "none"; return; }
    tip.style.display = "block";
    tip.style.left = `${e.offsetX + 14}px`;
    tip.style.top = `${e.offsetY + 14}px`;
    tip.innerHTML = `<b>#${i}</b> ${L.label[i] ? esc(L.label[i]) : `<span class="muted">“${esc(showTok(L.top[i]))}”</span>`}<br><span class="muted">${fmtFreq(L.freq[i])} · → ${esc(L.promotes[i])}</span>`;
  };
  canvas.onmouseleave = () => (tip.style.display = "none");
  canvas.onwheel = (e) => {
    e.preventDefault();
    const k = Math.exp(-e.deltaY * 0.0015);
    const cx = canvas.clientWidth / 2, cy = canvas.clientHeight / 2;
    mapView.ox = (mapView.ox - (e.offsetX - cx)) * k + (e.offsetX - cx);
    mapView.oy = (mapView.oy - (e.offsetY - cy)) * k + (e.offsetY - cy);
    mapView.scale *= k;
    drawMap();
  };
  canvas.ondblclick = () => { mapView = { scale: 1, ox: 0, oy: 0 }; drawMap(); };
  $<HTMLSelectElement>("#mapcolor").onchange = drawMap;
  window.addEventListener("resize", () => state.tab === "map" && drawMap());
}

// ------------------------------------------------------------------ notes

function renderNotes() {
  const cur = state.saes.find((s) => s.key === state.sae);
  $("#notes").innerHTML = cur?.notes
    ? `<pre>${esc(cur.notes)}</pre>`
    : `<div class="empty">No notes for this SAE yet (notes.md next to features.json).</div>`;
}

// ------------------------------------------------------------------ navigation

function setTab(tab: string) {
  state.tab = tab;
  document.querySelectorAll<HTMLButtonElement>("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  $("#detail").hidden = tab !== "feature";
  $("#mapwrap").hidden = tab !== "map";
  $("#notes").hidden = tab !== "notes";
  if (tab === "map") requestAnimationFrame(drawMap);
  if (tab === "notes") renderNotes();
  if (tab === "feature") renderFeature();
  writeHash();
}

async function selectFeature(id: number, toFeature = true) {
  state.id = id;
  writeHash();
  renderList();
  if (state.tab === "map") drawMap();
  if (toFeature && state.tab !== "feature") setTab("feature");
  else if (state.tab === "feature") renderFeature();
}

async function selectSae(key: string, id = -1) {
  state.sae = key;
  state.id = id;
  state.cats.clear();
  mapView = { scale: 1, ox: 0, oy: 0 };
  renderHeader();
  $("#detail").innerHTML = `<div class="empty">Loading…</div>`;
  state.list = await getJson<List>(`/api/list?sae=${encodeURIComponent(key)}`);
  renderCats();
  filter();
  if (id >= 0) {
    const r = state.rows.indexOf(id);
    if (r >= 0) $("#list").scrollTop = Math.max(0, r * ROW - 100);
    renderList();
  }
  setTab(state.tab);
}

async function main() {
  state.saes = await getJson<SaeInfo[]>("/api/saes");
  if (!state.saes.length) {
    $("#detail").innerHTML = `<div class="empty">No SAEs with features yet. Run features.py on a checkpoint.</div>`;
    return;
  }
  const h = readHash();
  state.tab = h.tab;
  $("#q").oninput = filter;
  $("#sort").onchange = filter;
  $("#onlyLabeled").onchange = filter;
  $("#list").onscroll = () => requestAnimationFrame(renderList);
  document.querySelectorAll<HTMLButtonElement>("#tabs button").forEach((b) => (b.onclick = () => setTab(b.dataset.tab!)));
  setupMap();
  window.addEventListener("keydown", (e) => {
    if ((e.key === "j" || e.key === "k") && document.activeElement?.tagName !== "INPUT") {
      const r = state.rows.indexOf(state.id) + (e.key === "j" ? 1 : -1);
      if (r >= 0 && r < state.rows.length) {
        selectFeature(state.rows[r]);
        const el = $("#list");
        if (r * ROW < el.scrollTop || r * ROW > el.scrollTop + el.clientHeight - ROW) el.scrollTop = r * ROW - 100;
      }
    }
  });
  await selectSae(state.saes.some((s) => s.key === h.sae) ? h.sae : state.saes[0].key, h.id);
}

main();
