// Client for the latent explorer (bundled by Bun via index.html).

type Points = {
  dataset: string;
  model: string;
  split: string;
  classes: string[];
  ids: number[]; // stable image ids from the dataset's index.json; tags are keyed by these
  paths: string[];
  thumbs: string[];
  label: number[];
  clusterings: Record<string, { cluster: number[]; dist: number[] }>; // keyed by k, attached by the server
  projections: Record<string, number[]>; // flat [x0, y0, x1, y1, ...]
};
type Tags = Record<string, string[]>; // image id -> tags
type Kind = "labels" | "rejections"; // tags an image has / tags a human said it does NOT have
type TagStats = {
  threshold: number | null; // lowest score with CV precision >= target, null if none reached it
  metrics: { ap: number; precision: number; recall: number; at: number; folds: number; target_precision: number };
  n_pos: number;
  n_neg: number;
};
/** autotag.py's output for this model (null until it has been run), scores cut to these points. */
type Predictions = {
  version: string;
  command: string;
  tags: Record<string, TagStats>;
  skipped: Record<string, { n_pos: number; n_neg: number }>;
  scores: Record<string, Record<string, number>>; // image id -> tag -> score; scores < 0.05 left out
} | null;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const P: Points = await (await fetch("/api/points")).json();
let tags: Tags = await (await fetch("/api/labels")).json();
let rejections: Tags = await (await fetch("/api/rejections")).json();
const PRED: Predictions = await (await fetch("/api/predictions")).json();
const N = P.paths.length;
const PAGE = 120;
const key = (i: number) => String(P.ids[i]);
const tagsOf = (i: number) => tags[key(i)];
const rejectsOf = (i: number) => rejections[key(i)];
const scoreOf = (i: number, t: string) => PRED?.scores[key(i)]?.[t] ?? 0;
const mapOf = (kind: Kind) => (kind === "labels" ? tags : rejections);

// --- state ------------------------------------------------------------------

const projNames = Object.keys(P.projections);
let proj = projNames.includes("umap") ? "umap" : projNames[0]!;
const ks = Object.keys(P.clusterings).map(Number).sort((a, b) => a - b);
// Default to the k closest to 16: a few clusters are too coarse to browse, many too fine.
let k: number | null = ks.length ? ks.reduce((a, b) => (Math.abs(b - 16) < Math.abs(a - 16) ? b : a)) : null;
let cluster: number[] | null = null, dist: number[] | null = null; // for the current k
function setK(next: number | null) {
  k = next;
  cluster = k === null ? null : P.clusterings[k]!.cluster;
  dist = k === null ? null : P.clusterings[k]!.dist;
}
setK(k);
let colorBy = cluster ? "cluster" : "class";
let selection: number[] = []; // sorted for display
let selectedSet = new Set<number>();
let shown = PAGE;
let hoverIdx = -1; // point under the cursor
let gridHoverIdx = -1; // thumbnail under the cursor
let legendHover: number | null = null; // group under the cursor in the legend
let view = { scale: 1, tx: 0, ty: 0 };
let lasso: [number, number][] | null = null;
let tool: "pan" | "lasso" = "pan"; // what a plain drag does; shift-drag always lassos

// Fixed random draw order so no group is always painted on top.
const drawOrder = Array.from({ length: N }, (_, i) => i);
for (let i = N - 1, s = 1; i > 0; i--) {
  s = (s * 16807) % 2147483647;
  const j = s % (i + 1);
  [drawOrder[i], drawOrder[j]] = [drawOrder[j]!, drawOrder[i]!];
}

// --- canvas geometry --------------------------------------------------------

const canvas = $<HTMLCanvasElement>("canvas");
const ctx = canvas.getContext("2d")!;
const plot = $("plot");
let W = 0, H = 0, dpr = 1;
let fit = { s: 1, ox: 0, oy: 0 }; // data -> screen before the user's pan/zoom
const sx = new Float32Array(N), sy = new Float32Array(N); // screen coords, updated per render

function computeFit() {
  const xy = P.projections[proj]!;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < N; i++) {
    const x = xy[2 * i]!, y = xy[2 * i + 1]!;
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  const pad = 32;
  const s = Math.min((W - 2 * pad) / (x1 - x0 || 1), (H - 2 * pad) / (y1 - y0 || 1));
  fit = { s, ox: (W - s * (x0 + x1)) / 2, oy: (H + s * (y0 + y1)) / 2 }; // y flipped
}

function project() {
  const xy = P.projections[proj]!;
  for (let i = 0; i < N; i++) {
    sx[i] = (fit.ox + fit.s * xy[2 * i]!) * view.scale + view.tx;
    sy[i] = (fit.oy - fit.s * xy[2 * i + 1]!) * view.scale + view.ty;
  }
}

function resize() {
  dpr = window.devicePixelRatio || 1;
  W = plot.clientWidth;
  H = plot.clientHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  computeFit();
  render();
}

// --- grouping & color -------------------------------------------------------

type Grouping = { names: string[]; of: (i: number) => number; colors: string[] };
let theme: Record<string, string> = {};
function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  for (const v of ["--marker", "--point-muted", "--ink", "--sheet", "--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"])
    theme[v] = cs.getPropertyValue(v).trim();
}
const SERIES = () => [1, 2, 3, 4, 5, 6, 7, 8].map((i) => theme[`--s${i}`]!);

function grouping(): Grouping {
  if (colorBy.startsWith("tag:")) {
    const t = colorBy.slice(4);
    return {
      names: [t, "untagged"],
      of: (i) => (tagsOf(i)?.includes(t) ? 0 : 1),
      colors: [theme["--marker"]!, theme["--point-muted"]!],
    };
  }
  if (colorBy.startsWith("score:")) return scoreGrouping(colorBy.slice(6));
  return colorBy === "cluster" ? clusterGrouping() : classGrouping();
}
/** autotag.py's score for one tag in five bands, from the muted point colour up to --marker. */
function scoreGrouping(t: string): Grouping {
  const mix = (f: number) => {
    const rgb = (h: string) => [1, 3, 5].map((j) => parseInt(h.slice(j, j + 2), 16));
    const a = rgb(theme["--point-muted"]!), b = rgb(theme["--marker"]!);
    return `rgb(${a.map((v, j) => Math.round(v + (b[j]! - v) * f)).join(" ")})`;
  };
  return {
    names: ["score ≥ 0.8", "0.6 – 0.8", "0.4 – 0.6", "0.2 – 0.4", "< 0.2"],
    of: (i) => {
      const s = scoreOf(i, t), band = [0.8, 0.6, 0.4, 0.2].findIndex((e) => s >= e);
      return band < 0 ? 4 : band;
    },
    colors: [mix(1), mix(0.75), mix(0.5), mix(0.25), theme["--point-muted"]!],
  };
}
const clusterNames = () => Array.from({ length: k! }, (_, c) => `cluster ${c}`);
const clusterGrouping = (): Grouping => ({ names: clusterNames(), of: (i) => cluster![i]!, colors: groupColors(k!) });
const classGrouping = (): Grouping => ({ names: P.classes, of: (i) => P.label[i]!, colors: groupColors(P.classes.length) });

/** Up to 8 groups: the fixed palette. More: `n` hues spread evenly around the OKLCH
 * circle, cycling through three lightness levels so neighboring hues also differ in
 * lightness. Past ~12 groups some pairs get hard to tell apart; hover the legend to
 * isolate one. */
function groupColors(n: number): string[] {
  if (n <= 8) return SERIES().slice(0, n);
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  const levels = dark ? [0.72, 0.58, 0.84] : [0.62, 0.48, 0.76];
  return Array.from({ length: n }, (_, g) => {
    const hue = (250 + (360 * g) / n) % 360; // start at blue, like the palette
    return `oklch(${levels[g % 3]} 0.15 ${hue.toFixed(1)})`;
  });
}

// --- render -----------------------------------------------------------------

let raf = 0;
function render() {
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame(draw);
}

function draw() {
  project();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const g = grouping();
  const r = (N > 20000 ? 1.6 : 2.2) * Math.min(3, Math.sqrt(view.scale));

  // Batch points into one path per color.
  const muted = theme["--point-muted"]!;
  const buckets = new Map<string, number[]>();
  const front: number[] = [];
  for (const i of drawOrder) {
    const gi = g.of(i);
    if (legendHover !== null && gi !== legendHover) (buckets.get(muted) ?? buckets.set(muted, []).get(muted)!).push(i);
    else if (legendHover !== null) front.push(i);
    else {
      const c = g.colors[gi]!;
      (buckets.get(c) ?? buckets.set(c, []).get(c)!).push(i);
    }
  }
  const fill = (idx: number[], color: string, rad: number, alpha = 0.85) => {
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.beginPath();
    for (const i of idx) {
      const x = sx[i]!, y = sy[i]!;
      if (x < -5 || y < -5 || x > W + 5 || y > H + 5) continue;
      ctx.moveTo(x + rad, y);
      ctx.arc(x, y, rad, 0, 2 * Math.PI);
    }
    ctx.fill();
  };
  // Muted points first so colored groups sit on top. With a selection, everything else
  // fades back so the picked points read as one shape.
  const base = selection.length && legendHover === null ? 0.32 : 0.85;
  if (buckets.has(muted)) fill(buckets.get(muted)!, muted, r, legendHover !== null ? 0.5 : base);
  for (const [c, idx] of buckets) if (c !== muted) fill(idx, c, r, base);
  if (front.length) fill(front, g.colors[legendHover!]!, r * 1.3, 1);
  ctx.globalAlpha = 1;

  // Selection: an ink halo underneath, then the points again in their own colors, so a
  // dense selection gets an outline instead of turning into a solid blob.
  if (selection.length && legendHover === null) {
    fill(selection, theme["--ink"]!, r + 1.4, 1);
    const bySel = new Map<string, number[]>();
    for (const i of selection) {
      const c = g.colors[g.of(i)]!;
      (bySel.get(c) ?? bySel.set(c, []).get(c)!).push(i);
    }
    for (const [c, idx] of bySel) fill(idx, c, r, 1);
    ctx.globalAlpha = 1;
  }
  // Hovered point (canvas or grid) gets a big ring with a surface halo.
  for (const i of [hoverIdx, gridHoverIdx]) {
    if (i < 0) continue;
    ctx.lineWidth = 4;
    ctx.strokeStyle = theme["--sheet"]!;
    ctx.beginPath();
    ctx.arc(sx[i]!, sy[i]!, 7, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.lineWidth = 2;
    ctx.strokeStyle = theme["--ink"]!;
    ctx.stroke();
  }
  if (lasso && lasso.length > 1) {
    // Drawn like a grease-pencil loop: a heavy marker stroke over a faint wash.
    ctx.strokeStyle = theme["--marker"]!;
    ctx.fillStyle = theme["--marker"]!;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(...lasso[0]!);
    for (const p of lasso) ctx.lineTo(...p);
    ctx.closePath();
    ctx.globalAlpha = 0.08;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.stroke();
  }
}

// --- legend -----------------------------------------------------------------

function renderLegend() {
  const g = grouping();
  const counts = new Array(g.names.length).fill(0);
  for (let i = 0; i < N; i++) counts[g.of(i)]++;
  const el = $("legend");
  el.innerHTML = "";
  g.names.forEach((name, gi) => {
    const row = document.createElement("button");
    row.className = "row";
    row.title = `Select every image in ${name} (⌘/Ctrl adds to the selection)`;
    row.innerHTML = `<span class="sw"></span><span></span><span class="n">${fmt(counts[gi])}</span>`;
    (row.children[0] as HTMLElement).style.background = g.colors[gi]!;
    row.children[1]!.textContent = name;
    row.onmouseenter = row.onfocus = () => { legendHover = gi; render(); };
    row.onmouseleave = row.onblur = () => { legendHover = null; render(); };
    row.onclick = (e) => {
      const idx = [];
      for (let i = 0; i < N; i++) if (g.of(i) === gi) idx.push(i);
      select(idx, e.metaKey || e.ctrlKey);
    };
    el.appendChild(row);
  });
}

// --- selection & grid -------------------------------------------------------

function fmt(n: number) {
  return n.toLocaleString();
}

function select(idx: number[], add = false) {
  selectedSet = add ? new Set([...selectedSet, ...idx]) : new Set(idx);
  sortSelection();
}

function sortSelection() {
  const mode = $<HTMLSelectElement>("sort").value;
  selection = [...selectedSet];
  if (mode === "random") {
    for (let i = selection.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [selection[i], selection[j]] = [selection[j]!, selection[i]!];
    }
  } else if (mode === "index") {
    selection.sort((a, b) => a - b);
  } else if (dist) {
    const d = dist, sign = mode === "-dist" ? -1 : 1;
    selection.sort((a, b) => sign * (d[a]! - d[b]!));
  }
  shown = PAGE;
  $("grid-wrap").scrollTop = 0;
  renderSelection();
  render();
}

/** One stacked bar: every group's share of the selection, plus a key for the largest few. */
function compBar(title: string, g: Grouping, keyCount: number): HTMLElement {
  const counts = new Map<number, number>();
  for (const i of selection) counts.set(g.of(i), (counts.get(g.of(i)) ?? 0) + 1);
  const parts = [...counts].sort((a, b) => b[1] - a[1]);
  const pct = (n: number) => `${Math.round((100 * n) / selection.length)}%`;

  const wrap = document.createElement("div");
  const label = document.createElement("div");
  label.className = "label";
  label.innerHTML = `<span></span><span></span>`;
  label.children[0]!.textContent = title;
  label.children[1]!.textContent = parts.length === 1 ? "all one group" : `${parts.length} groups`;
  const bar = document.createElement("div");
  bar.className = "bar";
  const key = document.createElement("div");
  key.className = "key";
  parts.forEach(([gi, n], j) => {
    const seg = document.createElement("span");
    seg.style.flex = String(n);
    seg.style.background = g.colors[gi]!;
    seg.title = `${g.names[gi]}: ${fmt(n)} (${pct(n)})`;
    bar.appendChild(seg);
    if (j < keyCount) {
      const item = document.createElement("span");
      item.innerHTML = `<i></i><span></span><b></b>`;
      (item.children[0] as HTMLElement).style.background = g.colors[gi]!;
      item.children[1]!.textContent = g.names[gi]!;
      item.children[2]!.textContent = pct(n);
      key.appendChild(item);
    }
  });
  if (parts.length > keyCount) {
    const more = document.createElement("span");
    more.textContent = `+${parts.length - keyCount} more`;
    key.appendChild(more);
  }
  wrap.append(label, bar, key);
  return wrap;
}

function renderComposition() {
  const el = $("composition");
  el.innerHTML = "";
  el.className = selection.length ? "comp" : "";
  if (!selection.length) {
    el.innerHTML = `<p class="empty">Drag a lasso around points, click a point, or click a group in the legend. The images you pick appear here to inspect and tag.</p>`;
    return;
  }
  el.appendChild(compBar("Class", classGrouping(), 6));
  if (cluster) el.appendChild(compBar(`Clusters at k = ${k}`, clusterGrouping(), 5));
}

function renderSelectionHeader() {
  const n = selection.length;
  $("sel-count").textContent = n ? `${fmt(n)} ${n === 1 ? "image" : "images"}` : "";
  $("clear").hidden = !n;
  $("sort-ctl").hidden = n < 2;
  $<HTMLButtonElement>("tag-add").disabled = !n;
  $<HTMLButtonElement>("untag").disabled = !n;
  $<HTMLButtonElement>("reject").disabled = !n;
  $<HTMLInputElement>("tag-input").placeholder = n
    ? `Tag ${n === 1 ? "this image" : `these ${fmt(n)} images`}…`
    : "Select images to tag them";
  renderComposition();
}

function makeCell(i: number): HTMLElement {
  const cell = document.createElement("div");
  cell.className = "cell";
  cell.tabIndex = 0;
  cell.dataset.i = String(i);
  const info = [`#${P.ids[i]}  ${P.paths[i]!}`, `class: ${P.classes[P.label[i]!]}`];
  if (cluster) info.push(`cluster ${cluster[i]}, distance to centre ${dist![i]!.toFixed(3)}`);
  cell.title = info.join("\n") + "\nDouble-click or Enter: open the full image";
  cell.innerHTML =
    `<div class="frame"><img loading="lazy" src="/file/${encodeURI(P.thumbs[i]!)}" alt=""></div>` +
    `<div class="cap"><span class="id">#${P.ids[i]}</span><span class="t"></span><span class="r"></span></div>` +
    `<button class="x" title="Remove from selection" aria-label="Remove from selection">×</button>`;
  updateCaption(cell, i);
  const open = () => window.open(`/file/${encodeURI(P.paths[i]!)}`, "_blank");
  cell.onmouseenter = cell.onfocus = () => { gridHoverIdx = i; render(); };
  cell.onmouseleave = cell.onblur = () => { if (gridHoverIdx === i) { gridHoverIdx = -1; render(); } };
  cell.ondblclick = open;
  const remove = () => {
    const next = (cell.nextElementSibling ?? cell.previousElementSibling) as HTMLElement | null;
    selectedSet.delete(i);
    selection = selection.filter((j) => j !== i);
    gridHoverIdx = -1;
    cell.remove();
    next?.classList.contains("cell") && next.focus();
    renderSelectionHeader();
    render();
  };
  cell.onkeydown = (e) => {
    if (e.target !== cell) return;
    if (e.key === "Enter") open();
    else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); remove(); }
  };
  (cell.querySelector(".x") as HTMLElement).onclick = (e) => { e.stopPropagation(); remove(); };
  return cell;
}

/** Tags in --marker, rejected tags struck through after them. */
function updateCaption(cell: Element, i: number) {
  cell.querySelector(".t")!.textContent = (tagsOf(i) ?? []).join(", ");
  const r = cell.querySelector(".r")!;
  r.textContent = (rejectsOf(i) ?? []).join(", ");
  (r as HTMLElement).title = r.textContent ? `Not: ${r.textContent}` : "";
}

function renderSelection() {
  renderSelectionHeader();
  const grid = $("grid");
  grid.innerHTML = "";
  for (const i of selection.slice(0, shown)) grid.appendChild(makeCell(i));
  const more = $<HTMLButtonElement>("more");
  more.hidden = selection.length <= shown;
  more.textContent = `Show ${fmt(Math.min(PAGE, selection.length - shown))} more of ${fmt(selection.length - shown)}`;
}

function showMore() {
  const grid = $("grid");
  for (const i of selection.slice(shown, shown + PAGE)) grid.appendChild(makeCell(i));
  shown += PAGE;
  const more = $<HTMLButtonElement>("more");
  more.hidden = selection.length <= shown;
  more.textContent = `Show ${fmt(Math.min(PAGE, selection.length - shown))} more of ${fmt(selection.length - shown)}`;
}
$("more").onclick = showMore;
// Load the next page as the bottom of the sheet scrolls into view.
new IntersectionObserver((es) => { if (es[0]!.isIntersecting && !$("more").hidden) showMore(); }, { root: $("grid-wrap"), rootMargin: "300px" })
  .observe($("more"));
$("sort").onchange = sortSelection;
$("clear").onclick = () => select([]);
function renderSortOptions() {
  const sel = $<HTMLSelectElement>("sort");
  const prev = sel.value;
  // Without k-means distances, sorting by them is meaningless.
  const opts: [string, string][] = dist
    ? [["dist", "Closest to cluster centre"], ["-dist", "Farthest from centre"], ["index", "Dataset order"], ["random", "Shuffled"]]
    : [["index", "Dataset order"], ["random", "Shuffled"]];
  sel.innerHTML = opts.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  if (opts.some(([v]) => v === prev)) sel.value = prev;
}

// --- tags -------------------------------------------------------------------

function setStatus(html: string, error = false) {
  // The selection and review panes each have a status line; both show the same message.
  for (const el of document.querySelectorAll<HTMLElement>(".status")) {
    el.innerHTML = html;
    el.classList.toggle("error", error);
  }
}

// Edits are sent as {id: tags} patches per file; the server merges them into the shared
// files and returns the full map, which also brings in edits made in other sessions.
type Patch = Record<Kind, Tags>;
const KINDS: Kind[] = ["labels", "rejections"];
const emptyPatch = (): Patch => ({ labels: {}, rejections: {} });
let pending = emptyPatch();
let saveTimer = 0;
function saveTags(patch: Patch, done: string) {
  for (const kind of KINDS) Object.assign(pending[kind], patch[kind]);
  setStatus("Saving…");
  clearTimeout(saveTimer);
  saveTimer = window.setTimeout(() => flush(done), 300);
}

async function flush(done: string) {
  const body = pending;
  pending = emptyPatch();
  const server: Partial<Patch> = {};
  let error = "";
  // Each patch carries this session's state of both files for the images it touches, so the
  // server's "accepting drops the rejection" rule agrees with it. A file with nothing to send
  // is fetched instead, since a patch to the other one may have changed it on the server.
  for (const kind of KINDS) {
    const n = Object.keys(body[kind]).length;
    const res = await fetch(`/api/${kind}`, n ? { method: "PATCH", body: JSON.stringify(body[kind]) } : {}).catch(() => null);
    if (res?.ok) server[kind] = await res.json();
    else if (n) {
      pending[kind] = { ...body[kind], ...pending[kind] };
      error ||= res ? await res.text() : "the explorer server isn't reachable";
    }
  }
  applyServerTags(server);
  if (error) {
    setStatus(`Not saved: ${esc(error)}. Your edits are kept here; tag again to retry.`, true);
    return;
  }
  setStatus(done);
  if (undoPatch) {
    for (const el of document.querySelectorAll(".status")) {
      const btn = document.createElement("button");
      btn.className = "quiet";
      btn.textContent = "Undo";
      btn.style.marginLeft = "6px";
      btn.onclick = undo;
      el.appendChild(btn);
    }
  }
}

/** Replace local tags with the server's, keeping edits that haven't been sent yet. */
function applyServerTags(server: Partial<Patch>) {
  if (server.labels) tags = server.labels;
  if (server.rejections) rejections = server.rejections;
  for (const kind of KINDS) applyPatchLocally(kind, pending[kind]);
  refreshAfterTagChange();
}

function refreshAfterTagChange() {
  if (colorBy.startsWith("tag:") && !allTags().has(colorBy.slice(4))) colorBy = cluster ? "cluster" : "class";
  renderTags();
  renderControls();
  // Update captions (and review buttons) in place so the sheet doesn't lose its scroll position.
  for (const cell of document.querySelectorAll<HTMLElement>(".cell")) {
    const i = Number(cell.dataset.i);
    updateCaption(cell, i);
    if (cell.dataset.tag !== undefined) updateReviewCell(cell, i, cell.dataset.tag);
  }
  render();
}

// Pick up tags added in other sessions when coming back to this tab.
window.addEventListener("focus", async () => {
  const [l, r] = await Promise.all(KINDS.map((kind) => fetch(`/api/${kind}`).catch(() => null)));
  if (l?.ok && r?.ok) applyServerTags({ labels: await l.json(), rejections: await r.json() });
});

function allTags(): Map<string, number> {
  const counts = new Map<string, number>();
  for (let i = 0; i < N; i++) for (const t of tagsOf(i) ?? []) counts.set(t, (counts.get(t) ?? 0) + 1);
  return counts;
}

let colorBeforeTag = colorBy; // restored when a pressed tag chip is clicked again
function renderTags() {
  const counts = allTags();
  const list = $("tag-list");
  list.innerHTML = "";
  $("tag-options").innerHTML = "";
  $("tags-title").textContent = counts.size ? `Tags in ${P.dataset}` : "Tags";
  for (const [t, n] of [...counts].sort((a, b) => b[1] - a[1])) {
    const chip = document.createElement("button");
    chip.className = "chip";
    const on = colorBy === `tag:${t}`;
    chip.setAttribute("aria-pressed", String(on));
    chip.innerHTML = `<span></span><span class="n">${fmt(n)}</span>`;
    chip.children[0]!.textContent = t;
    chip.title = on ? "Back to the previous coloring" : `Select the ${fmt(n)} images tagged “${t}” and highlight them`;
    chip.onclick = () => {
      if (on) {
        colorBy = colorBeforeTag;
      } else {
        if (!colorBy.startsWith("tag:")) colorBeforeTag = colorBy;
        colorBy = `tag:${t}`;
        $<HTMLInputElement>("tag-input").value = t;
        select(P.ids.flatMap((_, i) => (tagsOf(i)?.includes(t) ? [i] : [])));
      }
      renderTags();
      renderControls();
      render();
    };
    list.appendChild(chip);
    const opt = document.createElement("option");
    opt.value = t;
    $("tag-options").appendChild(opt);
  }
  if (!counts.size) list.innerHTML = `<span class="empty" style="margin:0">None yet. Select images, type a name above, and press Tag.</span>`;
}

let undoPatch: Patch | null = null;
function applyPatchLocally(kind: Kind, patch: Tags) {
  const m = mapOf(kind);
  for (const [id, t] of Object.entries(patch)) { if (t.length) m[id] = t; else delete m[id]; }
}

/** accept: tag it (and drop a rejection); reject: mark it as NOT having the tag (and drop
 * the tag); untag: drop the tag; clear: drop both. */
type Action = "accept" | "reject" | "untag" | "clear";

/** Apply `action` for tag `t` to images `idx` in both files, locally and on the server.
 * Returns how many images changed; `done(n)` is the status message once saved. */
function changeTag(idx: number[], t: string, action: Action, done: (n: number) => string): number {
  const patch = emptyPatch(), before = emptyPatch();
  let n = 0;
  for (const i of idx) {
    const id = key(i);
    const lab = new Set(tags[id] ?? []), rej = new Set(rejections[id] ?? []);
    const l0 = lab.has(t), r0 = rej.has(t);
    if (action === "accept") { lab.add(t); rej.delete(t); }
    else if (action === "reject") { rej.add(t); lab.delete(t); }
    else { lab.delete(t); if (action === "clear") rej.delete(t); }
    if (lab.has(t) !== l0) { before.labels[id] = [...(tags[id] ?? [])]; patch.labels[id] = [...lab]; }
    if (rej.has(t) !== r0) { before.rejections[id] = [...(rejections[id] ?? [])]; patch.rejections[id] = [...rej]; }
    if (lab.has(t) !== l0 || rej.has(t) !== r0) n++;
  }
  if (!n) return 0;
  for (const kind of KINDS) applyPatchLocally(kind, patch[kind]);
  undoPatch = before;
  refreshAfterTagChange();
  saveTags(patch, done(n));
  return n;
}

const images = (n: number) => `${fmt(n)} ${n === 1 ? "image" : "images"}`;

function editTags(action: "accept" | "untag" | "reject") {
  const input = $<HTMLInputElement>("tag-input");
  const t = input.value.trim();
  if (!selection.length) return;
  if (!t) { input.focus(); setStatus("Type a tag name first."); return; }
  const q = `“${esc(t)}”`;
  const n = changeTag(selection, t, action, (n) =>
    action === "accept" ? `Tagged ${images(n)} as ${q}.`
    : action === "untag" ? `Untagged ${images(n)} from ${q}.`
    : `Marked ${images(n)} as not ${q}.`);
  if (!n)
    setStatus(action === "accept" ? `All selected images are already tagged ${q}.`
      : action === "untag" ? `None of the selected images are tagged ${q}.`
      : `All selected images are already marked as not ${q}.`);
}

function undo() {
  if (!undoPatch) return;
  const patch = undoPatch;
  undoPatch = null;
  for (const kind of KINDS) applyPatchLocally(kind, patch[kind]);
  refreshAfterTagChange();
  saveTags(patch, "Undone.");
}

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
$("tag-form").onsubmit = (e) => { e.preventDefault(); editTags("accept"); };
$("untag").onclick = () => editTags("untag");
$("reject").onclick = () => editTags("reject");

// --- review (autotag.py suggestions) ---------------------------------------

let mode: "select" | "review" = "select";
let reviewTag: string | null = PRED ? Object.keys(PRED.tags).sort()[0] ?? null : null;
let likely: number[] = [], uncertain: number[] = [];
const REVIEW_PAGE = 48;
const reviewShown = { likely: REVIEW_PAGE, uncertain: REVIEW_PAGE };
const pct = (x: number) => `${Math.round(100 * x)}%`;

function setMode(next: "select" | "review") {
  mode = next;
  $("select-pane").hidden = mode !== "select";
  $("review-pane").hidden = mode !== "review";
  $("tab-select").setAttribute("aria-selected", String(mode === "select"));
  $("tab-review").setAttribute("aria-selected", String(mode === "review"));
  if (mode === "review") renderReview();
  savePrefs();
}
$("tab-select").onclick = () => setMode("select");
$("tab-review").onclick = () => setMode("review");

/** Images without a human decision on `reviewTag`: (a) likely, at or above the tag's
 * threshold, highest first; (b) uncertain, the rest ordered by closeness to 0.5. */
function computeReview() {
  likely = [];
  uncertain = [];
  reviewShown.likely = reviewShown.uncertain = REVIEW_PAGE;
  if (!PRED || reviewTag === null) return;
  const t = reviewTag, thr = PRED.tags[t]!.threshold;
  for (let i = 0; i < N; i++) {
    const s = PRED.scores[key(i)]?.[t];
    if (s === undefined || tagsOf(i)?.includes(t) || rejectsOf(i)?.includes(t)) continue;
    (thr !== null && s >= thr ? likely : uncertain).push(i);
  }
  likely.sort((a, b) => scoreOf(b, t) - scoreOf(a, t));
  uncertain.sort((a, b) => Math.abs(scoreOf(a, t) - 0.5) - Math.abs(scoreOf(b, t) - 0.5));
}

function renderReview() {
  const sel = $<HTMLSelectElement>("review-tag");
  const names = PRED ? Object.keys(PRED.tags).sort() : [];
  sel.innerHTML = names.map((t) => `<option value="${esc(t)}"></option>`).join("");
  [...sel.options].forEach((o, j) => (o.textContent = names[j]!));
  if (reviewTag !== null) sel.value = reviewTag;
  sel.disabled = !names.length;
  $("review-lists").hidden = !names.length;
  $<HTMLButtonElement>("review-color").disabled = !names.length;

  const meta = $("review-meta");
  meta.innerHTML = "";
  const para = (html: string, cls = "") => {
    const el = document.createElement("p");
    el.innerHTML = html;
    if (cls) el.className = cls;
    meta.appendChild(el);
    return el;
  };
  const skipped = PRED ? Object.entries(PRED.skipped) : [];
  if (!PRED) {
    para(`No suggestions yet for this model. Tag at least 5 images with a tag, mark at least 5 as not having it
      (select them and press Reject), then run <code>uv run projects/latent_analysis/autotag.py --dataset ${esc(P.dataset)}</code>
      with the same <code>--model</code> and <code>--image-size</code> as this embedding, and reload this page.`, "empty");
  } else if (reviewTag !== null) {
    const st = PRED.tags[reviewTag]!, m = st.metrics;
    para(`<b>${fmt(st.n_pos)}</b> tagged, <b>${fmt(st.n_neg)}</b> rejected · cross-validated (${m.folds} folds): AP <b>${m.ap.toFixed(2)}</b>`);
    if (st.threshold !== null)
      para(`Precision <b>${pct(m.precision)}</b>, recall <b>${pct(m.recall)}</b> at score ≥ <b>${st.threshold.toFixed(3)}</b>`);
    else
      para(`No threshold reached ${pct(m.target_precision)} precision (at 0.5: precision ${pct(m.precision)}, recall ${pct(m.recall)}),
        so nothing is listed as likely. Accept or reject the uncertain images to give it more examples.`, "warn");
  } else {
    para(`No tag has enough examples for a probe yet.`, "empty");
  }
  if (skipped.length) {
    const el = para("", "hint");
    el.textContent = `Too few examples: ${skipped.map(([t, c]) => `${t} (${c.n_pos} tagged, ${c.n_neg} rejected)`).join(", ")}.`;
  }
  if (PRED)
    para(`Suggestions from ${esc(PRED.version)}. After reviewing, re-run <code>${esc(PRED.command)}</code> and reload this page.`, "hint");

  computeReview();
  renderReviewList("likely");
  renderReviewList("uncertain");
}

function renderReviewList(which: "likely" | "uncertain") {
  const all = which === "likely" ? likely : uncertain;
  const grid = $(which);
  grid.innerHTML = "";
  for (const i of all.slice(0, reviewShown[which])) grid.appendChild(makeReviewCell(i, reviewTag!));
  $(`${which}-count`).textContent = all.length ? fmt(all.length) : "";
  $(`${which}-empty`).hidden = !!all.length;
  const more = $<HTMLButtonElement>(`${which}-more`);
  more.hidden = all.length <= reviewShown[which];
  more.textContent = `Show ${fmt(Math.min(REVIEW_PAGE, all.length - reviewShown[which]))} more of ${fmt(all.length - reviewShown[which])}`;
  if (which === "likely") $<HTMLButtonElement>("accept-all").disabled = !all.length;
}
for (const which of ["likely", "uncertain"] as const)
  $(`${which}-more`).onclick = () => { reviewShown[which] += REVIEW_PAGE; renderReviewList(which); };

function makeReviewCell(i: number, t: string): HTMLElement {
  const cell = document.createElement("div");
  cell.className = "cell review";
  cell.tabIndex = 0;
  cell.dataset.i = String(i);
  cell.dataset.tag = t;
  cell.title = `#${P.ids[i]}  ${P.paths[i]!}\nclass: ${P.classes[P.label[i]!]}\nY: yes, N: no, Enter: open the full image`;
  cell.innerHTML =
    `<div class="frame"><img loading="lazy" src="/file/${encodeURI(P.thumbs[i]!)}" alt=""><span class="score">${scoreOf(i, t).toFixed(2)}</span></div>` +
    `<div class="cap"><span class="id">#${P.ids[i]}</span><span class="t"></span><span class="r"></span></div>` +
    `<div class="acts"><button class="yes" title="It is “${esc(t)}” (Y)">Yes</button><button class="no" title="It is not “${esc(t)}” (N)">No</button></div>`;
  const decide = (action: "accept" | "reject") => {
    const on = action === "accept" ? tagsOf(i)?.includes(t) : rejectsOf(i)?.includes(t);
    const q = `“${esc(t)}”`;
    changeTag([i], t, on ? "clear" : action, () =>
      on ? `Cleared #${P.ids[i]}.` : action === "accept" ? `Tagged #${P.ids[i]} as ${q}.` : `Marked #${P.ids[i]} as not ${q}.`);
  };
  (cell.querySelector(".yes") as HTMLElement).onclick = () => decide("accept");
  (cell.querySelector(".no") as HTMLElement).onclick = () => decide("reject");
  cell.onmouseenter = cell.onfocus = () => { gridHoverIdx = i; render(); };
  cell.onmouseleave = cell.onblur = () => { if (gridHoverIdx === i) { gridHoverIdx = -1; render(); } };
  cell.ondblclick = () => window.open(`/file/${encodeURI(P.paths[i]!)}`, "_blank");
  cell.onkeydown = (e) => {
    if (e.target !== cell || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "Enter") window.open(`/file/${encodeURI(P.paths[i]!)}`, "_blank");
    else if (e.key === "y" || e.key === "n") {
      e.stopPropagation();
      decide(e.key === "y" ? "accept" : "reject");
      (cell.nextElementSibling as HTMLElement | null)?.focus();
    }
  };
  updateCaption(cell, i);
  updateReviewCell(cell, i, t);
  return cell;
}

/** Reviewed images stay in place (dimmed, with their answer pressed) until the list is rebuilt. */
function updateReviewCell(cell: HTMLElement, i: number, t: string) {
  const yes = !!tagsOf(i)?.includes(t), no = !!rejectsOf(i)?.includes(t);
  cell.classList.toggle("done", yes || no);
  cell.querySelector(".yes")!.setAttribute("aria-pressed", String(yes));
  cell.querySelector(".no")!.setAttribute("aria-pressed", String(no));
}

$("review-tag").onchange = (e) => {
  reviewTag = (e.target as HTMLSelectElement).value;
  if (colorBy.startsWith("score:")) { colorBy = `score:${reviewTag}`; renderControls(); render(); }
  renderReview();
  savePrefs();
};
$("review-refresh").onclick = renderReview;
$("review-color").onclick = () => {
  if (reviewTag === null) return;
  colorBy = `score:${reviewTag}`;
  renderControls();
  renderTags();
  render();
};
$("accept-all").onclick = () => {
  if (reviewTag === null) return;
  const t = reviewTag;
  // Only the ones still without an answer: a "No" given in this list stays a no.
  const open = likely.slice(0, reviewShown.likely).filter((i) => !tagsOf(i)?.includes(t) && !rejectsOf(i)?.includes(t));
  changeTag(open, t, "accept", (n) => `Tagged ${images(n)} as “${esc(t)}”.`);
};

// --- controls ---------------------------------------------------------------

// Remember view choices per points file, so a refresh (e.g. to pick up a new k) keeps them.
const PREFS = `latent-explorer:${P.dataset}/${P.model}/${P.split}`;
function savePrefs() {
  try { localStorage.setItem(PREFS, JSON.stringify({ proj, k, colorBy, sort: $<HTMLSelectElement>("sort").value, mode, reviewTag })); } catch {}
}
function loadPrefs() {
  try {
    const p = JSON.parse(localStorage.getItem(PREFS) ?? "{}");
    if (projNames.includes(p.proj)) proj = p.proj;
    if (ks.includes(p.k)) setK(p.k);
    if (p.colorBy === "class" || (p.colorBy === "cluster" && cluster) || (p.colorBy?.startsWith("tag:") && allTags().has(p.colorBy.slice(4))) ||
        (p.colorBy?.startsWith("score:") && PRED?.tags[p.colorBy.slice(6)]))
      colorBy = p.colorBy;
    if (PRED?.tags[p.reviewTag]) reviewTag = p.reviewTag;
    return { sort: p.sort as string | undefined, mode: p.mode === "review" ? "review" as const : "select" as const };
  } catch { return { sort: undefined, mode: "select" as const }; }
}

function renderControls() {
  const pSel = $<HTMLSelectElement>("projection");
  const label = (m: string) => m.replace(/^tsne_p(\d+)$/, "t-SNE (perplexity $1)").replace(/^(umap|pca)$/, (x) => x.toUpperCase());
  pSel.innerHTML = projNames.map((m) => `<option value="${m}"></option>`).join("");
  [...pSel.options].forEach((o) => (o.textContent = label(o.value)));
  pSel.value = proj;
  const kSel = $<HTMLSelectElement>("k");
  kSel.innerHTML = ks.map((x) => `<option value="${x}">k = ${x}</option>`).join("");
  kSel.value = String(k);
  kSel.disabled = ks.length < 2;
  $("k-ctl").hidden = !ks.length;
  const cSel = $<HTMLSelectElement>("color-by");
  const opts = [
    ...(cluster ? ["cluster"] : []), "class",
    ...[...allTags().keys()].map((t) => `tag:${t}`),
    ...Object.keys(PRED?.tags ?? {}).sort().map((t) => `score:${t}`),
  ];
  cSel.innerHTML = opts.map(() => `<option></option>`).join("");
  [...cSel.options].forEach((o, j) => {
    o.value = opts[j]!;
    o.textContent = o.value.startsWith("tag:") ? `Tag: ${o.value.slice(4)}`
      : o.value.startsWith("score:") ? `Suggested: ${o.value.slice(6)}`
      : o.value === "cluster" ? "Cluster" : "Class";
  });
  cSel.value = colorBy;
  renderLegend();
  savePrefs();
}
$("projection").onchange = (e) => {
  proj = (e.target as HTMLSelectElement).value;
  view = { scale: 1, tx: 0, ty: 0 };
  computeFit();
  render();
  savePrefs();
};
$("k").onchange = (e) => {
  setK(Number((e.target as HTMLSelectElement).value));
  renderControls();
  sortSelection(); // re-sorts by the new distances and refreshes composition + plot
};
$("color-by").onchange = (e) => {
  colorBy = (e.target as HTMLSelectElement).value;
  renderLegend();
  renderTags();
  render();
  savePrefs();
};

$("export").onclick = () => {
  const esc = (s: string) => (/[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s);
  const rows = [["id", "path", "class", "cluster", "dist", "tags", "rejected"].join(",")];
  for (let i = 0; i < N; i++)
    rows.push([P.ids[i], esc(P.paths[i]!), P.classes[P.label[i]!], cluster?.[i] ?? "", dist?.[i] ?? "",
      esc((tagsOf(i) ?? []).join("|")), esc((rejectsOf(i) ?? []).join("|"))].join(","));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv" }));
  a.download = `${P.dataset}_${P.split}${k ? `_k${k}` : ""}_${P.model}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
};

// --- plot tools -------------------------------------------------------------

function setTool(next: "pan" | "lasso") {
  tool = next;
  plot.dataset.tool = next;
  document.querySelectorAll<HTMLButtonElement>("#tools .tool").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.tool === next)));
}
document.querySelectorAll<HTMLButtonElement>("#tools .tool").forEach((b) => (b.onclick = () => setTool(b.dataset.tool as "pan" | "lasso")));

function zoomAt(f: number, x = W / 2, y = H / 2) {
  const scale = Math.min(200, Math.max(0.5, view.scale * f));
  const r = scale / view.scale;
  view = { scale, tx: x - (x - view.tx) * r, ty: y - (y - view.ty) * r };
  render();
}
const fitView = () => { view = { scale: 1, tx: 0, ty: 0 }; render(); };
$("zoom-in").onclick = () => zoomAt(1.5);
$("zoom-out").onclick = () => zoomAt(1 / 1.5);
$("fit").onclick = fitView;

function toggleKeys(open = $("keys").hidden) {
  $("keys").hidden = !open;
  $("keys-btn").setAttribute("aria-expanded", String(open));
  $("keys-btn").setAttribute("aria-pressed", String(open));
}
$("keys-btn").onclick = () => toggleKeys();

// --- pointer interaction ----------------------------------------------------

const tooltip = $("tooltip");
function nearest(x: number, y: number, maxPx = 8): number {
  let best = -1, bd = maxPx * maxPx;
  for (let i = 0; i < N; i++) {
    const dx = sx[i]! - x, dy = sy[i]! - y, d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function showTooltip(i: number, x: number, y: number) {
  if (i < 0) { tooltip.style.display = "none"; return; }
  const img = tooltip.querySelector("img")!;
  const src = `/file/${encodeURI(P.thumbs[i]!)}`;
  if (img.getAttribute("src") !== src) img.setAttribute("src", src);
  const info = tooltip.querySelector(".info")!;
  info.textContent = "";
  const line = (text: string, cls?: string) => {
    const d = document.createElement("div");
    d.textContent = text;
    if (cls) d.className = cls;
    info.appendChild(d);
  };
  line(`#${P.ids[i]}  ${P.classes[P.label[i]!]}`, "name");
  if (cluster) line(`Cluster ${cluster[i]}, distance ${dist![i]!.toFixed(3)}`);
  const t = tagsOf(i);
  if (t?.length) line(t.join(", "), "tagged");
  const r = rejectsOf(i);
  if (r?.length) line(`not: ${r.join(", ")}`);
  if (colorBy.startsWith("score:")) line(`suggested ${colorBy.slice(6)}: ${scoreOf(i, colorBy.slice(6)).toFixed(3)}`);
  tooltip.style.display = "block";
  const tw = 196, th = tooltip.offsetHeight;
  tooltip.style.left = `${x + 16 + tw > W ? x - tw - 8 : x + 16}px`;
  tooltip.style.top = `${Math.max(4, Math.min(H - th - 4, y - th / 2))}px`;
}

function inPolygon(x: number, y: number, poly: [number, number][]) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]!, [xj, yj] = poly[j]!;
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

let drag: { x: number; y: number; tx: number; ty: number; moved: boolean } | null = null;
canvas.onpointerdown = (e) => {
  canvas.setPointerCapture(e.pointerId);
  if (e.shiftKey || tool === "lasso") lasso = [[e.offsetX, e.offsetY]];
  else drag = { x: e.offsetX, y: e.offsetY, tx: view.tx, ty: view.ty, moved: false };
};
canvas.onpointermove = (e) => {
  const x = e.offsetX, y = e.offsetY;
  plot.classList.toggle("shift", e.shiftKey);
  if (lasso) {
    lasso.push([x, y]);
    tooltip.style.display = "none";
    render();
    return;
  }
  if (drag) {
    if (Math.hypot(x - drag.x, y - drag.y) > 3) drag.moved = true;
    if (drag.moved) {
      plot.classList.add("dragging");
      view.tx = drag.tx + x - drag.x;
      view.ty = drag.ty + y - drag.y;
      tooltip.style.display = "none";
      render();
      return;
    }
  }
  const i = nearest(x, y);
  if (i !== hoverIdx) { hoverIdx = i; render(); }
  showTooltip(i, x, y);
};
canvas.onpointerup = (e) => {
  const add = e.metaKey || e.ctrlKey;
  plot.classList.remove("dragging");
  if (lasso) {
    const poly = lasso;
    lasso = null;
    if (poly.length > 2) {
      const idx: number[] = [];
      for (let i = 0; i < N; i++) if (inPolygon(sx[i]!, sy[i]!, poly)) idx.push(i);
      select(idx, add);
    } else {
      // A click (no drag) with the lasso tool still selects the point under it.
      const i = nearest(e.offsetX, e.offsetY);
      if (i >= 0) select([i], add); else render();
    }
  } else if (drag && !drag.moved) {
    const i = nearest(e.offsetX, e.offsetY);
    if (i >= 0) {
      if (add && selectedSet.has(i)) { selectedSet.delete(i); sortSelection(); }
      else select([i], add);
    }
  }
  drag = null;
};
canvas.onpointerleave = () => { hoverIdx = -1; tooltip.style.display = "none"; render(); };
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  zoomAt(Math.exp(-e.deltaY * 0.0015), e.offsetX, e.offsetY);
}, { passive: false });
canvas.ondblclick = fitView;

window.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName ?? "");
  if (e.key === "Shift") plot.classList.add("shift");
  if (e.key === "Escape") {
    if (!$("keys").hidden) return toggleKeys(false);
    if (typing) return (document.activeElement as HTMLElement).blur();
    return select([]);
  }
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  if ((e.key === "z" || e.key === "Z") && undoPatch) undo();
  else if (e.key === "v" || e.key === "h") setTool("pan");
  else if (e.key === "l") setTool("lasso");
  else if (e.key === "0") fitView();
  else if (e.key === "+" || e.key === "=") zoomAt(1.5);
  else if (e.key === "-") zoomAt(1 / 1.5);
  else if (e.key === "?") toggleKeys();
  else if (e.key === "t" && selection.length) { e.preventDefault(); $("tag-input").focus(); }
});
window.addEventListener("keyup", (e) => { if (e.key === "Shift") plot.classList.remove("shift"); });

// --- boot -------------------------------------------------------------------

{
  const meta = $("meta");
  meta.innerHTML = `<b></b> ${P.split}, ${fmt(N)} images, <span></span>`;
  meta.children[0]!.textContent = P.dataset;
  meta.children[1]!.textContent = P.model;
  meta.title = `${P.dataset}/${P.split}, embedded with ${P.model}`;
}
readTheme();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readTheme(); renderLegend(); render(); });
const saved = loadPrefs();
renderSortOptions();
if (saved.sort && [...$<HTMLSelectElement>("sort").options].some((o) => o.value === saved.sort)) $<HTMLSelectElement>("sort").value = saved.sort;
renderControls();
renderTags();
renderSelection();
setMode(saved.mode);
new ResizeObserver(resize).observe(plot);
