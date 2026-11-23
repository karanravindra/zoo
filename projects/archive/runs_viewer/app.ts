// Client for the runs viewer (bundled by Bun via index.html).
//
// A workspace is one run group (the parent path of the runs, e.g. pixel_art/ae), like a
// project. Every run in it has a fixed color and an eye toggle in the sidebar; the charts
// show the visible runs. The Workspace tab has a panel per metrics.csv column, grouped in
// sections by prefix (train_/val_/test_), plus custom panels built in the panel editor
// (per-step or per-run data, line/scatter/bar). The Runs table tab lists summary metrics
// and config per run, and clicking a run opens its page. Everything polls and redraws live.

// --- types ----------------------------------------------------------------------

type Status = "done" | "running" | "stopped";
type RunSummary = {
  path: string;
  group: string;
  name: string;
  mtime: number;
  status: Status;
  config: Record<string, unknown> | null;
  meta: Record<string, unknown>;
  results: Record<string, unknown> | null;
};
type Metrics = { columns: string[]; rows: (number | null)[][] };
type RunDetail = RunSummary & { metrics: Metrics | null };

type Source = "rows" | "runs"; // one point per metrics.csv row, or one per run
type PlotType = "line" | "scatter" | "bar";
type Panel = {
  id: string; // "m:<column>" for metric panels, "c:<n>" for custom ones
  title?: string;
  source: Source;
  type: PlotType;
  x: string; // "" = the workspace x-axis (rows) / run name (runs)
  y: string[];
  logX?: boolean;
  logY?: boolean;
};
type View = { tab: "charts" } | { tab: "table" } | { tab: "run"; path: string };

// --- helpers ----------------------------------------------------------------------

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const PALETTE = ["#5387dd", "#e0605b", "#3aa57a", "#9a6fd1", "#e8a33d", "#d55a9a", "#2aa3b8", "#a97d50", "#7d9a2e", "#c4569b", "#5f6fc9", "#de7b38"];
const DASHES = ["", "6 3", "2 2.5", "9 3 2 3"];
const POLL_MS = 3000;

function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
}
function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {}
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (!isFinite(v)) return String(v);
    if (Number.isInteger(v) && Math.abs(v) < 1e7) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
    return String(Number(v.toPrecision(4)));
  }
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
function ago(ms: number): string {
  const s = (Date.now() - ms) / 1000;
  if (s < 60) return `${Math.max(0, Math.round(s))}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return new Date(ms).toLocaleDateString();
}
/** Whether lower is better for a metric name (null: no opinion). */
function lowerIsBetter(name: string): boolean | null {
  if (/epoch|step|time|rows/i.test(name)) return null;
  if (/loss|mse|mae|err|fid|lpips|nll|bpd|perplex/i.test(name)) return true;
  if (/acc|psnr|ssim|auc|(^|_)ap($|_)|f1|precision|recall|iou|score/i.test(name)) return false;
  return null;
}
function flatten(obj: unknown, prefix: string, out: Record<string, unknown>) {
  if (obj && typeof obj === "object" && !Array.isArray(obj)) {
    for (const [k, v] of Object.entries(obj)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  } else if (prefix) out[prefix] = obj;
}

const ICON = {
  eye: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M1 8s2.6-5 7-5 7 5 7 5-2.6 5-7 5-7-5-7-5z"/><circle cx="8" cy="8" r="2.1"/></svg>`,
  eyeOff: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M1 8s2.6-5 7-5 7 5 7 5-2.6 5-7 5-7-5-7-5z" opacity=".45"/><path d="M2 2l12 12"/></svg>`,
  full: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"/></svg>`,
  edit: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M11 2.5l2.5 2.5-8 8H3v-2.5z"/></svg>`,
  trash: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M3 4h10M6.5 4V2.5h3V4M4.5 4l.8 9.5h5.4l.8-9.5"/></svg>`,
  chev: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M6 3l5 5-5 5"/></svg>`,
  reset: `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 8a5 5 0 1 0 1.5-3.5M3 2.5v2.5h2.5"/></svg>`,
};

// --- state ------------------------------------------------------------------------

let runs: RunSummary[] = [];
let group = "";
let view: View = { tab: "charts" };
const details = new Map<string, RunDetail>();
const colors = new Map<string, string>();
const hidden = new Set<string>(load<string[]>("rv:hidden", []));
const collapsed = new Set<string>(load<string[]>("rv:collapsed", []));
let custom: Record<string, Panel[]> = load("rv:custom", {}); // per group
let prefs: Record<string, { logY?: boolean }> = load("rv:prefs", {}); // metric panel overrides
let smoothing: number = load("rv:smoothing", 0);
let xChoice: string = load("rv:x", ""); // "" = auto
let runQuery = "";
let panelQuery = "";
let sort = { key: "mtime", asc: false };
let tableAllConfig = false;
let hoverRun: string | null = null;
const zooms = new Map<string, [number, number]>();
let drawers: (() => void)[] = []; // redraw every chart on the page
let pageSig = "";
let uid = 0;

const groups = () => [...new Set(runs.map((r) => r.group))].sort();
const inGroup = () => runs.filter((r) => r.group === group);
function matches(r: RunSummary): boolean {
  if (!runQuery) return true;
  const q = runQuery.toLowerCase();
  return r.name.toLowerCase().includes(q) || JSON.stringify(r.config ?? {}).toLowerCase().includes(q);
}
/** Runs in the workspace that pass the search (hidden ones included). */
const shownRuns = () => inGroup().filter(matches);
const detailsOf = (rs: RunSummary[]) => rs.map((r) => details.get(r.path)).filter((d): d is RunDetail => !!d);
/** What the charts plot: searched and not hidden. */
const visibleDetails = () => detailsOf(shownRuns().filter((r) => !hidden.has(r.path)));
const colorOf = (path: string) => colors.get(path) ?? "#888";

function assignColors() {
  // Stable per workspace: by path, so timestamped runs keep their color as new ones arrive.
  for (const g of groups()) runs.filter((r) => r.group === g).map((r) => r.path).sort()
    .forEach((p, i) => colors.set(p, PALETTE[i % PALETTE.length]!));
}

function runRecord(d: RunDetail): Record<string, unknown> {
  const out: Record<string, unknown> = { run: d.name, status: d.status, mtime: d.mtime };
  flatten(d.config, "config", out);
  flatten(d.results, "results", out);
  flatten(d.meta, "meta", out);
  const m = d.metrics;
  if (m) {
    out.rows = m.rows.length;
    m.columns.forEach((c, j) => {
      const vals = m.rows.map((r) => r[j]).filter((v): v is number => v !== null && isFinite(v));
      if (!vals.length) return;
      out[`final.${c}`] = vals[vals.length - 1];
      out[`min.${c}`] = Math.min(...vals);
      out[`max.${c}`] = Math.max(...vals);
    });
  }
  return out;
}

// --- data loading -------------------------------------------------------------------

async function fetchDetail(path: string): Promise<RunDetail | null> {
  const res = await fetch(`/api/run/${path.split("/").map(encodeURIComponent).join("/")}`);
  return res.ok ? res.json() : null;
}

/** Load the workspace's runs that are new or changed since last fetched. */
async function syncDetails(): Promise<boolean> {
  const stale = inGroup().filter((r) => {
    const d = details.get(r.path);
    return !d || d.mtime !== r.mtime || d.status !== r.status;
  });
  await Promise.all(stale.map(async (r) => {
    const d = await fetchDetail(r.path);
    if (d) details.set(r.path, d);
  }));
  return stale.length > 0;
}

/** Re-scan runs; returns whether anything changed (and the page was refreshed). */
async function poll(): Promise<boolean> {
  try {
    const next: RunSummary[] = await (await fetch("/api/runs")).json();
    $("live").classList.remove("off");
    const sig = (rs: RunSummary[]) => rs.map((r) => `${r.path}:${r.status}:${r.mtime}`).join("|");
    if (sig(next) === sig(runs)) return false;
    runs = next;
    assignColors();
    for (const p of [...details.keys()]) if (!runs.some((r) => r.path === p)) details.delete(p);
    if (!groups().includes(group)) group = defaultGroup();
    if (view.tab === "run" && !runs.some((r) => r.path === (view as { path: string }).path)) view = { tab: "charts" };
    await syncDetails();
    renderHeader();
    renderSidebar();
    refreshMain();
    return true;
  } catch {
    $("live").classList.add("off");
    return false;
  }
}

const defaultGroup = () => runs[0]?.group ?? ""; // runs come newest first

// --- routing ------------------------------------------------------------------------

function writeHash(push: boolean) {
  const p = new URLSearchParams({ g: group, v: view.tab });
  if (view.tab === "run") p.set("run", view.path);
  const url = `#${p}`;
  if (url === location.hash) return;
  if (push) history.pushState(null, "", url);
  else history.replaceState(null, "", url);
}
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const g = p.get("g");
  group = g !== null && groups().includes(g) ? g : defaultGroup();
  const v = p.get("v"), run = p.get("run");
  view = v === "table" ? { tab: "table" } : v === "run" && run && runs.some((r) => r.path === run) ? { tab: "run", path: run } : { tab: "charts" };
}
async function go(next: Partial<{ group: string; view: View }>) {
  if (next.group !== undefined) group = next.group;
  if (next.view) view = next.view;
  writeHash(true);
  await syncDetails();
  renderHeader();
  renderSidebar();
  refreshMain(true);
  $("main").scrollTop = 0;
}

// --- header & sidebar -----------------------------------------------------------------

function renderHeader() {
  const sel = $<HTMLSelectElement>("group");
  sel.innerHTML = groups().map((g) => `<option value="${esc(g)}" ${g === group ? "selected" : ""}>${esc(g || "runs")} (${runs.filter((r) => r.group === g).length})</option>`).join("");
  $("tab-charts").setAttribute("aria-current", view.tab === "table" ? "false" : "page");
  $("tab-table").setAttribute("aria-current", view.tab === "table" ? "page" : "false");
}

function headline(r: RunSummary): string {
  const entries = Object.entries(r.results ?? {}).filter(([, v]) => typeof v === "number");
  const e = entries.find(([k]) => lowerIsBetter(k) !== null) ?? entries[0];
  return e ? fmt(e[1]) : "";
}

function renderSidebar() {
  const rs = shownRuns();
  const all = inGroup();
  const visible = rs.filter((r) => !hidden.has(r.path)).length;
  $("run-count").textContent = runQuery ? `${rs.length} of ${all.length}` : String(all.length);
  $("side-foot").textContent = `${visible} visible · ${all.filter((r) => r.status === "running").length} running`;
  const list = $("run-list");
  list.replaceChildren();
  if (!all.length) {
    list.innerHTML = `<p class="empty">No runs under <code>runs/</code> yet.</p>`;
    return;
  }
  for (const r of rs) {
    const off = hidden.has(r.path);
    const row = document.createElement("div");
    row.className = `run-row${off ? " off" : ""}${hoverRun === r.path ? " hl" : ""}`;
    row.dataset.path = r.path;
    row.title = r.path;
    row.innerHTML = `<button class="ghost icon eye" aria-label="${off ? "Show" : "Hide"} ${esc(r.name)}" aria-pressed="${!off}">${off ? ICON.eyeOff : ICON.eye}</button>
      <span class="swatch" style="background:${colorOf(r.path)}"></span>
      <span class="name"><a>${esc(r.name)}</a><span class="state ${r.status}" title="${r.status}"></span></span>
      <span class="metric">${esc(headline(r))}</span>`;
    row.querySelector(".eye")!.addEventListener("click", () => toggleRuns([r.path], !off));
    row.querySelector("a")!.addEventListener("click", () => go({ view: { tab: "run", path: r.path } }));
    row.addEventListener("mouseenter", () => setHover(r.path));
    row.addEventListener("mouseleave", () => setHover(null));
    list.append(row);
  }
}

function toggleRuns(paths: string[], hide: boolean) {
  for (const p of paths) if (hide) hidden.add(p); else hidden.delete(p);
  save("rv:hidden", [...hidden]);
  renderSidebar();
  refreshMain();
}

/** Highlight one run in every chart and in the sidebar (null clears). */
function setHover(path: string | null) {
  if (path === hoverRun) return;
  hoverRun = path;
  document.querySelectorAll<SVGElement>(".chart [data-run]").forEach((el) => {
    const mine = el.dataset.run === path;
    el.classList.toggle("dim", path !== null && !mine);
    if (el.classList.contains("line")) el.classList.toggle("hl", path !== null && mine);
  });
  document.querySelectorAll<HTMLElement>(".run-row").forEach((el) => el.classList.toggle("hl", el.dataset.path === path));
}

// --- panels -------------------------------------------------------------------------

function metricColumns(ds: RunDetail[]): string[] {
  const cols: string[] = [];
  for (const d of ds) for (const c of d.metrics?.columns ?? []) if (!cols.includes(c)) cols.push(c);
  return cols;
}
function xAxis(cols: string[]): string {
  if (xChoice && cols.includes(xChoice)) return xChoice;
  return cols.includes("epoch") ? "epoch" : cols.includes("step") ? "step" : cols[0] ?? "";
}

type Section = { name: string; panels: Panel[] };

function sections(ds: RunDetail[], withCustom: boolean): Section[] {
  const cols = metricColumns(ds);
  const x = xAxis(cols);
  const by = new Map<string, Panel[]>();
  const order = ["Train", "Val", "Test", "Other"];
  for (const c of cols) {
    if (c === x) continue;
    const m = /^(train|val|valid|test)_/.exec(c);
    const name = m ? (m[1] === "valid" ? "Val" : m[1]![0]!.toUpperCase() + m[1]!.slice(1)) : "Other";
    if (!by.has(name)) by.set(name, []);
    by.get(name)!.push({ id: `m:${c}`, source: "rows", type: "line", x: "", y: [c], logY: prefs[`m:${c}`]?.logY ?? /(^|_)(loss|lr)($|_)/.test(c) });
  }
  const out: Section[] = [];
  if (withCustom && custom[group]?.length) out.push({ name: "Custom", panels: custom[group]! });
  for (const name of order) if (by.has(name)) out.push({ name, panels: by.get(name)! });
  let re: RegExp | null = null;
  try {
    re = panelQuery ? new RegExp(panelQuery, "i") : null;
  } catch {
    re = null;
  }
  const q = panelQuery.toLowerCase();
  const keep = (p: Panel) => !panelQuery || (re ? re.test(panelTitle(p)) : panelTitle(p).toLowerCase().includes(q));
  return out.map((s) => ({ ...s, panels: s.panels.filter(keep) })).filter((s) => s.panels.length);
}

function panelTitle(p: Panel): string {
  if (p.title) return p.title;
  if (p.id.startsWith("m:")) return p.y[0]!;
  return `${p.y.join(", ") || "(no y)"} vs ${p.x || (p.source === "runs" ? "run" : "step")}`;
}

function setPanelLog(p: Panel, logY: boolean) {
  if (p.id.startsWith("m:")) {
    prefs[p.id] = { ...prefs[p.id], logY };
    save("rv:prefs", prefs);
  } else {
    custom[group] = (custom[group] ?? []).map((c) => (c.id === p.id ? { ...c, logY } : c));
    save("rv:custom", custom);
  }
}

/** A panel card: header with hover tools, and a chart that redraws itself. */
function panelEl(p: Panel, getRuns: () => RunDetail[], keyPrefix: string): HTMLElement {
  const el = document.createElement("div");
  el.className = "panel";
  const isCustom = p.id.startsWith("c:");
  el.innerHTML = `<div class="ph"><span class="t" title="${esc(panelTitle(p))}">${esc(panelTitle(p))}</span><div class="tools">
      <button class="ghost icon zoom-reset" title="Reset zoom (or double-click the chart)" hidden>${ICON.reset}</button>
      <button class="ghost icon log" title="Log scale y" aria-pressed="${!!p.logY}">log</button>
      ${isCustom ? `<button class="ghost icon edit" title="Edit panel">${ICON.edit}</button>` : ""}
      <button class="ghost icon full" title="View full screen">${ICON.full}</button>
      ${isCustom ? `<button class="ghost icon del" title="Delete panel">${ICON.trash}</button>` : ""}
    </div></div><div class="chart"></div>`;
  const chart = el.querySelector<HTMLElement>(".chart")!;
  const key = `${keyPrefix}${p.id}`;
  const zoomBtn = el.querySelector<HTMLElement>(".zoom-reset")!;
  const draw = () => {
    drawPanel(chart, p, getRuns(), { height: 230, zoomKey: key, redraw: draw });
    zoomBtn.hidden = !zooms.has(key);
  };
  el.querySelector(".log")!.addEventListener("click", (e) => {
    p.logY = !p.logY;
    setPanelLog(p, p.logY);
    (e.currentTarget as HTMLElement).setAttribute("aria-pressed", String(p.logY));
    draw();
  });
  zoomBtn.addEventListener("click", () => {
    zooms.delete(key);
    draw();
  });
  el.querySelector(".full")!.addEventListener("click", () => openFullscreen(p, getRuns));
  el.querySelector(".edit")?.addEventListener("click", () => openEditor(p));
  el.querySelector(".del")?.addEventListener("click", () => {
    custom[group] = (custom[group] ?? []).filter((c) => c.id !== p.id);
    save("rv:custom", custom);
    refreshMain(true);
  });
  drawers.push(draw);
  return el;
}

function sectionsEl(secs: Section[], getRuns: () => RunDetail[], keyPrefix: string): HTMLElement {
  const wrap = document.createElement("div");
  for (const s of secs) {
    const det = document.createElement("details");
    det.className = "section";
    const ckey = `${group}|${keyPrefix}|${s.name}`;
    det.open = !collapsed.has(ckey);
    det.innerHTML = `<summary><span class="chev">${ICON.chev}</span>${esc(s.name)} <span class="n">${s.panels.length}</span></summary>`;
    const grid = document.createElement("div");
    grid.className = "grid";
    const before = drawers.length;
    for (const p of s.panels) grid.append(panelEl(p, getRuns, keyPrefix));
    const mine = drawers.slice(before);
    det.append(grid);
    det.addEventListener("toggle", () => {
      if (det.open) {
        collapsed.delete(ckey);
        mine.forEach((d) => d()); // charts aren't drawn while their section is closed
      } else collapsed.add(ckey);
      save("rv:collapsed", [...collapsed]);
    });
    wrap.append(det);
  }
  return wrap;
}

// --- main views ---------------------------------------------------------------------

function chartToolbar(extraLeft = ""): string {
  const cols = metricColumns(detailsOf(inGroup()));
  const x = xAxis(cols);
  return `<div class="toolbar">
    ${extraLeft}
    <input type="search" id="panel-q" placeholder="Search panels (regex)" value="${esc(panelQuery)}" aria-label="Search panels" />
    <label title="Exponential moving average, like TensorBoard">Smoothing
      <input type="range" id="smooth" min="0" max="0.99" step="0.01" value="${smoothing}" />
      <input type="number" id="smooth-n" min="0" max="0.99" step="0.01" value="${smoothing}" /></label>
    ${cols.length ? `<label>X axis <select id="xaxis">${cols.map((c) => `<option ${c === x ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></label>` : ""}
    <span class="spacer"></span>
    ${view.tab === "charts" ? `<button class="primary" id="add-panel">Add panel</button>` : ""}
  </div>`;
}

function wireChartToolbar() {
  const q = document.getElementById("panel-q") as HTMLInputElement | null;
  q?.addEventListener("input", () => {
    panelQuery = q.value;
    rebuildContent();
  });
  const s = document.getElementById("smooth") as HTMLInputElement | null;
  const n = document.getElementById("smooth-n") as HTMLInputElement | null;
  const setSmooth = (v: number) => {
    smoothing = Math.min(0.99, Math.max(0, isFinite(v) ? v : 0));
    save("rv:smoothing", smoothing);
    if (s) s.value = String(smoothing);
    if (n) n.value = String(smoothing);
    drawers.forEach((d) => d());
  };
  s?.addEventListener("input", () => setSmooth(Number(s.value)));
  n?.addEventListener("change", () => setSmooth(Number(n.value)));
  const x = document.getElementById("xaxis") as HTMLSelectElement | null;
  x?.addEventListener("change", () => {
    xChoice = x.value;
    save("rv:x", xChoice);
    zooms.clear();
    refreshMain(true);
  });
  document.getElementById("add-panel")?.addEventListener("click", () => openEditor(null));
}

/** What the page is built from; when it's unchanged, a refresh only redraws the charts. */
function computeSig(): string {
  const base = `${view.tab}|${group}|${view.tab === "run" ? view.path : ""}|${panelQuery}`;
  if (view.tab === "table") return `${base}|${Math.random()}`; // tables just rebuild
  const ds = view.tab === "run" ? detailsOf(runs.filter((r) => r.path === (view as { path: string }).path)) : detailsOf(inGroup());
  const secs = sections(ds, view.tab === "charts");
  return `${base}|${metricColumns(ds).join(",")}|${secs.map((s) => `${s.name}:${s.panels.map((p) => p.id).join(",")}`).join(";")}`;
}

function refreshMain(force = false) {
  const sig = computeSig();
  if (!force && sig === pageSig) {
    drawers.forEach((d) => d());
    if (view.tab === "run") renderRunCards();
    return;
  }
  pageSig = sig;
  const main = $("main");
  const tableScroll = main.querySelector(".table-wrap");
  const [st, sl] = tableScroll ? [tableScroll.scrollTop, tableScroll.scrollLeft] : [0, 0];
  drawers = [];
  if (!runs.length) {
    main.innerHTML = `<div class="empty">No runs yet. Training scripts write to <code>runs/&lt;project&gt;/…</code>; they show up here as soon as they start.</div>`;
    return;
  }
  if (view.tab === "table") {
    renderTable(main);
    const w = main.querySelector(".table-wrap");
    if (w) [w.scrollTop, w.scrollLeft] = [st, sl];
  } else if (view.tab === "run") renderRunPage(main, view.path);
  else renderCharts(main);
}

/** Re-make the panel grid only, keeping the toolbar (and the search box's focus). */
function rebuildContent() {
  drawers = [];
  pageSig = computeSig();
  if (view.tab === "run") {
    const old = document.querySelector(".run-charts");
    old?.replaceWith(runCharts(view.path));
  } else document.querySelector(".content")?.replaceChildren(chartsBody());
  drawers.forEach((d) => d());
}

function chartsBody(): HTMLElement {
  const secs = sections(detailsOf(inGroup()), true);
  if (!secs.length) {
    const p = document.createElement("div");
    p.className = "empty";
    p.innerHTML = panelQuery ? "No panels match the search." : "No metrics.csv in this workspace yet. Use <b>Add panel</b> to plot config and results across runs.";
    return p;
  }
  return sectionsEl(secs, visibleDetails, "");
}

function renderCharts(main: HTMLElement) {
  main.innerHTML = `${chartToolbar()}<div class="content"></div>`;
  main.querySelector(".content")!.append(chartsBody());
  wireChartToolbar();
  drawers.forEach((d) => d());
}

// --- run page -----------------------------------------------------------------------

function runCharts(path: string): HTMLElement {
  const d = () => detailsOf(runs.filter((r) => r.path === path));
  const el = sectionsEl(sections(d(), false), d, "r:");
  el.className = "run-charts";
  return el;
}

function renderRunPage(main: HTMLElement, path: string) {
  main.innerHTML = `${chartToolbar(`<a class="back" id="back">← Workspace</a>`)}<div class="content"><div id="run-cards"></div></div>`;
  wireChartToolbar();
  $("back").addEventListener("click", () => go({ view: { tab: "charts" } }));
  renderRunCards();
  main.querySelector(".content")!.append(runCharts(path));
  drawers.forEach((dr) => dr());
}

function kvRows(entries: [string, unknown][]): string {
  return entries.map(([k, v]) => `<div>${esc(k)}</div><div class="${typeof v === "number" ? "num" : ""}">${esc(fmt(v))}</div>`).join("");
}

function renderRunCards() {
  const host = document.getElementById("run-cards");
  if (!host || view.tab !== "run") return;
  const d = details.get(view.path);
  if (!d) {
    host.innerHTML = `<p class="empty">Loading…</p>`;
    return;
  }
  const git = typeof d.meta.git_commit === "string" ? d.meta.git_commit : null;
  const dirty = git?.endsWith("-dirty");
  const cfg: Record<string, unknown> = {};
  flatten(d.config, "", cfg);
  const rec = runRecord(d);
  const summary: [string, unknown][] = [
    ...Object.entries(d.results ?? {}),
    ...(d.metrics?.columns ?? []).map((c) => [`final ${c}`, rec[`final.${c}`]] as [string, unknown]).filter(([, v]) => v !== undefined),
  ];
  const meta = Object.entries(d.meta).filter(([k]) => k !== "git_commit" && k !== "device");
  host.innerHTML = `<div style="display:grid;gap:14px">
    <div class="run-head"><span class="swatch" style="background:${colorOf(d.path)}"></span><h2>${esc(d.name)}</h2>
      <span class="pill"><span class="state ${d.status}"></span>${d.status}</span>
      ${hidden.has(d.path) ? `<span class="pill">hidden in workspace</span>` : ""}</div>
    <div class="cards">
      <div class="card"><h3>Overview</h3><div class="body kv">${kvRows([
        ["path", `runs/${d.path}`], ["state", d.status], ["updated", `${new Date(d.mtime).toLocaleString()} (${ago(d.mtime)})`],
        ["rows logged", d.metrics?.rows.length ?? 0], ["device", d.meta.device ?? "—"],
      ])}<div>git commit</div><div class="mono">${git ? `${esc(git.replace(/-dirty$/, "").slice(0, 12))}${dirty ? ` <span class="pill warn">dirty</span>` : ""}` : "—"}</div>
      ${kvRows(meta)}</div></div>
      <div class="card"><h3>Summary</h3><div class="body kv">${summary.length ? kvRows(summary) : `<div>—</div><div></div>`}</div></div>
      <div class="card"><h3>Config</h3><div class="body kv">${Object.keys(cfg).length ? kvRows(Object.entries(cfg)) : `<div>—</div><div></div>`}</div></div>
    </div></div>`;
}

// --- runs table -----------------------------------------------------------------------

function renderTable(main: HTMLElement) {
  const ds = detailsOf(shownRuns());
  const recs = ds.map((d) => ({ d, r: runRecord(d) }));
  const keys = (prefix: string) => [...new Set(recs.flatMap(({ r }) => Object.keys(r).filter((k) => k.startsWith(prefix))))];
  const cols = metricColumns(ds);
  const x = xAxis(cols);
  const summaryKeys = [...keys("results."), ...cols.filter((c) => c !== x).map((c) => `final.${c}`)];
  const differs = (k: string) => new Set(recs.map(({ r }) => JSON.stringify(r[k]))).size > 1;
  const configKeys = keys("config.").filter((k) => tableAllConfig || differs(k));
  const label = (k: string) => k.replace(/^(results|config|final)\./, "");

  const val = (r: Record<string, unknown>, k: string) => (k === "name" ? r.run : r[k]);
  recs.sort((a, b) => {
    const va = val(a.r, sort.key), vb = val(b.r, sort.key);
    if (va === undefined) return 1;
    if (vb === undefined) return -1;
    const c = typeof va === "number" && typeof vb === "number" ? va - vb : String(va).localeCompare(String(vb), undefined, { numeric: true });
    return sort.asc ? c : -c;
  });
  const best = new Map<string, number>();
  for (const k of summaryKeys) {
    const lower = lowerIsBetter(k);
    const nums = recs.map(({ r }) => r[k]).filter((v): v is number => typeof v === "number");
    if (lower !== null && nums.length > 1) best.set(k, lower ? Math.min(...nums) : Math.max(...nums));
  }
  const th = (k: string, text: string, cls = "") =>
    `<th data-k="${esc(k)}" class="${cls} ${sort.key === k ? `sorted${sort.asc ? " asc" : ""}` : ""}" title="${esc(k)}">${esc(text)}</th>`;
  const td = (v: unknown, cls = "") => `<td class="${cls} ${typeof v === "number" ? "num" : ""}">${esc(fmt(v))}</td>`;

  main.innerHTML = `<div class="toolbar"><strong>${recs.length} runs</strong><span class="spacer"></span>
      <label><input type="checkbox" id="all-config" ${tableAllConfig ? "checked" : ""}/> Show config that is the same for every run</label></div>
    <div class="content"><div class="card"><div class="table-wrap"><table>
      <thead><tr class="grp"><th colspan="5">Run</th>
        ${summaryKeys.length ? `<th class="sep" colspan="${summaryKeys.length}">Summary</th>` : ""}
        ${configKeys.length ? `<th class="sep" colspan="${configKeys.length}">Config</th>` : ""}</tr>
      <tr class="cols"><th></th>${th("name", "Name")}${th("status", "State")}${th("mtime", "Updated")}${th("rows", "Rows", "num")}
        ${summaryKeys.map((k, i) => th(k, label(k), `num${i === 0 ? " sep" : ""}`)).join("")}
        ${configKeys.map((k, i) => th(k, label(k), i === 0 ? "sep" : "")).join("")}</tr></thead>
      <tbody>${recs.map(({ d, r }) => `<tr class="${hidden.has(d.path) ? "off" : ""}" data-path="${esc(d.path)}">
        <td><button class="ghost icon eye" aria-label="Toggle ${esc(d.name)}">${hidden.has(d.path) ? ICON.eyeOff : ICON.eye}</button></td>
        <td><span class="swatch" style="background:${colorOf(d.path)}"></span><a class="open">${esc(d.name)}</a></td>
        <td><span class="state ${d.status}" style="margin:0 6px 0 0"></span>${d.status}</td>
        <td>${ago(d.mtime)}</td>${td(r.rows)}
        ${summaryKeys.map((k, i) => td(r[k], `${best.get(k) === r[k] ? "best" : ""}${i === 0 ? " sep" : ""}`)).join("")}
        ${configKeys.map((k, i) => td(r[k], i === 0 ? "sep" : "")).join("")}</tr>`).join("")}</tbody>
    </table></div></div></div>`;

  main.querySelectorAll<HTMLElement>("tr.cols th[data-k]").forEach((h) => h.addEventListener("click", () => {
    const k = h.dataset.k!;
    sort = { key: k, asc: sort.key === k ? !sort.asc : k === "name" };
    refreshMain(true);
  }));
  main.querySelector("#all-config")!.addEventListener("change", (e) => {
    tableAllConfig = (e.target as HTMLInputElement).checked;
    refreshMain(true);
  });
  main.querySelectorAll<HTMLElement>("tbody tr").forEach((tr) => {
    const path = tr.dataset.path!;
    tr.querySelector(".eye")!.addEventListener("click", () => toggleRuns([path], !hidden.has(path)));
    tr.querySelector(".open")!.addEventListener("click", () => go({ view: { tab: "run", path } }));
    tr.addEventListener("mouseenter", () => setHover(path));
    tr.addEventListener("mouseleave", () => setHover(null));
  });
}

// --- charts -------------------------------------------------------------------------

type DrawOpts = { height: number; zoomKey: string; redraw: () => void };

function drawPanel(host: HTMLElement, p: Panel, ds: RunDetail[], o: DrawOpts) {
  if (host.closest("details:not([open])")) return; // drawn when its section opens
  if (p.source === "rows") drawRows(host, p, ds, o);
  else drawPerRun(host, p, ds, o);
}

function niceTicks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo];
  const step0 = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(Number(t.toPrecision(12)));
  return out;
}
function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  const mult = hi - lo < 1.2 ? [1, 1.5, 2, 3, 5, 7] : hi - lo < 2.5 ? [1, 2, 5] : [1];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++)
    for (const m of mult) {
      const v = Math.log10(m) + e;
      if (v >= lo - 1e-9 && v <= hi + 1e-9) out.push(v);
    }
  return out.length >= 2 ? out : niceTicks(lo, hi, 4);
}

const NS = "http://www.w3.org/2000/svg";
function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  parent?.append(el);
  return el;
}

function legendEl(items: { label: string; dash: string }[]): HTMLElement {
  const el = document.createElement("div");
  el.className = "legend";
  el.innerHTML = items.map((i) => `<span><svg viewBox="0 0 18 6"><line x1="0" x2="18" y1="3" y2="3" stroke="var(--ink-2)" stroke-width="2.2" ${i.dash ? `stroke-dasharray="${i.dash}"` : ""}/></svg>${esc(i.label)}</span>`).join("");
  return el;
}

type Margins = { l: number; r: number; t: number; b: number };
type Tick = { v: number; label: string };

/** Axes and grid; returns the plot group, clipped to the plot area. */
function frame(root: SVGSVGElement, W: number, H: number, M: Margins, sx: (v: number) => number, sy: (v: number) => number,
  xTicks: Tick[], yTicks: Tick[], xLabel: string): SVGGElement {
  const axis = svg("g", { class: "axis" }, root);
  for (const t of yTicks) {
    svg("line", { class: "gridl", x1: M.l, x2: W - M.r, y1: sy(t.v), y2: sy(t.v) }, axis);
    svg("text", { x: M.l - 6, y: sy(t.v) + 3.5, "text-anchor": "end" }, axis).textContent = t.label;
  }
  for (const t of xTicks) svg("text", { x: sx(t.v), y: H - M.b + 15, "text-anchor": "middle" }, axis).textContent = t.label;
  svg("line", { class: "gridl", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }, axis);
  svg("text", { x: M.l + (W - M.l - M.r) / 2, y: H - 3, "text-anchor": "middle", class: "label" }, axis).textContent = xLabel;
  const id = `clip${uid++}`;
  const clip = svg("clipPath", { id }, svg("defs", {}, root));
  svg("rect", { x: M.l, y: M.t - 4, width: W - M.l - M.r, height: H - M.t - M.b + 8 }, clip);
  return svg("g", { "clip-path": `url(#${id})` }, root);
}

function showTip(e: MouseEvent, html: string) {
  const tip = $("tip");
  tip.innerHTML = html;
  tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + 18, y = e.clientY + 14;
  if (x + w > innerWidth - 8) x = e.clientX - w - 18;
  if (y + h > innerHeight - 8) y = Math.max(8, innerHeight - h - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function hideTip() {
  $("tip").style.display = "none";
}

/** TensorBoard-style smoothing: debiased exponential moving average. */
function smooth(pts: [number, number][], w: number): [number, number][] {
  if (w <= 0) return pts;
  let last = 0, n = 0;
  return pts.map(([x, y]) => {
    last = last * w + (1 - w) * y;
    n++;
    return [x, last / (1 - w ** n)];
  });
}

function nearestIdx(pts: [number, number][], x: number): number {
  let lo = 0, hi = pts.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (pts[mid]![0] < x) lo = mid; else hi = mid;
  }
  return Math.abs(pts[lo]![0] - x) <= Math.abs(pts[hi]![0] - x) ? lo : hi;
}

const chartSize = (host: HTMLElement, height: number) => ({ W: Math.max(240, Math.round(host.clientWidth || 400)), H: height });
const M: Margins = { l: 54, r: 12, t: 8, b: 32 };

type LineSeries = { run: RunDetail; field: string; color: string; dash: string; raw: [number, number][]; sm: [number, number][] };

/** Per-step panels: one point per metrics.csv row, a line per visible run (and y field). */
function drawRows(host: HTMLElement, p: Panel, ds: RunDetail[], o: DrawOpts) {
  const xKey = p.x || xAxis(metricColumns(detailsOf(inGroup())));
  const series: LineSeries[] = [];
  for (const d of ds) {
    const m = d.metrics;
    if (!m) continue;
    const xi = m.columns.indexOf(xKey);
    if (xi < 0) continue;
    p.y.forEach((f, fi) => {
      const j = m.columns.indexOf(f);
      if (j < 0) return;
      const raw = m.rows
        .map((r) => [r[xi], r[j]] as [number | null, number | null])
        .filter((q): q is [number, number] => q[0] !== null && q[1] !== null && isFinite(q[0]) && isFinite(q[1]))
        .sort((a, b) => a[0] - b[0]);
      if (raw.length) series.push({ run: d, field: f, color: colorOf(d.path), dash: DASHES[fi % DASHES.length]!, raw, sm: p.type === "line" ? smooth(raw, smoothing) : raw });
    });
  }
  host.replaceChildren();
  if (!series.length) {
    host.innerHTML = `<div class="nodata">${ds.length ? "No data for the visible runs" : "No visible runs"}</div>`;
    return;
  }
  if (p.y.length > 1) host.append(legendEl(p.y.map((f, i) => ({ label: f, dash: DASHES[i % DASHES.length]! }))));

  const showRaw = p.type === "line" && smoothing > 0;
  const logX = !!p.logX && series.every((s) => s.raw.every((q) => q[0] > 0));
  const logY = !!p.logY && series.every((s) => s.raw.every((q) => q[1] > 0));
  const tx = (v: number) => (logX ? Math.log10(v) : v), ty = (v: number) => (logY ? Math.log10(v) : v);
  const allX = series.flatMap((s) => s.raw.map((q) => q[0]));
  const zoom = zooms.get(o.zoomKey);
  const [x0, x1] = zoom ?? [Math.min(...allX), Math.max(...allX)];
  let X0 = tx(x0), X1 = tx(x1);
  if (X0 === X1) [X0, X1] = [X0 - 1, X1 + 1];
  const inView = (q: [number, number]) => q[0] >= x0 && q[0] <= x1;
  const ys = series.flatMap((s) => [...s.sm, ...(showRaw ? s.raw : [])].filter(inView).map((q) => ty(q[1])));
  let Y0 = ys.length ? Math.min(...ys) : 0, Y1 = ys.length ? Math.max(...ys) : 1;
  if (Y0 === Y1) [Y0, Y1] = [Y0 - (Math.abs(Y0) || 1) * 0.1, Y1 + (Math.abs(Y1) || 1) * 0.1];
  const pad = (Y1 - Y0) * 0.06;
  Y0 -= pad;
  Y1 += pad;

  const { W, H } = chartSize(host, o.height);
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const sx = (v: number) => M.l + ((v - X0) / (X1 - X0)) * pw;
  const sy = (v: number) => M.t + ph - ((v - Y0) / (Y1 - Y0)) * ph;
  const root = svg("svg", { class: "plot", viewBox: `0 0 ${W} ${H}`, height: H });
  const xt = (logX ? logTicks(X0, X1) : niceTicks(X0, X1, Math.max(3, Math.floor(pw / 80)))).map((v) => ({ v, label: fmt(logX ? 10 ** v : v) }));
  const yt = (logY ? logTicks(Y0, Y1) : niceTicks(Y0, Y1, Math.max(3, Math.floor(ph / 45)))).map((v) => ({ v, label: fmt(logY ? 10 ** v : v) }));
  const g = frame(root, W, H, M, sx, sy, xt, yt, xKey);

  const pathD = (pts: [number, number][]) => pts.map((q, i) => `${i ? "L" : "M"}${sx(tx(q[0])).toFixed(1)},${sy(ty(q[1])).toFixed(1)}`).join("");
  const dash = (s: LineSeries): Record<string, string> => (s.dash ? { "stroke-dasharray": s.dash } : {});
  const state = (s: LineSeries) => (hoverRun === null ? "" : hoverRun === s.run.path ? " hl" : " dim");
  if (showRaw) for (const s of series) svg("path", { d: pathD(s.raw), class: `raw${state(s).replace(" hl", "")}`, stroke: s.color, "data-run": s.run.path, ...dash(s) }, g);
  for (const s of series) {
    if (p.type === "scatter") {
      for (const q of s.raw) svg("circle", { cx: sx(tx(q[0])), cy: sy(ty(q[1])), r: 3.5, fill: s.color, class: `pt${state(s).replace(" hl", "")}`, "data-run": s.run.path }, g);
    } else {
      svg("path", { d: pathD(s.sm), class: `line${state(s)}`, stroke: s.color, "data-run": s.run.path, ...dash(s) }, g);
      if (s.sm.length === 1) svg("circle", { cx: sx(tx(s.sm[0]![0])), cy: sy(ty(s.sm[0]![1])), r: 3, fill: s.color, "data-run": s.run.path }, g);
    }
  }
  const xhair = svg("line", { class: "xhair", y1: M.t, y2: M.t + ph, visibility: "hidden" }, root);
  const dots = svg("g", {}, root);
  const brush = svg("rect", { class: "brush", y: M.t, height: ph, visibility: "hidden" }, root);
  host.append(root);
  if (zoom) {
    const z = document.createElement("div");
    z.className = "zoomed";
    z.textContent = "zoomed · double-click to reset";
    host.append(z);
  }

  const local = (e: MouseEvent) => {
    const r = root.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H] as const;
  };
  const clampX = (v: number) => Math.min(Math.max(v, M.l), W - M.r);
  const toData = (px: number) => {
    const t = X0 + ((px - M.l) / pw) * (X1 - X0);
    return logX ? 10 ** t : t;
  };
  const leave = () => {
    xhair.setAttribute("visibility", "hidden");
    dots.replaceChildren();
    hideTip();
    setHover(null);
  };

  // Drag to zoom the x range, double-click to reset (like TensorBoard).
  let dragFrom: number | null = null;
  const onUp = (e: MouseEvent) => {
    window.removeEventListener("mouseup", onUp);
    if (dragFrom === null) return;
    const from = dragFrom, to = clampX(local(e)[0]);
    dragFrom = null;
    brush.setAttribute("visibility", "hidden");
    if (Math.abs(to - from) > 6) {
      zooms.set(o.zoomKey, [toData(Math.min(from, to)), toData(Math.max(from, to))]);
      leave();
      o.redraw();
    }
  };
  root.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    dragFrom = clampX(local(e)[0]);
    window.addEventListener("mouseup", onUp);
  });
  root.addEventListener("dblclick", () => {
    if (!zooms.has(o.zoomKey)) return;
    zooms.delete(o.zoomKey);
    o.redraw();
  });

  root.addEventListener("mousemove", (e) => {
    const [mx, my] = local(e);
    if (dragFrom !== null) {
      const to = clampX(mx);
      brush.setAttribute("x", String(Math.min(dragFrom, to)));
      brush.setAttribute("width", String(Math.abs(to - dragFrom)));
      brush.setAttribute("visibility", "visible");
      return;
    }
    if (mx < M.l || mx > W - M.r) return leave();
    const xv = toData(mx);
    const hits = series.map((s) => {
      const i = nearestIdx(s.sm, xv);
      return { s, q: s.sm[i]!, rawY: s.raw[i]![1] };
    }).filter((h) => inView(h.q));
    if (!hits.length) return leave();
    const pxOf = (h: { q: [number, number] }) => sx(tx(h.q[0]));
    const snapPx = pxOf(hits.reduce((a, b) => (Math.abs(pxOf(b) - mx) < Math.abs(pxOf(a) - mx) ? b : a)));
    const at = hits.filter((h) => Math.abs(pxOf(h) - snapPx) < 12);
    const closest = at.reduce((a, b) => (Math.abs(sy(ty(b.q[1])) - my) < Math.abs(sy(ty(a.q[1])) - my) ? b : a));
    setHover(closest.s.run.path);
    xhair.setAttribute("x1", String(snapPx));
    xhair.setAttribute("x2", String(snapPx));
    xhair.setAttribute("visibility", "visible");
    dots.replaceChildren();
    for (const h of at) svg("circle", { cx: pxOf(h), cy: sy(ty(h.q[1])), r: h === closest ? 4.5 : 3, fill: h.s.color, stroke: "var(--panel)", "stroke-width": 1.5 }, dots);
    at.sort((a, b) => b.q[1] - a.q[1]);
    showTip(e, `<div class="h">${esc(xKey)} ${esc(fmt(closest.q[0]))}</div>${at.map((h) => `<div class="r${h === closest ? " hl" : ""}">
      <span class="sw" style="background:${h.s.color}"></span>
      <span class="nm">${esc(h.s.run.name)}${p.y.length > 1 ? ` · ${esc(h.s.field)}` : ""}</span>
      <span class="v">${esc(fmt(h.q[1]))}${showRaw ? `<small>${esc(fmt(h.rawY))}</small>` : ""}</span></div>`).join("")}`);
  });
  root.addEventListener("mouseleave", () => {
    if (dragFrom === null) leave();
  });
}

/** Per-run panels: one point per visible run, from config / results / metric summaries. */
function drawPerRun(host: HTMLElement, p: Panel, ds: RunDetail[], o: DrawOpts) {
  type Pt = { d: RunDetail; field: string; fi: number; x: number | string; y: number };
  const pts: Pt[] = [];
  for (const d of ds) {
    const r = runRecord(d);
    const xv = p.x ? r[p.x] : d.name;
    if (xv === undefined || xv === null) continue;
    p.y.forEach((f, fi) => {
      const yv = r[f];
      if (typeof yv === "number" && isFinite(yv)) pts.push({ d, field: f, fi, x: typeof xv === "number" ? xv : String(xv), y: yv });
    });
  }
  host.replaceChildren();
  if (!pts.length) {
    host.innerHTML = `<div class="nodata">${p.y.length ? "No data for the visible runs" : "Pick a y field"}</div>`;
    return;
  }
  if (p.y.length > 1) host.append(legendEl(p.y.map((f, i) => ({ label: f, dash: DASHES[i % DASHES.length]! }))));

  const categorical = p.type === "bar" || pts.some((q) => typeof q.x === "string");
  const cats = categorical ? [...new Set(pts.map((q) => String(q.x)))] : [];
  if (categorical && cats.every((c) => c !== "" && !isNaN(Number(c)))) cats.sort((a, b) => Number(a) - Number(b));
  const logX = !!p.logX && !categorical && pts.every((q) => (q.x as number) > 0);
  const logY = !!p.logY && pts.every((q) => q.y > 0);
  const tx = (v: number) => (logX ? Math.log10(v) : v), ty = (v: number) => (logY ? Math.log10(v) : v);

  let X0: number, X1: number;
  if (categorical) [X0, X1] = [0, cats.length];
  else {
    const xs = pts.map((q) => tx(q.x as number));
    [X0, X1] = [Math.min(...xs), Math.max(...xs)];
    if (X0 === X1) [X0, X1] = [X0 - 1, X1 + 1];
    const padX = (X1 - X0) * 0.06;
    X0 -= padX;
    X1 += padX;
  }
  const ys = pts.map((q) => ty(q.y));
  let Y0 = Math.min(...ys), Y1 = Math.max(...ys);
  const barFromZero = p.type === "bar" && !logY;
  if (barFromZero) [Y0, Y1] = [Math.min(0, Y0), Math.max(0, Y1)];
  if (Y0 === Y1) [Y0, Y1] = [Y0 - (Math.abs(Y0) || 1) * 0.1, Y1 + (Math.abs(Y1) || 1) * 0.1];
  const pad = (Y1 - Y0) * 0.08;
  if (!(barFromZero && Y0 === 0)) Y0 -= pad;
  Y1 += pad;

  const { W, H } = chartSize(host, o.height);
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const sx = (v: number) => M.l + ((v - X0) / (X1 - X0)) * pw;
  const sy = (v: number) => M.t + ph - ((v - Y0) / (Y1 - Y0)) * ph;
  const px = (q: Pt) => (categorical ? sx(cats.indexOf(String(q.x)) + 0.5) : sx(tx(q.x as number)));
  const root = svg("svg", { class: "plot", viewBox: `0 0 ${W} ${H}`, height: H });
  const every = Math.max(1, Math.ceil(cats.length / Math.max(2, Math.floor(pw / 70))));
  const xt = categorical
    ? cats.map((c, i) => ({ v: i + 0.5, label: c.length > 14 ? `${c.slice(0, 13)}…` : c })).filter((_, i) => i % every === 0)
    : (logX ? logTicks(X0, X1) : niceTicks(X0, X1, Math.max(3, Math.floor(pw / 80)))).map((v) => ({ v, label: fmt(logX ? 10 ** v : v) }));
  const yt = (logY ? logTicks(Y0, Y1) : niceTicks(Y0, Y1, Math.max(3, Math.floor(ph / 45)))).map((v) => ({ v, label: fmt(logY ? 10 ** v : v) }));
  const g = frame(root, W, H, M, sx, sy, xt, yt, p.x || "run");
  const dim = (q: Pt) => (hoverRun && hoverRun !== q.d.path ? " dim" : "");

  const marks: { q: Pt; x: number; y: number }[] = [];
  if (p.type === "bar") {
    // Bars sharing an x value (several fields, or runs with the same config) sit side by side.
    const band = pw / cats.length;
    const slots = new Map<string, Pt[]>();
    for (const q of pts) {
      const k = String(q.x);
      if (!slots.has(k)) slots.set(k, []);
      slots.get(k)!.push(q);
    }
    const zero = sy(logY ? Y0 : 0);
    for (const [k, items] of slots) {
      const bw = Math.min(46, (band * 0.84) / items.length);
      items.forEach((q, i) => {
        const cx = sx(cats.indexOf(k) + 0.5) + (i - (items.length - 1) / 2) * bw;
        const y = sy(ty(q.y));
        svg("rect", { x: cx - bw / 2 + 1, width: Math.max(1, bw - 2), y: Math.min(y, zero), height: Math.max(1, Math.abs(zero - y)), rx: 2, fill: colorOf(q.d.path), "fill-opacity": q.fi ? 0.55 : 1, class: dim(q), "data-run": q.d.path }, g);
        marks.push({ q, x: cx, y });
      });
    }
  } else {
    if (p.type === "line")
      p.y.forEach((f, fi) => {
        const line = pts.filter((q) => q.field === f).sort((a, b) => px(a) - px(b));
        svg("path", { d: line.map((q, i) => `${i ? "L" : "M"}${px(q).toFixed(1)},${sy(ty(q.y)).toFixed(1)}`).join(""), fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.4, ...(DASHES[fi % DASHES.length] ? { "stroke-dasharray": DASHES[fi % DASHES.length]! } : {}) }, g);
      });
    for (const q of pts) {
      const x = px(q), y = sy(ty(q.y));
      svg("circle", { cx: x, cy: y, r: 5, fill: colorOf(q.d.path), class: `pt${dim(q)}`, "data-run": q.d.path, ...(q.fi ? { "fill-opacity": 0.55 } : {}) }, g);
      marks.push({ q, x, y });
    }
  }
  host.append(root);

  const local = (e: MouseEvent) => {
    const r = root.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H] as const;
  };
  root.addEventListener("mousemove", (e) => {
    const [mx, my] = local(e);
    const near = marks.reduce((a, b) => (Math.hypot(b.x - mx, b.y - my) < Math.hypot(a.x - mx, a.y - my) ? b : a));
    if (Math.abs(near.x - mx) > (p.type === "bar" ? 30 : 28) || (p.type !== "bar" && Math.hypot(near.x - mx, near.y - my) > 28)) {
      hideTip();
      setHover(null);
      return;
    }
    setHover(near.q.d.path);
    const same = marks.filter((m) => m.q.d === near.q.d);
    showTip(e, `<div class="h"><span class="sw" style="display:inline-block;background:${colorOf(near.q.d.path)};margin-right:6px"></span>${esc(near.q.d.name)}</div>
      <div class="r"><span></span><span class="nm">${esc(p.x || "run")}</span><span class="v">${esc(fmt(near.q.x))}</span></div>
      ${same.map((m) => `<div class="r${m === near ? " hl" : ""}"><span></span><span class="nm">${esc(m.q.field)}</span><span class="v">${esc(fmt(m.q.y))}</span></div>`).join("")}`);
  });
  root.addEventListener("mouseleave", () => {
    hideTip();
    setHover(null);
  });
}

// --- dialogs: fullscreen & panel editor ------------------------------------------------

const dlg = () => $<HTMLDialogElement>("dlg");
let dialogDraw: (() => void) | null = null;
dlg().addEventListener("close", () => {
  dialogDraw = null;
  hideTip();
});

function openFullscreen(p: Panel, getRuns: () => RunDetail[]) {
  const d = dlg();
  d.innerHTML = `<div class="dh">${esc(panelTitle(p))}<span class="spacer"></span><button class="ghost" id="dlg-close">Close</button></div><div class="db"><div class="chart"></div></div>`;
  d.querySelector("#dlg-close")!.addEventListener("click", () => d.close());
  d.showModal();
  const chart = d.querySelector<HTMLElement>(".chart")!;
  const draw = () => drawPanel(chart, p, getRuns(), { height: Math.max(320, Math.min(620, innerHeight - 220)), zoomKey: `full:${p.id}`, redraw: draw });
  dialogDraw = draw;
  draw();
}

function fieldsFor(source: Source): string[] {
  const ds = detailsOf(inGroup());
  if (source === "rows") return metricColumns(ds);
  const set = new Set<string>();
  for (const d of ds) Object.keys(runRecord(d)).forEach((k) => set.add(k));
  return [...set].filter((k) => k !== "run");
}
function numericFor(source: Source): string[] {
  if (source === "rows") return fieldsFor("rows");
  const recs = detailsOf(inGroup()).map(runRecord);
  return fieldsFor("runs").filter((f) => f !== "mtime" && recs.some((r) => typeof r[f] === "number"));
}
function defaultPanel(source: Source): Panel {
  const id = `c:${Date.now()}`;
  const nums = numericFor(source);
  if (source === "rows") {
    const x = xAxis(nums);
    const y = nums.find((f) => f !== x && lowerIsBetter(f) !== null) ?? nums.find((f) => f !== x);
    return { id, source, type: "line", x: "", y: y ? [y] : [] };
  }
  const recs = detailsOf(inGroup()).map(runRecord);
  const differs = (f: string) => new Set(recs.map((r) => JSON.stringify(r[f]))).size > 1;
  const x = fieldsFor("runs").find((f) => f.startsWith("config.") && differs(f)) ?? "";
  const ys = nums.filter((f) => f.startsWith("results.") || f.startsWith("final."));
  const y = ys.find((f) => lowerIsBetter(f) !== null) ?? ys[0];
  return { id, source, type: x ? "scatter" : "bar", x, y: y ? [y] : [] };
}

function openEditor(existing: Panel | null) {
  const hasRows = fieldsFor("rows").length > 0;
  let p: Panel = existing ? { ...existing, y: [...existing.y] } : defaultPanel(hasRows ? "rows" : "runs");
  const d = dlg();
  d.innerHTML = `<div class="dh">${existing ? "Edit panel" : "Add panel"}<span class="spacer"></span></div>
    <div class="db"><div class="editor"><div class="form"></div><div class="panel"><div class="ph"><span class="t"></span></div><div class="chart"></div></div></div></div>
    <div class="df"><button class="ghost" id="ed-cancel">Cancel</button><button class="primary" id="ed-save">${existing ? "Save" : "Add panel"}</button></div>`;
  const form = d.querySelector<HTMLElement>(".form")!;
  const chart = d.querySelector<HTMLElement>(".chart")!;
  const title = d.querySelector<HTMLElement>(".ph .t")!;
  const preview = () => {
    title.textContent = panelTitle(p);
    drawPanel(chart, p, visibleDetails(), { height: 320, zoomKey: "editor", redraw: preview });
  };
  const seg = (k: string, opts: [string, string][], cur: string, disabled: string[] = []) =>
    `<div class="seg" data-seg="${k}">${opts.map(([v, l]) => `<button type="button" data-v="${v}" aria-pressed="${v === cur}" ${disabled.includes(v) ? "disabled" : ""}>${l}</button>`).join("")}</div>`;
  const opt = (v: string, l: string, cur: string) => `<option value="${esc(v)}" ${v === cur ? "selected" : ""}>${esc(l)}</option>`;

  const renderForm = () => {
    const fields = fieldsFor(p.source);
    const ys = numericFor(p.source).filter((f) => !p.y.includes(f));
    form.innerHTML = `
      <div class="fld">Data${seg("source", [["rows", "Per step"], ["runs", "Per run"]], p.source, hasRows ? [] : ["rows"])}</div>
      <div class="fld">Chart${seg("type", [["line", "Line"], ["scatter", "Scatter"], ["bar", "Bar"]], p.type, p.source === "rows" ? ["bar"] : [])}</div>
      <label>X axis <select data-k="x">${p.source === "rows"
        ? opt("", `workspace x-axis (${xAxis(fields)})`, p.x) + fields.map((f) => opt(f, f, p.x)).join("")
        : opt("", "run name", p.x) + fields.map((f) => opt(f, f, p.x)).join("")}</select></label>
      <div class="fld">Y axis
        <div class="ychips">${p.y.map((y) => `<button type="button" class="ychip" data-y="${esc(y)}" title="Remove">${esc(y)} ✕</button>`).join("")}</div>
        <select data-k="add-y"><option value="">+ add a field</option>${ys.map((f) => opt(f, f, "")).join("")}</select></div>
      <div class="row"><label><input type="checkbox" data-k="logX" ${p.logX ? "checked" : ""}/> log x</label>
        <label><input type="checkbox" data-k="logY" ${p.logY ? "checked" : ""}/> log y</label></div>
      <label>Title <input type="search" data-k="title" placeholder="${esc(panelTitle({ ...p, title: undefined }))}" value="${esc(p.title ?? "")}" /></label>
      <p style="margin:0;color:var(--ink-3);font-size:12px">${p.source === "runs"
        ? "Per run: one point per visible run. Fields are config.*, results.*, meta.*, and final./min./max. of each metric."
        : "Per step: one point per metrics.csv row for each visible run. Smoothing applies to line charts."}</p>`;
    form.querySelectorAll<HTMLElement>("[data-seg]").forEach((s) => s.querySelectorAll<HTMLButtonElement>("button").forEach((b) => b.addEventListener("click", () => {
      if (s.dataset.seg === "source") p = { ...defaultPanel(b.dataset.v as Source), id: p.id, title: p.title };
      else p.type = b.dataset.v as PlotType;
      renderForm();
      preview();
    })));
    form.querySelectorAll<HTMLInputElement | HTMLSelectElement>("[data-k]").forEach((el) => {
      const k = el.dataset.k!;
      el.addEventListener(k === "title" ? "input" : "change", () => {
        if (k === "add-y") {
          if (el.value) p.y.push(el.value);
          renderForm();
        } else if (k === "logX" || k === "logY") p[k] = (el as HTMLInputElement).checked;
        else if (k === "title") p.title = el.value || undefined;
        else if (k === "x") p.x = el.value;
        preview();
      });
    });
    form.querySelectorAll<HTMLElement>(".ychip").forEach((b) => b.addEventListener("click", () => {
      p.y = p.y.filter((y) => y !== b.dataset.y);
      renderForm();
      preview();
    }));
  };
  d.querySelector("#ed-cancel")!.addEventListener("click", () => d.close());
  d.querySelector("#ed-save")!.addEventListener("click", () => {
    const list = custom[group] ?? [];
    custom[group] = existing ? list.map((c) => (c.id === p.id ? p : c)) : [...list, p];
    save("rv:custom", custom);
    d.close();
    collapsed.delete(`${group}||Custom`);
    refreshMain(true);
  });
  d.showModal();
  renderForm();
  dialogDraw = preview;
  preview();
}

// --- boot ---------------------------------------------------------------------------

$("group").addEventListener("change", (e) => go({ group: (e.target as HTMLSelectElement).value, view: { tab: view.tab === "table" ? "table" : "charts" } }));
$("tab-charts").addEventListener("click", (e) => {
  e.preventDefault();
  go({ view: { tab: "charts" } });
});
$("tab-table").addEventListener("click", (e) => {
  e.preventDefault();
  go({ view: { tab: "table" } });
});
$("filter").addEventListener("input", (e) => {
  runQuery = (e.target as HTMLInputElement).value.trim();
  renderSidebar();
  refreshMain();
});
$("show-all").addEventListener("click", () => toggleRuns(shownRuns().map((r) => r.path), false));
$("hide-all").addEventListener("click", () => toggleRuns(shownRuns().map((r) => r.path), true));
addEventListener("popstate", async () => {
  readHash();
  await syncDetails();
  renderHeader();
  renderSidebar();
  refreshMain(true);
});

let lastWidth = 0, resizeTimer: ReturnType<typeof setTimeout> | undefined;
new ResizeObserver(([entry]) => {
  const w = Math.round(entry!.contentRect.width);
  if (w === lastWidth) return;
  lastWidth = w;
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    drawers.forEach((d) => d());
    dialogDraw?.();
  }, 100);
}).observe($("main"));

runs = await (await fetch("/api/runs")).json();
assignColors();
readHash();
writeHash(false);
await syncDetails();
renderHeader();
renderSidebar();
refreshMain(true);
setInterval(async () => {
  if (await poll()) dialogDraw?.();
}, POLL_MS);
