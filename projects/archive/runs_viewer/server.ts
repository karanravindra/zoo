/**
 * Runs viewer: browse, watch and compare everything under <repo>/runs.
 *
 *   bun run projects/archive/runs_viewer/server.ts [--port 3001]
 *
 * A run is any directory under runs/ holding config.json, metrics.csv or results.json,
 * at any depth. Runs are re-discovered on every request, so new runs and new metrics rows
 * show up on refresh (the page also polls while a run is still training).
 */
import { readdirSync, statSync } from "node:fs";
import { dirname, relative, resolve, sep } from "node:path";
import { parseArgs } from "node:util";
import index from "./index.html";

const REPO = resolve(import.meta.dir, "../../..");
const RUNS = resolve(REPO, "runs");
const RUN_FILES = ["config.json", "metrics.csv", "results.json"];
const RUNNING_WITHIN_MS = 2 * 60 * 1000; // metrics.csv touched this recently -> still training

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: { port: { type: "string", default: "3001" } },
});

type Status = "done" | "running" | "stopped";
type RunSummary = {
  path: string; // relative to runs/, with "/" separators
  group: string; // parent path, e.g. "pixel_art/ae"
  name: string;
  mtime: number; // newest of the run files, ms
  status: Status;
  config: Record<string, unknown> | null;
  meta: Record<string, unknown>; // everything in config.json besides "config" (git_commit, device, ...)
  results: Record<string, unknown> | null;
};

const toPosix = (p: string) => p.split(sep).join("/");

/** Run dirs (relative to runs/), found by their marker files. */
async function runDirs(): Promise<string[]> {
  const dirs = new Set<string>();
  for await (const rel of new Bun.Glob(`**/{${RUN_FILES.join(",")}}`).scan(RUNS)) dirs.add(toPosix(dirname(rel)));
  dirs.delete(".");
  return [...dirs];
}

async function readJson(path: string): Promise<any> {
  const f = Bun.file(path);
  if (!(await f.exists())) return null;
  try {
    return await f.json();
  } catch {
    return null; // half-written or not JSON
  }
}

function mtimeOf(path: string): number | null {
  try {
    return statSync(path).mtimeMs;
  } catch {
    return null;
  }
}

async function summarize(rel: string): Promise<RunSummary> {
  const dir = resolve(RUNS, rel);
  const raw = await readJson(resolve(dir, "config.json"));
  const results = await readJson(resolve(dir, "results.json"));
  const { config = null, ...meta } = raw && typeof raw === "object" && "config" in raw ? raw : { config: raw };
  const metricsTime = mtimeOf(resolve(dir, "metrics.csv"));
  const mtime = Math.max(0, ...RUN_FILES.map((f) => mtimeOf(resolve(dir, f)) ?? 0));
  const status: Status = results
    ? "done"
    : metricsTime !== null && Date.now() - metricsTime < RUNNING_WITHIN_MS ? "running" : "stopped";
  const slash = rel.lastIndexOf("/");
  return {
    path: rel,
    group: slash < 0 ? "" : rel.slice(0, slash),
    name: slash < 0 ? rel : rel.slice(slash + 1),
    mtime,
    status,
    config,
    meta,
    results,
  };
}

/** metrics.csv as columns + numeric rows (non-numbers become null; a half-written last line is dropped). */
async function readMetrics(dir: string): Promise<{ columns: string[]; rows: (number | null)[][] } | null> {
  const f = Bun.file(resolve(dir, "metrics.csv"));
  if (!(await f.exists())) return null;
  const text = await f.text();
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  if (!lines.length) return { columns: [], rows: [] };
  const columns = lines[0]!.split(",").map((c) => c.trim());
  const rows = lines
    .slice(1)
    .map((l) => l.split(","))
    .filter((cells) => cells.length === columns.length)
    .map((cells) => cells.map((c) => (c.trim() === "" || isNaN(Number(c)) ? null : Number(c))));
  return { columns, rows };
}

type FileEntry = { path: string; size: number; kind: "image" | "json" | "csv" | "checkpoint" | "other" };

function kindOf(name: string): FileEntry["kind"] {
  const ext = name.slice(name.lastIndexOf(".") + 1).toLowerCase();
  if (["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(ext)) return "image";
  if (ext === "json") return "json";
  if (ext === "csv") return "csv";
  if (["pt", "pth", "ckpt", "safetensors"].includes(ext)) return "checkpoint";
  return "other";
}

/** Files in a run dir, recursing into subfolders that aren't runs themselves. */
function listFiles(rel: string, nestedRuns: Set<string>): FileEntry[] {
  const out: FileEntry[] = [];
  const walk = (sub: string) => {
    for (const e of readdirSync(resolve(RUNS, sub), { withFileTypes: true })) {
      const p = `${sub}/${e.name}`;
      if (e.isDirectory()) {
        if (!nestedRuns.has(p)) walk(p);
      } else if (e.isFile()) {
        out.push({ path: p, size: statSync(resolve(RUNS, p)).size, kind: kindOf(e.name) });
      }
    }
  };
  walk(rel);
  return out.sort((a, b) => a.path.localeCompare(b.path));
}

/** Resolve a request path under runs/, or null if it escapes it. */
function underRuns(rel: string): string | null {
  const path = resolve(RUNS, rel);
  return path.startsWith(RUNS + sep) ? path : null;
}

const server = Bun.serve({
  port: Number(values.port),
  development: true,
  routes: {
    "/": index,
    "/api/runs": async () => {
      const runs = await Promise.all((await runDirs()).map(summarize));
      runs.sort((a, b) => b.mtime - a.mtime);
      return Response.json(runs);
    },
    "/api/run/*": async (req) => {
      const rel = decodeURIComponent(new URL(req.url).pathname.slice("/api/run/".length)).replace(/\/+$/, "");
      const dir = underRuns(rel);
      if (!dir) return new Response("forbidden", { status: 403 });
      const all = new Set(await runDirs());
      if (!all.has(rel)) return new Response("not a run", { status: 404 });
      const nested = new Set([...all].filter((r) => r.startsWith(`${rel}/`)));
      return Response.json({ ...(await summarize(rel)), metrics: await readMetrics(dir), files: listFiles(rel, nested) });
    },
    // Files under runs/ only; checkpoints are listed but not served.
    "/file/*": async (req) => {
      const rel = decodeURIComponent(new URL(req.url).pathname.slice("/file/".length));
      const path = underRuns(rel);
      if (!path || kindOf(path) === "checkpoint") return new Response("forbidden", { status: 403 });
      const f = Bun.file(path);
      if (!(await f.exists())) return new Response("not found", { status: 404 });
      return new Response(f, { headers: { "Cache-Control": "no-cache" } });
    },
  },
});

console.log(`${(await runDirs()).length} runs under ${relative(process.cwd(), RUNS) || "."}`);
console.log(`runs viewer at ${server.url}`);
