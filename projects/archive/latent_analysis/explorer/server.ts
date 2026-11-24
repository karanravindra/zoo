/**
 * Latent explorer: browse projected embeddings, inspect the images behind the points,
 * and tag them.
 *
 *   bun run projects/archive/latent_analysis/explorer/server.ts [points.json] [--port 3000]
 *
 * With no points file, opens the most recently exported one under runs/latent_analysis
 * (<dataset>/<model>/explorer/<split>.json, written by export.py). Every k-means run next
 * to it (k<k>/clusters_<split>.json, written by kmeans.py) is attached each time the page
 * loads, so a new k shows up on refresh without restarting.
 *
 * Serves the UI, the points file, thumbnails/images under <repo>/data, and reads/writes
 * the tags file named in the points file (`labelsFile`, a JSON map of image id -> tags; ids
 * come from the dataset's index.json, see image_index.py) and rejections.json next to it
 * (image id -> tags a human said the image does NOT have). Both are shared by every
 * model/split/k of the dataset, so writes are per-image patches merged into whatever is on
 * disk, and several explorers can run at once without overwriting each other's tags.
 * Adding a tag to one file for an image removes it from the other.
 *
 * Also serves autotag/predictions.json for the points file's model (written by autotag.py),
 * re-read on every page load like the clusters, for the review mode.
 */
import { mkdirSync, renameSync, statSync } from "node:fs";
import { basename, dirname, relative, resolve, sep } from "node:path";
import { parseArgs } from "node:util";
import index from "./index.html";

const REPO = resolve(import.meta.dir, "../../../..");
const DATA = resolve(REPO, "data");

const { values, positionals } = parseArgs({
  args: Bun.argv.slice(2),
  options: { port: { type: "string", default: "3000" } },
  allowPositionals: true,
});
if (positionals.length > 1) {
  console.error("usage: bun run server.ts [points.json] [--port 3000]");
  process.exit(1);
}

/** Newest points file from export.py (skips older formats without stable image ids). */
async function latestPoints(): Promise<string> {
  const found: { path: string; mtime: number }[] = [];
  for await (const rel of new Bun.Glob("*/*/explorer/*.json").scan(resolve(REPO, "runs/latent_analysis"))) {
    const path = resolve(REPO, "runs/latent_analysis", rel);
    if ("ids" in (await Bun.file(path).json())) found.push({ path, mtime: statSync(path).mtimeMs });
  }
  if (!found.length) {
    console.error("No explorer points file found; run projects/latent_analysis/export.py first.");
    process.exit(1);
  }
  found.sort((a, b) => b.mtime - a.mtime);
  if (found.length > 1)
    console.log(`other points files (pass one as an argument to open it):\n${found.slice(1).map((f) => `  ${relative(REPO, f.path)}`).join("\n")}`);
  return found[0]!.path;
}

const pointsPath = positionals[0] ? resolve(positionals[0]) : await latestPoints();
const points = await Bun.file(pointsPath).json();
const labelsPath = resolve(REPO, points.labelsFile);
const rejectionsPath = resolve(dirname(labelsPath), "rejections.json");
const split = basename(pointsPath, ".json");
const runDir = resolve(dirname(pointsPath), "..");
const predictionsPath = resolve(runDir, "autotag", "predictions.json");

type Clustering = { k: number; pathsKey: string; cluster: number[]; dist: number[] };

/** Every k-means run for this model and split whose image order matches the points. */
async function clusterings(): Promise<Record<string, Clustering>> {
  const out: Record<string, Clustering> = {};
  for await (const rel of new Bun.Glob(`k*/clusters_${split}.json`).scan(runDir)) {
    const c: Clustering = await Bun.file(resolve(runDir, rel)).json();
    if (c.pathsKey === points.pathsKey && c.cluster.length === points.ids.length) out[c.k] = c;
    else console.warn(`skipping ${rel}: made from different embeddings (re-run kmeans.py)`);
  }
  return out;
}

type Tags = Record<string, string[]>; // image id -> tags

function isTags(x: unknown): x is Tags {
  return (
    typeof x === "object" && x !== null && !Array.isArray(x) &&
    Object.keys(x).every((k) => /^\d+$/.test(k)) &&
    Object.values(x).every((v) => Array.isArray(v) && v.every((t) => typeof t === "string"))
  );
}

type Kind = "labels" | "rejections";
const tagFiles: Record<Kind, string> = { labels: labelsPath, rejections: rejectionsPath };

async function readTags(kind: Kind): Promise<Tags> {
  const f = Bun.file(tagFiles[kind]);
  return (await f.exists()) ? f.json() : {};
}

async function writeTags(kind: Kind, tags: Tags) {
  // Write to a temp file then rename so a crash never truncates tags.
  const path = tagFiles[kind];
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.${process.pid}.tmp`;
  await Bun.write(tmp, JSON.stringify(tags, null, 1));
  renameSync(tmp, path);
}

// Patches are applied one at a time (across both files) so concurrent requests can't
// interleave read/write.
let queue: Promise<unknown> = Promise.resolve();
function applyPatch(kind: Kind, patch: Tags): Promise<Tags> {
  const run = queue.then(async () => {
    const other: Kind = kind === "labels" ? "rejections" : "labels";
    const [tags, opposite] = await Promise.all([readTags(kind), readTags(other)]);
    let oppositeChanged = false;
    for (const [id, t] of Object.entries(patch)) {
      // A newly accepted tag drops its rejection; a newly rejected tag drops its label.
      const added = t.filter((x) => !tags[id]?.includes(x));
      const rest = opposite[id]?.filter((x) => !added.includes(x));
      if (rest && rest.length !== opposite[id]!.length) {
        if (rest.length) opposite[id] = rest; else delete opposite[id];
        oppositeChanged = true;
      }
      if (t.length) tags[id] = [...new Set(t)]; else delete tags[id];
    }
    await writeTags(kind, tags);
    if (oppositeChanged) await writeTags(other, opposite);
    return tags;
  });
  queue = run.catch(() => {});
  return run;
}

function tagRoutes(kind: Kind) {
  return {
    GET: async () => Response.json(await readTags(kind)),
    // Body: {id: tags} for the images that changed; [] removes an image's entry.
    // Responds with the full merged map so the client also picks up other sessions' edits.
    PATCH: async (req: Request) => {
      const body = await req.json();
      if (!isTags(body)) return new Response("expected {id: string[]}", { status: 400 });
      return Response.json(await applyPatch(kind, body));
    },
  };
}

/** autotag.py's predictions for this model, with scores cut down to the loaded points;
 * null if autotag.py hasn't been run for it. */
async function predictions() {
  const f = Bun.file(predictionsPath);
  if (!(await f.exists())) return null;
  const p = await f.json();
  const scores: Record<string, Record<string, number>> = {};
  for (const id of points.ids as number[]) if (p.scores[id]) scores[id] = p.scores[id];
  return { ...p, scores };
}

const server = Bun.serve({
  port: Number(values.port),
  development: true,
  routes: {
    "/": index,
    "/api/points": async () => Response.json({ ...points, clusterings: await clusterings() }),
    "/api/labels": tagRoutes("labels"),
    "/api/rejections": tagRoutes("rejections"),
    "/api/predictions": async () => Response.json(await predictions()),
    // Repo-relative file paths, restricted to <repo>/data.
    "/file/*": async (req) => {
      const rel = decodeURIComponent(new URL(req.url).pathname.slice("/file/".length));
      const path = resolve(REPO, rel);
      if (!path.startsWith(DATA + sep)) return new Response("forbidden", { status: 403 });
      const f = Bun.file(path);
      if (!(await f.exists())) return new Response("not found", { status: 404 });
      return new Response(f, { headers: { "Cache-Control": "max-age=86400" } });
    },
  },
});

console.log(`${points.dataset}/${points.split} · ${points.model} · ${points.paths.length} points`);
console.log(`k: ${Object.keys(await clusterings()).join(", ") || "none (run kmeans.py)"}`);
console.log(`tags: ${labelsPath}`);
const pred = await predictions();
console.log(`autotag: ${pred ? `${Object.keys(pred.tags).length} tags from ${pred.version}` : "none (run autotag.py)"}`);
console.log(`explorer at ${server.url}`);
