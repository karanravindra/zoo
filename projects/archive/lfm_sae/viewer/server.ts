/**
 * SAE feature viewer for lfm_sae: browse the latents of every SAE under runs/lfm_sae/train/.
 *
 *   bun run projects/archive/lfm_sae/viewer/server.ts [--port 3002] [--runs runs/lfm_sae/train]
 *
 * An SAE shows up once features.py has written <run>/sae_<n>/features/features.json. Optional files next to it:
 * labels.json (labels and notes from reading the latents), notes.md, umap.json and neighbors.json (layout.py).
 */

import { existsSync, statSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { parseArgs } from "node:util";
import index from "./index.html";

const REPO = resolve(import.meta.dir, "../../../..");

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    port: { type: "string", default: "3002" },
    runs: { type: "string", default: resolve(REPO, "runs/lfm_sae/train") }, // train_sae.py runs directory
  },
});
const TRAIN = resolve(values.runs!);

type Example = { doc: number; pos: number; act: number; source: string; left: string; token: string; right: string };
type Feature = {
  id: number; freq: number; mean_act: number; max_act: number; examples: Example[];
  promotes: string[]; suppresses: string[];
};
type Label = { label: string; category?: string; confidence?: string; explanation?: string };
type Sae = {
  key: string; run: string; n: number; dir: string; mtime: number;
  hook: string; tokens: number; docs: number; eval: Record<string, number> | null;
  features: Feature[]; labels: Record<string, Label>; notes: string;
  umap: { x: number[]; y: number[] } | null; neighbors: Record<string, { ids: number[][]; cos: number[][] }>;
};

const cache = new Map<string, Sae>();

async function readJson<T>(path: string, fallback: T): Promise<T> {
  return existsSync(path) ? ((await Bun.file(path).json()) as T) : fallback;
}

function mtimeOf(dir: string): number {
  let t = 0;
  for (const f of ["features.json", "labels.json", "notes.md", "umap.json", "neighbors.json"]) {
    const p = resolve(dir, f);
    if (existsSync(p)) t = Math.max(t, statSync(p).mtimeMs);
  }
  return t;
}

async function discover(): Promise<{ key: string; run: string; n: number; dir: string }[]> {
  if (!existsSync(TRAIN)) return [];
  const out = [];
  for await (const rel of new Bun.Glob("*/sae_*/features/features.json").scan(TRAIN)) {
    const dir = resolve(TRAIN, dirname(rel));
    const [run, sae] = relative(TRAIN, dir).split("/");
    out.push({ key: `${run}/${sae}`, run, n: Number(sae.replace("sae_", "")), dir });
  }
  return out.sort((a, b) => b.run.localeCompare(a.run) || a.n - b.n);
}

async function load(key: string): Promise<Sae | null> {
  const found = (await discover()).find((s) => s.key === key);
  if (!found) return null;
  const mtime = mtimeOf(found.dir);
  const hit = cache.get(key);
  if (hit && hit.mtime === mtime) return hit;
  const f = await readJson<any>(resolve(found.dir, "features.json"), null);
  const sae: Sae = {
    ...found, mtime, hook: f.hook, tokens: f.tokens, docs: f.docs, eval: f.eval ?? null, features: f.features,
    labels: await readJson(resolve(found.dir, "labels.json"), {}),
    notes: existsSync(resolve(found.dir, "notes.md")) ? await Bun.file(resolve(found.dir, "notes.md")).text() : "",
    umap: await readJson(resolve(found.dir, "umap.json"), null),
    neighbors: await readJson(resolve(found.dir, "neighbors.json"), {}),
  };
  cache.set(key, sae);
  return sae;
}

function siblingKey(sae: Sae, n: string) {
  return `${sae.run}/sae_${n}`;
}

const json = (x: unknown) => Response.json(x, { headers: { "cache-control": "no-store" } });

const server = Bun.serve({
  port: Number(values.port),
  routes: {
    "/": index,
    "/api/saes": async () => {
      const list = [];
      for (const s of await discover()) {
        const sae = await load(s.key);
        if (!sae) continue;
        list.push({
          key: sae.key, run: sae.run, n: sae.n, hook: sae.hook, tokens: sae.tokens, docs: sae.docs, eval: sae.eval,
          labeled: Object.keys(sae.labels).length, hasUmap: !!sae.umap, notes: sae.notes,
          alive: sae.features.filter((f) => f.freq > 0).length,
        });
      }
      return json(list);
    },
    // Compact columns for every latent; the client filters and sorts.
    "/api/list": async (req) => {
      const sae = await load(new URL(req.url).searchParams.get("sae") ?? "");
      if (!sae) return new Response("unknown sae", { status: 404 });
      const F = sae.features;
      return json({
        freq: F.map((f) => f.freq),
        max: F.map((f) => Math.round(f.max_act * 100) / 100),
        top: F.map((f) => (f.examples[0]?.token ?? "").slice(0, 24)),
        promotes: F.map((f) => f.promotes.slice(0, 3).join(" ")),
        label: F.map((f) => sae.labels[f.id]?.label ?? ""),
        category: F.map((f) => sae.labels[f.id]?.category ?? ""),
        umap: sae.umap,
      });
    },
    "/api/feature": async (req) => {
      const p = new URL(req.url).searchParams;
      const sae = await load(p.get("sae") ?? "");
      const id = Number(p.get("id"));
      const f = sae?.features[id];
      if (!sae || !f) return new Response("unknown feature", { status: 404 });
      const neighbors: Record<string, unknown[]> = {};
      for (const [n, nb] of Object.entries(sae.neighbors)) {
        const other = await load(siblingKey(sae, n));
        neighbors[n] = nb.ids[id].map((j, r) => {
          const g = other?.features[j];
          return {
            id: j, cos: nb.cos[id][r], freq: g?.freq ?? 0, label: other?.labels[j]?.label ?? "",
            top: g ? g.examples.slice(0, 3).map((e) => e.token) : [],
          };
        });
      }
      return json({ ...f, label: sae.labels[id] ?? null, neighbors });
    },
  },
  fetch: () => new Response("not found", { status: 404 }),
});

console.log(`SAE feature viewer at ${server.url}`);
