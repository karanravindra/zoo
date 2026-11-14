# runs_viewer

A local experiment dashboard for everything under `<repo>/runs/`, in the style of TensorBoard / Weights & Biases:
live metric charts, run comparison, a runs table and per-run pages. Bun only, no dependencies to install.

```sh
bun run projects/runs_viewer/server.ts            # http://localhost:3001
bun run projects/runs_viewer/server.ts --port 4000
```

## What counts as a run

Any directory under `runs/`, at any depth, that holds `config.json`, `metrics.csv` or `results.json` (the files
every training script writes). Runs with the same parent path (e.g. `pixel_art/ae`) form a **workspace**, picked
in the top bar. Run state: **done** once `results.json` exists, **running** while `metrics.csv` has changed in the
last 2 minutes, otherwise **stopped**.

## Using it

- **Sidebar:** the workspace's runs, each with a fixed color, an eye toggle that shows or hides it in charts, and a
  search (by name or config value) that also filters the charts. Hovering a run highlights it everywhere. Click a
  name to open its run page.
- **Workspace tab:** one panel per `metrics.csv` column, in collapsible sections by prefix (Train / Val / Test /
  Other).
  - Hovering a chart lists every run's value at that step, sorted by value.
  - Drag across a chart to zoom the x range; double-click to reset.
  - Each panel has log-y and full-screen buttons.
  - The toolbar has a panel search (regex), TensorBoard-style smoothing (debiased EMA, with the raw curve drawn
    faintly behind it) and the x-axis column.
- **Add panel:** an editor with a live preview.
  - Data is either *per step* (each `metrics.csv` row) or *per run* (one point per run, from `config.*`,
    `results.*`, `meta.*` and the `final.`/`min.`/`max.` of each metric column).
  - Chart type is line, scatter or bar, with any x, one or more y fields, log x/y and a title.
- **Runs table tab:** sortable columns: state, last update, rows logged, summary (`results.json` plus the final
  value of each metric, with the best value per column highlighted) and the config fields that differ between
  runs.
- **Run page:** overview (path, state, git commit flagged when dirty, device), summary, config, and the run's charts.
- The page checks for changes every 3 s and redraws only what changed, so charts of a running job grow as
  `metrics.csv` does. Hidden runs, smoothing, x-axis, log scales, collapsed sections and custom panels are saved
  in the browser (localStorage).

## Files

`server.ts` (run discovery, JSON API, serves files under `runs/` except checkpoints), `index.html` (layout and
styles), `app.ts` (client, bundled by Bun; hand-drawn SVG charts).
