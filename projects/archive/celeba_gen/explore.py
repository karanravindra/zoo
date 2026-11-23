# /// notebook
# Run with: uv run marimo edit projects/celeba_gen/explore.py
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CelebA-HQ embeddings

    SigLIP 2 image embeddings of all 30,000 CelebA-HQ faces, written by `embed.py`. Two linked views of the same
    embeddings, both over the top 8 principal components, and both colored by the folder labels:

    - **Tour and UMAP**: a dtour tour through the PCs (left) next to a UMAP (right). Lasso in either view, or click a
      class in the legend.
    - **Pair plot**: a scatterplot matrix of the PCs, with box and lasso selection over every face.

    The embeddings must exist first: `uv run projects/celeba_gen/embed.py`.
    """)
    return


@app.cell
def _():
    import dtour
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    from PIL import Image
    from sklearn.decomposition import PCA
    from wigglystuff import PairPlot

    from zoo.paths import REPO_ROOT

    return Image, PCA, PairPlot, REPO_ROOT, dtour, mo, np, pl, torch


@app.cell
def _(REPO_ROOT, np, pl, torch):
    MODEL = "siglip2-base-256"
    LATENTS = REPO_ROOT / "data" / "latents" / "celeba_hq" / f"vl-{MODEL}"
    SPLITS = ("train", "val")

    parts = {s: torch.load(LATENTS / f"{s}.pt", weights_only=True) for s in SPLITS}
    classes = parts["train"]["classes"]
    assert parts["val"]["classes"] == classes, "train and val class lists differ"
    text = torch.load(LATENTS / "text.pt", weights_only=True)
    assert text["classes"] == classes, "text.pt was embedded with different classes"

    X = np.concatenate([parts[s]["features"].numpy() for s in SPLITS])
    T = text["features"].numpy()
    label_ids = np.concatenate([parts[s]["labels"].numpy() for s in SPLITS])
    df = pl.DataFrame(
        {
            "split": np.concatenate([[s] * len(parts[s]["labels"]) for s in SPLITS]),
            "class": [classes[i] for i in label_ids],
            "path": sum((parts[s]["paths"] for s in SPLITS), []),
        }
    )
    return LATENTS, T, X, classes, df


@app.cell
def _(LATENTS, PCA, X, df, np, pl):
    PCS = [f"pc{i}" for i in range(1, 9)]
    pcs = PCA(n_components=len(PCS), random_state=0).fit_transform(X)

    # UMAP takes a minute or so on CPU, so the 2-D result is cached next to the embeddings.
    cache = LATENTS / "umap-2d.npy"
    if cache.exists():
        xy = np.load(cache)
    else:
        import umap

        xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X)
        np.save(cache, xy)

    table = df.with_columns(
        [pl.Series(name, pcs[:, i]) for i, name in enumerate(PCS)]
        + [pl.Series("umap_x", xy[:, 0]), pl.Series("umap_y", xy[:, 1])]
    )
    return PCS, pcs, table


@app.cell
def _(T, X, classes, mo, np, table):
    # Zero-shot: each image gets the class whose prompt embedding is closest. Compared with the folder label.
    zero_shot = np.array(classes)[(X @ T.T).argmax(axis=1)]
    correct = zero_shot == table["class"].to_numpy()
    accuracy = {s: correct[table["split"].to_numpy() == s].mean() for s in ("train", "val")}
    mo.md(
        "**Zero-shot accuracy** (`a photo of a {class}`): "
        + ", ".join(f"{s} {a:.1%}" for s, a in accuracy.items())
    )
    return (zero_shot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Tour and UMAP

    The tour animates through the PCs. The UMAP is on the right. Selections are linked.
    """)
    return


@app.cell
def _(PCS, classes, dtour, table):
    class_colors = dtour.build_color_map(classes, theme="light")

    tour_widget = dtour.Widget(
        data=table.select(*PCS, "class", "split"),
        tour=dtour.little_tour(table.select(PCS)),
        tour_dimensions=PCS,
        point_color_by="class",
        color_map=class_colors,
        point_opacity=0.5,
        preview_size="small",
        preview_count=8,
        theme_mode="light",
        height=600,
    )
    umap_widget = dtour.Widget(
        data=table.select("umap_x", "umap_y", "class", "split"),
        tour_dimensions=["umap_x", "umap_y"],
        point_color_by="class",
        color_map=class_colors,
        point_opacity=0.5,
        show_legend=False,
        theme_mode="light",
        height=600,
    )
    dtour.link(tour_widget, umap_widget)
    return tour_widget, umap_widget


@app.cell
def _(mo, tour_widget, umap_widget):
    mo.hstack([tour_widget, umap_widget], widths="equal", gap=0)
    return


@app.cell
def _(mo, np, table, tour_widget, umap_widget):
    def selected_rows(widget):
        """Rows selected in `widget`, by lasso or by clicking a class in its legend."""
        if widget.selected_labels:
            return np.flatnonzero(table["class"].is_in(widget.selected_labels).to_numpy())
        return np.asarray(widget.selected_indices, dtype=int)

    get_rows, set_rows = mo.state(np.array([], dtype=int))
    for _w in (tour_widget, umap_widget):
        _w.observe(lambda _, w=_w: set_rows(selected_rows(w)), names=["selected_indices", "selected_labels"])
    return (get_rows,)


@app.cell
def _(Image, REPO_ROOT, get_rows, mo, np, pl, table, zero_shot):
    _rows = get_rows()
    if len(_rows) == 0:
        _view = mo.md("_Lasso faces in either view, or click a class in the tour's legend, to see them here._")
    else:
        _mask = np.zeros(len(table), dtype=bool)
        _mask[_rows] = True
        _picked = table.filter(pl.Series(_mask))
        _composition = _picked.group_by("class").len().sort("len", descending=True)
        _agree = (zero_shot[_rows] == _picked["class"].to_numpy()).mean()

        _shown = _rows[:120]
        _tiles = [
            mo.image(Image.open(REPO_ROOT / table["path"][int(i)]).convert("RGB").resize((96, 96))) for i in _shown
        ]
        _view = mo.vstack(
            [
                mo.md(
                    f"**{len(_rows):,} selected**. Zero-shot agrees with the folder label on {_agree:.1%}. "
                    f"Showing the first {len(_shown)}."
                ),
                mo.ui.table(_composition, selection=None),
                mo.hstack(_tiles, wrap=True, gap=0.25),
            ]
        )
    _view
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Try this:**

    - Press play on the tour and watch which projection separates `male` from `female` best.
    - Lasso a region of the UMAP where the two classes overlap. The faces appear below, with the zero-shot agreement.
    - Click a class in the tour's legend to see its faces in the UMAP.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Pair plot

    A scatterplot matrix of the same PCs, with linked box and lasso selection over all 30,000 faces.

    - **Drag** inside any cell to draw a box. The selected faces keep their color in every cell.
    - **Keep** narrows the plot to the selection and **Exclude** drops it. Filters stack; **Reset** brings every row back.
    - **Lasso** switches to a freehand outline.
    """)
    return


@app.cell(expand_output=True)
def _(PCS, PairPlot, df, mo, pcs):
    pair_plot = PairPlot(
        pcs.astype("float32"),
        columns=PCS,
        color=df["class"].to_numpy(),
    )
    plot = mo.ui.anywidget(pair_plot)
    plot
    return pair_plot, plot


@app.cell
def _(Image, REPO_ROOT, T, X, classes, df, mo, np, pair_plot, pl, plot):
    _selection = plot.value["selection"]
    _filters = plot.value["filters"]
    _remaining = int(pair_plot.get_active_mask().sum())
    _filter_note = (
        f"{len(_filters)} filter(s): " + ", ".join(step["action"] for step in _filters) + f" → {_remaining:,} rows remain."
        if _filters
        else ""
    )

    if not _selection:
        _view = mo.md(_filter_note or "_Drag a box in any cell to select faces._")
    else:
        _idx = np.asarray(pair_plot.get_indices(), dtype=int)
        _mask = np.zeros(len(df), dtype=bool)
        _mask[_idx] = True
        _picked = df.filter(pl.Series(_mask))
        _composition = _picked.group_by("class", "split").len().sort("len", descending=True)

        _zero_shot = np.array(classes)[(X[_idx] @ T.T).argmax(axis=1)]
        _agree = (_zero_shot == _picked["class"].to_numpy()).mean()

        _shown = _idx[:60]
        _tiles = [
            mo.image(Image.open(REPO_ROOT / df["path"][int(i)]).convert("RGB").resize((96, 96))) for i in _shown
        ]
        _view = mo.vstack(
            [
                mo.md(
                    f"{_filter_note}\n\n**{len(_idx):,}** of {_remaining:,} faces selected with a "
                    f"{_selection['type']} in `{pair_plot.columns[_selection['x_index']]}` × "
                    f"`{pair_plot.columns[_selection['y_index']]}`. "
                    f"Zero-shot agrees with the folder label on {_agree:.1%}. Showing the first {len(_shown)}."
                ),
                mo.ui.table(_composition, selection=None),
                mo.hstack(_tiles, wrap=True, gap=0.25),
            ]
        )
    _view
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
