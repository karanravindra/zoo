# /// notebook
# Run with: uv run marimo edit projects/tagger/app.py -- --name <dataset> [--model <embedding>]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tagger

    A tour through the top PCs next to a UMAP of the same embeddings, linked: lasso points in any plot (or click a
    class in the legend), look at what you picked, and give it a tag. Tags are saved to `tags.parquet` in the dataset
    folder on every change. An item can carry several tags. Hover a point to see it.

    - **Zoom in** re-fits the PCA and UMAP on the selection alone, which separates structure the global view squeezes.
    - **Find similar** lists the items nearest to the selection (its centroid, if several) and can select them.
    - **Compare embedding** adds a UMAP of a second embedding of the same items, linked to the others.
    - **Probes** train a logistic regression per task on your tags, then suggest labels and pick the items to label next.

    Embed first: `uv run projects/tagger/cli.py embed --name <dataset> --source <folder|file|hf id>`.
    """)
    return


@app.cell
def _():
    import dtour
    import marimo as mo
    import numpy as np
    import polars as pl

    import probe
    import projection
    import store

    return dtour, mo, np, pl, probe, projection, store


@app.cell
def _(mo, store):
    _args = mo.cli_args()
    _names = store.list_datasets()
    mo.stop(not _names, mo.md("**No datasets yet.** Run the `embed` command first."))
    name_picker = mo.ui.dropdown(_names, value=_args.get("name") or _names[0], label="Dataset")
    name_picker
    return (name_picker,)


@app.cell
def _(mo, name_picker, store):
    _models = store.list_models(name_picker.value)
    mo.stop(not _models, mo.md(f"**{name_picker.value}** has no embeddings: run `embed` on it."))
    _want = mo.cli_args().get("model")
    model_picker = mo.ui.dropdown(_models, value=_want if _want in _models else _models[0], label="Embedding")
    model_picker
    return (model_picker,)


@app.cell
def _(mo, model_picker, name_picker, store):
    _others = [m for m in store.list_models(name_picker.value) if m != model_picker.value]
    cmp_picker = mo.ui.dropdown(["(none)", *_others], value="(none)", label="Compare embedding")
    cmp_picker
    return (cmp_picker,)


@app.cell
def _(model_picker, name_picker, np, pl, probe, projection, store):
    name, model = name_picker.value, model_picker.value
    items = store.load_items(name)
    meta = store.load_meta(name)
    X = store.load_embeddings(name, model)
    assert len(X) == len(items), "embedding and items.parquet differ in length: re-run embed with --overwrite"
    # What the hover tooltip shows for an item: its picture (a thumbnail; cached after the first load) or its text.
    if meta["modality"] == "image":
        items = items.with_columns(image=pl.Series(store.load_thumbs(name, items)))
    else:
        items = items.with_columns(text=pl.col("value").str.slice(0, 300))

    PCS = [f"pc{i}" for i in range(1, 9)][: X.shape[1]]
    _pcs = projection.pca(X, len(PCS))
    _xy = projection.umap_2d(name, model, X)  # cached; `cli.py project` precomputes it with other settings
    base = items.with_columns(
        *[pl.Series(c, _pcs[:, i]) for i, c in enumerate(PCS)],
        umap_x=pl.Series(_xy[:, 0]),
        umap_y=pl.Series(_xy[:, 1]),
    )
    # State the buttons below change, kept in a plain dict so that changing it does not rebuild the (selection-holding)
    # widgets: tags, the current selection, the probes' tasks and results, and the last "find similar" result.
    _tasks = probe.load_tasks(name)
    holder = {
        "tags": store.load_tags(name),
        "rows": np.array([], dtype=int),
        "tasks": _tasks,
        "task_sel": next(iter(_tasks), None),
        "models": {},
        "pred": None,
        "sim": None,
        "msg": "",
    }
    return PCS, X, base, holder, meta, name


@app.cell
def _(cmp_picker, name_picker, projection, store):
    if cmp_picker.value == "(none)":
        cmp_xy = None
    else:
        _name, _m = name_picker.value, cmp_picker.value
        cmp_xy = projection.umap_2d(_name, _m, store.load_embeddings(_name, _m))
    return (cmp_xy,)


@app.cell
def _(base, mo):
    # Columns with few enough values to color by.
    _cols = [c for c in base.columns if c not in ("id", "value") and not c.startswith(("pc", "umap_"))]
    _cols = [c for c in _cols if base[c].n_unique() <= 50]
    color_by = mo.ui.dropdown(["(none)", "tag", "prediction", *_cols], value="(none)", label="Color by")
    recolor = mo.ui.button(label="Recolor (picks up new tags and predictions)")
    mo.hstack([color_by, recolor], justify="start")
    return color_by, recolor


@app.cell
def _(PCS, base, cmp_xy, color_by, dtour, holder, np, pl, recolor):
    recolor.value  # rebuilding the widgets drops the selection, so it only happens when asked
    _tags = holder["tags"].group_by("id").agg(pl.col("tag").sort().str.join(", ").alias("tag"))
    table = base.join(_tags, on="id", how="left").with_columns(pl.col("tag").fill_null("untagged"))
    _pred = holder["pred"]
    _n = len(table)
    table = table.with_columns(
        prediction=pl.Series(_pred["label"].tolist() if _pred else ["(no model)"] * _n, dtype=pl.String),
        confidence=pl.Series([f"{c:.2f}" for c in _pred["conf"]] if _pred else [""] * _n, dtype=pl.String),
    )
    if cmp_xy is not None:
        table = table.with_columns(cmp_x=pl.Series(cmp_xy[:, 0]), cmp_y=pl.Series(cmp_xy[:, 1]))

    _color = color_by.value
    # The tooltip, in order. A prediction only shows once a probe has been trained.
    _hover = ["image", "text", "tag", *(["prediction", "confidence"] if _pred else []), "source", "label"]
    _hover = [c for c in _hover if c in table.columns]
    kw = {"tooltip_columns": _hover}
    if _color != "(none)":
        table = table.with_columns(pl.col(_color).cast(pl.String).fill_null("(missing)").alias(_color))
        kw["point_color_by"] = _color
        kw["color_map"] = dtour.build_color_map(sorted(table[_color].unique().to_list()), theme="light")
    show_cols = list(dict.fromkeys([*([_color] if _color != "(none)" else []), *_hover]))
    # The hover picture is inlined in the page, so only the UMAP carries it; the other plots hover as text.
    kw_lite = {**kw, "tooltip_columns": [c for c in _hover if c != "image"]}
    lite_cols = [c for c in show_cols if c != "image"]

    tour_widget = dtour.Widget(
        data=table.select(*PCS, *lite_cols),
        tour=dtour.little_tour(table.select(PCS)),
        tour_dimensions=PCS,
        point_opacity=0.5,
        theme_mode="light",
        height=560,
        **kw_lite,
    )
    umap_widget = dtour.Widget(
        data=table.select("umap_x", "umap_y", *show_cols),
        tour_dimensions=["umap_x", "umap_y"],
        point_opacity=0.5,
        show_legend=False,
        theme_mode="light",
        height=760,
        **kw,
    )
    cmp_widget = None
    if cmp_xy is not None:
        cmp_widget = dtour.Widget(
            data=table.select("cmp_x", "cmp_y", *lite_cols),
            tour_dimensions=["cmp_x", "cmp_y"],
            point_opacity=0.5,
            show_legend=False,
            theme_mode="light",
            height=560,
            **kw_lite,
        )
    # All of them show the same items in the same order, so their selections link. Keep the return value: a bare call
    # would show it as the cell's output.
    _unlink = dtour.link(*[w for w in (tour_widget, umap_widget, cmp_widget) if w is not None])
    return cmp_widget, kw, kw_lite, lite_cols, show_cols, table, tour_widget, umap_widget


@app.cell
def _(cmp_picker, cmp_widget, mo, tour_widget, umap_widget):
    _lower = [mo.vstack([mo.md("**Tour** (top PCs)"), tour_widget])]
    if cmp_widget is not None:
        _lower.append(mo.vstack([mo.md(f"**UMAP of {cmp_picker.value}**"), cmp_widget]))
    mo.vstack(
        [mo.md("**UMAP**"), umap_widget, mo.hstack(_lower, widths="equal", gap=1)],
        gap=1,
    )
    return


@app.cell
def _(cmp_widget, color_by, holder, mo, np, table, tour_widget, umap_widget):
    _color = color_by.value
    _main = [w for w in (tour_widget, umap_widget, cmp_widget) if w is not None]
    get_rows, set_rows = mo.state(np.array([], dtype=int))

    def observe(widgets, frame, mapping=None):
        """Keep `holder["rows"]` (the items to tag) at the selection of `widgets`, which show the rows of `frame`;
        `mapping` turns a position in `frame` into an item id (None: it is the item id)."""

        def selected(w):
            if w.selected_labels and _color != "(none)":
                pos = np.flatnonzero(frame[_color].is_in(w.selected_labels).to_numpy())
            else:
                pos = np.asarray(w.selected_indices, dtype=int)
            return pos if mapping is None else mapping[pos]

        def on_change(_, w):
            holder["rows"] = selected(w)
            set_rows(holder["rows"])

        for w in widgets:
            w.observe(lambda c, w=w: on_change(c, w), names=["selected_indices", "selected_labels"])

    observe(_main, table)

    def select_rows(rows):
        """Select items in the main plots, as if lassoed."""
        for w in _main:
            w.selected_indices = [int(r) for r in rows]

    return get_rows, observe, select_rows


@app.cell
def _(mo):
    tag_input = mo.ui.text(placeholder="tag name", label="Tag")
    img_size = mo.ui.slider(96, 256, value=192, step=16, label="Image size", show_value=True)
    get_version, set_version = mo.state(0)  # bumped when the tags, the tasks, or a probe/similar result change
    return get_version, img_size, set_version, tag_input


@app.cell
def _(holder, mo, name, pl, set_version, store, tag_input):
    def _apply(_, add: bool):
        tag, rows = tag_input.value.strip(), holder["rows"]
        if not tag or not len(rows):
            return
        _have = holder["tags"]
        _sel = pl.DataFrame({"id": rows.astype("int64"), "tag": tag})
        if add:
            _new = pl.concat([_have, _sel], how="vertical").unique()
        else:
            _new = _have.join(_sel, on=["id", "tag"], how="anti")
        holder["tags"] = _new
        store.save_tags(name, _new)
        set_version(lambda v: v + 1)

    add_btn = mo.ui.button(label="Add tag to selection", on_click=lambda _: _apply(_, True), kind="success")
    remove_btn = mo.ui.button(label="Remove tag from selection", on_click=lambda _: _apply(_, False), kind="danger")
    return add_btn, remove_btn


@app.cell
def _(X, holder, mo, np, select_rows, set_version):
    # Zoom: re-fit the PCA and UMAP on the selection.
    get_zoom, set_zoom = mo.state(None)

    def _zoom(_):
        if len(holder["rows"]) >= 15:
            set_zoom(holder["rows"].copy())

    zoom_btn = mo.ui.button(label="Zoom in on selection (15+ items)", on_click=_zoom)
    unzoom_btn = mo.ui.button(label="Close zoom", on_click=lambda _: set_zoom(None))

    # Find similar: the items nearest the selection's centroid, by cosine similarity (X is unit-norm).
    k_slider = mo.ui.slider(8, 120, value=48, step=4, label="Neighbors")

    def _similar(_):
        rows = holder["rows"]
        if not len(rows):
            return
        q = X[rows].mean(axis=0)
        sims = X @ (q / np.linalg.norm(q))
        sims[rows] = -np.inf  # the selection is not its own neighbor
        nn = np.argsort(-sims)[: k_slider.value]
        holder["sim"] = {"rows": nn, "sims": sims[nn], "query": len(rows)}
        set_version(lambda v: v + 1)

    similar_btn = mo.ui.button(label="Find similar to selection", on_click=_similar)
    select_sim_btn = mo.ui.button(
        label="Select the neighbors in the plots",
        on_click=lambda _: select_rows(holder["sim"]["rows"]) if holder["sim"] else None,
    )
    return get_zoom, k_slider, select_sim_btn, similar_btn, unzoom_btn, zoom_btn


@app.cell
def _(
    add_btn, get_rows, get_version, holder, img_size, k_slider, meta, mo, np, pl, remove_btn, select_rows,
    select_sim_btn, similar_btn, store, table, tag_input, unzoom_btn, zoom_btn,
):  # fmt: skip
    get_version()  # redraw after a tag is added or removed
    _rows = get_rows()
    _tags = holder["tags"].group_by("id").agg(pl.col("tag").sort().str.join(", ").alias("tags"))
    # Dropping items from the selection: tick some of the items shown, then drop them (or keep only them). The result
    # becomes the selection in the plots, so tagging, zoom, find-similar and the probes all use it.
    drop_ui, drop_ids, drop_btn, keep_btn = None, [], None, None
    if len(_rows) == 0:
        _view = mo.md("_Lasso points in any plot, or click a class in the tour's legend, to see them here._")
    else:
        _mask = np.zeros(len(table), dtype=bool)
        _mask[_rows] = True
        _picked = table.filter(pl.Series(_mask)).join(_tags, on="id", how="left")
        _shown = _picked.head(120)
        drop_ids = _shown["id"].to_list()
        if meta["modality"] == "image":
            drop_ui = mo.ui.array([mo.ui.checkbox() for _ in drop_ids])
            _body = mo.hstack(
                [
                    mo.vstack([mo.image(store.load_image(p, img_size.value)), c], gap=0.1, align="center")
                    for p, c in zip(_shown["value"], drop_ui.elements)
                ],
                wrap=True,
                gap=0.25,
            )
        else:
            drop_ui = mo.ui.table(
                _shown.select("id", pl.col("value").str.slice(0, 300), pl.col("tags").fill_null("")),
                selection="multi",
                page_size=10,
            )
            _body = drop_ui

        def _ticked():
            if meta["modality"] == "image":
                return [i for i, on in zip(drop_ids, drop_ui.value) if on]
            _v = drop_ui.value
            return [int(r["id"]) for r in (_v.iter_rows(named=True) if hasattr(_v, "iter_rows") else _v)]

        def _drop(_, keep: bool):
            _ids = _ticked()
            if not _ids:
                return
            _cur = holder["rows"]
            select_rows(np.array(_ids, dtype=int) if keep else _cur[~np.isin(_cur, _ids)])

        drop_btn = mo.ui.button(label="Drop ticked from selection", on_click=lambda _: _drop(_, False), kind="danger")
        keep_btn = mo.ui.button(label="Keep only ticked", on_click=lambda _: _drop(_, True))
        _view = mo.vstack(
            [
                mo.md(f"**{len(_rows):,} selected**. Showing the first {len(_shown)}."),
                mo.hstack([tag_input, add_btn, remove_btn], justify="start", align="end"),
                mo.hstack([zoom_btn, unzoom_btn, similar_btn, k_slider, select_sim_btn], justify="start", align="end"),
                mo.hstack([drop_btn, keep_btn, img_size, mo.md("_Tick items below first._")], justify="start", align="center"),
                _body,
            ]
        )
    _view
    return drop_btn, drop_ids, drop_ui, keep_btn


@app.cell
def _(PCS, X, dtour, get_zoom, kw, kw_lite, lite_cols, mo, observe, pl, projection, show_cols, table):
    _rows = get_zoom()
    mo.stop(_rows is None)
    _pcs = projection.pca(X[_rows], len(PCS))
    _xy = projection.umap_fit(X[_rows])
    _sub = (
        table[_rows.tolist()]
        .select("id", *show_cols)
        .with_columns(
            *[pl.Series(c, _pcs[:, i]) for i, c in enumerate(PCS)],
            zoom_x=pl.Series(_xy[:, 0]),
            zoom_y=pl.Series(_xy[:, 1]),
        )
    )
    zoom_tour = dtour.Widget(
        data=_sub.select(*PCS, *lite_cols),
        tour=dtour.little_tour(_sub.select(PCS)),
        tour_dimensions=PCS,
        point_opacity=0.6,
        theme_mode="light",
        height=520,
        **kw_lite,
    )
    zoom_umap = dtour.Widget(
        data=_sub.select("zoom_x", "zoom_y", *show_cols),
        tour_dimensions=["zoom_x", "zoom_y"],
        point_opacity=0.6,
        show_legend=False,
        theme_mode="light",
        height=520,
        **kw,
    )
    _unlink = dtour.link(zoom_tour, zoom_umap)
    observe([zoom_tour, zoom_umap], _sub, _rows)  # selecting here selects the same items for tagging
    mo.vstack(
        [
            mo.md(
                f"## Zoom: {len(_rows):,} items\n\nA tour and UMAP fit on the selection only. Lassoing here selects "
                "those items for tagging; the plots above keep their own selection."
            ),
            mo.hstack([zoom_tour, zoom_umap], widths="equal", gap=0),
        ]
    )
    return


@app.cell
def _(get_version, holder, img_size, meta, mo, pl, store, table):
    get_version()
    _sim = holder["sim"]
    if _sim is None:
        _out = None
    else:
        _near = table[_sim["rows"].tolist()]
        if meta["modality"] == "image":
            _body = mo.hstack(
                [
                    mo.image(
                        store.load_image(p, img_size.value), caption=f"{s:.2f}" + (f" {t}" if t != "untagged" else "")
                    )
                    for p, s, t in zip(_near["value"], _sim["sims"], _near["tag"])
                ],
                wrap=True,
                gap=0.25,
            )
        else:
            _body = mo.ui.table(
                _near.select("id", pl.col("value").str.slice(0, 300), "tag").with_columns(
                    similarity=pl.Series(_sim["sims"])
                ),
                selection=None,
                page_size=10,
            )
        _out = mo.vstack(
            [
                mo.md(
                    f"## Similar to the {_sim['query']:,} selected\n\nNearest {len(_near)} items by cosine similarity "
                    '(with their tags). "Select the neighbors" above puts them in the plots, ready to tag.'
                ),
                _body,
            ]
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Probes

    A **task** is a set of your tags used as classes: *species* = lion, tiger, jaguar. Save one, train it, and a
    logistic regression on the embedding learns to tell its tags apart, using the items that carry exactly one of them.
    Cross-validated accuracy tells you how separable the classes are; it is not an independent test if you tagged by
    lassoing clusters of this same embedding. A task needs at least two tags. For a yes/no attribute, tag both sides
    (`male` and `female`). Predictions only reach the plots on **Recolor** with *Color by → prediction*.
    """)
    return


@app.cell
def _(get_version, holder, mo, name, probe, set_version):
    get_version()
    _options = sorted(holder["tags"]["tag"].unique().to_list())
    task_name = mo.ui.text(placeholder="e.g. species", label="Task name")
    task_tags = mo.ui.multiselect(_options, label="Classes (tags)")

    def _save(_):
        _n, _t = task_name.value.strip(), list(task_tags.value)
        if not _n or len(_t) < 2:
            holder["msg"] = "A task needs a name and at least two tags."
        else:
            holder["tasks"][_n] = _t
            holder["task_sel"] = _n
            holder["models"].pop(_n, None)
            probe.save_tasks(name, holder["tasks"])
            holder["msg"] = f"Saved task {_n}: {', '.join(_t)}."
        set_version(lambda v: v + 1)

    save_task_btn = mo.ui.button(label="Save task", on_click=_save, kind="success")
    mo.hstack([task_name, task_tags, save_task_btn], justify="start", align="end")
    return


@app.cell
def _(X, get_version, holder, mo, np, probe, set_version):
    get_version()
    c_picker = mo.ui.dropdown(
        {"0.1": 0.1, "1": 1.0, "10": 10.0, "100": 100.0, "1000": 1000.0}, value="10", label="C (higher = weaker regularization)"
    )
    balanced = mo.ui.checkbox(value=True, label="Balance classes")

    def _set_pred(task):
        _res = holder["models"].get(task)
        if _res and "proba" in _res:
            _p = _res["proba"]
            holder["pred"] = {
                "task": task,
                "label": np.array(_res["classes"], dtype=object)[_p.argmax(1)],
                "conf": _p.max(1),
            }
        else:
            holder["pred"] = None

    def _train(task):
        _classes = holder["tasks"][task]
        _y, _conflicts = probe.labels(holder["tags"], _classes, len(X))
        try:
            _res = probe.train(X, _y, C=c_picker.value, balanced=balanced.value)
            _res["conflicts"] = _conflicts
        except ValueError as e:
            _res = {"error": str(e)}
        holder["models"][task] = _res

    def _train_selected(_):
        if holder["task_sel"]:
            _train(holder["task_sel"])
            _set_pred(holder["task_sel"])
            holder["msg"] = ""
            set_version(lambda v: v + 1)

    def _train_all(_):
        for _t in holder["tasks"]:
            _train(_t)
        _set_pred(holder["task_sel"])
        holder["msg"] = f"Trained {len(holder['tasks'])} tasks."
        set_version(lambda v: v + 1)

    def _pick(task):
        holder["task_sel"] = task
        _set_pred(task)
        set_version(lambda v: v + 1)

    task_picker = mo.ui.dropdown(list(holder["tasks"]), value=holder["task_sel"], label="Task", on_change=_pick)
    train_btn = mo.ui.button(label="Train this task", on_click=_train_selected, kind="success")
    train_all_btn = mo.ui.button(label="Train all tasks", on_click=_train_all)
    mo.hstack([task_picker, c_picker, balanced, train_btn, train_all_btn], justify="start", align="end")
    return


@app.cell
def _(X, get_version, holder, mo, name, np, pl, probe, select_rows, set_version, store):
    get_version()
    _task = holder["task_sel"]
    _res = holder["models"].get(_task) if _task else None
    _classes = (_res or {}).get("classes", [])
    n_show = mo.ui.slider(8, 200, value=48, step=4, label="How many")
    min_conf = mo.ui.slider(0.5, 0.99, value=0.9, step=0.01, label="Min confidence")
    cls_picker = mo.ui.dropdown(_classes, value=_classes[0] if _classes else None, label="Predicted as")

    def _unlabeled():
        _y, _ = probe.labels(holder["tags"], holder["tasks"][holder["task_sel"]], len(X))
        return _y == ""

    def _uncertain(_):
        if _res and "proba" in _res:
            select_rows(probe.uncertain(_res["proba"], _unlabeled(), n_show.value))

    def _by_class(_):
        if _res and "proba" in _res and cls_picker.value:
            _p = _res["proba"]
            _k = _classes.index(cls_picker.value)
            _ok = _unlabeled() & (_p.argmax(1) == _k) & (_p.max(1) >= min_conf.value)
            _rows = np.flatnonzero(_ok)
            select_rows(_rows[np.argsort(-_p[_rows, _k])][: n_show.value])

    def _accept(_):
        if not (_res and "proba" in _res):
            return
        _un = _unlabeled()
        _todo = [int(r) for r in holder["rows"] if _un[r]]
        if not _todo:
            holder["msg"] = "Nothing to accept: every selected item already has a tag in this task."
        else:
            _labels = np.array(_classes, dtype=object)[_res["proba"][_todo].argmax(1)]
            _add = pl.DataFrame({"id": _todo, "tag": _labels.tolist()}, schema=holder["tags"].schema)
            _new = pl.concat([holder["tags"], _add]).unique()
            holder["tags"] = _new
            store.save_tags(name, _new)
            holder["msg"] = f"Tagged {len(_todo)} selected items with the probe's predictions. Train again to use them."
        set_version(lambda v: v + 1)

    uncertain_btn = mo.ui.button(label="Select the most uncertain", on_click=_uncertain)
    class_btn = mo.ui.button(label="Select confident predictions", on_click=_by_class)
    accept_btn = mo.ui.button(label="Accept predictions for the selection", on_click=_accept, kind="success")
    return accept_btn, class_btn, cls_picker, min_conf, n_show, uncertain_btn


@app.cell
def _(
    X, accept_btn, class_btn, cls_picker, get_version, holder, min_conf, mo, n_show, np, pl, probe, uncertain_btn,
):  # fmt: skip
    get_version()
    _task = holder["task_sel"]
    _parts = []
    if holder["msg"]:
        _parts.append(mo.callout(mo.md(holder["msg"]), kind="info"))
    if not holder["tasks"]:
        _parts.append(mo.md("_No tasks yet. Pick two or more tags above and save a task._"))
    elif _task is None:
        _parts.append(mo.md("_Pick a task._"))
    else:
        _res = holder["models"].get(_task)
        _y, _conflicts = probe.labels(holder["tags"], holder["tasks"][_task], len(X))
        _counts = ", ".join(f"{t} {int((_y == t).sum())}" for t in holder["tasks"][_task])
        _note = f"; {_conflicts} carry several of its tags and are left out" if _conflicts else ""
        _parts.append(mo.md(f"**{_task}**: {_counts}; {int((_y == '').sum()):,} unlabeled{_note}."))
        if _res is None:
            _parts.append(mo.md("_Not trained yet._"))
        elif "error" in _res:
            _parts.append(mo.callout(mo.md(f"Could not train: {_res['error']}"), kind="warn"))
        else:
            _p = _res["proba"]
            _acc = (
                f"{_res['accuracy']:.1%} cross-validated accuracy ({_res['folds']}-fold)"
                if _res["accuracy"] is not None
                else "no cross-validation (a class has a single example)"
            )
            _un = _y == ""
            _dist = (
                pl.DataFrame({"predicted": np.array(_res["classes"], dtype=object)[_p[_un].argmax(1)].tolist()})
                .group_by("predicted")
                .len()
                .sort("len", descending=True)
                .rename({"len": "unlabeled items"})
            )
            _report = mo.ui.table(_res["report"], selection=None) if _res["report"] is not None else mo.md("-")
            _parts += [
                mo.md(
                    f"Trained on {_res['n']:,} labeled items: **{_acc}**. "
                    f"Mean confidence on the unlabeled: {_p[_un].max(1).mean():.2f}."
                ),
                mo.hstack(
                    [
                        mo.vstack([mo.md("Per class (cross-validated)"), _report]),
                        mo.vstack([mo.md("What it predicts for the unlabeled"), mo.ui.table(_dist, selection=None)]),
                    ],
                    justify="start",
                    gap=2,
                ),
                mo.md(
                    "**Pick what to label next.** It lands in the plots as the selection; tag it above, "
                    "or accept the predictions."
                ),
                mo.hstack(
                    [n_show, uncertain_btn, cls_picker, min_conf, class_btn, accept_btn],
                    justify="start",
                    align="end",
                    wrap=True,
                ),
            ]
    mo.vstack(_parts)
    return


@app.cell
def _(get_version, holder, mo, name, store):
    get_version()
    _counts = holder["tags"].group_by("tag").len().sort("len", descending=True).rename({"len": "items"})
    mo.vstack(
        [
            mo.md(f"## Tags\n\nSaved to `{store.dataset_dir(name) / 'tags.parquet'}`."),
            mo.ui.table(_counts, selection=None) if len(_counts) else mo.md("_No tags yet._"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
