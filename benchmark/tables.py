# Reading the benchmark results and turning them into the tables and figures
# the summary reports.

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

# separates the two column blocks: the comparison against other tools, then our
# own alternative implementations
SPLIT = "┆"


def read_results():
    results = pd.read_csv(config.RESULTS_FILE)
    results["method"] = pd.Categorical(results["method"], config.method_order, ordered=True)
    return results


def unrestricted_forest(results):
    # the negative controls are run at two forest depths; keep only the unrestricted
    # forest (max_depth None), the same setting as the pattern probe, so the two
    # depths are never averaged into one number
    depth = results["params_json"].map(lambda s: json.loads(s).get("max_depth"))
    return results[depth.isna()]


def mean_per_method(results, dataset, benchmark, metrics, target=None):
    # one row per metric, one column per method, in the fixed method order.
    # ELLA has no embedding, so it runs its own gene-level test and lands under
    # "<benchmark>_native" -- same question, so it belongs in the same table.
    wanted = [benchmark, f"{benchmark}_native"]
    rows = results[(results.dataset == dataset) & (results.benchmark.isin(wanted))
                   & (results.metric.isin(metrics))]
    if target is not None:
        rows = rows[rows.target == target]
    wide = rows.pivot_table(index="metric", columns="method", values="value", observed=False)
    wide = wide.reindex([m for m in metrics if m in wide.index])
    return wide[[m for m in config.method_order if m in wide.columns and wide[m].notna().any()]]


def label_columns(wide, split=True):
    # put the competitor block first, then a divider, then our alternatives
    ordered = [m for m in config.method_order if m in wide.columns]
    out = wide[ordered].rename(columns=config.method_names)
    if split:
        left = [config.method_names[m] for m in ordered if m in config.competitors]
        right = [config.method_names[m] for m in ordered if m in config.alternatives]
        if left and right:
            out.insert(len(left), SPLIT, "")
    return out


def md_table(frame, floatfmt="{:.4f}", best=None, index_name=None):
    # best is None, "max" or "min" -- bolds the winning cell in each row
    frame = frame.copy()
    if frame.index.name or index_name:
        frame.insert(0, index_name or frame.index.name, frame.index.astype(str))
        frame = frame.reset_index(drop=True)

    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]

    # rows holding counts are written as counts -- these tables put one metric per
    # row, so a row of whole numbers is a count and should not read as 69.0000
    whole_row = []
    for r in range(len(frame)):
        vals = [frame[c].iloc[r] for c in numeric]
        finite = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        whole_row.append(bool(finite) and all(float(v).is_integer() for v in finite))

    cells = {}
    for c in frame.columns:
        out = []
        for r, v in enumerate(frame[c]):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append("")
            elif isinstance(v, (bool, np.bool_)):
                out.append("yes" if v else "no")
            elif isinstance(v, (int, np.integer, float, np.floating)) and c in numeric:
                out.append(f"{int(v):,}" if whole_row[r] else floatfmt.format(v))
            elif isinstance(v, (float, np.floating)):
                out.append(floatfmt.format(v))
            else:
                out.append(str(v))
        cells[c] = out

    if best in ("max", "min") and numeric:
        vals = frame[numeric].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            pick = np.nanargmax(vals, axis=1) if best == "max" else np.nanargmin(vals, axis=1)
        for row, j in enumerate(pick):
            if np.isfinite(vals[row]).any():
                c = numeric[int(j)]
                cells[c][row] = f"**{cells[c][row]}**"

    widths = [max(len(str(c)), *(len(x) for x in cells[c])) if len(frame) else len(str(c))
              for c in frame.columns]
    head = "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(frame.columns)) + " |"
    rule = "|" + "|".join("-" * (widths[i] + 1) + (":" if c in numeric else "-")
                          for i, c in enumerate(frame.columns)) + "|"
    body = ["| " + " | ".join(
        (cells[c][r].rjust(widths[i]) if c in numeric else cells[c][r].ljust(widths[i]))
        for i, c in enumerate(frame.columns)) + " |" for r in range(len(frame))]
    return "\n".join([head, rule] + body)


def show(frame, caption="", **kw):
    from IPython.display import Markdown, display
    text = md_table(frame, **kw)
    if caption:
        text = f"**{caption}**\n\n{text}"
    if SPLIT in frame.columns:
        text += ("\n\nLeft of the divider: SubCellSpace against the other published tools. "
                 "Right: our own alternative implementations of the same idea.")
    display(Markdown(text))


def bars(wide, ax, ylabel="", floor=0.0, annotate=True, labels=None):
    # one group per row of `wide`, one bar per method
    groups, methods = list(wide.index), list(wide.columns)
    width = min(0.8 / max(len(methods), 1), 0.3)
    x = np.arange(len(groups))
    # a gap where the table puts its divider, so the bars read in the same two
    # blocks: the published tools, then our own alternatives
    slots = []
    gap = 0
    for m in methods:
        if m in config.alternatives and gap == 0:
            gap = 0.6
        slots.append(len(slots) + gap)
    centre = (max(slots) + min(slots)) / 2
    for i, m in enumerate(methods):
        offset = (slots[i] - centre) * (width + 0.01)
        values = wide[m].to_numpy(dtype=float)
        hatch = "///" if config.method_block[m] == "ours" else None
        ax.bar(x + offset, values - floor, width, bottom=floor,
               color=config.palette[m], label=config.method_names[m],
               edgecolor="white", linewidth=0.8, hatch=hatch, zorder=3)
        if annotate:
            # `labels` lets a bar show a share while its annotation gives the raw
            # count, which is what the two halves of a call need: the shares are
            # comparable across rows, the counts are what a reader quotes
            written = labels[m] if labels is not None else None
            for j, (xi, v) in enumerate(zip(x + offset, values)):
                if np.isfinite(v):
                    text = (written[j] if written is not None else
                            f"{int(v):,}" if float(v).is_integer() else f"{v:.2f}")
                    ax.annotate(text, (xi, v), textcoords="offset points",
                                xytext=(0, 2), ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", lw=0.6, color="#e3e2df", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save_figure(fig, name, dpi=300):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(config.FIGURES / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    return config.FIGURES / f"{name}.png"


def confusion(counts, ax, title="", row_labels=True,
              truth=("patterned", "random"), call=("called patterned", "called random")):
    # a 2x2 of what the method called against what the panel says. Rows are the
    # ground truth, columns the call; both use the paper's patterned/random names
    import matplotlib as mpl
    grid = np.array([[counts["tp"], counts["fn"]],
                     [counts["fp"], counts["tn"]]], dtype=float)
    share = grid / grid.sum(axis=1, keepdims=True).clip(1)
    image = ax.imshow(share, cmap=mpl.colors.LinearSegmentedColormap.from_list(
        "b", ["#ffffff", "#2a78d6"]), vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center",
                    fontsize=15, color="white" if share[i, j] > 0.55 else "#0b0b0b")
    # keep every cell visible, including the empty ones
    for edge in (-0.5, 0.5, 1.5):
        ax.axhline(edge, color="#b8b7b3", lw=1.0)
        ax.axvline(edge, color="#b8b7b3", lw=1.0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(list(call), fontsize=9)
    ax.set_yticks([0, 1])
    # the rows are the same in every panel, so only the leftmost is labelled
    ax.set_yticklabels(list(truth) if row_labels else ["", ""], fontsize=9)
    ax.set_title(title, fontsize=10.5, loc="left", pad=8)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    return image
