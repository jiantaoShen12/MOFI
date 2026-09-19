#!/usr/bin/env python3
from __future__ import annotations
#python downstream/run_hc_author5_lineage_sankey.py --run-dir results/five_data_rerun_20260318_182337/rerun/hc_author5_2/coarse_dense/sample

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys

import anndata as ad
import matplotlib as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go

mpl.rcParams.update({
    "font.family": "Arial",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.tl.analysis_dense_time import _build_classifier_space_latent
from downstream.plotting.hc_author5_palette import AUTHOR5_COLORS, AUTHOR5_ORDER


def _export_plotly_html_with_edge(html_path: Path, *, width: int, height: int) -> None:
    """Render the original Plotly Sankey when Kaleido cannot start on Windows."""
    edge_candidates = (
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    )
    edge = next((path for path in edge_candidates if path.exists()), None)
    if edge is None:
        raise RuntimeError("Kaleido failed and Microsoft Edge was not found for static export")
    with tempfile.TemporaryDirectory(prefix="mofi_sankey_edge_") as profile:
        common = [
            str(edge),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--allow-file-access-from-files",
            f"--user-data-dir={profile}",
            "--virtual-time-budget=5000",
        ]
        subprocess.run(
            [
                *common,
                f"--window-size={int(width)},{int(height)}",
                f"--screenshot={html_path.with_suffix('.png')}",
                html_path.resolve().as_uri(),
            ],
            check=True,
            timeout=90,
        )
        dumped = subprocess.run(
            [
                *common,
                "--dump-dom",
                html_path.resolve().as_uri(),
            ],
            check=True,
            timeout=90,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
        match = re.search(r'(<svg[^>]*class="main-svg"[^>]*>.*?</svg>)', dumped, flags=re.DOTALL)
        if match is None:
            raise RuntimeError("Edge DOM export did not contain the Plotly main SVG")
        svg_text = match.group(1)
        if "xmlns=" not in svg_text.split(">", 1)[0]:
            svg_text = svg_text.replace(
                "<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1
            )
        html_path.with_suffix(".svg").write_text(svg_text, encoding="utf-8")
        subprocess.run(
            [
                *common,
                f"--print-to-pdf={html_path.with_suffix('.pdf')}",
                "--print-to-pdf-no-header",
                html_path.resolve().as_uri(),
            ],
            check=True,
            timeout=90,
        )


CELLTYPE_TO_AUTHOR = {
    "radial glial cell": "RG",
    "neural progenitor cell": "IPC",
    "inhibitory interneuron": "IN-fetal",
    "caudal ganglionic eminence derived interneuron": "IN-CGE",
    "medial ganglionic eminence derived interneuron": "IN-MGE",
}
ALLOWED_FORWARD = {
    "RG": {"RG", "IPC",},
    "IPC": {"IPC", "IN-fetal"},
    "IN-fetal": {"IN-fetal", "IN-CGE", "IN-MGE"},
    "IN-CGE": {"IN-CGE"},
    "IN-MGE": {"IN-MGE"},
}

ALLOWED_FORWARD = {
    "RG": {"RG", "IPC","IN-fetal", "IN-CGE", "IN-MGE"},
    "IPC": {"RG", "IPC","IN-fetal", "IN-CGE", "IN-MGE"},
    "IN-fetal": {"RG", "IPC","IN-fetal", "IN-CGE", "IN-MGE"},
    "IN-CGE": {"RG", "IPC","IN-fetal", "IN-CGE"},
    "IN-MGE": {"RG", "IPC","IN-fetal", "IN-CGE", "IN-MGE"},
}

TYPE_ORDER = {
    "RG": 0,
    "IPC": 1,
    "IN-fetal": 2,
    "IN-CGE": 3,
    "IN-MGE": 4,
}


def make_palette(n: int) -> list[str]:
    base = [
        "#B7D98C",
        "#4E8B57",
        "#F3BCCB",
        "#D96C93",
        "#7B6FB2",
    ]
    out: list[str] = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def hex_to_rgba(h: str, alpha: float = 0.48) -> str:
    h = h.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"


def fit_knn_backend(X: np.ndarray, y: np.ndarray, n_neighbors: int) -> tuple[str, object]:
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        algorithm="auto",
        metric="euclidean",
        n_jobs=-1,
    )
    model.fit(X, y)
    return "sklearn_knn", model


def predict_knn_backend(backend_name: str, backend: object, Xq: np.ndarray) -> np.ndarray:
    _ = backend_name
    model = backend  # type: ignore
    return np.asarray(model.predict(np.asarray(Xq, dtype=np.float32))).astype(str)


def build_long_transitions(
    pred_labels: list[np.ndarray],
    weights: np.ndarray,
    sampled_idx: list[int],
    *,
    time_labels: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    n = len(sampled_idx)
    if time_labels is None:
        time_labels = [str(i + 1) for i in range(n)]
    for i in range(n - 1):
        src = pred_labels[i]
        tgt = pred_labels[i + 1]
        w_source = weights[sampled_idx[i]]
        w_target = weights[sampled_idx[i + 1]]
        g = pd.DataFrame(
            {
                "source": src,
                "target": tgt,
                "value": w_source,
                "value_source_weighted": w_source,
                "value_target_weighted": w_target,
            }
        )
        g = g.groupby(["source", "target"], as_index=False)[
            ["value", "value_source_weighted", "value_target_weighted"]
        ].sum()
        g["source_total"] = g.groupby("source")["value_source_weighted"].transform("sum")
        g["proportion_source"] = np.where(
            g["source_total"] > 0,
            g["value_source_weighted"] / g["source_total"],
            0.0,
        )
        g["target_total"] = g.groupby("target")["value_target_weighted"].transform("sum")
        g["proportion_target"] = np.where(
            g["target_total"] > 0,
            g["value_target_weighted"] / g["target_total"],
            0.0,
        )
        stage_total = float(g["value"].sum()) if len(g) else 0.0
        g["proportion_stage"] = (g["value"] / stage_total) if stage_total > 0 else 0.0
        g["stage_from"] = time_labels[i]
        g["stage_to"] = time_labels[i + 1]
        rows.append(
            g[
                [
                    "stage_from",
                    "stage_to",
                    "source",
                    "target",
                    "value",
                    "source_total",
                    "target_total",
                    "proportion_source",
                    "proportion_target",
                    "proportion_stage",
                ]
            ]
        )
    if not rows:
        raise RuntimeError("No transitions aggregated; cannot build sankey.")
    return pd.concat(rows, axis=0, ignore_index=True)


def prune_with_constraints(
    long_df: pd.DataFrame,
    *,
    min_prop_source: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    raw = long_df.copy()
    pruned = raw[raw["proportion_source"] >= min_prop_source].copy()
    if pruned.empty:
        pruned = raw.sort_values("value", ascending=False).head(max(1, min(20, len(raw)))).copy()
    stats = {
        "n_edges_raw": int(raw.shape[0]),
        "n_edges_after_prop_prune": int((raw["proportion_source"] >= min_prop_source).sum()),
        "n_edges_final": int(pruned.shape[0]),
        "min_prop_source_threshold": float(min_prop_source),
    }
    return pruned.reset_index(drop=True), stats


def apply_forward_rule_filter(
    df: pd.DataFrame,
    *,
    allowed_forward_edges: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    keep = []
    for _, row in out.iterrows():
        s = str(row["source"])
        t = str(row["target"])
        allowed = set(allowed_forward_edges.get(s, {s})) | {s}
        keep.append(t in allowed)
    kept = out[np.asarray(keep, dtype=bool)].copy()
    stats = {
        "n_edges_before": int(out.shape[0]),
        "n_edges_after": int(kept.shape[0]),
        "n_edges_removed": int(out.shape[0] - kept.shape[0]),
    }
    return kept.reset_index(drop=True), stats


def predict_labels_on_indices(
    trajectories_main: np.ndarray,
    trajectories_sub: np.ndarray,
    sampled_idx: list[int],
    *,
    classifier: Any,
    train_space: str,
) -> list[np.ndarray]:
    pred_labels: list[np.ndarray] = []
    for tidx in sampled_idx:
        X_main = trajectories_main[tidx]
        X_sub = trajectories_sub[tidx]
        Xq = _build_classifier_space_latent(
            train_space=train_space,
            main_latent=X_main,
            sub_latent=X_sub,
            mapper=None,
            map_batch=2048,
        )
        yhat = np.asarray(classifier.predict(np.asarray(Xq, dtype=np.float32))).astype(str)
        mapped = np.asarray([CELLTYPE_TO_AUTHOR.get(v, v) for v in yhat], dtype=object)
        pred_labels.append(mapped)
    return pred_labels


def render_sankey_html(plot_df: pd.DataFrame, html_path: Path, *, write_static: bool = True) -> dict:
    plot_df = plot_df.copy()
    plot_df["source_node"] = plot_df["source"].astype(str) + "_T" + plot_df["stage_from"].astype(str)
    plot_df["target_node"] = plot_df["target"].astype(str) + "_T" + plot_df["stage_to"].astype(str)

    all_nodes = pd.unique(plot_df[["source_node", "target_node"]].values.ravel("K"))

    def parse_node(node: str) -> tuple[str, float]:
        ct, t = node.rsplit("_T", 1)
        return ct, float(t)

    sorted_nodes = sorted(
        all_nodes,
        key=lambda n: (parse_node(n)[1], TYPE_ORDER.get(parse_node(n)[0], 999), parse_node(n)[0]),
    )
    node_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
    plot_df["source_idx"] = plot_df["source_node"].map(node_to_idx)
    plot_df["target_idx"] = plot_df["target_node"].map(node_to_idx)

    base_types = sorted({parse_node(n)[0] for n in sorted_nodes}, key=lambda x: TYPE_ORDER.get(x, 999))
    palette = [AUTHOR5_COLORS.get(ct, c) for ct, c in zip(base_types, make_palette(len(base_types)))]
    color_map = {ct: palette[i] for i, ct in enumerate(base_types)}
    node_colors = [color_map.get(parse_node(n)[0], "#999999") for n in sorted_nodes]
    link_colors = [hex_to_rgba(color_map.get(s, "#999999"), 0.45) for s in plot_df["source"].astype(str).tolist()]

    time_to_nodes: dict[int, list[str]] = {}
    for n in sorted_nodes:
        t = parse_node(n)[1]
        time_to_nodes.setdefault(t, []).append(n)
    ordered_time_cols = sorted(time_to_nodes.keys())
    time_col_index = {t: i for i, t in enumerate(ordered_time_cols)}
    num_tp = len(ordered_time_cols)
    max_nodes_per_col = max(len(v) for v in time_to_nodes.values()) if time_to_nodes else 1

    x_positions: list[float] = []
    y_positions: list[float] = []
    for n in sorted_nodes:
        ct, t = parse_node(n)
        col_i = time_col_index[t]
        x = 0.04 + (col_i / max(1, num_tp - 1)) * 0.92
        col_nodes = sorted(
            time_to_nodes[t],
            key=lambda k: (TYPE_ORDER.get(parse_node(k)[0], 999), parse_node(k)[0]),
        )
        j = col_nodes.index(n)
        y = (j + 0.5) / max(1, len(col_nodes))
        x_positions.append(x)
        y_positions.append(y)

    pad = max(6, int(16 - 0.3 * max(0, max_nodes_per_col - 8)))
    thickness = max(12, int(18 - 0.25 * max(0, max_nodes_per_col - 8)))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=pad,
                    thickness=thickness,
                    line=dict(color="black", width=0.35),
                    label=sorted_nodes,
                    color=node_colors,
                    x=x_positions,
                    y=y_positions,
                ),
                link=dict(
                    source=plot_df["source_idx"],
                    target=plot_df["target_idx"],
                    value=plot_df["value"],
                    color=link_colors,
                ),
            )
        ]
    )

    width = max(2200, 500 + 320 * num_tp)
    height = max(2000, 500 + 130 * max_nodes_per_col)

    fig.update_layout(
        title_text="HC author5 lineage transition Sankey (simulated + classifier + mass)",
        font=dict(family="Arial", size=12),
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    # Inline Plotly so the release page and static renderer work offline.
    fig.write_html(str(html_path), include_plotlyjs=True)
    if write_static:
        # Kaleido 0.2.x is bundled in the release environment and works
        # offline on Windows.  Prefer it over launching a second Edge profile:
        # concurrent Jupyter sessions otherwise collide on Edge's singleton
        # lock and make static export appear nondeterministic.
        try:
            fig.write_image(str(html_path.with_suffix(".png")))
            fig.write_image(str(html_path.with_suffix(".pdf")))
        except Exception as exc:
            print(f"Kaleido export failed ({exc}); trying the local Edge renderer", flush=True)
            try:
                _export_plotly_html_with_edge(html_path, width=width, height=height)
            except Exception as edge_exc:
                # A pre-rendered static panel may already be distributed with
                # the release.  Keep it intact when a workstation blocks
                # browser subprocesses; the interactive HTML remains freshly
                # regenerated from the original transition table.
                static_paths = [html_path.with_suffix(ext) for ext in (".png", ".pdf", ".svg")]
                if all(path.exists() and path.stat().st_size > 0 for path in static_paths):
                    print(f"Static export unavailable ({edge_exc}); retaining existing static panel", flush=True)
                else:
                    raise
        for static_path in (
            html_path.with_suffix(".png"),
            html_path.with_suffix(".pdf"),
            html_path.with_suffix(".svg"),
        ):
            if not static_path.exists() or static_path.stat().st_size == 0:
                raise RuntimeError(f"Static Sankey export was not created: {static_path}")
    return {
        "num_time_points": int(num_tp),
        "max_nodes_per_column": int(max_nodes_per_col),
        "width": int(width),
        "height": int(height),
        "node_pad": int(pad),
        "node_thickness": int(thickness),
    }


def _load_classifier(run_dir: Path) -> tuple[Any, dict]:
    import pickle

    manifest_path = run_dir / "reports" / "classifiers" / "classifier_manifest.json"
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    cls_path = Path(str(meta["classifier_path"]))
    with cls_path.open("rb") as f:
        payload = pickle.load(f)
    return payload["classifier"], meta


def _load_trajectories(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    main = np.load(run_dir / "assets" / "traj_main.npy")
    sub = np.load(run_dir / "assets" / "traj_sub.npy")
    weights = np.load(run_dir / "assets" / "weights.npy")
    sampled_idx = list(range(main.shape[0]))
    return main, sub, weights, sampled_idx


def _interp_axis(arr: np.ndarray, pos: float) -> np.ndarray:
    """Linearly interpolate arr at position pos along axis 0.
    pos=0 means first step, pos=n-1 means last step.
    """
    n = arr.shape[0]
    lo = int(np.clip(np.floor(pos), 0, n - 2))
    hi = lo + 1
    alpha = pos - lo
    return (1.0 - alpha) * arr[lo] + alpha * arr[hi]


def _interpolate_trajectories(
    main: np.ndarray, sub: np.ndarray, weights: np.ndarray,
    original_indices: list[int],
    interp_positions: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Insert interpolated steps at specified positions into trajectory arrays.

    interp_positions are fractional indices into the original array, e.g.
    3.5 means halfway between step 3 and step 4.
    """
    n = main.shape[0]
    insert = sorted(set(interp_positions))
    insert = [p for p in insert if 0 < p < n - 1]
    if not insert:
        return main, sub, weights, original_indices

    positions = sorted(set(original_indices) | set(insert))
    new_main = np.stack([_interp_axis(main, p) for p in positions])
    new_sub = np.stack([_interp_axis(sub, p) for p in positions])
    new_weights = np.stack([_interp_axis(weights, p) for p in positions])
    new_idx = list(range(len(positions)))
    return new_main, new_sub, new_weights, new_idx


def render(**overrides: Any) -> Path:
    """Render the author-5 lineage from Python without a CLI process."""

    if "run_dir" in overrides:
        args = parse_args(["--run-dir", str(overrides["run_dir"])])
    else:
        args = parse_args()
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise TypeError(f"Unknown lineage option: {key}")
        setattr(args, key, value)
    main(args)
    return (Path(args.run_dir).resolve() / "figures" / "author5_lineage")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HC author5 lineage Sankey from simulated trajectories.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-prop-source", type=float, default=0.08)
    parser.add_argument(
        "--skip-static", action="store_true",
        help="Write the interactive HTML without invoking the optional Kaleido exporter.",
    )
    parser.add_argument(
        "--time-labels", type=str, default=None,
        help="Comma-separated time labels for each step, e.g. '0,1,2,3,4' or '0,1,2,3,3.5,4'. "
             "If used with --interpolate-times, labels count must include the interpolated steps.",
    )
    parser.add_argument(
        "--interpolate-times", type=str, default=None,
        help="Comma-separated fractional positions to interpolate, e.g. '3.5' inserts a new step "
             "halfway between original step 3 and step 4 (0-indexed). Multiple values supported.",
    )
    return parser.parse_args(argv)


def main(args: Optional[argparse.Namespace] = None) -> None:
    args = parse_args() if args is None else args

    run_dir = args.run_dir.resolve()
    outdir = run_dir / "figures" / "author5_lineage"
    outdir.mkdir(parents=True, exist_ok=True)

    classifier, cls_meta = _load_classifier(run_dir)
    train_space = str(cls_meta.get("train_space", "joint")).strip().lower()
    traj_main, traj_sub, weights, sampled_idx = _load_trajectories(run_dir)

    interp_positions = None
    if args.interpolate_times is not None:
        interp_positions = [float(s.strip()) for s in args.interpolate_times.split(",")]
        traj_main, traj_sub, weights, sampled_idx = _interpolate_trajectories(
            traj_main, traj_sub, weights, sampled_idx, interp_positions,
        )

    time_labels = None
    if args.time_labels is not None:
        time_labels = [s.strip() for s in args.time_labels.split(",")]
        if len(time_labels) != len(sampled_idx):
            raise ValueError(
                f"--time-labels has {len(time_labels)} entries but trajectory has "
                f"{len(sampled_idx)} time steps"
            )

    pred_labels = predict_labels_on_indices(
        traj_main,
        traj_sub,
        sampled_idx,
        classifier=classifier,
        train_space=train_space,
    )
    long_df = build_long_transitions(pred_labels, weights, sampled_idx, time_labels=time_labels)
    long_csv = outdir / "author5_lineage_transitions_long.csv"
    long_df.to_csv(long_csv, index=False)

    pruned_df, prune_stats = prune_with_constraints(
        long_df,
        min_prop_source=float(args.min_prop_source),
    )
    pruned_df, rule_filter_stats = apply_forward_rule_filter(
        pruned_df,
        allowed_forward_edges=ALLOWED_FORWARD,
    )
    pruned_csv = outdir / "author5_lineage_transitions_pruned.csv"
    pruned_df.to_csv(pruned_csv, index=False)

    html_path = outdir / "author5_lineage_sankey.html"
    render = render_sankey_html(pruned_df, html_path, write_static=not args.skip_static)

    manifest = {
        "run_dir": str(run_dir),
        "train_space": train_space,
        "time_labels": time_labels,
        "pruning": prune_stats,
        "rule_filter": rule_filter_stats,
        "render": render,
        "artifacts": {
            "lineage_transitions_long_csv": str(long_csv),
            "lineage_transitions_pruned_csv": str(pruned_csv),
            "sankey_html": str(html_path),
            "sankey_pdf": str(html_path.with_suffix(".pdf")),
            "sankey_png": str(html_path.with_suffix(".png")),
            "sankey_svg": str(html_path.with_suffix(".svg")),
        },
    }
    (outdir / "author5_lineage_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
