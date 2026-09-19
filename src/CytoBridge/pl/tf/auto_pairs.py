from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from .context import TFRunContext


def _split_entities(text: str) -> List[str]:
    x = str(text).strip()
    x = x.replace("、", ",").replace("，", ",").replace("；", ",")
    x = re.sub(r"\band\b", ",", x, flags=re.IGNORECASE)
    parts = re.split(r"[,/;\s]+", x)
    return [p.strip() for p in parts if p.strip()]


def _extract_pair_candidates(line: str) -> List[Tuple[str, str]]:
    s = str(line).strip().replace("*", "")
    if not s:
        return []

    m = re.sub(r"^[-\d\.\)\s]+", "", s).strip()
    out: List[Tuple[str, str]] = []

    if "↔" in m:
        left, right = m.split("↔", 1)
        for a in _split_entities(left):
            for b in _split_entities(right):
                out.append((a, b))
        return out

    for sep in ["->", "→", ":"]:
        if sep in m:
            left, right = m.split(sep, 1)
            for a in _split_entities(left):
                for b in _split_entities(right):
                    out.append((a, b))
            return out

    return out


def _load_var_names(path: Path) -> List[str]:
    adata = ad.read_h5ad(str(path), backed="r")
    try:
        og = adata.uns.get("original_gene_info", {})
        var_names = [str(x) for x in list(og.get("var_names", []))]
        if not var_names:
            var_names = [str(x) for x in adata.var_names.tolist()]
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()
    return var_names


def _auto_pairs_from_jacobian(ctx: TFRunContext, *, top_k: int) -> pd.DataFrame:
    path = ctx.tables_dir / "jacobian_grn_edges.csv"
    if not path.exists():
        return pd.DataFrame(columns=["primary_var", "secondary_var", "score", "source"])

    df = pd.read_csv(path)
    required = {"main_var_feature", "sec_var_feature"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(columns=["primary_var", "secondary_var", "score", "source"])

    if "abs_weight" in df.columns:
        score = np.asarray(df["abs_weight"], dtype=float)
    elif "weight" in df.columns:
        score = np.abs(np.asarray(df["weight"], dtype=float))
    else:
        score = np.zeros(df.shape[0], dtype=float)

    work = pd.DataFrame(
        {
            "primary_var": df["main_var_feature"].astype(str),
            "secondary_var": df["sec_var_feature"].astype(str),
            "score": score,
            "source": "jacobian_grn_edges",
        }
    )
    work = work.drop_duplicates(subset=["primary_var", "secondary_var"], keep="first")
    work = work.sort_values("score", ascending=False)
    if int(top_k) > 0:
        work = work.head(int(top_k))
    return work.reset_index(drop=True)


def _fallback_pairs_from_markers(ctx: TFRunContext, *, n: int = 20) -> pd.DataFrame:
    main_path = ctx.tables_dir / "main_var_marker_pred.csv"
    sec_path = ctx.tables_dir / "sec_var_marker_pred.csv"
    if (not main_path.exists()) or (not sec_path.exists()):
        return pd.DataFrame(columns=["primary_var", "secondary_var", "score", "source"])

    mdf = pd.read_csv(main_path)
    sdf = pd.read_csv(sec_path)
    if "marker" not in mdf.columns or "marker" not in sdf.columns:
        return pd.DataFrame(columns=["primary_var", "secondary_var", "score", "source"])

    if "mean_expr" in mdf.columns:
        mscore = mdf.groupby("marker", as_index=False)["mean_expr"].mean().sort_values("mean_expr", ascending=False)
        mlist = mscore["marker"].astype(str).head(int(n)).tolist()
    else:
        mlist = mdf["marker"].astype(str).head(int(n)).tolist()

    if "mean_expr" in sdf.columns:
        sscore = sdf.groupby("marker", as_index=False)["mean_expr"].mean().sort_values("mean_expr", ascending=False)
        slist = sscore["marker"].astype(str).head(int(n)).tolist()
    else:
        slist = sdf["marker"].astype(str).head(int(n)).tolist()

    out: List[Dict[str, Any]] = []
    for i, a in enumerate(mlist):
        if i >= len(slist):
            break
        out.append(
            {
                "primary_var": str(a),
                "secondary_var": str(slist[i]),
                "score": float(len(mlist) - i),
                "source": "marker_fallback",
            }
        )
    return pd.DataFrame(out)


def _manual_pairs_from_txt(
    txt_paths: Sequence[Path],
    *,
    primary_vars: set[str],
    secondary_vars: set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []

    for path in txt_paths:
        if not path.exists():
            unresolved.append({"file": str(path), "line": "", "left": "", "right": "", "reason": "file_not_found"})
            continue
        for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for left, right in _extract_pair_candidates(raw):
                a = str(left).strip()
                b = str(right).strip()
                if not a or not b:
                    continue
                if (a in primary_vars) and (b in secondary_vars):
                    rows.append(
                        {
                            "primary_var": a,
                            "secondary_var": b,
                            "score": 0.0,
                            "source": "manual_txt",
                            "source_file": str(path),
                            "source_line": int(i),
                        }
                    )
                elif (b in primary_vars) and (a in secondary_vars):
                    rows.append(
                        {
                            "primary_var": b,
                            "secondary_var": a,
                            "score": 0.0,
                            "source": "manual_txt_reoriented",
                            "source_file": str(path),
                            "source_line": int(i),
                        }
                    )
                else:
                    unresolved.append(
                        {
                            "file": str(path),
                            "line": int(i),
                            "left": a,
                            "right": b,
                            "reason": "not_in_primary_secondary_vocab",
                        }
                    )

    manual_df = pd.DataFrame(rows)
    if not manual_df.empty:
        manual_df = manual_df.drop_duplicates(subset=["primary_var", "secondary_var"], keep="first")
    unresolved_df = pd.DataFrame(unresolved)
    return manual_df, unresolved_df


def build_pair_tables(
    ctx: TFRunContext,
    *,
    top_k_pairs: int = 50,
    manual_pairs_txt: Sequence[str] | None = None,
) -> Dict[str, Any]:
    out_dir = ctx.output_table_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    auto_df = _auto_pairs_from_jacobian(ctx, top_k=int(top_k_pairs))
    if auto_df.empty:
        auto_df = _fallback_pairs_from_markers(ctx, n=max(20, int(top_k_pairs)))

    primary_vocab = set(_load_var_names(ctx.primary_processed_path))
    secondary_vocab = set(_load_var_names(ctx.secondary_processed_path))

    txt_paths = [Path(str(x)).resolve() for x in (manual_pairs_txt or []) if str(x).strip()]
    manual_df, unresolved_df = _manual_pairs_from_txt(
        txt_paths,
        primary_vars=primary_vocab,
        secondary_vars=secondary_vocab,
    )

    final_df = pd.concat([auto_df, manual_df], axis=0, ignore_index=True)
    if not final_df.empty:
        final_df = final_df.drop_duplicates(subset=["primary_var", "secondary_var"], keep="first")
        final_df = final_df.sort_values(["score", "primary_var", "secondary_var"], ascending=[False, True, True]).reset_index(drop=True)

    auto_csv = out_dir / "tf_pairs_auto.csv"
    manual_csv = out_dir / "tf_pairs_manual.csv"
    unresolved_csv = out_dir / "tf_pairs_manual_unresolved.csv"
    final_csv = out_dir / "tf_pairs_final.csv"
    auto_df.to_csv(auto_csv, index=False)
    manual_df.to_csv(manual_csv, index=False)
    unresolved_df.to_csv(unresolved_csv, index=False)
    final_df.to_csv(final_csv, index=False)

    summary = {
        "auto_pairs_csv": str(auto_csv),
        "manual_pairs_csv": str(manual_csv),
        "manual_unresolved_csv": str(unresolved_csv),
        "final_pairs_csv": str(final_csv),
        "n_auto": int(auto_df.shape[0]),
        "n_manual": int(manual_df.shape[0]),
        "n_manual_unresolved": int(unresolved_df.shape[0]),
        "n_final": int(final_df.shape[0]),
    }
    return summary
