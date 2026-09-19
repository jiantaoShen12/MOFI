from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def _safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}


def _latest_existing_path(paths: Sequence[Path]) -> Optional[Path]:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return sorted(existing, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _resolve_perturb_manifest_path(run_dir: Path, *, kind: str) -> Path:
    base = run_dir / "figures" / "tf" / "perturb_pipeline"
    if str(kind) == "figure":
        cands = [
            base / "figure_manifest__single.json",
            base / "figure_manifest__batch.json",
            base / "figure_manifest.json",
        ]
    else:
        cands = [
            base / "perturb_pipeline_manifest__single.json",
            base / "perturb_pipeline_manifest__batch.json",
            base / "perturb_pipeline_manifest.json",
        ]
    resolved = _latest_existing_path(cands)
    return resolved if resolved is not None else cands[0]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        v = float(value)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _dig(obj: Dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _normalize_path(path_text: str) -> str:
    return str(Path(path_text).resolve())


def _find_workflow_summary(run_dir: Path) -> Optional[Path]:
    for base in [run_dir, *run_dir.parents]:
        with_fig = base / "workflow_summary_with_figures.json"
        if with_fig.exists():
            return with_fig
        plain = base / "workflow_summary.json"
        if plain.exists():
            return plain
    return None


def _find_run_record(summary_obj: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    runs = summary_obj.get("runs", [])
    if not isinstance(runs, list):
        return {}
    run_resolved = run_dir.resolve()
    for row in runs:
        if not isinstance(row, dict):
            continue
        p = str(row.get("run_dir", "")).strip()
        if not p:
            continue
        try:
            if Path(p).resolve() == run_resolved:
                return row
        except Exception:
            continue
    if len(runs) == 1 and isinstance(runs[0], dict):
        return runs[0]
    return {}


def _count_list(value: Any) -> Optional[int]:
    if isinstance(value, list):
        return int(len(value))
    return None


def _read_report_json_metric(run_dir: Path, report_name: str, *keys: str) -> Any:
    report_path = run_dir / "reports" / report_name
    obj = _safe_load_json(report_path)
    if not obj:
        return None
    return _dig(obj, *keys)


def _extract_dense_metrics(run_record: Dict[str, Any], run_dir: Path) -> Dict[str, Optional[float]]:
    train_accuracy = _as_float(_dig(run_record, "classifier_eval", "train_accuracy"))
    cell_type_score_median = _as_float(_dig(run_record, "cell_type_eval", "metrics", "cell_type_score_median"))
    program_score_median = _as_float(_dig(run_record, "mechanism_eval", "program_score_median"))

    jacobian_top_pairs_count = _count_list(_dig(run_record, "jacobian_grn_eval", "top_pairs"))
    key_molecule_count = _count_list(_dig(run_record, "key_molecule_eval", "selected_union"))

    if train_accuracy is None:
        train_accuracy = _as_float(_read_report_json_metric(run_dir, "classifier_eval.json", "train_accuracy"))
    if cell_type_score_median is None:
        cell_type_score_median = _as_float(
            _read_report_json_metric(run_dir, "analysis_summary.json", "cell_type_eval", "metrics", "cell_type_score_median")
        )
    if program_score_median is None:
        program_score_median = _as_float(_read_report_json_metric(run_dir, "analysis_summary.json", "mechanism_eval", "program_score_median"))

    if jacobian_top_pairs_count is None:
        top_pairs = _read_report_json_metric(run_dir, "jacobian_grn_eval.json", "top_pairs")
        jacobian_top_pairs_count = int(len(top_pairs)) if isinstance(top_pairs, list) else None
    if key_molecule_count is None:
        sel = _read_report_json_metric(run_dir, "key_molecule_eval.json", "selected_union")
        key_molecule_count = int(len(sel)) if isinstance(sel, list) else None

    return {
        "classifier_train_accuracy": train_accuracy,
        "cell_type_score_median": cell_type_score_median,
        "program_score_median": program_score_median,
        "jacobian_top_pairs_count": float(jacobian_top_pairs_count) if jacobian_top_pairs_count is not None else None,
        "key_molecule_count": float(key_molecule_count) if key_molecule_count is not None else None,
    }


def _csv_has_data(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return False
            first = next(reader, None)
            return first is not None
    except Exception:
        return False


def _read_curve_csv(path: Path, *, time_col: str, value_col: str) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = _as_float(row.get(time_col))
            v = _as_float(row.get(value_col))
            if t is None or v is None:
                continue
            rows.append((float(t), float(v)))
    if not rows:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    # Multiple rows can share the same time; aggregate by mean for a stable comparison.
    by_t: Dict[float, List[float]] = {}
    for t, v in rows:
        by_t.setdefault(float(t), []).append(float(v))
    times = np.asarray(sorted(by_t.keys()), dtype=float)
    vals = np.asarray([float(np.mean(by_t[float(t)])) for t in times], dtype=float)
    return times, vals


def _normalized_rmse(
    new_csv: Path,
    base_csv: Path,
    *,
    time_col: str,
    value_col: str,
) -> Optional[float]:
    if (not new_csv.exists()) or (not base_csv.exists()):
        return None
    try:
        t_new, y_new = _read_curve_csv(new_csv, time_col=time_col, value_col=value_col)
        t_base, y_base = _read_curve_csv(base_csv, time_col=time_col, value_col=value_col)
        if t_new.size < 2 or t_base.size < 2:
            return None
        y_base_interp = np.interp(t_new, t_base, y_base)
        rmse = float(np.sqrt(np.mean((y_new - y_base_interp) ** 2)))
        denom = float(max(np.std(y_base_interp), 1e-12))
        return float(rmse / denom)
    except Exception:
        return None


def _relative_diff(new_v: Optional[float], base_v: Optional[float]) -> Optional[float]:
    if new_v is None or base_v is None:
        return None
    denom = max(abs(float(base_v)), 1e-12)
    return float(abs(float(new_v) - float(base_v)) / denom)


def _detect_dataset_id(run_dir: Path, dataset_id: str) -> str:
    if str(dataset_id).strip():
        return str(dataset_id).strip()
    tokens = {"gse253582", "gse268609", "hc04", "31800", "ho"}
    parts_lower = [p.lower() for p in run_dir.parts]
    for token in tokens:
        if token in parts_lower:
            return token
    return run_dir.name


def _compose_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# RESULTS REPORT - {payload.get('dataset_id','unknown')}")
    lines.append("")
    lines.append(f"- status: {payload.get('status','UNKNOWN')}")
    lines.append(f"- source_result_subdir: {payload.get('source_result_subdir','')}")
    lines.append(f"- baseline_result_subdir: {payload.get('baseline_result_subdir','')}")
    lines.append(f"- generated_at_utc: {payload.get('generated_at_utc','')}")
    lines.append("")

    dm = payload.get("dense_metrics", {})
    bm = payload.get("baseline_dense_metrics", {})
    lines.append("## Dense Metrics")
    lines.append("")
    lines.append("| metric | new | baseline |")
    lines.append("|---|---:|---:|")
    for key in [
        "classifier_train_accuracy",
        "cell_type_score_median",
        "program_score_median",
        "jacobian_top_pairs_count",
        "key_molecule_count",
    ]:
        lines.append(f"| {key} | {dm.get(key)} | {bm.get(key)} |")
    lines.append("")

    per = payload.get("perturbation", {})
    lines.append("## Perturbation")
    lines.append("")
    lines.append(f"- draw_status: {per.get('draw_status','unknown')}")
    lines.append(f"- pair_main_gene_valid: {payload.get('pair_main_gene_valid', False)}")
    lines.append(f"- pair_png: {per.get('pair_png','')}")
    lines.append(f"- main_gene_png: {per.get('main_gene_png','')}")
    lines.append(f"- pair_sim_csv: {per.get('pair_sim_csv','')}")
    lines.append(f"- main_sim_csv: {per.get('main_sim_csv','')}")
    lines.append("")

    diff = payload.get("diff_check", {})
    lines.append("## Difference Check")
    lines.append("")
    lines.append(f"- threshold: {diff.get('threshold')} ")
    lines.append(f"- difference_too_large: {diff.get('difference_too_large')} ")
    warn_metrics = diff.get("warn_metrics", [])
    lines.append(f"- warn_metrics: {', '.join(warn_metrics) if warn_metrics else '(none)'}")
    curve_warns = diff.get("curve_warns", [])
    lines.append(f"- curve_warns: {', '.join(curve_warns) if curve_warns else '(none)'}")
    lines.append("")

    reasons = payload.get("reasons", [])
    lines.append("## Reasons")
    lines.append("")
    if reasons:
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def generate_run_results_report(
    run_dir: str,
    *,
    dataset_id: str = "",
    baseline_run_dir: str = "",
    diff_threshold: float = 1e-3,
    perturb_draw_row: Optional[Dict[str, Any]] = None,
    expected_partial_patterns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    run = Path(run_dir).resolve()
    dataset = _detect_dataset_id(run, dataset_id)
    baseline = Path(baseline_run_dir).resolve() if str(baseline_run_dir).strip() else None
    patterns = [str(x).strip() for x in (expected_partial_patterns or []) if str(x).strip()]

    reasons: List[str] = []

    summary_path = _find_workflow_summary(run)
    summary_obj = _safe_load_json(summary_path) if summary_path is not None else {}
    run_record = _find_run_record(summary_obj, run) if summary_obj else {}
    dense_metrics = _extract_dense_metrics(run_record, run)
    dense_ok = bool(summary_path is not None and run_record)
    if not dense_ok:
        reasons.append("dense workflow summary missing or run record not found")

    base_metrics: Dict[str, Optional[float]] = {}
    base_run_record: Dict[str, Any] = {}
    if baseline is not None and baseline.exists():
        base_summary_path = _find_workflow_summary(baseline)
        if base_summary_path is not None:
            base_summary_obj = _safe_load_json(base_summary_path)
            base_run_record = _find_run_record(base_summary_obj, baseline) if base_summary_obj else {}
        base_metrics = _extract_dense_metrics(base_run_record, baseline)
    else:
        if baseline is not None:
            reasons.append(f"baseline run_dir not found: {baseline}")

    figure_manifest_path = _resolve_perturb_manifest_path(run, kind="figure")
    perturb_manifest_path = _resolve_perturb_manifest_path(run, kind="perturb")
    figure_manifest = _safe_load_json(figure_manifest_path)
    perturb_manifest = _safe_load_json(perturb_manifest_path)

    draw_status = "success" if figure_manifest else "failed"
    draw_reason = ""
    if perturb_draw_row is not None:
        draw_status = str(perturb_draw_row.get("status", draw_status))
        draw_reason = str(perturb_draw_row.get("reason", "")).strip()
        if draw_reason:
            reasons.append(draw_reason)

    generated = figure_manifest.get("generated", {}) if isinstance(figure_manifest.get("generated"), dict) else {}
    outputs = perturb_manifest.get("outputs", {}) if isinstance(perturb_manifest.get("outputs"), dict) else {}

    pair_png = Path(str(generated.get("pair_png", ""))).resolve() if str(generated.get("pair_png", "")).strip() else None
    main_png = (
        Path(str(generated.get("main_gene_png", ""))).resolve()
        if str(generated.get("main_gene_png", "")).strip()
        else None
    )
    pair_sim_csv = Path(str(outputs.get("pair_sim_csv", ""))).resolve() if str(outputs.get("pair_sim_csv", "")).strip() else None
    pair_real_csv = (
        Path(str(outputs.get("pair_real_csv", ""))).resolve()
        if str(outputs.get("pair_real_csv", "")).strip()
        else None
    )
    main_sim_csv = Path(str(outputs.get("main_sim_csv", ""))).resolve() if str(outputs.get("main_sim_csv", "")).strip() else None
    main_real_csv = (
        Path(str(outputs.get("main_real_csv", ""))).resolve()
        if str(outputs.get("main_real_csv", "")).strip()
        else None
    )

    pair_curve_valid = bool(
        pair_png is not None
        and pair_png.exists()
        and pair_sim_csv is not None
        and pair_real_csv is not None
        and _csv_has_data(pair_sim_csv)
        and _csv_has_data(pair_real_csv)
    )
    main_curve_valid = bool(
        main_png is not None
        and main_png.exists()
        and main_sim_csv is not None
        and main_real_csv is not None
        and _csv_has_data(main_sim_csv)
        and _csv_has_data(main_real_csv)
    )
    pair_main_gene_valid = bool(pair_curve_valid and main_curve_valid)

    if draw_status != "success":
        reasons.append("perturbation figure drawing failed")
    if not perturb_manifest:
        reasons.append("perturbation pipeline manifest missing")
    if not pair_curve_valid:
        reasons.append("pair perturbation artifacts missing or empty")
    if not main_curve_valid:
        reasons.append("main-gene perturbation artifacts missing or empty")

    source_result_subdir = str(
        figure_manifest.get("source_result_subdir")
        or perturb_manifest.get("source_result_subdir")
        or str(run)
    )

    diff_details: Dict[str, Dict[str, Optional[float]]] = {}
    warn_metrics: List[str] = []
    for key, new_v in dense_metrics.items():
        base_v = base_metrics.get(key)
        rel = _relative_diff(new_v, base_v)
        diff_details[key] = {
            "new": new_v,
            "baseline": base_v,
            "relative_diff": rel,
        }
        if rel is not None and rel > float(diff_threshold):
            warn_metrics.append(key)

    curve_diffs: Dict[str, Optional[float]] = {
        "pair_primary_nrmse": None,
        "pair_secondary_nrmse": None,
        "main_gene_nrmse": None,
    }
    curve_warns: List[str] = []
    if baseline is not None and baseline.exists():
        base_perturb = _resolve_perturb_manifest_path(baseline, kind="perturb")
        base_manifest = _safe_load_json(base_perturb)
        base_outputs = base_manifest.get("outputs", {}) if isinstance(base_manifest.get("outputs"), dict) else {}

        if pair_sim_csv is not None and str(base_outputs.get("pair_sim_csv", "")).strip():
            base_pair_sim = Path(str(base_outputs["pair_sim_csv"])).resolve()
            curve_diffs["pair_primary_nrmse"] = _normalized_rmse(
                pair_sim_csv,
                base_pair_sim,
                time_col="time",
                value_col="primary_value",
            )
            curve_diffs["pair_secondary_nrmse"] = _normalized_rmse(
                pair_sim_csv,
                base_pair_sim,
                time_col="time",
                value_col="secondary_value",
            )
        if main_sim_csv is not None and str(base_outputs.get("main_sim_csv", "")).strip():
            base_main_sim = Path(str(base_outputs["main_sim_csv"])).resolve()
            curve_diffs["main_gene_nrmse"] = _normalized_rmse(
                main_sim_csv,
                base_main_sim,
                time_col="time",
                value_col="value",
            )

    for key, v in curve_diffs.items():
        if v is not None and v > float(diff_threshold):
            curve_warns.append(key)

    difference_too_large = bool(warn_metrics or curve_warns)

    reason_text = " | ".join(reasons).lower()
    matches_partial_pattern = any(p.lower() in reason_text for p in patterns)

    if dense_ok and draw_status == "success" and pair_main_gene_valid:
        status = "SUCCESS"
    elif dense_ok and matches_partial_pattern:
        status = "PARTIAL"
    else:
        status = "FAILED"

    payload: Dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "dataset_id": dataset,
        "status": status,
        "run_dir": str(run),
        "source_result_subdir": source_result_subdir,
        "baseline_result_subdir": str(baseline) if baseline is not None else "",
        "expected_partial_patterns": patterns,
        "dense_summary_path": str(summary_path) if summary_path is not None else "",
        "dense_metrics": dense_metrics,
        "baseline_dense_metrics": base_metrics,
        "diff_check": {
            "threshold": float(diff_threshold),
            "metrics": diff_details,
            "warn_metrics": warn_metrics,
            "curve_nrmse": curve_diffs,
            "curve_warns": curve_warns,
            "difference_too_large": difference_too_large,
        },
        "perturbation": {
            "draw_status": draw_status,
            "draw_reason": draw_reason,
            "pair_png": str(pair_png) if pair_png is not None else "",
            "main_gene_png": str(main_png) if main_png is not None else "",
            "pair_sim_csv": str(pair_sim_csv) if pair_sim_csv is not None else "",
            "main_sim_csv": str(main_sim_csv) if main_sim_csv is not None else "",
            "pair_curve_valid": pair_curve_valid,
            "main_gene_curve_valid": main_curve_valid,
            "figure_manifest_json": str(figure_manifest_path) if figure_manifest_path.exists() else "",
            "perturb_pipeline_manifest_json": str(perturb_manifest_path) if perturb_manifest_path.exists() else "",
            "source_files_read": perturb_manifest.get("source_files_read", []),
        },
        "pair_main_gene_valid": pair_main_gene_valid,
        "reasons": sorted(set([x for x in reasons if x])),
    }

    out_json = run / "results_report.json"
    out_md = run / "RESULTS_REPORT.md"
    _write_json(out_json, payload)
    out_md.write_text(_compose_markdown(payload), encoding="utf-8")

    payload["results_report_json"] = str(out_json)
    payload["results_report_md"] = str(out_md)
    return payload


def _compose_index_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# RESULTS INDEX")
    lines.append("")
    lines.append(f"- generated_at_utc: {payload.get('generated_at_utc','')}")
    lines.append(f"- num_runs: {payload.get('num_runs',0)}")
    lines.append(f"- overall_pass: {payload.get('acceptance',{}).get('overall_pass')}")
    lines.append("")
    lines.append("| dataset | status | source_result_subdir | pair_main_gene_valid | difference_too_large |")
    lines.append("|---|---|---|---:|---:|")
    for row in payload.get("runs", []):
        lines.append(
            "| {dataset} | {status} | {src} | {valid} | {diff} |".format(
                dataset=row.get("dataset_id", ""),
                status=row.get("status", ""),
                src=row.get("source_result_subdir", ""),
                valid=row.get("pair_main_gene_valid", False),
                diff=row.get("difference_too_large", False),
            )
        )
    lines.append("")
    lines.append("## Acceptance")
    lines.append("")
    for k, v in payload.get("acceptance", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def generate_results_index(
    reports: Sequence[Dict[str, Any]],
    *,
    index_root: str,
) -> Dict[str, Any]:
    root = Path(index_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for rep in reports:
        rows.append(
            {
                "dataset_id": rep.get("dataset_id", ""),
                "status": rep.get("status", ""),
                "run_dir": rep.get("run_dir", ""),
                "source_result_subdir": rep.get("source_result_subdir", ""),
                "pair_main_gene_valid": bool(rep.get("pair_main_gene_valid", False)),
                "difference_too_large": bool(rep.get("diff_check", {}).get("difference_too_large", False)),
                "results_report_json": rep.get("results_report_json", ""),
                "results_report_md": rep.get("results_report_md", ""),
                "reasons": rep.get("reasons", []),
                "expected_partial_patterns": rep.get("expected_partial_patterns", []),
            }
        )

    by_dataset = {str(x.get("dataset_id", "")).lower(): x for x in rows}

    all_5_runs_produced = bool(len(rows) == 5 and all(Path(str(x.get("run_dir", ""))).exists() for x in rows))
    all_5_reports_exist = bool(
        len(rows) == 5
        and all(
            Path(str(x.get("results_report_json", ""))).exists() and Path(str(x.get("results_report_md", ""))).exists()
            for x in rows
        )
    )
    all_include_source_result_subdir = bool(all(str(x.get("source_result_subdir", "")).strip() for x in rows))

    required_valid_datasets = ["gse268609", "hc04", "31800"]
    required_pair_main_gene_valid = bool(
        all(bool(by_dataset.get(k, {}).get("pair_main_gene_valid", False)) for k in required_valid_datasets)
    )

    partial_allowed = ["gse253582", "ho"]
    partial_rule_rows_ok = True
    for k in partial_allowed:
        row = by_dataset.get(k, {})
        st = str(row.get("status", ""))
        if st not in {"SUCCESS", "PARTIAL"}:
            partial_rule_rows_ok = False
            continue
        if st == "PARTIAL":
            reasons = " | ".join([str(x) for x in row.get("reasons", [])]).lower()
            patterns = [str(x).lower() for x in row.get("expected_partial_patterns", [])]
            if (not patterns) or (not any(p in reasons for p in patterns)):
                partial_rule_rows_ok = False

    non_partial_must_success = True
    for k in required_valid_datasets:
        row = by_dataset.get(k, {})
        if str(row.get("status", "")) != "SUCCESS":
            non_partial_must_success = False

    difference_too_large_any = bool(any(bool(x.get("difference_too_large", False)) for x in rows))

    acceptance = {
        "all_5_runs_produced": all_5_runs_produced,
        "all_5_reports_exist": all_5_reports_exist,
        "all_include_source_result_subdir": all_include_source_result_subdir,
        "required_pair_main_gene_valid": required_pair_main_gene_valid,
        "partial_rule_rows_ok": partial_rule_rows_ok,
        "non_partial_must_success": non_partial_must_success,
        "difference_too_large_any": difference_too_large_any,
    }
    acceptance["overall_pass"] = bool(
        acceptance["all_5_runs_produced"]
        and acceptance["all_5_reports_exist"]
        and acceptance["all_include_source_result_subdir"]
        and acceptance["required_pair_main_gene_valid"]
        and acceptance["partial_rule_rows_ok"]
        and acceptance["non_partial_must_success"]
    )

    payload: Dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "num_runs": int(len(rows)),
        "runs": rows,
        "acceptance": acceptance,
    }

    out_json = root / "results_index.json"
    out_md = root / "RESULTS_INDEX.md"
    _write_json(out_json, payload)
    out_md.write_text(_compose_index_markdown(payload), encoding="utf-8")

    payload["results_index_json"] = str(out_json)
    payload["results_index_md"] = str(out_md)
    return payload


__all__ = [
    "generate_run_results_report",
    "generate_results_index",
]
