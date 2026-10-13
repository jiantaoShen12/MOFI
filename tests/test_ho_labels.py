"""Synthetic regression tests; no biological release files are modified."""
import copy
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from CytoBridge.reproducibility import (
    PAPER_DATASETS,
    apply_ho_release_label_fix,
    correct_ho_stage_labels,
    validate_ho_label_consistency,
    validate_ho_release_labels,
)


def make_ho(labels=("EB", "nepi", "nect", "organoid")):
    obj = ad.AnnData(
        X=np.arange(len(labels) * 2, dtype=float).reshape(len(labels), 2),
        obs=pd.DataFrame({
            "cell_type": pd.Categorical(labels),
            "time_point_processed": np.arange(len(labels), dtype=float),
        }, index=[f"cell_{i}" for i in range(len(labels))]),
    )
    obj.obsm["X_latent"] = obj.X.copy()
    return obj


def labels(obj):
    return obj.obs["cell_type"].astype(str).tolist()


def test_first_correction_preserves_data_and_other_labels():
    obj = make_ho()
    original = obj.copy()
    corrected, audit = correct_ho_stage_labels(obj)
    assert corrected is obj
    assert labels(obj) == ["EB", "nect", "nepi", "organoid"]
    assert audit["changed_cells"] == 2
    assert audit["status"] == "applied"
    assert audit["before_counts"]["nepi"] == 1
    np.testing.assert_array_equal(obj.X, original.X)
    np.testing.assert_array_equal(obj.obsm["X_latent"], original.obsm["X_latent"])
    pd.testing.assert_series_equal(obj.obs["time_point_processed"], original.obs["time_point_processed"])


@pytest.mark.parametrize("original", [
    ["nepi", "nect"], ["nepi", "nepi"], ["nect"], ["EB", "organoid"],
])
def test_repeated_correction_preserves_labels_and_first_audit(original):
    obj, first = correct_ho_stage_labels(make_ho(original))
    saved_labels, saved_audit = labels(obj), copy.deepcopy(first)
    for _ in range(3):
        result, audit = correct_ho_stage_labels(obj)
        assert result is obj
        assert labels(obj) == saved_labels
        assert audit == saved_audit
        assert obj.uns["mofi_ho_label_correction"] == saved_audit


def test_roundtrip_and_copy_keep_correction(tmp_path):
    obj, audit = correct_ho_stage_labels(make_ho())
    path = tmp_path / "ho.h5ad"
    obj.write_h5ad(path)
    reloaded = ad.read_h5ad(path)
    for target in (reloaded, reloaded.copy(), reloaded[:3].copy()):
        before = labels(target)
        _, second_audit = correct_ho_stage_labels(target)
        assert labels(target) == before
        assert second_audit == audit


def test_existing_legacy_applied_audit_is_respected():
    obj = make_ho(["nect", "nepi"])
    audit = {"status": "applied", "label_key": "cell_type", "changed_cells": 2}
    obj.uns["mofi_ho_label_correction"] = copy.deepcopy(audit)
    _, returned = correct_ho_stage_labels(obj)
    assert labels(obj) == ["nect", "nepi"]
    assert returned == audit


def test_marker_for_other_column_raises_without_changes():
    obj, _ = correct_ho_stage_labels(make_ho())
    obj.obs["alternative"] = ["EB", "nepi", "nect", "organoid"]
    before = obj.obs.copy()
    with pytest.raises(ValueError, match="different label column"):
        correct_ho_stage_labels(obj, label_key="alternative")
    pd.testing.assert_frame_equal(obj.obs, before)


def test_missing_column_raises():
    with pytest.raises(KeyError):
        correct_ho_stage_labels(make_ho(), label_key="missing")


def test_three_release_files_stay_correct_after_second_fix(tmp_path):
    spec = PAPER_DATASETS["ho"]
    for relative in (spec.primary_path, spec.secondary_path, spec.dynamics_path):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        make_ho().write_h5ad(path)
    first = apply_ho_release_label_fix(tmp_path)
    second = apply_ho_release_label_fix(tmp_path)
    assert first == second
    for path in second["files"]:
        obj = ad.read_h5ad(path)
        assert labels(obj) == ["EB", "nect", "nepi", "organoid"]
        correct_ho_stage_labels(obj)  # training / Figure 5 load-time call
        assert labels(obj) == ["EB", "nect", "nepi", "organoid"]
    report = validate_ho_release_labels(tmp_path)
    assert all(pair["status"] == "consistent" for pair in report["pairs"].values())
    assert json.loads((tmp_path / "validation/ho_label_correction.json").read_text()) == second


def test_consistency_aligns_cell_ids_not_row_order():
    obj, _ = correct_ho_stage_labels(make_ho())
    shuffled = obj[[3, 1, 0, 2]].copy()
    report = validate_ho_label_consistency({"RNA": obj, "ATAC": shuffled, "dynamics": obj.copy()})
    assert all(pair["shared_cells"] == 4 for pair in report["pairs"].values())


@pytest.mark.parametrize("column,value,match", [
    ("cell_type", "nepi", "labels disagree"),
    ("time_point_processed", 99., "times disagree"),
])
def test_consistency_rejects_conflicting_annotations(column, value, match):
    obj, _ = correct_ho_stage_labels(make_ho())
    other = obj.copy()
    other.obs.loc["cell_1", column] = value
    with pytest.raises(ValueError, match=match):
        validate_ho_label_consistency({"RNA": obj, "ATAC": other})


def test_consistency_rejects_mixed_provenance():
    obj, _ = correct_ho_stage_labels(make_ho())
    with pytest.raises(ValueError, match="missing applied"):
        validate_ho_label_consistency({"RNA": obj, "ATAC": make_ho()})


def test_disjoint_ids_are_reported_unverified():
    obj, _ = correct_ho_stage_labels(make_ho())
    other = obj.copy()
    other.obs_names = [f"atac_{i}" for i in range(other.n_obs)]
    report = validate_ho_label_consistency({"RNA": obj, "ATAC": other})
    assert report["pairs"]["RNA/ATAC"]["status"] == "no_shared_ids_unverified"


def test_cache_signature_tracks_labels_and_survives_roundtrip(tmp_path):
    from CytoBridge.reproducibility import ho_label_signature

    obj = make_ho()
    before = ho_label_signature(obj)
    correct_ho_stage_labels(obj)
    corrected = ho_label_signature(obj)
    assert before != corrected
    correct_ho_stage_labels(obj)
    assert ho_label_signature(obj) == corrected
    path = tmp_path / "ho.h5ad"
    obj.write_h5ad(path)
    assert ho_label_signature(ad.read_h5ad(path)) == corrected
