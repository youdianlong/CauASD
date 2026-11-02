"""Synthetic checks; do not require PyTorch, checkpoints, or real datasets."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "natural_speed" / "summarize_paired_uncertainty.py"
SPEC = importlib.util.spec_from_file_location("paired_uncertainty", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_record(tmp_path, split=1, identical=False):
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    dataset = "ntu60" if split <= 3 else "ntu120"
    sidecar = tmp_path / "data" / dataset / "split_{}".format(split)
    sidecar.mkdir(parents=True)
    labels = np.repeat([0, 1], 40).astype(np.int64)
    scores = np.tile(np.arange(40, dtype=float), 2)
    subjects = np.tile(np.arange(1, 9), 10)
    # S001 is the same for all samples; P is the subject identifier.
    names = np.array(["S001C001P{:03d}R{:03d}A{:03d}".format(subject, i + 1, label + 1)
                      for i, (subject, label) in enumerate(zip(subjects, labels))])
    wrong = 1 - labels
    pgfa = np.where(scores < 12, labels, wrong)
    aug = np.where(scores % 3 == 0, labels, wrong)
    cauasd = labels.copy()
    if identical:
        pgfa = aug = cauasd
    np.save(sidecar / "unseen_label.npy", labels)
    np.save(sidecar / "unseen_sample_names.npy", names)
    path = prediction_dir / "{}_split{}_natural_speed.npz".format(dataset, split)
    np.savez_compressed(path, labels=labels, sample_indices=np.arange(80), fixed_v50=scores,
                        pgfa_prediction=pgfa, pgfaaug_prediction=aug, cauasd_prediction=cauasd)
    groups = np.tile(np.repeat([0, 1, 2], [12, 16, 12]), 2)
    metadata = dict(dataset=dataset, split=str(split), speed_proxy_version="valid_body_transition_v2")
    for method, prediction in zip(MODULE.METHODS, (pgfa, aug, cauasd)):
        accuracy = [np.mean([(prediction[(labels == cls) & (groups == group)] == cls).mean()
                             for cls in (0, 1)]) for group in (0, 1, 2)]
        metadata[method] = dict(zip(("Naturally slower", "Middle", "Naturally faster", "Slow-fast gap"),
                                   accuracy + [abs(accuracy[0] - accuracy[2])]))
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return MODULE.load_split(prediction_dir, tmp_path / "data", split), path


def test_masks_match_table_v_stable_ties_and_counts():
    labels = np.repeat([0, 1], 10)
    groups = MODULE.strata(np.zeros(20), labels)
    assert groups.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2] * 2


@pytest.mark.parametrize("scores", [np.array([np.nan] * 4), np.array([-1.] * 4), np.zeros((4, 1))])
def test_invalid_scores_are_rejected(scores):
    with pytest.raises(ValueError):
        MODULE.strata(scores, np.zeros(4, dtype=int))


def test_small_class_cannot_silently_disappear():
    with pytest.raises(ValueError, match="all three strata"):
        MODULE.strata(np.arange(3.), np.zeros(3, dtype=int))


def test_loader_uses_performer_not_setup(tmp_path):
    record, _ = fixture_record(tmp_path)
    assert np.unique(record["subjects"]).tolist() == list(range(1, 9))
    assert record["point"][0, 3] == 1.0
    assert record["point"][2, 3] == 0.0


def test_known_paired_differences_and_intervals(tmp_path):
    record, _ = fixture_record(tmp_path)
    result = MODULE.bootstrap([record], repeats=100, seed=4)
    comparison = next(row for row in result["comparisons"] if row["baseline"] == "PGFA" and row["metric"] == "Gap")
    assert comparison["difference_pp"] == -100.0
    assert comparison["ci95_pp"] == [-100.0, -100.0]
    faster = next(row for row in result["comparisons"] if row["baseline"] == "PGFA" and row["metric"] == "Faster accuracy")
    assert faster["difference_pp"] == 100.0
    assert faster["ci95_pp"] == [100.0, 100.0]


def test_identical_predictions_have_zero_paired_uncertainty(tmp_path):
    record, _ = fixture_record(tmp_path, identical=True)
    result = MODULE.bootstrap([record], repeats=100, seed=4)
    for comparison in result["comparisons"]:
        assert comparison["difference_pp"] == 0
        assert comparison["ci95_pp"] == [0, 0]


def test_cross_split_resampling_is_shared_and_reproducible(tmp_path):
    records = [fixture_record(tmp_path, split)[0] for split in (1, 2, 3)]
    single = MODULE.bootstrap(records[:1], repeats=100, seed=42)
    repeated = MODULE.bootstrap(records, repeats=100, seed=42)
    assert single["comparisons"] == repeated["comparisons"]
    assert repeated == MODULE.bootstrap(records, repeats=100, seed=42)


def test_empty_cells_are_not_omitted_from_macro_average():
    counts = np.eye(3)
    successes = np.ones((3, 3, 3))
    assert MODULE.weighted_metrics(counts, successes, np.array([1., 0., 2.])) is None


def test_single_subject_and_mixed_datasets_are_rejected(tmp_path):
    record, _ = fixture_record(tmp_path)
    record["subjects"][:] = 1
    with pytest.raises(ValueError, match="two independent subject"):
        MODULE.bootstrap([record], repeats=100)
    with pytest.raises(ValueError, match="one dataset"):
        MODULE.bootstrap([record, dict(record, dataset="ntu120")], repeats=100)


def test_wrong_saved_metrics_are_rejected(tmp_path):
    _, path = fixture_record(tmp_path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["PGFA"]["Slow-fast gap"] = 0.4
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="do not match"):
        MODULE.load_split(path.parent, tmp_path / "data", 1)


@pytest.mark.parametrize("corruption", ["indices", "length", "nan", "missing", "version", "names"])
def test_invalid_archives_and_sidecars_fail_explicitly(tmp_path, corruption):
    _, path = fixture_record(tmp_path)
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if corruption == "indices":
        arrays["sample_indices"][1] = arrays["sample_indices"][0]
    elif corruption == "length":
        arrays["pgfa_prediction"] = arrays["pgfa_prediction"][:-1]
    elif corruption == "nan":
        arrays["fixed_v50"][0] = np.nan
    elif corruption == "missing":
        del arrays["fixed_v50"]
    elif corruption == "version":
        path.with_suffix(".json").write_text('{"speed_proxy_version": "old"}')
    elif corruption == "names":
        np.save(tmp_path / "data/ntu60/split_1/unseen_sample_names.npy", np.array(["unknown"] * 80))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        MODULE.load_split(path.parent, tmp_path / "data", 1)


def test_cli_writes_reports_and_refuses_overwrite(tmp_path):
    _, path = fixture_record(tmp_path)
    output = tmp_path / "reports"
    command = [sys.executable, str(SCRIPT), "--predictions-dir", str(path.parent),
               "--data-root", str(tmp_path / "data"), "--splits", "1",
               "--repeats", "100", "--output-dir", str(output)]
    success = subprocess.run(command, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    json_path = output / "paired_natural_speed_summary.json"
    data = json.loads(json_path.read_text())
    assert data["results"][0]["n_subjects"] == 8
    assert len(data["input_sha256"][0]["files"]) == 4
    assert "paired" in (output / "paired_natural_speed_summary.md").read_text()
    original = json_path.read_bytes()
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0
    assert "already exist" in again.stderr
    assert json_path.read_bytes() == original


@pytest.mark.parametrize("extra", [["--repeats", "0"], ["--seed", "-1"], ["--splits", "1,1"], ["--splits", "7"]])
def test_cli_rejects_invalid_arguments(extra):
    run = subprocess.run([sys.executable, str(SCRIPT)] + extra, capture_output=True, text=True)
    assert run.returncode != 0
    assert "Use distinct splits" in run.stderr
