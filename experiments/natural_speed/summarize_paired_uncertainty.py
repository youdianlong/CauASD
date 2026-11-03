#!/usr/bin/env python3
"""Table-V paired subject-cluster bootstrap from saved predictions; NumPy only.

Run from the project root:
    python experiments/natural_speed/summarize_paired_uncertainty.py

No training, inference, re-stratification during bootstrap, or checkpoint loading.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


METHODS = ("PGFA", "PGFAaug", "CauASD")
PREDICTIONS = ("pgfa_prediction", "pgfaaug_prediction", "cauasd_prediction")
SUBJECT = re.compile(r"S\d{3}C\d{3}P(\d{3})R\d{3}A\d{3}", re.IGNORECASE)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def integers(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
        raise ValueError("{} must be a nonempty one-dimensional integer array".format(name))
    return values.astype(np.int64)


def strata(scores, labels):
    """Exactly reproduce Table V's log-floor, stable sorting, and count rule."""
    if scores.ndim != 1 or len(scores) != len(labels):
        raise ValueError("Score/label shape mismatch")
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Scores must be finite and nonnegative; filter invalid scores upstream")
    rank_scores = np.log(np.maximum(scores, 1e-12))
    result = np.ones(len(labels), dtype=np.int64)
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        count = int(np.floor(0.30 * len(indices)))
        if count == 0:
            raise ValueError("Every class must have samples in all three strata")
        order = indices[np.argsort(rank_scores[indices], kind="stable")]
        result[order[:count]] = 0
        result[order[-count:]] = 2
    return result


def load_split(predictions_dir, data_root, split):
    dataset = "ntu60" if split <= 3 else "ntu120"
    path = Path(predictions_dir) / "{}_split{}_natural_speed.npz".format(dataset, split)
    sidecar = Path(data_root) / dataset / "split_{}".format(split)
    names_path = sidecar / "unseen_sample_names.npy"
    labels_path = sidecar / "unseen_label.npy"
    metadata_path = path.with_suffix(".json")
    for required in (path, names_path, labels_path, metadata_path):
        if not required.is_file():
            raise FileNotFoundError("Missing input: {}".format(required))
    with np.load(path, allow_pickle=False) as archive:
        required = ("labels", "sample_indices", "fixed_v50") + PREDICTIONS
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError("{} lacks {}; use the corrected prediction export".format(path, missing))
        labels = integers(archive["labels"], "labels")
        indices = integers(archive["sample_indices"], "sample_indices")
        scores = np.asarray(archive["fixed_v50"], dtype=np.float64)
        predictions = [integers(archive[key], key) for key in PREDICTIONS]
    if scores.ndim != 1:
        raise ValueError("fixed_v50 must be one-dimensional")
    if any(len(array) != len(labels) for array in [indices, scores] + predictions):
        raise ValueError("Prediction/label/index lengths differ: {}".format(path))
    names = np.asarray(np.load(names_path, allow_pickle=False)).reshape(-1)
    full_labels = integers(np.asarray(np.load(labels_path, allow_pickle=False)).reshape(-1), "source labels")
    if names.dtype.kind not in "US" or len(names) != len(full_labels):
        raise ValueError("Sample names must be strings aligned with the original labels")
    if indices.min() < 0 or indices.max() >= len(names) or len(np.unique(indices)) != len(indices):
        raise ValueError("Invalid or duplicate original sample indices")
    if not np.array_equal(full_labels[indices], labels):
        raise ValueError("Saved predictions are not aligned with the original labels")
    selected_names = names[indices].astype(str)
    if len(np.unique(selected_names)) != len(selected_names):
        raise ValueError("Duplicate sample names within a split")
    subjects = []
    for name in selected_names:
        match = SUBJECT.search(name)
        if not match:
            raise ValueError("Cannot parse NTU subject ID from {!r}".format(name))
        subjects.append(int(match.group(1)))  # P is performer; S is setup, NOT subject.
    if any(not np.isin(prediction, np.unique(labels)).all() for prediction in predictions):
        raise ValueError("Predictions contain labels outside this split's unseen classes")
    groups = strata(scores, labels)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if (metadata.get("speed_proxy_version") != "valid_body_transition_v2"
            or metadata.get("dataset") != dataset or str(metadata.get("split")) != str(split)):
        raise ValueError("JSON does not identify the matching corrected Table-V run")
    point = []
    for method, prediction in zip(METHODS, predictions):
        accuracy = [np.mean([(prediction[(labels == cls) & (groups == group)] == cls).mean()
                             for cls in np.unique(labels)]) for group in range(3)]
        accuracy.append(abs(accuracy[0] - accuracy[2]))
        expected = metadata.get(method, {})
        keys = ("Naturally slower", "Middle", "Naturally faster", "Slow-fast gap")
        if any(key not in expected for key in keys) or not np.allclose(
                accuracy, [expected[key] for key in keys], rtol=0, atol=1e-10):
            raise ValueError("Recomputed metrics do not match the saved JSON for {}".format(method))
        point.append(accuracy)
    return dict(dataset=dataset, split=split, labels=labels, groups=groups,
                subjects=np.asarray(subjects), predictions=np.asarray(predictions),
                point=np.asarray(point), provenance={
                    p.name: digest(p) for p in (path, names_path, labels_path, metadata_path)})


def counts_by_subject(record, all_subjects):
    class_index = np.searchsorted(np.unique(record["labels"]), record["labels"])
    subject_index = np.searchsorted(all_subjects, record["subjects"])
    cells = class_index * 3 + record["groups"]
    shape = (len(np.unique(record["labels"])) * 3, len(all_subjects))
    counts = np.zeros(shape, dtype=np.float64)
    np.add.at(counts, (cells, subject_index), 1)
    successes = np.zeros((3,) + shape, dtype=np.float64)
    for method in range(3):
        np.add.at(successes[method], (cells, subject_index),
                  record["predictions"][method] == record["labels"])
    return counts, successes


def weighted_metrics(counts, successes, weights):
    denominators = counts @ weights
    if np.any(denominators <= 0):
        return None  # Never silently drop an original class or stratum.
    accuracy = ((successes @ weights) / denominators).reshape(3, -1, 3).mean(axis=1)
    return np.column_stack((accuracy, np.abs(accuracy[:, 0] - accuracy[:, 2])))


def bootstrap(records, repeats=2000, seed=2026):
    if not records or repeats < 100 or seed < 0:
        raise ValueError("Need records, at least 100 replicates, and a nonnegative seed")
    if len({r["dataset"] for r in records}) != 1:
        raise ValueError("Bootstrap one dataset at a time")
    subjects = np.unique(np.concatenate([record["subjects"] for record in records]))
    if len(subjects) < 2:
        raise ValueError("At least two independent subject clusters are required")
    prepared = [counts_by_subject(record, subjects) for record in records]
    point = np.mean([record["point"] for record in records], axis=0)
    # The same subject multiplicities are used for ALL methods and ALL splits.
    rng = np.random.default_rng(seed)
    draws = []
    rejected = 0
    attempts = 0
    while len(draws) < repeats and attempts < repeats * 20:
        attempts += 1
        weights = rng.multinomial(len(subjects), np.full(len(subjects), 1.0 / len(subjects)))
        values = [weighted_metrics(counts, successes, weights) for counts, successes in prepared]
        if any(value is None for value in values):
            rejected += 1
            continue
        draws.append(np.mean(values, axis=0))
    if len(draws) != repeats:
        raise ValueError("Too many cluster draws omit a class-stratum; cannot report this interval reliably")
    draws = np.asarray(draws)
    comparisons = []
    for baseline in (0, 1):
        for metric, column in (("Gap", 3), ("Faster accuracy", 2)):
            differences = 100 * (draws[:, 2, column] - draws[:, baseline, column])
            interval = np.percentile(differences, [2.5, 97.5])
            comparisons.append(dict(baseline=METHODS[baseline], metric=metric,
                                    difference_pp=float(100 * (point[2, column] - point[baseline, column])),
                                    ci95_pp=interval.tolist()))
    return dict(dataset=records[0]["dataset"], splits=[r["split"] for r in records],
                n_subjects=len(subjects), repeats=repeats, seed=seed,
                rejected_empty_cell_draws=rejected,
                excluded_draw_fraction=rejected / attempts,
                mean_metrics_percent={method: dict(zip(
                    ("Slower", "Middle", "Faster", "Gap"), (100 * point[i]).tolist()))
                    for i, method in enumerate(METHODS)}, comparisons=comparisons)


def markdown(results, records):
    lines = ["# Table V: paired subject-cluster uncertainty", "",
             "Units: percentage points. Differences are CauASD minus baseline.",
             "Negative Gap differences and positive Faster-accuracy differences favor CauASD.", "",
             "## Per-split gaps", "",
             "| Dataset | Split | PGFA | PGFA-aug | CauASD |",
             "| --- | --- | ---: | ---: | ---: |"]
    for record in records:
        lines.append("| {} | {} | {} |".format(record["dataset"], record["split"],
                     " | ".join("{:.2f}".format(100 * x) for x in record["point"][:, 3])))
    lines += ["", "## Paired differences and percentile 95% intervals", "",
              "| Dataset | Baseline | Metric | Difference | 95% interval |",
              "| --- | --- | --- | ---: | --- |"]
    for result in results:
        for row in result["comparisons"]:
            lines.append("| {} | {} | {} | {:.2f} | [{:.2f}, {:.2f}] |".format(
                result["dataset"], row["baseline"], row["metric"], row["difference_pp"], *row["ci95_pp"]))
    lines += ["", "## Interpretation", "",
              "Fixed class-stratum assignments and saved predictions are used. Subjects (P IDs), not frames, batches, or class splits, are resampled. Subject multiplicities are shared across methods and across selected splits within each dataset. Class accuracies are macro-averaged before the equal-weight split average; absolute gaps are formed within splits before averaging.", "",
              "These intervals are conditional on the selected checkpoints, class splits, exclusions, and strata. They do not quantify retraining variability, model-selection bias, or uncertainty from re-estimating strata. Intervals containing zero do not establish a consistent direction of the paired difference. No multiplicity correction is applied.", "",
              "Draws lacking any original class-stratum are rejected and redrawn; rejection rates are reported below. A high rejection rate limits interpretation."]
    for result in results:
        lines.append("\n- {}: splits {}; {} subjects; {} accepted draws; seed {}; rejected {:.2%} of attempted draws.".format(
            result["dataset"], result["splits"], result["n_subjects"], result["repeats"],
            result["seed"], result["excluded_draw_fraction"]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=Path("analysis/natural_speed_v2"))
    parser.add_argument("--data-root", type=Path, default=Path("data/zeroshot"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/reviewer5_paired"))
    parser.add_argument("--splits", default="1,2,3,4,5,6")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    splits = [int(value) for value in args.splits.split(",")]
    if (not splits or len(set(splits)) != len(splits) or any(s not in range(1, 7) for s in splits)
            or args.repeats < 100 or args.seed < 0):
        parser.error("Use distinct splits 1--6, repeats >=100, and seed >=0")
    outputs = [args.output_dir / ("paired_natural_speed_summary." + suffix) for suffix in ("md", "json")]
    if any(path.exists() for path in outputs):
        parser.error("Outputs already exist; choose a new --output-dir (existing results are not overwritten)")
    records = [load_split(args.predictions_dir, args.data_root, split) for split in sorted(splits)]
    results = []
    for dataset in ("ntu60", "ntu120"):
        selected = [record for record in records if record["dataset"] == dataset]
        if selected:
            print("{}: bootstrapping {} splits without model inference...".format(dataset, len(selected)), flush=True)
            results.append(bootstrap(selected, args.repeats, args.seed))
    report = markdown(results, records)
    payload = dict(method="paired multinomial subject-cluster percentile bootstrap; fixed strata",
                   confidence=0.95, results=results,
                   input_sha256=[dict(dataset=r["dataset"], split=r["split"], files=r["provenance"]) for r in records])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("x", encoding="utf-8") as handle:
        handle.write(report)
    with outputs[1].open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(report)
    print("Saved: {} and {}".format(*outputs))


if __name__ == "__main__":
    main()
