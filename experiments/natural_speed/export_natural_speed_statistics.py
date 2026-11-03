#!/usr/bin/env python3
"""Export the statistics needed to document the natural-speed strata.

This is a post-processing script.  It does not train a model and does not
modify any input data.  The fixed-length proxy is computed on the original
unwarped ``unseen_data.npy`` using the same body-scale normalization,
zero-padding handling, and within-class lower/upper 30-percent ranking used
by ``run_natural_speed_gap_stgcn.py``. Static samples with valid body geometry
and valid adjacent-frame transitions are retained. Samples without valid
transitions or with degenerate body extent are explicitly excluded.

The script writes:

* ``natural_speed_summary_<dataset>_split<split>.csv``
    Dataset-level counts and score summaries.
* ``natural_speed_by_class_<dataset>_split<split>.csv``
    One row per action class, including the numerical q30/q70 values, exact
    rank cutoffs, stratum counts, and the class-level score distribution.
* ``natural_speed_by_stratum_<dataset>_split<split>.csv``
    One row per action class and stratum with distribution statistics.
* ``natural_speed_class_macro_<dataset>_split<split>.csv`` and
  ``natural_speed_s1_<dataset>_split<split>.tex``
    Table-S1 summaries: every score/threshold column is an unweighted mean
    over the same action classes; stratum counts are summed, not averaged.
    The existing ``summary`` files remain pooled sample statistics, NOT
    class-macro statistics. Do not mix their columns into Table S1.
* ``natural_speed_correlations_<dataset>_split<split>.csv``
    Spearman correlations with the raw-time speed and original duration when
    ``--raw-time-speed-npz`` is supplied.
* ``natural_speed_accuracy_<dataset>_split<split>.csv``
    Optional macro accuracies under each speed definition when prediction NPZ
    files are supplied with ``--prediction NAME=FILE``.
* ``natural_speed_metadata_<dataset>_split<split>.json``
    Input paths and a machine-readable summary.

Example (fixed-50 statistics only):

  python export_natural_speed_statistics.py \
      --data data/zeroshot/ntu60/split_1/unseen_data.npy \
      --labels data/zeroshot/ntu60/split_1/unseen_label.npy \
      --dataset ntu60 --split 1 \
      --output-dir analysis/natural_speed_statistics

Example including raw-time validation and prediction exports:

  python export_natural_speed_statistics.py \
      --data data/zeroshot/ntu60/split_1/unseen_data.npy \
      --labels data/zeroshot/ntu60/split_1/unseen_label.npy \
      --sample-names data/zeroshot/ntu60/split_1/unseen_sample_names.npy \
      --raw-time-speed-npz analysis/raw_duration_speed_xsub/ntu60/raw.npz \
      --prediction PGFA=analysis/pgfa_split1_features.npz \
      --prediction CauASD=analysis/cauasd_split1_features.npz \
      --dataset ntu60 --split 1 \
      --output-dir analysis/natural_speed_statistics

Run all six NTU splits in one command (the default only needs the original
unwarped data and labels):

  python export_natural_speed_statistics.py --all-splits

Rebuild Table S1 from an existing class-level export without loading skeletons:

  python export_natural_speed_statistics.py \
      --class-statistics-csv analysis/natural_speed_statistics/natural_speed_by_class_all_splits.csv \
      --output-dir analysis/natural_speed_s1_checked

This only checks and aggregates existing scores. It does not repair errors
in the upstream speed definition or certify that those scores match Table V.

In all-splits mode, the default relative layout is:

  data/zeroshot/<dataset>/split_<split>/unseen_data.npy
  data/zeroshot/<dataset>/split_<split>/unseen_label.npy

The raw-time archive should contain ``speed_time`` (or ``speed``), and may
also contain ``duration_sec``, ``names``, and ``labels``.  If the archive has
sample names, pass the matching ``--sample-names`` sidecar so that samples are
joined by ID rather than by an assumed row order.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np


STRATA = ("slower", "middle", "faster")
STAT_NAMES = ("mean", "std", "min", "q25", "median", "q75", "max")
SPLIT_DATASET = {"1": "ntu60", "2": "ntu60", "3": "ntu60",
                 "4": "ntu120", "5": "ntu120", "6": "ntu120"}
SPEED_PROXY_VERSION = "valid_body_transition_v2"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-speed-proxy", action="store_true",
                        help="Run deterministic synthetic padding/scale/missing-body checks; no data needed.")
    parser.add_argument("--class-statistics-csv", default=None,
                        help="Only validate/aggregate an existing by-class CSV into Table S1; "
                             "no skeleton data or model checkpoints needed.")
    parser.add_argument("--all-splits", action="store_true",
                        help="Process NTU splits 1-6 in one command using --data-root.")
    parser.add_argument("--data-root", default="data/zeroshot",
                        help="Root containing <dataset>/split_<split>/ files in all-splits mode.")
    parser.add_argument("--splits", nargs="+", default=list(SPLIT_DATASET),
                        help="Splits for all-splits mode; default: 1 2 3 4 5 6.")
    parser.add_argument("--data", required=False,
                        help="Original unwarped unseen_data.npy, shape (N,C,T,V,M).")
    parser.add_argument("--labels", required=False,
                        help="Labels matching --data row-for-row.")
    parser.add_argument("--sample-names", default=None,
                        help="Optional sample-ID sidecar matching --data.")
    parser.add_argument("--sample-names-pattern", default=None,
                        help="All-splits pattern with {dataset} and {split}.")
    parser.add_argument("--raw-time-speed-npz", default=None,
                        help="Optional raw-time archive containing speed_time/speed.")
    parser.add_argument("--raw-time-speed-pattern", default=None,
                        help="All-splits raw-time NPZ pattern with {dataset} and {split}.")
    parser.add_argument("--duration-npy", default=None,
                        help="Optional duration_sec .npy aligned with --data.")
    parser.add_argument("--duration-pattern", default=None,
                        help="All-splits duration NPY pattern with {dataset} and {split}.")
    parser.add_argument("--prediction", action="append", default=[],
                        metavar="NAME=NPZ",
                        help="Optional prediction archive; repeat for PGFA and CauASD. "
                             "The NPZ must contain labels and predictions.")
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--split", default="unknown")
    parser.add_argument("--fraction", type=float, default=0.30,
                        help="Lower/upper within-class fraction; default: 0.30.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="analysis/natural_speed_statistics")
    parser.add_argument("--allow-row-alignment", action="store_true",
                        help="Allow raw-time archives without names to align by row "
                             "after checking labels and lengths.")
    parser.add_argument("--keep-going", action="store_true",
                        help="All-splits mode: continue after a split fails.")
    return parser.parse_args()


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {name: None for name in STAT_NAMES}
    return {
        "mean": finite_or_none(values.mean()),
        "std": finite_or_none(values.std()),
        "min": finite_or_none(values.min()),
        "q25": finite_or_none(np.quantile(values, 0.25)),
        "median": finite_or_none(np.median(values)),
        "q75": finite_or_none(np.quantile(values, 0.75)),
        "max": finite_or_none(values.max()),
    }


def speed_proxy_batch(x):
    """Match the current fixed-50 body-scale-normalized speed proxy."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 5 or x.shape[1] != 3:
        raise ValueError("Expected (N,3,T,V,M), got {}".format(x.shape))

    if x.shape[2] < 2 or x.shape[3] < 1 or x.shape[4] < 1:
        raise ValueError("Expected at least two frames, one joint and one body slot")
    # Missing bodies are all-zero joint coordinates in a particular frame.
    # Preserve the person axis: another active person must not validate an
    # empty slot. Non-finite body frames are also excluded.
    active = (np.any(np.abs(x) > 1e-6, axis=(1, 3))
              & np.all(np.isfinite(x), axis=(1, 3)))  # N,T,M
    transition = active[:, 1:, :] & active[:, :-1, :]  # N,T-1,M
    with np.errstate(invalid="ignore", over="ignore"):
        delta = np.linalg.norm(x[:, :, 1:] - x[:, :, :-1], axis=1)
        extent = np.linalg.norm(x - x[:, :, :, :1, :], axis=1)
    total = np.where(transition[:, :, None, :], delta, 0.0).sum(
        axis=(1, 2, 3), dtype=np.float64)
    count = transition.sum(axis=(1, 2), dtype=np.float64) * x.shape[3]
    extent = np.where(active[:, :, None, :], extent, np.nan)
    no_body = ~active.any(axis=(1, 2))
    if no_body.any():
        extent[no_body, 0, 0, 0] = 0.0  # avoid an all-NaN median warning
    scale = np.nanmedian(extent.reshape(len(x), -1), axis=1)
    valid = ((count > 0) & np.isfinite(total)
             & np.isfinite(scale) & (scale > 1e-6))
    # A degenerate body has no meaningful normalized speed. Exclude it
    # explicitly, rather than silently using coordinate-dependent scale=1.
    result = np.full(len(x), np.nan, dtype=np.float64)
    result[valid] = total[valid] / count[valid] / scale[valid]
    return result, count


def compute_fixed_speed(path, batch_size):
    data = np.load(path, mmap_mode="r")
    if data.ndim != 5 or data.shape[1] != 3:
        raise ValueError("Expected data shape (N,3,T,V,M), got {}".format(data.shape))
    values = np.empty(len(data), dtype=np.float64)
    transitions = np.empty(len(data), dtype=np.float64)
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        values[start:stop], transitions[start:stop] = speed_proxy_batch(data[start:stop])
        print("fixed speed: {}/{}".format(stop, len(data)), flush=True)
    return values, transitions


def check_speed_proxy():
    """Small, deterministic public checks for the corrected proxy."""
    x = np.zeros((1, 3, 50, 5, 1), dtype=np.float32)
    x[0, 0, :, :, 0] = (0.2 * np.arange(50)[:, None]
                         + 5.0 * np.arange(5)[None, :])
    padded = np.concatenate([x, np.zeros_like(x)], axis=-1)
    base, count = speed_proxy_batch(x)
    one_empty, _ = speed_proxy_batch(padded)
    doubled, _ = speed_proxy_batch(2.0 * padded)
    swapped, _ = speed_proxy_batch(padded[..., ::-1])
    for label, value in (("base", base), ("empty_slot", one_empty),
                         ("double_coordinates", doubled), ("slot_order", swapped)):
        if not np.allclose(value, 0.02, rtol=1e-6, atol=1e-9):
            raise AssertionError("{} invariance check failed: {}".format(label, value))
    if count[0] != 49 * 5:
        raise AssertionError("Invalid adjacent-frame joint count")
    missing = padded.copy()
    missing[:, :, 20:30, :, :] = 0
    missing_speed, missing_count = speed_proxy_batch(missing)
    if not np.allclose(missing_speed, base, rtol=1e-6, atol=1e-9):
        raise AssertionError("Missing-body entry/exit was included in displacement")
    if missing_count[0] != (19 + 19) * 5:
        raise AssertionError("Missing-body transition count is incorrect")
    static = np.repeat(x[:, :, :1], 50, axis=2)
    static_speed, _ = speed_proxy_batch(static)
    empty_speed, _ = speed_proxy_batch(np.zeros_like(padded))
    degenerate_speed, _ = speed_proxy_batch(np.ones_like(padded))
    if static_speed[0] != 0 or not np.isnan(empty_speed[0]) or not np.isnan(degenerate_speed[0]):
        raise AssertionError("Static/empty/degenerate sample policy failed")
    return {
        "speed_proxy_version": SPEED_PROXY_VERSION,
        "one_valid_body": float(base[0]),
        "with_empty_slot": float(one_empty[0]),
        "with_empty_slot_coordinates_x2": float(doubled[0]),
        "body_slot_order_invariant": True,
        "missing_body_transitions_excluded": True,
        "static_valid_body_speed": float(static_speed[0]),
        "empty_or_degenerate_samples_excluded": True,
        "checks_passed": True,
    }


def normalise_name(value):
    value = str(value).replace("\\", "/")
    if value.endswith(".skeleton"):
        value = value[:-9]
    return value


def load_names(path):
    if path is None:
        return None
    values = np.asarray(np.load(path, allow_pickle=False)).reshape(-1)
    names = np.asarray([normalise_name(value) for value in values])
    if len(set(names.tolist())) != len(names):
        raise ValueError("Duplicate names in {}".format(path))
    return names


def parse_predictions(items):
    result = []
    for item in items:
        if "=" not in item:
            raise ValueError("Prediction must have the form NAME=NPZ: {}".format(item))
        name, path = item.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError("Prediction must have the form NAME=NPZ: {}".format(item))
        result.append((name.strip(), path.strip()))
    return result


def format_pattern(pattern, dataset, split):
    if pattern is None:
        return None
    try:
        return pattern.format(dataset=dataset, split=split)
    except KeyError as exc:
        raise ValueError(
            "Pattern may use only {dataset} and {split}: {}".format(pattern)
        ) from exc


def parse_prediction_patterns(items):
    result = []
    for item in items:
        if "=" not in item:
            raise ValueError(
                "Prediction pattern must have the form NAME=PATTERN: {}".format(item)
            )
        name, pattern = item.split("=", 1)
        if not name.strip() or not pattern.strip():
            raise ValueError(
                "Prediction pattern must have the form NAME=PATTERN: {}".format(item)
            )
        result.append((name.strip(), pattern.strip()))
    return result


def merge_csvs(paths, output):
    rows = []
    fieldnames = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
            rows.extend(reader)
    if not rows:
        return None
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def run_all_splits(args):
    requested = [str(value) for value in args.splits]
    invalid = [value for value in requested if value not in SPLIT_DATASET]
    if invalid:
        raise ValueError(
            "Unsupported split(s): {}. Use 1-6.".format(", ".join(invalid))
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    data_root = args.data_root
    os.makedirs(args.output_dir, exist_ok=True)
    single_script = os.path.abspath(__file__)
    prediction_patterns = parse_prediction_patterns(args.prediction)
    completed = []
    failed = []

    for split in requested:
        dataset = SPLIT_DATASET[split]
        split_root = os.path.join(data_root, dataset, "split_" + split)
        data_path = os.path.join(split_root, "unseen_data.npy")
        labels_path = os.path.join(split_root, "unseen_label.npy")
        if not os.path.isfile(data_path) or not os.path.isfile(labels_path):
            message = "Missing data or labels for {} split {} under {}".format(
                dataset, split, split_root
            )
            if args.keep_going:
                print("WARNING: " + message, flush=True)
                failed.append({"dataset": dataset, "split": split, "error": message})
                continue
            raise FileNotFoundError(message)

        command = [
            sys.executable,
            single_script,
            "--data", data_path,
            "--labels", labels_path,
            "--dataset", dataset,
            "--split", split,
            "--fraction", str(args.fraction),
            "--batch-size", str(args.batch_size),
            "--output-dir", args.output_dir,
        ]
        sample_names = format_pattern(args.sample_names_pattern, dataset, split)
        raw_speed = format_pattern(args.raw_time_speed_pattern, dataset, split)
        duration = format_pattern(args.duration_pattern, dataset, split)
        if sample_names:
            command.extend(["--sample-names", sample_names])
        if raw_speed:
            command.extend(["--raw-time-speed-npz", raw_speed])
        if duration:
            command.extend(["--duration-npy", duration])
        if args.allow_row_alignment:
            command.append("--allow-row-alignment")
        for name, pattern in prediction_patterns:
            prediction_path = format_pattern(pattern, dataset, split)
            command.extend(["--prediction", "{}={}".format(name, prediction_path)])

        print("\nRunning {} split {}".format(dataset, split), flush=True)
        try:
            subprocess.run(command, check=True)
            completed.append({"dataset": dataset, "split": split})
        except subprocess.CalledProcessError as exc:
            message = "statistics failed with exit code {}".format(exc.returncode)
            if not args.keep_going:
                raise
            print("WARNING: {} split {}: {}".format(dataset, split, message), flush=True)
            failed.append({"dataset": dataset, "split": split, "error": message})

    prefixes = {
        "summary": "natural_speed_summary_",
        "by_class": "natural_speed_by_class_",
        "by_stratum": "natural_speed_by_stratum_",
        "correlations": "natural_speed_correlations_",
        "accuracy": "natural_speed_accuracy_",
    }
    combined = {}
    for key, prefix in prefixes.items():
        paths = [
            os.path.join(
                args.output_dir,
                prefix + "{}_split{}.csv".format(SPLIT_DATASET[split], split),
            )
            for split in (item["split"] for item in completed)
        ]
        output = os.path.join(args.output_dir, prefix + "all_splits.csv")
        merged = merge_csvs(paths, output)
        if merged is not None:
            combined[key] = os.path.abspath(merged)

    if "by_class" in combined:
        with open(combined["by_class"], "r", newline="", encoding="utf-8-sig") as handle:
            class_rows = list(csv.DictReader(handle))
        combined.update(write_s1_outputs(class_rows, args.output_dir, "all_splits"))

    metadata = {
        "data_root": os.path.abspath(data_root),
        "output_dir": os.path.abspath(args.output_dir),
        "requested_splits": requested,
        "completed": completed,
        "failed": failed,
        "combined_outputs": combined,
    }
    metadata_path = os.path.join(
        args.output_dir, "natural_speed_all_splits_metadata.json"
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print("\nCombined outputs:")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


def load_prediction(path, labels):
    with np.load(path, allow_pickle=False) as archive:
        if "labels" not in archive or "predictions" not in archive:
            raise ValueError("{} must contain labels and predictions".format(path))
        stored_labels = np.asarray(archive["labels"]).reshape(-1).astype(np.int64)
        predictions = np.asarray(archive["predictions"]).reshape(-1).astype(np.int64)
    if len(stored_labels) != len(labels) or not np.array_equal(stored_labels, labels):
        raise ValueError("Prediction labels do not match --labels: {}".format(path))
    return predictions


def align_raw_archive(path, base_labels, base_names, allow_row_alignment):
    with np.load(path, allow_pickle=False) as archive:
        speed_key = "speed_time" if "speed_time" in archive else "speed"
        if speed_key not in archive:
            raise ValueError("{} must contain speed_time or speed".format(path))
        raw_speed = np.asarray(archive[speed_key]).reshape(-1).astype(np.float64)
        raw_duration = (np.asarray(archive["duration_sec"]).reshape(-1).astype(np.float64)
                        if "duration_sec" in archive else None)
        raw_names = (np.asarray([normalise_name(value) for value in archive["names"]])
                     if "names" in archive else None)
        raw_labels = (np.asarray(archive["labels"]).reshape(-1).astype(np.int64)
                      if "labels" in archive else None)

    if raw_names is not None:
        if base_names is None:
            raise ValueError(
                "Raw archive has names; provide --sample-names for exact ID alignment.")
        if len(set(raw_names.tolist())) != len(raw_names):
            raise ValueError("Duplicate names in raw archive: {}".format(path))
        raw_by_name = {name: index for index, name in enumerate(raw_names.tolist())}
        missing = [name for name in base_names.tolist() if name not in raw_by_name]
        if missing:
            raise ValueError("{} sample names missing from raw archive; first: {}"
                             .format(len(missing), missing[0]))
        indices = np.asarray([raw_by_name[name] for name in base_names.tolist()], dtype=np.int64)
    else:
        if not allow_row_alignment:
            raise ValueError(
                "Raw archive has no names. Use --allow-row-alignment only when row "
                "order is known to match --data.")
        if len(raw_speed) != len(base_labels):
            raise ValueError("Raw speed and data lengths differ: {} vs {}"
                             .format(len(raw_speed), len(base_labels)))
        indices = np.arange(len(base_labels), dtype=np.int64)

    if raw_labels is not None:
        aligned_labels = raw_labels[indices]
        if not np.array_equal(aligned_labels, base_labels):
            raise ValueError("Raw archive labels do not match --labels after alignment")

    aligned_speed = raw_speed[indices]
    aligned_duration = raw_duration[indices] if raw_duration is not None else None
    return aligned_speed, aligned_duration


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return None
    xr, yr = rankdata(x), rankdata(y)
    xr -= xr.mean()
    yr -= yr.mean()
    denominator = np.linalg.norm(xr) * np.linalg.norm(yr)
    return finite_or_none(np.dot(xr, yr) / denominator) if denominator > 0 else None


def within_class_spearman(x, y, labels):
    values = []
    for cls in np.unique(labels):
        mask = labels == cls
        value = spearman(x[mask], y[mask])
        if value is not None:
            values.append(value)
    return finite_or_none(np.mean(values)) if values else None


def make_groups(speed, labels, fraction):
    if not 0 < fraction < 0.5:
        raise ValueError("--fraction must be between 0 and 0.5")
    masks = {name: np.zeros(len(labels), dtype=bool) for name in STRATA}
    details = {}
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        # The original accuracy script ranks log(max(speed, 1e-12)).  This is
        # monotonic for non-negative speed and also retains zero-speed samples.
        ranking_speed = np.log(np.maximum(speed, 1e-12))
        order = indices[np.argsort(ranking_speed[indices], kind="stable")]
        n = int(np.floor(len(indices) * fraction))
        if n < 1:
            raise ValueError("Class {} has too few samples for fraction {}"
                             .format(int(cls), fraction))
        slow_indices = order[:n]
        fast_indices = order[-n:]
        masks["slower"][slow_indices] = True
        masks["faster"][fast_indices] = True
        masks["middle"][order[n:-n]] = True

        class_speed = speed[indices]
        details[int(cls)] = {
            "class_n": int(len(indices)),
            "q30": finite_or_none(np.quantile(class_speed, 0.30)),
            "q70": finite_or_none(np.quantile(class_speed, 0.70)),
            "rank_slow_max": finite_or_none(speed[slow_indices[-1]]),
            "rank_fast_min": finite_or_none(speed[fast_indices[0]]),
            "slower_n": int(len(slow_indices)),
            "middle_n": int(len(indices) - 2 * n),
            "faster_n": int(len(fast_indices)),
        }
    return masks, details


def add_stats(row, values):
    row.update(stats(values))
    return row


def class_macro_rows(class_rows):
    """Aggregate the same complete class rows for EVERY S1 score column."""
    columns = ("q30", "q70", "mean", "std", "median")
    count_columns = ("class_n", "slower_n", "middle_n", "faster_n")
    grouped = {}
    seen = set()
    for row in class_rows:
        try:
            key = tuple(str(row[name]) for name in ("dataset", "split", "speed_definition"))
            class_id = int(row["class_id"])
            values = {name: float(row[name]) for name in columns}
            counts = {name: int(row[name]) for name in count_columns}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Incomplete or invalid class-level statistics: {}".format(row)) from exc
        identity = key + (class_id,)
        if identity in seen:
            raise ValueError("Duplicate class-level record: {}".format(identity))
        seen.add(identity)
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Non-finite class-level statistics: {}".format(identity))
        if not values["q30"] <= values["median"] <= values["q70"]:
            raise ValueError("Expected q30 <= median <= q70 for {}".format(identity))
        if values["std"] < 0 or min(counts.values()) < 0 or counts["class_n"] == 0:
            raise ValueError("Invalid standard deviation or sample count: {}".format(identity))
        if counts["class_n"] != sum(counts[name + "_n"] for name in STRATA):
            raise ValueError("Stratum counts do not sum to class_n: {}".format(identity))
        grouped.setdefault(key, []).append((values, counts))
    if not grouped:
        raise ValueError("No class-level statistics to aggregate")

    output = []
    for key, entries in sorted(grouped.items()):
        result = dict(zip(("dataset", "split", "speed_definition"), key))
        result["n_classes"] = len(entries)
        for name in columns:
            result["avg_" + name] = math.fsum(values[name] for values, _ in entries) / len(entries)
        for name in count_columns:
            result[name] = sum(counts[name] for _, counts in entries)
        if not result["avg_q30"] <= result["avg_median"] <= result["avg_q70"]:
            raise ValueError("Class-macro quantile order failed: {}".format(key))
        output.append(result)
    return output


def s1_latex(rows):
    """Render fixed-length S1 without substituting pooled split summaries."""
    rows = [row for row in rows if row["speed_definition"] == "fixed50_body_scale_proxy"]
    if not rows:
        raise ValueError("Table S1 needs fixed50_body_scale_proxy class-level rows")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Statistics of within-class speed stratification. Threshold and score "
        r"summaries are unweighted arithmetic means over action classes within each split. "
        r"S/M/F denotes sample counts. Averaged thresholds are descriptive summaries, "
        r"not shared cutoffs; class-specific thresholds are provided in the class-level export.}",
        r"\label{tab:natural_speed_statistics}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrr}",
        r"\hline",
        r"Dataset & Split & \#Cls. & \multicolumn{2}{c}{Class-wise thresholds} & "
        r"\multicolumn{3}{c}{Class-wise score summaries} & Sample counts \\",
        r" & & & Avg. $q_{30}$ & Avg. $q_{70}$ & Avg. Mean & Avg. Std & Avg. Median & S/M/F \\",
        r"\hline",
    ]
    datasets = {"ntu60": "NTU-60", "ntu120": "NTU-120"}
    for row in rows:
        dataset = datasets.get(row["dataset"])
        if dataset is None:
            raise ValueError("Unsupported Table-S1 dataset: {}".format(row["dataset"]))
        fields = [dataset, str(int(row["split"])), str(row["n_classes"])]
        fields.extend("{:.4f}".format(row["avg_" + name])
                      for name in ("q30", "q70", "mean", "std", "median"))
        fields.append("{}/{}/{}".format(row["slower_n"], row["middle_n"], row["faster_n"]))
        lines.append(" & ".join(fields) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", "}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def write_s1_outputs(class_rows, output_dir, tag):
    rows = class_macro_rows(class_rows)
    latex = s1_latex(rows)
    os.makedirs(output_dir, exist_ok=True)
    macro_path = os.path.join(output_dir, "natural_speed_class_macro_{}.csv".format(tag))
    latex_path = os.path.join(output_dir, "natural_speed_s1_{}.tex".format(tag))
    write_csv(macro_path, rows)
    with open(latex_path, "w", encoding="utf-8") as handle:
        handle.write(latex)
    return {"class_macro": os.path.abspath(macro_path), "s1_latex": os.path.abspath(latex_path)}


def build_rows(dataset, split, definition, speed, labels, fraction):
    groups, details = make_groups(speed, labels, fraction)
    class_rows = []
    stratum_rows = []
    for cls in np.unique(labels):
        cls = int(cls)
        class_mask = labels == cls
        detail = details[cls]
        class_row = {
            "dataset": dataset,
            "split": split,
            "speed_definition": definition,
            "class_id": cls,
            **detail,
        }
        add_stats(class_row, speed[class_mask])
        class_rows.append(class_row)
        for stratum in STRATA:
            mask = class_mask & groups[stratum]
            row = {
                "dataset": dataset,
                "split": split,
                "speed_definition": definition,
                "class_id": cls,
                "stratum": stratum,
                "n": int(mask.sum()),
            }
            add_stats(row, speed[mask])
            stratum_rows.append(row)

    summary = {
        "dataset": dataset,
        "split": split,
        "speed_definition": definition,
        "n_valid": int(len(labels)),
        "n_classes": int(len(np.unique(labels))),
        "slower_n": int(groups["slower"].sum()),
        "middle_n": int(groups["middle"].sum()),
        "faster_n": int(groups["faster"].sum()),
    }
    add_stats(summary, speed)
    return summary, class_rows, stratum_rows, groups


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prediction_accuracy_rows(definition, speed, labels, groups, predictions):
    rows = []
    for method, prediction in predictions.items():
        values = {"method": method, "speed_definition": definition}
        accuracies = {}
        for stratum in STRATA:
            mask = groups[stratum]
            per_class = []
            for cls in np.unique(labels):
                class_mask = mask & (labels == cls)
                if class_mask.any():
                    per_class.append(float((prediction[class_mask] == labels[class_mask]).mean()))
            accuracies[stratum] = float(np.mean(per_class)) if per_class else None
            values[stratum + "_n"] = int(mask.sum())
            values[stratum + "_macro_accuracy"] = accuracies[stratum]
        if accuracies["slower"] is not None and accuracies["faster"] is not None:
            values["slow_fast_gap"] = abs(accuracies["faster"] - accuracies["slower"])
        else:
            values["slow_fast_gap"] = None
        rows.append(values)
    return rows


def main():
    args = parse_args()
    if args.check_speed_proxy:
        print(json.dumps(check_speed_proxy(), indent=2))
        return
    if args.class_statistics_csv:
        if args.all_splits or args.data or args.labels:
            raise ValueError("--class-statistics-csv is a standalone aggregation mode")
        with open(args.class_statistics_csv, "r", newline="", encoding="utf-8-sig") as handle:
            class_rows = list(csv.DictReader(handle))
        outputs = write_s1_outputs(class_rows, args.output_dir, "all_splits")
        print("Checked class-macro aggregation only; upstream speed scores are unchanged.")
        print("Source: " + os.path.abspath(args.class_statistics_csv))
        print(json.dumps(outputs, indent=2, ensure_ascii=False))
        print(s1_latex(class_macro_rows(class_rows)))
        return
    if args.all_splits:
        run_all_splits(args)
        return
    if not args.data or not args.labels:
        raise SystemExit("--data and --labels are required unless --all-splits is used")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    labels = np.asarray(np.load(args.labels, allow_pickle=False)).reshape(-1).astype(np.int64)
    data = np.load(args.data, mmap_mode="r")
    if len(data) != len(labels):
        raise ValueError("Data/label length mismatch: {} vs {}".format(len(data), len(labels)))

    names = load_names(args.sample_names)
    if names is not None and len(names) != len(labels):
        raise ValueError("Sample-name/data length mismatch")

    fixed_speed, transitions = compute_fixed_speed(args.data, args.batch_size)
    # Keep genuine static samples, but exclude invalid geometry/transitions.
    fixed_valid = np.isfinite(fixed_speed)
    if not fixed_valid.any():
        raise ValueError("No samples have valid body extent and adjacent-frame transitions")
    if not fixed_valid.all():
        print("warning: excluding {} samples without a valid fixed-speed value"
              .format(int((~fixed_valid).sum())), flush=True)

    raw_speed = raw_duration = None
    if args.raw_time_speed_npz:
        raw_speed, raw_duration = align_raw_archive(
            args.raw_time_speed_npz, labels, names, args.allow_row_alignment)

    if args.duration_npy:
        duration = np.asarray(np.load(args.duration_npy, allow_pickle=False)).reshape(-1).astype(np.float64)
        if len(duration) != len(labels):
            raise ValueError("Duration/data length mismatch")
        raw_duration = duration

    definitions = [("fixed50_body_scale_proxy", fixed_speed, fixed_valid)]
    if raw_speed is not None:
        raw_valid = np.isfinite(raw_speed) & (raw_speed > 0)
        definitions.append(("raw_time_body_scale_proxy", raw_speed, raw_valid))

    prediction_items = parse_predictions(args.prediction)
    prediction_map = {
        name: load_prediction(path, labels) for name, path in prediction_items
    }

    all_summary = []
    all_class_rows = []
    all_stratum_rows = []
    all_accuracy_rows = []
    for definition, values, valid in definitions:
        valid = valid & np.isfinite(labels)
        local_speed = values[valid]
        local_labels = labels[valid]
        summary, class_rows, stratum_rows, groups = build_rows(
            args.dataset, args.split, definition, local_speed, local_labels, args.fraction)
        summary["n_total"] = int(len(labels))
        summary["n_excluded"] = int(len(labels) - len(local_labels))
        version = SPEED_PROXY_VERSION if definition == "fixed50_body_scale_proxy" else "raw_time_archive"
        for row in [summary] + class_rows + stratum_rows:
            row["speed_proxy_version"] = version
        all_summary.append(summary)
        all_class_rows.extend(class_rows)
        all_stratum_rows.extend(stratum_rows)
        if prediction_map:
            local_predictions = {name: prediction[valid]
                                 for name, prediction in prediction_map.items()}
            all_accuracy_rows.extend(prediction_accuracy_rows(
                definition, local_speed, local_labels, groups, local_predictions))

    correlation_rows = []
    if raw_speed is not None:
        valid = fixed_valid & np.isfinite(raw_speed) & (raw_speed > 0)
        correlation_rows.append({
            "dataset": args.dataset,
            "split": args.split,
            "x": "fixed50_body_scale_proxy",
            "y": "raw_time_body_scale_proxy",
            "n": int(valid.sum()),
            "spearman": spearman(fixed_speed[valid], raw_speed[valid]),
            "within_class_spearman_macro": within_class_spearman(
                fixed_speed[valid], raw_speed[valid], labels[valid]),
        })
    if raw_duration is not None:
        valid = fixed_valid & np.isfinite(raw_duration) & (raw_duration > 0)
        correlation_rows.append({
            "dataset": args.dataset,
            "split": args.split,
            "x": "fixed50_body_scale_proxy",
            "y": "original_duration_sec",
            "n": int(valid.sum()),
            "spearman": spearman(fixed_speed[valid], raw_duration[valid]),
            "within_class_spearman_macro": within_class_spearman(
                fixed_speed[valid], raw_duration[valid], labels[valid]),
        })

    os.makedirs(args.output_dir, exist_ok=True)
    tag = "{}_split{}".format(args.dataset, args.split)
    summary_path = os.path.join(args.output_dir, "natural_speed_summary_{}.csv".format(tag))
    class_path = os.path.join(args.output_dir, "natural_speed_by_class_{}.csv".format(tag))
    stratum_path = os.path.join(args.output_dir, "natural_speed_by_stratum_{}.csv".format(tag))
    corr_path = os.path.join(args.output_dir, "natural_speed_correlations_{}.csv".format(tag))
    accuracy_path = os.path.join(args.output_dir, "natural_speed_accuracy_{}.csv".format(tag))
    metadata_path = os.path.join(args.output_dir, "natural_speed_metadata_{}.json".format(tag))

    write_csv(summary_path, all_summary)
    write_csv(class_path, all_class_rows)
    write_csv(stratum_path, all_stratum_rows)
    s1_outputs = write_s1_outputs(all_class_rows, args.output_dir, tag)
    write_csv(corr_path, correlation_rows)
    if all_accuracy_rows:
        write_csv(accuracy_path, all_accuracy_rows)

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "data": os.path.abspath(args.data),
        "labels": os.path.abspath(args.labels),
        "sample_names": os.path.abspath(args.sample_names) if args.sample_names else None,
        "raw_time_speed_npz": (os.path.abspath(args.raw_time_speed_npz)
                                if args.raw_time_speed_npz else None),
        "fraction": args.fraction,
        "fixed_speed_proxy_version": SPEED_PROXY_VERSION,
        "fixed_speed_validity": {
            "missing_body": "all-zero joint coordinates in a person slot/frame",
            "active_body": "finite coordinates with at least one absolute coordinate > 1e-6",
            "transition": "same body slot active in both adjacent frames",
            "body_scale": "median root-relative joint extent over active body frames only",
            "invalid_policy": "exclude no-transition or non-finite/extent<=1e-6 samples; no scale=1 fallback",
            "n_excluded": int((~fixed_valid).sum()),
            "excluded_sample_indices": np.flatnonzero(~fixed_valid).tolist(),
        },
        "definition_rows": all_summary,
        "outputs": {
            "summary": os.path.abspath(summary_path),
            "by_class": os.path.abspath(class_path),
            "by_stratum": os.path.abspath(stratum_path),
            "correlations": os.path.abspath(corr_path),
            "accuracy": os.path.abspath(accuracy_path) if all_accuracy_rows else None,
            **s1_outputs,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
