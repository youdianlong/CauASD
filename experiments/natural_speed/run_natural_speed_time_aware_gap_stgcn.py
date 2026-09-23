#!/usr/bin/env python3
"""Re-evaluate natural-speed strata with an independent time-aware speed proxy.

This script is a post-processing/evaluation script.  It does not train any
model.  It uses the original fixed-50-frame test samples for inference, but
constructs the slower/middle/faster strata with ``speed_time`` from the raw
variable-length sequences produced by ``build_raw_time_speed.py``.

The three checkpoints are evaluated on exactly the same original samples:
PGFA, PGFA augmented (PGFA^ddagger), and CauASD.  The output has the same
columns as the natural-speed Table-V script, so it can be compared directly
with the existing proxy-defined result.

Example with sample-name alignment:

  python 第二次大修/CauASD/experiments/natural_speed/run_natural_speed_time_aware_gap_stgcn.py \
      --split 1 --device 0 \
      --baseline path/to/pgfa_split1.pt \
      --pgfaaug path/to/pgfa_aug_split1.pt \
      --cauasd path/to/cauasd_split1.pt \
      --output analysis/natural_speed_time_aware/ntu60_split1_natural_speed_time_aware.csv

The raw-time archive and sample-name sidecar are selected automatically from
the split.  By default, the script expects the raw-time files in the sibling
``SkeletonGCL-main/analysis/raw_time_speed`` directory and the sample-name
sidecars under the current project's ``data/zeroshot`` directory.  The
locations can be overridden with ``--raw-time-root`` and ``--data-root``.
"""

import argparse
import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from export_cauasd_speed_features import NpyDataset, load_models


UNSEEN = {
    ("ntu60", "1"): [4, 19, 31, 47, 51],
    ("ntu60", "2"): [12, 29, 32, 44, 59],
    ("ntu60", "3"): [7, 20, 28, 39, 58],
    ("ntu120", "4"): [3, 18, 26, 38, 41, 60, 87, 99, 102, 110],
    ("ntu120", "5"): [5, 12, 14, 15, 17, 42, 67, 82, 100, 119],
    ("ntu120", "6"): [6, 20, 27, 33, 42, 55, 71, 97, 104, 118],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--baseline", required=True,
                        help="PGFA checkpoint with encoder/adapter keys.")
    parser.add_argument("--pgfaaug", required=True,
                        help="PGFA augmented checkpoint with encoder/adapter keys.")
    parser.add_argument("--cauasd", required=True,
                        help="CauASD checkpoint with encoder/adapter keys.")
    parser.add_argument("--data-root", default=None,
                        help="Root containing <dataset>/split_<split>/ files.")
    parser.add_argument("--raw-time-root", default=None,
                        help="Root containing ntu60_raw_time_speed.npz and "
                             "ntu120_val_raw_time_speed.npz.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fraction", type=float, default=0.30,
                        help="Lower/upper within-class fraction; default: 0.30.")
    parser.add_argument("--output", default="",
                        help="Output CSV path. A JSON and NPZ are written beside it.")
    return parser.parse_args()


def project_root():
    """Find the repository root from the working directory or script path."""
    candidates = []
    for start in (os.getcwd(), os.path.dirname(__file__)):
        current = os.path.abspath(start)
        for _ in range(8):
            candidates.append(current)
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "data", "zeroshot")):
            return candidate

    # Keep a deterministic fallback for a clear error message later.
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", ".."
    ))


def default_raw_time_path(dataset, raw_time_root=None):
    filename = {
        "ntu60": "ntu60_raw_time_speed.npz",
        "ntu120": "ntu120_val_raw_time_speed.npz",
    }[dataset]
    root = raw_time_root or os.path.join(
        project_root(), "..", "SkeletonGCL-main", "analysis", "raw_time_speed"
    )
    return os.path.abspath(os.path.join(root, filename))


def normalise_name(value):
    value = str(value).replace("\\", "/")
    if value.endswith(".skeleton"):
        value = value[:-9]
    return value


def load_names(path):
    values = np.asarray(np.load(path, allow_pickle=False)).reshape(-1)
    names = np.asarray([normalise_name(value) for value in values])
    if len(set(names.tolist())) != len(names):
        raise ValueError("Duplicate sample names in {}".format(path))
    return names


def speed_proxy_batch(data):
    """The fixed-50 body-scale-normalized proxy used by the original Table V."""
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 5 or x.shape[1] != 3:
        raise ValueError("Expected data shape (N,3,T,V,M), got {}".format(x.shape))

    active = np.any(np.abs(x) > 1e-6, axis=(1, 3, 4))
    delta = np.linalg.norm(x[:, :, 1:] - x[:, :, :-1], axis=1)
    transition = active[:, 1:] & active[:, :-1]
    total = (delta * transition[:, :, None, None]).sum(
        axis=(1, 2, 3), dtype=np.float64)
    count = transition.sum(axis=1, dtype=np.float64) * x.shape[3] * x.shape[4]
    speed = total / np.maximum(count, 1.0)

    root = x[:, :, :, :1, :]
    extent = np.linalg.norm(x - root, axis=1)
    extent = np.where(active[:, :, None, None], extent, np.nan)
    invalid = ~active.any(axis=1)
    if invalid.any():
        extent[invalid, 0, 0, 0] = 1.0
    scale = np.nanmedian(extent.reshape(len(x), -1), axis=1)
    scale = np.where(
        np.isfinite(scale) & (scale > 1e-6),
        scale,
        1.0,
    )
    return speed / scale


def compute_fixed_speed(data_path, batch_size):
    data = np.load(data_path, mmap_mode="r")
    speed = np.empty(len(data), dtype=np.float64)
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        speed[start:stop] = speed_proxy_batch(data[start:stop])
        print("fixed speed: {}/{}".format(stop, len(data)), flush=True)
    return speed


def load_raw_time_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        speed_key = "speed_time" if "speed_time" in archive.files else "speed"
        if speed_key not in archive.files:
            raise ValueError("{} must contain speed_time or speed".format(path))

        speed = archive[speed_key].reshape(-1).astype(np.float64)
        names = None
        if "names" in archive.files:
            names = np.asarray([
                normalise_name(value) for value in archive["names"].reshape(-1)
            ])
            if len(set(names.tolist())) != len(names):
                raise ValueError("Duplicate names in {}".format(path))

        labels = None
        if "labels" in archive.files:
            labels = archive["labels"].reshape(-1).astype(np.int64)

        duration = None
        if "duration_sec" in archive.files:
            duration = archive["duration_sec"].reshape(-1).astype(np.float64)

    lengths = {len(speed)}
    if names is not None:
        lengths.add(len(names))
    if labels is not None:
        lengths.add(len(labels))
    if duration is not None:
        lengths.add(len(duration))
    if len(lengths) != 1:
        raise ValueError("Raw-time arrays have different lengths in {}".format(path))
    return names, labels, speed, duration


def align_raw_time_values(raw_names, raw_labels, raw_speed, raw_duration,
                          fixed_names, fixed_labels, allow_row_alignment):
    if fixed_names is not None:
        if raw_names is None:
            raise ValueError(
                "The raw-time archive has no names; omit --sample-names or "
                "use --allow-row-alignment after verifying row order."
            )
        raw_index = {name: i for i, name in enumerate(raw_names.tolist())}
        missing = [name for name in fixed_names if name not in raw_index]
        if missing:
            raise ValueError(
                "{} fixed samples are missing from the raw-time archive; first: {}"
                .format(len(missing), missing[0])
            )
        order = np.asarray([raw_index[name] for name in fixed_names], dtype=np.int64)
        speed = raw_speed[order]
        duration = None if raw_duration is None else raw_duration[order]
        if raw_labels is not None:
            aligned_labels = raw_labels[order]
            if not np.array_equal(aligned_labels, fixed_labels):
                raise ValueError("Sample-name alignment produced label mismatches.")
        return speed, duration, "sample_name_alignment"

    if not allow_row_alignment:
        raise ValueError(
            "Provide --sample-names, or explicitly add --allow-row-alignment "
            "after verifying that the raw-time archive uses the same row order."
        )
    if len(raw_speed) != len(fixed_labels):
        raise ValueError("Raw-time and fixed-data lengths differ.")
    if raw_labels is not None and not np.array_equal(raw_labels, fixed_labels):
        raise ValueError("Row alignment rejected because labels do not match.")
    return raw_speed, raw_duration, "row_alignment_after_label_check"


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
    x = rankdata(x)
    y = rankdata(y)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 0 else None


def within_class_spearman(x, y, labels):
    values = []
    for cls in np.unique(labels):
        mask = labels == cls
        if mask.sum() >= 2:
            value = spearman(x[mask], y[mask])
            if value is not None:
                values.append(value)
    return float(np.mean(values)) if values else None


def make_groups(speed, labels, fraction):
    slow = np.zeros(len(labels), dtype=bool)
    fast = np.zeros(len(labels), dtype=bool)
    for cls in np.unique(labels):
        indices = np.flatnonzero(labels == cls)
        n = int(np.floor(len(indices) * fraction))
        if n < 1:
            continue
        order = indices[np.argsort(speed[indices], kind="stable")]
        slow[order[:n]] = True
        fast[order[-n:]] = True
    return {
        "Naturally slower": slow,
        "Middle": ~(slow | fast),
        "Naturally faster": fast,
    }


def macro_accuracy(labels, prediction, conditions):
    result = {}
    for name, mask in conditions.items():
        per_class = []
        for cls in np.unique(labels):
            selected = (labels == cls) & mask
            if selected.any():
                per_class.append(float((prediction[selected] == cls).mean()))
        result[name] = float(np.mean(per_class)) if per_class else None
    result["Slow-fast gap"] = abs(
        result["Naturally faster"] - result["Naturally slower"]
    )
    return result


def predict(checkpoint, loader, device, text, unseen):
    model_args = SimpleNamespace(
        backbone="stgcn",
        checkpoint=checkpoint,
        stgcn_hidden_channels=16,
        stgcn_hidden_dim=256,
        stgcn_dropout=0.5,
        stgcn_layout="ntu-rgb+d",
        stgcn_strategy="spatial",
    )
    encoder, adapter = load_models(model_args, device)
    prediction = []
    with torch.no_grad():
        for x, _, _ in loader:
            feature = F.normalize(
                adapter(encoder(x.to(device, non_blocking=True))), dim=1
            )
            prediction.append(
                unseen[(feature @ text.T).argmax(dim=1).cpu().numpy()]
            )
    return np.concatenate(prediction)


def write_csv(path, results):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Method", "Naturally slower", "Middle", "Naturally faster",
            "Slow-fast gap",
        ])
        for name, values in results:
            writer.writerow([
                name,
                "{:.2f}".format(100.0 * values["Naturally slower"]),
                "{:.2f}".format(100.0 * values["Middle"]),
                "{:.2f}".format(100.0 * values["Naturally faster"]),
                "{:.2f}".format(100.0 * values["Slow-fast gap"]),
            ])


def main():
    args = parse_args()
    split = str(args.split)
    dataset = (
        "ntu60" if split in {"1", "2", "3"}
        else "ntu120" if split in {"4", "5", "6"}
        else ""
    )
    key = (dataset, split)
    if key not in UNSEEN:
        raise ValueError("Supported dataset/split pairs: {}".format(sorted(UNSEEN)))
    if not 0.0 < args.fraction < 0.5:
        raise ValueError("--fraction must be between 0 and 0.5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint evaluation.")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda:{}".format(args.device))

    repo = project_root()
    data_root = os.path.abspath(
        args.data_root or os.path.join(repo, "data", "zeroshot")
    )
    split_root = os.path.join(data_root, dataset, "split_" + split)
    data_path = os.path.join(split_root, "unseen_data.npy")
    labels_path = os.path.join(split_root, "unseen_label.npy")
    if not os.path.isfile(data_path) or not os.path.isfile(labels_path):
        raise FileNotFoundError(
            "Missing unseen_data.npy or unseen_label.npy under {}".format(split_root)
        )

    labels = np.load(labels_path).reshape(-1).astype(np.int64)
    unseen = np.asarray(UNSEEN[key], dtype=np.int64)
    if not np.isin(labels, unseen).all():
        raise ValueError("Unexpected labels in {}".format(labels_path))

    fixed_speed = compute_fixed_speed(data_path, args.batch_size)
    if len(fixed_speed) != len(labels):
        raise ValueError("Fixed speed and label lengths differ.")

    raw_time_path = default_raw_time_path(dataset, args.raw_time_root)
    if not os.path.isfile(raw_time_path):
        raise FileNotFoundError(
            "Automatically selected raw-time file does not exist: {}. "
            "Use --raw-time-root to override its location.".format(raw_time_path)
        )
    raw_names, raw_labels, raw_speed, raw_duration = load_raw_time_archive(
        raw_time_path
    )

    sample_names_path = os.path.join(split_root, "unseen_sample_names.npy")
    if not os.path.isfile(sample_names_path):
        raise FileNotFoundError(
            "Missing automatically selected sample-name file: {}".format(
                sample_names_path
            )
        )
    fixed_names = load_names(sample_names_path)
    if len(fixed_names) != len(labels):
        raise ValueError("Sample-name and label lengths differ.")

    time_speed, duration, alignment = align_raw_time_values(
        raw_names,
        raw_labels,
        raw_speed,
        raw_duration,
        fixed_names,
        labels,
        False,
    )

    valid = np.isfinite(fixed_speed) & np.isfinite(time_speed)
    if duration is not None:
        valid &= np.isfinite(duration)
    excluded = int((~valid).sum())
    if not valid.any():
        raise ValueError("No valid samples remain after raw-time filtering.")
    if excluded:
        print("warning: excluding {} samples without valid time-aware speed".format(excluded))

    loader = DataLoader(
        NpyDataset(data_path, labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    # ``data_root`` points to ``<project>/data/zeroshot``.  Derive the
    # language directory from it so the script also works when copied to or
    # launched from a different directory.
    language_path = os.path.join(
        os.path.dirname(data_root),
        "language",
        dataset + "_des_embeddings.npy",
    )
    language = torch.as_tensor(
        np.load(language_path), dtype=torch.float32, device=device
    )
    text = F.normalize(language[torch.as_tensor(unseen, device=device)], dim=1)

    baseline_prediction = predict(args.baseline, loader, device, text, unseen)
    pgfaaug_prediction = predict(args.pgfaaug, loader, device, text, unseen)
    cauasd_prediction = predict(args.cauasd, loader, device, text, unseen)

    labels = labels[valid]
    fixed_speed = fixed_speed[valid]
    time_speed = time_speed[valid]
    if duration is not None:
        duration = duration[valid]
    baseline_prediction = baseline_prediction[valid]
    pgfaaug_prediction = pgfaaug_prediction[valid]
    cauasd_prediction = cauasd_prediction[valid]

    groups = make_groups(time_speed, labels, args.fraction)
    baseline_result = macro_accuracy(labels, baseline_prediction, groups)
    pgfaaug_result = macro_accuracy(labels, pgfaaug_prediction, groups)
    cauasd_result = macro_accuracy(labels, cauasd_prediction, groups)
    results = [
        ("PGFA", baseline_result),
        ("PGFAaug", pgfaaug_result),
        ("CauASD", cauasd_result),
    ]

    output = args.output or os.path.join(
        "analysis", "natural_speed_time_aware",
        "{}_split{}_natural_speed_time_aware.csv".format(dataset, split),
    )
    write_csv(output, results)

    stem = os.path.splitext(output)[0]
    metadata = {
        "dataset": dataset,
        "split": split,
        "speed_definition": "body-scale-normalized trajectory length divided by nominal duration",
        "raw_time_speed_npz": raw_time_path,
        "alignment": alignment,
        "fraction": args.fraction,
        "n_total": int(len(valid)),
        "n_valid": int(valid.sum()),
        "n_excluded": excluded,
        "group_counts": {
            name: int(mask.sum()) for name, mask in groups.items()
        },
        "correlations": {
            "fixed_v50_vs_v_time_spearman": spearman(fixed_speed, time_speed),
            "fixed_v50_vs_v_time_within_class_spearman": within_class_spearman(
                fixed_speed, time_speed, labels
            ),
            "fixed_v50_vs_duration_spearman": (
                spearman(fixed_speed, duration) if duration is not None else None
            ),
            "v_time_vs_duration_spearman": (
                spearman(time_speed, duration) if duration is not None else None
            ),
        },
        "results": {
            name: values for name, values in results
        },
        "outputs": {
            "csv": output,
            "metadata": stem + ".json",
            "npz": stem + ".npz",
        },
    }
    with open(stem + ".json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    np.savez_compressed(
        stem + ".npz",
        labels=labels,
        fixed_v50=fixed_speed,
        v_time=time_speed,
        duration_sec=duration if duration is not None else np.asarray([]),
        pgfa_prediction=baseline_prediction,
        pgfaaug_prediction=pgfaaug_prediction,
        cauasd_prediction=cauasd_prediction,
        slower=groups["Naturally slower"],
        middle=groups["Middle"],
        faster=groups["Naturally faster"],
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
