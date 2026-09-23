#!/usr/bin/env python3
"""Correlate the existing fixed-50 speed proxy with raw-time speed.

This is a post-processing-only diagnostic. It does not train or evaluate a
recognition model and does not change Table V.
"""

import argparse
import json
import os
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-data", required=True,
                        help="Fixed 50-frame unseen_data.npy.")
    parser.add_argument("--labels", required=True,
                        help="Labels in the same order as fixed-data.")
    parser.add_argument("--sample-names", required=True,
                        help="Sample names in the same order as fixed-data.")
    parser.add_argument("--raw-speed-npz", required=True,
                        help="NPZ produced by build_raw_time_speed.py.")
    parser.add_argument("--output", required=True,
                        help="Output JSON path.")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def normalise_name(name):
    name = str(name)
    return name[:-9] if name.endswith(".skeleton") else name


def fixed_speed_batch(x):
    """Current Table-V body-scale-normalised speed on fixed 50-frame input."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 5 or x.shape[1] != 3 or x.shape[2] != 50:
        raise ValueError("Expected (N,3,50,V,M), got {}".format(x.shape))

    active = np.any(np.abs(x) > 1e-6, axis=(1, 3, 4))
    delta = np.linalg.norm(x[:, :, 1:] - x[:, :, :-1], axis=1)
    transition = active[:, 1:] & active[:, :-1]
    total = (delta * transition[:, :, None, None]).sum(axis=(1, 2, 3), dtype=np.float64)
    count = transition.sum(axis=1, dtype=np.float64) * x.shape[3] * x.shape[4]
    speed = total / np.maximum(count, 1.0)

    root = x[:, :, :, :1, :]
    extent = np.linalg.norm(x - root, axis=1)
    extent = np.where(active[:, :, None, None], extent, np.nan)
    invalid = ~active.any(axis=1)
    if invalid.any():
        extent[invalid, 0, 0, 0] = 1.0
    scale = np.nanmedian(extent.reshape(len(x), -1), axis=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return speed / scale


def fixed_speed(path, batch_size):
    data = np.load(path, mmap_mode="r")
    values = np.empty(len(data), dtype=np.float64)
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        values[start:stop] = fixed_speed_batch(data[start:stop])
    return values


def rankdata(values):
    """Average ranks, including ties, without requiring scipy."""
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


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 0 else None


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def within_class_spearman(x, y, labels):
    values = []
    for cls in np.unique(labels):
        mask = labels == cls
        if mask.sum() >= 2:
            value = spearman(x[mask], y[mask])
            if value is not None:
                values.append(value)
    return float(np.mean(values)) if values else None


def main():
    args = parse_args()
    labels = np.asarray(np.load(args.labels)).reshape(-1).astype(np.int64)
    names = np.asarray(np.load(args.sample_names)).astype(str)
    names = np.asarray([normalise_name(name) for name in names])
    fixed = fixed_speed(args.fixed_data, args.batch_size)

    if not (len(labels) == len(names) == len(fixed)):
        raise ValueError("fixed-data, labels, and sample-names lengths differ")
    if len(set(names.tolist())) != len(names):
        raise ValueError("Duplicate sample names in fixed test sidecar")

    with np.load(args.raw_speed_npz, allow_pickle=False) as archive:
        raw_names = np.asarray([normalise_name(name) for name in archive["names"].astype(str)])
        raw_speed = archive["speed_time"].reshape(-1).astype(np.float64)
    if len(set(raw_names.tolist())) != len(raw_names):
        raise ValueError("Duplicate sample names in raw speed file")
    raw_by_name = dict(zip(raw_names, raw_speed))
    missing = [name for name in names if name not in raw_by_name]
    if missing:
        raise ValueError("Missing raw-time speed for {} samples; first is {}".format(len(missing), missing[0]))
    time_speed = np.asarray([raw_by_name[name] for name in names], dtype=np.float64)

    valid = np.isfinite(fixed) & np.isfinite(time_speed) & (fixed > 0) & (time_speed > 0)
    excluded = names[~valid].tolist()
    if not valid.any():
        raise ValueError("No valid samples remain after filtering")
    fixed = fixed[valid]
    time_speed = time_speed[valid]
    labels = labels[valid]

    result = {
        "n_samples_total": int(len(names)),
        "n_samples_valid": int(len(labels)),
        "n_samples_excluded": int(len(excluded)),
        "excluded_sample_names": excluded,
        "n_classes": int(len(np.unique(labels))),
        "fixed_speed": "body-scale-normalised mean inter-frame displacement on fixed 50-frame input",
        "raw_time_speed": "mean-joint raw trajectory length divided by nominal duration and body scale",
        "nominal_fps": 30.0,
        "pearson": pearson(fixed, time_speed),
        "spearman": spearman(fixed, time_speed),
        "within_class_spearman_macro": within_class_spearman(fixed, time_speed, labels),
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
