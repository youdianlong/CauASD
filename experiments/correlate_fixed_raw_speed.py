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
    """Corrected Table-V proxy with per-person and per-transition masks."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 5 or x.shape[1] != 3 or x.shape[2] != 50:
        raise ValueError("Expected (N,3,50,V,M), got {}".format(x.shape))

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
    return result


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
        "fixed_speed_proxy_version": "valid_body_transition_v2",
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
