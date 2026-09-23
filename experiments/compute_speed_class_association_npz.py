#!/usr/bin/env python3
"""Compute class--speed association (eta-squared) from an official NTU NPZ.

This is a data-only analysis: it neither reads a ZSL split nor loads a model.
It accepts the common NTU layouts ``(N,T,150)``, ``(N,T,V,C[,M])`` and
``(N,C,T,V,M)``.  Input should be the official training partition, e.g.
``NTU60_CS.npz`` with ``x_train`` and ``y_train``.

Example:
  python compute_speed_class_association_npz.py \
    --npz /path/to/NTU60_CS.npz \
    --output-dir analysis/ntu60_xsub_speed_association
"""
import argparse
import csv
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz", help="Official NTU .npz archive.")
    source.add_argument("--data-npy", help="Skeleton tensor .npy, e.g. train_position.npy.")
    p.add_argument("--label-npy", help="Labels for --data-npy, e.g. train_label.npy.")
    p.add_argument("--data-key", default="x_train")
    p.add_argument("--label-key", default="y_train")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Samples per speed-computation batch; keeps 50-frame NTU inputs memory-safe.")
    p.add_argument("--no-scale-normalise", action="store_true",
                   help="Use displacement in stored coordinate units instead of body-scale-normalised displacement.")
    return p.parse_args()


def labels_from_array(y):
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim == 2:
        if y.shape[1] == 1:
            return y[:, 0].astype(np.int64)
        return y.argmax(axis=1).astype(np.int64)
    raise ValueError("Unsupported label shape {}.".format(y.shape))


def to_ntvmc(x):
    """Convert common NTU tensors to (N,T,V,M,C)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3:  # (N,T,75) or (N,T,150)
        n, t, d = x.shape
        if d % 75:
            raise ValueError("Flattened last dimension must be 75 or 150, got {}.".format(d))
        return x.reshape(n, t, d // 75, 25, 3).transpose(0, 1, 3, 2, 4)
    if x.ndim == 4:
        # (N,T,V,C), (N,C,T,V), or (N,T,C,V)
        if x.shape[-1] == 3:
            return x[:, :, :, None, :]
        if x.shape[1] == 3:
            return x.transpose(0, 2, 3, 1)[:, :, :, None, :]
        if x.shape[2] == 3:
            return x.transpose(0, 1, 3, 2)[:, :, :, None, :]
    if x.ndim == 5:
        # find coordinate axis; the remaining axes are assumed T,V,M.
        coordinate_axes = [axis for axis, size in enumerate(x.shape[1:], start=1) if size == 3]
        if not coordinate_axes:
            raise ValueError("Cannot find a coordinate axis of size 3 in {}.".format(x.shape))
        c_axis = coordinate_axes[-1]
        # Standard (N,C,T,V,M)
        if c_axis == 1:
            return x.transpose(0, 2, 3, 4, 1)
        # Standard (N,T,V,M,C)
        if c_axis == 4:
            return x
        # Common (N,T,V,C,M)
        if c_axis == 3:
            return x.transpose(0, 1, 2, 4, 3)
    raise ValueError("Unsupported skeleton array shape {}.".format(x.shape))


def speed_proxy(x, scale_normalise=True):
    """Mean joint displacement over consecutive valid frames, per sample."""
    n, t, v, m, c = x.shape
    if t < 2 or c != 3:
        raise ValueError("Expected at least two 3-D frames, got {}.".format(x.shape))
    # A body is active only when it has non-zero coordinates in the frame.
    active = np.any(np.abs(x) > 1e-6, axis=(2, 4))  # N,T,M
    delta = np.linalg.norm(x[:, 1:] - x[:, :-1], axis=-1)  # N,T-1,V,M
    transition = active[:, 1:] & active[:, :-1]
    weight = transition[:, :, None, :]
    total = (delta * weight).sum(axis=(1, 2, 3), dtype=np.float64)
    count = transition.sum(axis=(1, 2), dtype=np.float64) * v
    speed = total / np.maximum(count, 1.0)

    if scale_normalise:
        # Root-relative median body extent removes subject/camera scale.  NTU
        # joint 1 is index 0 in the official ordering.
        root = x[:, :, :1, :, :]
        extent = np.linalg.norm(x - root, axis=-1)  # N,T,V,M
        extent = np.where(active[:, :, None, :], extent, np.nan)
        scale = np.nanmedian(extent.reshape(n, -1), axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        speed = speed / scale
    return speed.astype(np.float64), count.astype(np.int64)


def eta_squared(labels, values):
    _, inverse = np.unique(labels, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=values)
    total = ((values - values.mean()) ** 2).sum()
    if total <= 1e-12:
        return 0.0
    between = (sums * sums / count).sum() - values.sum() ** 2 / len(values)
    return float(max(0.0, between / total))


def permutation_pvalue(labels, values, observed, permutations, rng):
    if permutations <= 0:
        return None
    ge = sum(eta_squared(labels, rng.permutation(values)) >= observed for _ in range(permutations))
    return float((ge + 1) / (permutations + 1))


def main():
    args = parse_args()
    if args.data_npy:
        if not args.label_npy:
            raise ValueError("--label-npy is required with --data-npy.")
        x = np.load(args.data_npy, mmap_mode="r")
        labels = labels_from_array(np.load(args.label_npy, mmap_mode="r"))
        source_info = {"data_npy": os.path.abspath(args.data_npy), "label_npy": os.path.abspath(args.label_npy)}
    else:
        with np.load(args.npz, allow_pickle=False) as data:
            if args.data_key not in data or args.label_key not in data:
                raise KeyError("Archive keys are {}; requested {}, {}.".format(data.files, args.data_key, args.label_key))
            x, labels = data[args.data_key], labels_from_array(data[args.label_key])
        source_info = {"npz": os.path.abspath(args.npz), "data_key": args.data_key, "label_key": args.label_key}
    if len(x) != len(labels):
        raise ValueError("Data/label length mismatch: {} vs {}.".format(len(x), len(labels)))
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    speed, transitions = np.empty(len(x), dtype=np.float64), np.empty(len(x), dtype=np.int64)
    for start in range(0, len(x), args.batch_size):
        stop = min(start + args.batch_size, len(x))
        batch_speed, batch_transitions = speed_proxy(
            to_ntvmc(x[start:stop]), scale_normalise=not args.no_scale_normalise)
        speed[start:stop], transitions[start:stop] = batch_speed, batch_transitions
        if start == 0 or stop == len(x) or stop % 5000 < args.batch_size:
            print("speed: {}/{}".format(stop, len(x)), flush=True)
    keep = np.isfinite(speed) & (speed > 0) & (transitions > 0)
    labels, speed, transitions = labels[keep], speed[keep], transitions[keep]
    log_speed = np.log(speed)
    eta = eta_squared(labels, log_speed)
    pvalue = permutation_pvalue(labels, log_speed, eta, args.permutations, np.random.default_rng(args.seed))
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        **source_info,
        "n_samples": int(len(labels)), "n_classes": int(len(np.unique(labels))),
        "speed_proxy": "log(mean consecutive valid-frame joint displacement / median root-relative body extent)",
        "eta_squared": eta, "permutation_p": pvalue, "permutations": args.permutations,
    }
    with open(os.path.join(args.output_dir, "speed_class_association.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "class_speed_statistics.csv"), "w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["class_id", "n", "mean_speed", "median_speed", "mean_log_speed", "q25_speed", "q75_speed"])
        for cls in np.unique(labels):
            s = speed[labels == cls]
            writer.writerow([int(cls), len(s), float(s.mean()), float(np.median(s)), float(np.log(s).mean()), float(np.quantile(s, .25)), float(np.quantile(s, .75))])
    print(json.dumps(summary, indent=2))
    print("wrote -> {}".format(args.output_dir))


if __name__ == "__main__":
    main()
