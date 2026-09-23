#!/usr/bin/env python3
"""Generate a small NTU-120 label-preservation audit package.

This is an audit-only script.  It does not load a model or a checkpoint.
It reads the already preprocessed 50-frame unseen test arrays from
split_4, split_5, and split_6, selects one sample per unseen class in each
split, and creates six legacy-TCI variants per selected sample.

The generated MP4 contains two panels (Original / Transformed) and displays
the action name.  The speed factor is omitted from the video and recorded
only in pair_key.csv.

Example (server):

  python3 generate_ntu120_label_identity_audit.py \
    --xsub-data /path/to/ntu120_frame50/xsub/val_position.npy \
    --xsub-label-pkl /path/to/ntu120_frame50/xsub/val_label.pkl \
    --output /path/to/label_identity_audit_ntu120 \
    --class-names /path/to/ntu120_des.txt

The script supports either complete X-Sub validation files:

  <xsub-dir>/val_position.npy
  <xsub-dir>/val_label.pkl

or split-specific files:

  <split-root>/split_4/unseen_data.npy
  <split-root>/split_4/unseen_label.npy
  <split-root>/split_5/unseen_data.npy
  <split-root>/split_5/unseen_label.npy
  <split-root>/split_6/unseen_data.npy
  <split-root>/split_6/unseen_label.npy

``unseen_sample_names.npy`` is optional.  If it is absent, the script writes
the split and array index as the sample identifier.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


FACTORS = (0.5, 0.75, 1.25, 1.5, 1.75, 2.0)
SPLITS = (4, 5, 6)
UNSEEN_CLASSES = {
    4: (3, 18, 26, 38, 41, 60, 87, 99, 102, 110),
    5: (5, 12, 14, 15, 17, 42, 67, 82, 100, 119),
    6: (6, 20, 27, 33, 42, 55, 71, 97, 104, 118),
}

# NTU RGB+D 25-joint graph, zero-based.
EDGES = (
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (22, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (24, 23),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split-root",
                        help="Directory containing split_4, split_5, and split_6.")
    source.add_argument("--xsub-data",
                        help="Complete NTU-120 X-Sub val_position.npy file.")
    parser.add_argument("--xsub-label-pkl", default=None,
                        help="val_label.pkl corresponding to --xsub-data.")
    parser.add_argument("--output", required=True,
                        help="Directory in which the audit package is written.")
    parser.add_argument("--class-names", default=None,
                        help="Optional NTU-120 description text file.")
    parser.add_argument("--per-class", type=int, default=1,
                        help="Samples selected per unseen class per split; default: 1.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--factors", default=",".join(str(x) for x in FACTORS))
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--no-render", action="store_true",
                        help="Only write metadata; do not render MP4 files.")
    return parser.parse_args()


def clean_name(value: object) -> str:
    value = str(value)
    return value[:-9] if value.endswith(".skeleton") else value


def load_class_names(path: str | None) -> Dict[int, str]:
    if not path:
        return {}
    result: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^\s*["“](.*?)["”]', line)
            result[index] = match.group(1) if match else "class_{}".format(index)
    return result


def load_split(split_root: Path, split_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    directory = split_root / "split_{}".format(split_id)
    data_path = directory / "unseen_data.npy"
    label_path = directory / "unseen_label.npy"
    if not data_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(
            "Missing input for split {}: {} and {}".format(split_id, data_path, label_path)
        )
    data = np.load(data_path, mmap_mode="r")
    labels = np.asarray(np.load(label_path)).reshape(-1).astype(np.int64)
    if data.ndim != 5 or tuple(data.shape[1:]) != (3, 50, 25, 2):
        raise ValueError("split {} has shape {}, expected (N,3,50,25,2)".format(split_id, data.shape))
    if len(data) != len(labels):
        raise ValueError("split {} data/label length mismatch: {} vs {}".format(split_id, len(data), len(labels)))

    names_path = directory / "unseen_sample_names.npy"
    if names_path.is_file():
        names = np.asarray(np.load(names_path, allow_pickle=True)).reshape(-1)
        names = np.asarray([clean_name(x) for x in names], dtype=str)
        if len(names) != len(data):
            raise ValueError("split {} data/name length mismatch: {} vs {}".format(split_id, len(data), len(names)))
    else:
        names = np.asarray(["split_{}_index_{}".format(split_id, i) for i in range(len(data))], dtype=str)
    return data, labels, names


def load_xsub(data_path: str, label_pkl: str | None) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load the complete X-Sub validation set and expose each unseen split."""
    if not label_pkl:
        raise ValueError("--xsub-label-pkl is required when --xsub-data is used.")
    data = np.load(data_path, mmap_mode="r")
    if data.ndim != 5 or tuple(data.shape[1:]) != (3, 50, 25, 2):
        raise ValueError("X-Sub data has shape {}, expected (N,3,50,25,2)".format(data.shape))

    with open(label_pkl, "rb") as handle:
        names_raw, labels_raw = pickle.load(handle)
    names = np.asarray([clean_name(x) for x in names_raw], dtype=str)
    labels = np.asarray(labels_raw).reshape(-1).astype(np.int64)
    if len(data) != len(labels) or len(data) != len(names):
        raise ValueError(
            "X-Sub data/name/label length mismatch: {} / {} / {}".format(
                len(data), len(names), len(labels)
            )
        )

    result: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split, unseen in UNSEEN_CLASSES.items():
        mask = np.isin(labels, unseen)
        result[split] = (data[mask], labels[mask], names[mask])
        if len(result[split][0]) == 0:
            raise ValueError("No X-Sub validation samples found for split {}.".format(split))
    return result


def select_one_per_class(labels: np.ndarray, per_class: int, rng: np.random.Generator) -> List[int]:
    selected: List[int] = []
    for label in sorted(np.unique(labels).tolist()):
        candidates = np.flatnonzero(labels == label).tolist()
        if len(candidates) < per_class:
            raise ValueError("Class {} has only {} samples; cannot select {}.".format(label, len(candidates), per_class))
        rng.shuffle(candidates)
        selected.extend(int(x) for x in candidates[:per_class])
    return selected


def legacy_warp(sequence: np.ndarray, factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply t' = clip(t / factor, 0, T-1) with linear interpolation."""
    if factor <= 0:
        raise ValueError("speed factors must be positive")
    frames = int(sequence.shape[1])
    output_t = np.arange(frames, dtype=np.float32)
    raw_t = output_t / float(factor)
    source_t = np.clip(raw_t, 0.0, float(frames - 1))
    grid = np.arange(frames, dtype=np.float32)
    warped = np.empty_like(sequence, dtype=np.float32)
    for channel in range(sequence.shape[0]):
        for joint in range(sequence.shape[2]):
            for body in range(sequence.shape[3]):
                warped[channel, :, joint, body] = np.interp(
                    source_t, grid, sequence[channel, :, joint, body].astype(np.float32)
                )
    return warped, source_t


def boundary_statistics(factor: float, frames: int) -> Dict[str, float]:
    output_t = np.arange(frames, dtype=np.float64)
    raw_t = output_t / float(factor)
    clipped = np.clip(raw_t, 0.0, float(frames - 1))
    outside = (raw_t < 0.0) | (raw_t > float(frames - 1))
    endpoint = (raw_t > float(frames - 1)) & (clipped == float(frames - 1))
    coverage = (clipped.max() - clipped.min()) / float(max(frames - 1, 1))
    return {
        "speed_factor": float(factor),
        "input_frames": int(frames),
        "trajectory_coverage_percent": float(100.0 * coverage),
        "out_of_range_percent": float(100.0 * outside.mean()),
        "endpoint_repeat_percent": float(100.0 * endpoint.mean()),
        "padding_percent": 0.0,
    }


def active_bodies(sequence: np.ndarray) -> List[int]:
    energy = np.sum(np.abs(sequence), axis=(0, 1, 2))
    active = [int(i) for i, value in enumerate(energy) if value > 1e-8]
    return active[:2] if active else [int(np.argmax(energy))]


def render_mp4(
    original: np.ndarray,
    transformed: np.ndarray,
    path: Path,
    action_name: str,
    fps: float,
) -> None:
    width, height = 900, 450
    panel_width = width // 2
    frame_count = int(original.shape[1])
    bodies = sorted(set(active_bodies(original) + active_bodies(transformed)))[:2]
    combined = np.concatenate([original, transformed], axis=1)
    x_values = combined[0]
    y_values = combined[1]
    finite_x = x_values[np.isfinite(x_values)]
    finite_y = y_values[np.isfinite(y_values)]
    x_min, x_max = float(finite_x.min()), float(finite_x.max())
    y_min, y_max = float(finite_y.min()), float(finite_y.max())
    x_pad = max((x_max - x_min) * 0.08, 1e-3)
    y_pad = max((y_max - y_min) * 0.08, 1e-3)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def project(x: float, y: float, left: int, right: int) -> Tuple[int, int]:
        px = left + int((x - x_min) / max(x_max - x_min, 1e-6) * (right - left))
        py = height - 18 - int((y - y_min) / max(y_max - y_min, 1e-6) * (height - 73))
        return px, py

    path.parent.mkdir(parents=True, exist_ok=True)
    # H.264 is generally more compatible with desktop and browser players
    # than the mp4v streams produced by some OpenCV builds.
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"avc1"), float(fps), (width, height)
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
        )
    if not writer.isOpened():
        raise RuntimeError("Cannot open MP4 writer: {}".format(path))
    try:
        for frame_index in range(frame_count):
            canvas = np.full((height, width, 3), 255, dtype=np.uint8)
            cv2.line(canvas, (panel_width, 0), (panel_width, height), (185, 185, 185), 2)
            cv2.putText(canvas, "Original", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Transformed", (panel_width + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(canvas, "frame {}".format(frame_index + 1), (width - 125, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            label_text = "Action: {}".format(action_name)
            if len(label_text) > 48:
                split_at = label_text.rfind(" ", 0, 48)
                split_at = split_at if split_at > 0 else 48
                label_lines = (label_text[:split_at], label_text[split_at + 1:])
            else:
                label_lines = (label_text,)
            for line_index, line in enumerate(label_lines):
                cv2.putText(canvas, line, (20, 45 + 17 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1, cv2.LINE_AA)

            for offset, sequence in ((0, original), (panel_width, transformed)):
                left, right = offset + 24, offset + panel_width - 24
                for body in bodies:
                    if body >= sequence.shape[3]:
                        continue
                    points: List[Tuple[int, int] | None] = []
                    for joint in range(sequence.shape[2]):
                        x = float(sequence[0, frame_index, joint, body])
                        y = float(sequence[1, frame_index, joint, body])
                        points.append(project(x, y, left, right) if math.isfinite(x) and math.isfinite(y) else None)
                    for a, b in EDGES:
                        if points[a] is not None and points[b] is not None:
                            cv2.line(canvas, points[a], points[b], (170, 90, 40), 2, cv2.LINE_AA)
                    for point in points:
                        if point is not None:
                            cv2.circle(canvas, point, 4, (45, 45, 190), -1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.per_class < 1:
        raise ValueError("--per-class must be at least 1")
    factors = tuple(float(x.strip()) for x in args.factors.split(",") if x.strip())
    if not factors:
        raise ValueError("At least one speed factor is required.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(args.class_names)
    rng = np.random.default_rng(args.seed)

    if args.xsub_data:
        loaded = load_xsub(args.xsub_data, args.xsub_label_pkl)
        source_description = "source: complete NTU-120 X-Sub validation files (local paths omitted)"
    else:
        split_root = Path(args.split_root)
        loaded = {split: load_split(split_root, split) for split in SPLITS}
        source_description = "source: split-specific NTU-120 files (local paths omitted)"
    selected: List[Tuple[int, int, int, str]] = []
    for split in SPLITS:
        data, labels, names = loaded[split]
        for index in select_one_per_class(labels, args.per_class, rng):
            selected.append((split, index, int(labels[index]), str(names[index])))
    selected.sort(key=lambda item: (item[0], item[2], item[1]))

    pair_rows: List[Dict[str, object]] = []
    key_rows: List[Dict[str, object]] = []
    for sample_number, (split, index, label, sample_name) in enumerate(selected, start=1):
        original = np.asarray(loaded[split][0][index], dtype=np.float32)
        action_name = class_names.get(label, "class_{}".format(label))
        for factor in factors:
            pair_id = "pair_{:04d}".format(len(pair_rows) + 1)
            transformed, _ = legacy_warp(original, factor)
            animation_rel = Path("pairs") / (pair_id + ".mp4")
            if not args.no_render:
                render_mp4(original, transformed, output / animation_rel, action_name, args.fps)
            pair_rows.append({
                "pair_id": pair_id,
                "animation": str(animation_rel),
                "annotator_1": "",
                "annotator_2": "",
                "notes": "",
            })
            key_rows.append({
                "pair_id": pair_id,
                "sample_number": sample_number,
                "dataset": "NTU-120",
                "split": split,
                "index": index,
                "class_label": label,
                "class_name": action_name,
                "sample_name": sample_name,
                "speed_factor": factor,
                "animation": str(animation_rel),
            })

    write_csv(output / "annotation_sheet.csv", pair_rows[0].keys(), pair_rows)
    write_csv(output / "pair_key.csv", key_rows[0].keys(), key_rows)
    boundary_rows = [boundary_statistics(factor, 50) for factor in factors]
    write_csv(output / "boundary_statistics.csv", boundary_rows[0].keys(), boundary_rows)

    split_counts = {str(split): sum(item[0] == split for item in selected) for split in SPLITS}
    summary_lines = [
        "NTU-120 action-identity audit package",
        "====================================",
        "",
        "Selected one sample per unseen class in each of split_4, split_5, and split_6.",
        "Each MP4 contains Original on the left and Transformed on the right.",
        "Action names are shown in the MP4; speed factors are omitted from the MP4 and recorded only in pair_key.csv.",
        "Annotators should fill annotation_sheet.csv with Yes, No, or Uncertain.",
        "Retention = Yes / (Yes + No + Uncertain).",
        "",
        "dataset: NTU-120",
        source_description,
        "seed: {}".format(args.seed),
        "per_class: {}".format(args.per_class),
        "selected_original_samples: {}".format(len(selected)),
        "pairs: {}".format(len(pair_rows)),
        "factors: {}".format(", ".join(str(x) for x in factors)),
        "split_counts: {}".format(split_counts),
        "annotation_values: Yes, No, Uncertain",
        "rendered: {}".format(not args.no_render),
    ]
    (output / "README.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("Generated {} original samples and {} pairs in {}".format(len(selected), len(pair_rows), output))
    print("Annotation sheet: {}".format(output / "annotation_sheet.csv"))
    print("Pair key: {}".format(output / "pair_key.csv"))


if __name__ == "__main__":
    main()
