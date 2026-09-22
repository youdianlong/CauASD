#!/usr/bin/env python3
"""Generate a blinded action-identity audit package for the legacy TCI.

The script uses already-preprocessed fixed 50-frame skeletons.  It does not
load a model or a checkpoint.  It samples a fixed, class-balanced set from
the NTU-60 X-Sub validation set, applies the legacy mapping

    t' = clip(t / speed_factor, 0, T - 1),

and writes side-by-side original/transformed animations plus annotation
forms.  The speed factor is hidden from annotators, while the action category
is shown because skeleton-only videos are otherwise difficult to judge.
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

import numpy as np


DEFAULT_SPLITS = {
    1: (4, 19, 31, 47, 51),
    2: (12, 29, 32, 44, 59),
    3: (7, 20, 28, 39, 58),
}
DEFAULT_FACTORS = (0.5, 0.75, 1.25, 1.5, 1.75, 2.0)

# NTU RGB+D 25-joint graph, converted to zero-based joint indices.
NTU25_EDGES = (
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (22, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (24, 23),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        help="Fixed 50-frame X-Sub validation position file.",
    )
    parser.add_argument(
        "--label-pkl",
        required=True,
        help="Validation sample-name/label pickle.",
    )
    parser.add_argument(
        "--descriptions",
        default=None,
        help="Optional NTU-60 class-description file used to add action names to pair_key.csv.",
    )
    parser.add_argument(
        "--output",
        default="analysis/label_identity_audit_ntu60",
        help="Output directory for the audit package.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Total original samples selected across the three unseen splits.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--factors",
        default=",".join(str(x) for x in DEFAULT_FACTORS),
        help="Comma-separated legacy speed factors.",
    )
    parser.add_argument(
        "--render-gifs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render side-by-side GIFs. Use --no-render-gifs for a quick dry run.",
    )
    parser.add_argument(
        "--max-gif-frames",
        type=int,
        default=50,
        help="Maximum frames per GIF; default keeps all 50 frames.",
    )
    return parser.parse_args()


def clean_name(value: object) -> str:
    name = str(value)
    return name[:-9] if name.endswith(".skeleton") else name


def load_labels(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as handle:
        names, labels = pickle.load(handle)
    names_arr = np.asarray([clean_name(x) for x in names], dtype=str)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(names_arr) != len(labels_arr):
        raise ValueError("Sample names and labels have different lengths.")
    return names_arr, labels_arr


def load_class_names(path: str) -> Dict[int, str]:
    if not path or not os.path.isfile(path):
        return {}
    result: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^["“](.*?)["”]', line)
            result[index] = match.group(1) if match else "class_{}".format(index)
    return result


def balanced_round_robin(
    indices: Sequence[int],
    labels: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> List[int]:
    """Select n indices with approximately equal representation per class."""
    by_class: Dict[int, List[int]] = {}
    for index in indices:
        by_class.setdefault(int(labels[index]), []).append(int(index))
    classes = sorted(by_class)
    for values in by_class.values():
        rng.shuffle(values)

    selected: List[int] = []
    cursor = 0
    while len(selected) < n:
        progressed = False
        for class_offset in range(len(classes)):
            class_id = classes[(cursor + class_offset) % len(classes)]
            values = by_class[class_id]
            if values:
                selected.append(values.pop())
                progressed = True
                if len(selected) == n:
                    break
        if not progressed:
            raise ValueError("Not enough candidate samples for class-balanced sampling.")
        cursor += 1
    return selected


def allocate_total(total: int, n_groups: int) -> List[int]:
    base, remainder = divmod(total, n_groups)
    return [base + (i < remainder) for i in range(n_groups)]


def legacy_warp(sequence: np.ndarray, speed_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Warp one (C,T,V,M) sequence using the exact legacy interpolation."""
    if sequence.ndim != 4:
        raise ValueError("Expected sequence shape (C,T,V,M), got {}".format(sequence.shape))
    channels, frames, joints, bodies = sequence.shape
    del channels, joints, bodies
    if frames < 1:
        raise ValueError("Temporal length must be positive.")
    if speed_factor <= 0:
        raise ValueError("Speed factor must be positive.")

    output_t = np.arange(frames, dtype=np.float32)
    raw_source_t = output_t / float(speed_factor)
    source_t = np.clip(raw_source_t, 0.0, float(frames - 1))
    source_grid = np.arange(frames, dtype=np.float32)

    warped = np.empty_like(sequence, dtype=np.float32)
    for c in range(sequence.shape[0]):
        for v in range(sequence.shape[2]):
            for m in range(sequence.shape[3]):
                warped[c, :, v, m] = np.interp(
                    source_t,
                    source_grid,
                    sequence[c, :, v, m].astype(np.float32),
                )
    return warped, source_t


def boundary_statistics(speed_factor: float, frames: int) -> Dict[str, float]:
    output_t = np.arange(frames, dtype=np.float64)
    raw = output_t / float(speed_factor)
    clipped = np.clip(raw, 0.0, float(frames - 1))
    out_of_range = (raw < 0.0) | (raw > float(frames - 1))
    endpoint_repeat = (raw > float(frames - 1)) & (clipped == float(frames - 1))
    coverage = (clipped.max() - clipped.min()) / float(max(frames - 1, 1))
    return {
        "speed_factor": float(speed_factor),
        "frames": int(frames),
        "coverage_percent": float(100.0 * coverage),
        "clipped_percent": float(100.0 * out_of_range.mean()),
        "endpoint_repeat_percent": float(100.0 * endpoint_repeat.mean()),
        "padding_percent": 0.0,
    }


def choose_body_indices(sequence: np.ndarray) -> List[int]:
    # Select bodies by total non-zero energy; retain both when both are present.
    energy = np.sum(np.abs(sequence), axis=(0, 1, 2))
    active = [int(i) for i, value in enumerate(energy) if value > 1e-8]
    return active or [int(np.argmax(energy))]


def make_animation(
    original: np.ndarray,
    warped: np.ndarray,
    output_path: Path,
    action_name: str,
    fps: int = 8,
    size: Tuple[int, int] = (900, 450),
    max_frames: int = 50,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
        use_pillow = True
    except ImportError:
        use_pillow = False

    if not use_pillow:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Animation rendering requires Pillow or OpenCV.") from exc
        make_mp4_with_opencv(original, warped, output_path.with_suffix(".mp4"), action_name=action_name, fps=fps, size=size, max_frames=max_frames)
        return

    width, height = size
    frame_count = min(int(original.shape[1]), int(max_frames))
    bodies = sorted(set(choose_body_indices(original) + choose_body_indices(warped)))
    bodies = bodies[:2]

    combined = np.concatenate([original[:, :frame_count], warped[:, :frame_count]], axis=1)
    x_values = combined[0]
    y_values = combined[1]
    x_min, x_max = float(np.nanmin(x_values)), float(np.nanmax(x_values))
    y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
    x_pad = max((x_max - x_min) * 0.08, 1e-3)
    y_pad = max((y_max - y_min) * 0.08, 1e-3)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    font = ImageFont.load_default()
    frames: List[Image.Image] = []
    panel_width = width // 2

    def project(x: float, y: float, left: int, right: int, top: int, bottom: int) -> Tuple[int, int]:
        px = left + int((x - x_min) / max(x_max - x_min, 1e-6) * (right - left))
        py = bottom - int((y - y_min) / max(y_max - y_min, 1e-6) * (bottom - top))
        return px, py

    for frame_index in range(frame_count):
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.line((panel_width, 0, panel_width, height), fill=(185, 185, 185), width=2)
        draw.text((20, 10), "Original", fill=(20, 20, 20), font=font)
        draw.text((panel_width + 20, 10), "Transformed", fill=(20, 20, 20), font=font)
        draw.text((20, 27), "Action category: {}".format(action_name), fill=(20, 20, 20), font=font)
        draw.text((width - 105, 10), "frame {}".format(frame_index + 1), fill=(20, 20, 20), font=font)

        for panel_offset, sequence in ((0, original), (panel_width, warped)):
            top, bottom = 55, height - 18
            left, right = panel_offset + 24, panel_offset + panel_width - 24
            for body in bodies:
                if body >= sequence.shape[3]:
                    continue
                points = []
                for joint in range(sequence.shape[2]):
                    x = float(sequence[0, frame_index, joint, body])
                    y = float(sequence[1, frame_index, joint, body])
                    if not (math.isfinite(x) and math.isfinite(y)):
                        points.append(None)
                    else:
                        points.append(project(x, y, left, right, top, bottom))
                for a, b in NTU25_EDGES:
                    if a < len(points) and b < len(points) and points[a] and points[b]:
                        draw.line((points[a][0], points[a][1], points[b][0], points[b][1]), fill=(40, 90, 170), width=2)
                for point in points:
                    if point:
                        radius = 3
                        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=(190, 45, 45))
        frames.append(canvas)

    if not frames:
        raise ValueError("No frames were available for GIF rendering.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(round(1000.0 / fps))),
        loop=0,
        optimize=True,
    )


def make_mp4_with_opencv(
    original: np.ndarray,
    warped: np.ndarray,
    output_path: Path,
    action_name: str,
    fps: int = 8,
    size: Tuple[int, int] = (900, 450),
    max_frames: int = 50,
) -> None:
    """Dependency-light fallback renderer for machines without Pillow."""
    import cv2

    width, height = size
    frame_count = min(int(original.shape[1]), int(max_frames))
    bodies = sorted(set(choose_body_indices(original) + choose_body_indices(warped)))[:2]
    combined = np.concatenate([original[:, :frame_count], warped[:, :frame_count]], axis=1)
    x_values = combined[0]
    y_values = combined[1]
    x_min, x_max = float(np.nanmin(x_values)), float(np.nanmax(x_values))
    y_min, y_max = float(np.nanmin(y_values)), float(np.nanmax(y_values))
    x_pad = max((x_max - x_min) * 0.08, 1e-3)
    y_pad = max((y_max - y_min) * 0.08, 1e-3)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    panel_width = width // 2

    def project(x: float, y: float, left: int, right: int, top: int, bottom: int) -> Tuple[int, int]:
        px = left + int((x - x_min) / max(x_max - x_min, 1e-6) * (right - left))
        py = bottom - int((y - y_min) / max(y_max - y_min, 1e-6) * (bottom - top))
        return px, py

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer AVC/H.264 for media-player compatibility.  Some OpenCV builds
    # transparently fall back to mp4v when AVC is unavailable.
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open MP4 writer at {}".format(output_path))

    try:
        for frame_index in range(frame_count):
            canvas = np.full((height, width, 3), 255, dtype=np.uint8)
            cv2.line(canvas, (panel_width, 0), (panel_width, height), (185, 185, 185), 2)
            cv2.putText(canvas, "Original", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Transformed", (panel_width + 20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Action category: {}".format(action_name), (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(canvas, "frame {}".format(frame_index + 1), (width - 130, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

            for panel_offset, sequence in ((0, original), (panel_width, warped)):
                top, bottom = 60, height - 18
                left, right = panel_offset + 24, panel_offset + panel_width - 24
                for body in bodies:
                    if body >= sequence.shape[3]:
                        continue
                    points = []
                    for joint in range(sequence.shape[2]):
                        x = float(sequence[0, frame_index, joint, body])
                        y = float(sequence[1, frame_index, joint, body])
                        points.append(project(x, y, left, right, top, bottom) if math.isfinite(x) and math.isfinite(y) else None)
                    for a, b in NTU25_EDGES:
                        if a < len(points) and b < len(points) and points[a] and points[b]:
                            cv2.line(canvas, points[a], points[b], (170, 90, 40), 2, cv2.LINE_AA)
                    for point in points:
                        if point:
                            cv2.circle(canvas, point, 4, (45, 45, 190), -1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    factors = tuple(float(x.strip()) for x in args.factors.split(",") if x.strip())
    if not factors:
        raise ValueError("At least one speed factor is required.")

    data = np.load(args.data, mmap_mode="r")
    names, labels = load_labels(args.label_pkl)
    class_names = load_class_names(args.descriptions)
    if data.ndim != 5 or data.shape[1:] != (3, 50, 25, 2):
        raise ValueError("Expected data shape (N,3,50,25,2), got {}".format(data.shape))
    if len(data) != len(labels):
        raise ValueError("Data and label lengths differ: {} vs {}".format(len(data), len(labels)))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    allocations = allocate_total(args.samples, len(DEFAULT_SPLITS))
    selected: List[Tuple[int, int, int]] = []
    for (split_id, unseen_classes), count in zip(sorted(DEFAULT_SPLITS.items()), allocations):
        candidate = np.flatnonzero(np.isin(labels, unseen_classes)).tolist()
        chosen = balanced_round_robin(candidate, labels, count, rng)
        selected.extend((split_id, int(index), int(labels[index])) for index in chosen)

    selected.sort(key=lambda item: (item[0], item[2], item[1]))
    selected_rows: List[Dict[str, object]] = []
    key_rows: List[Dict[str, object]] = []
    boundary_rows = [boundary_statistics(factor, int(data.shape[2])) for factor in factors]

    pairs_dir = output / "pairs"
    pair_counter = 0
    pair_arrays: Dict[str, np.ndarray] = {}
    for sample_number, (split_id, index, label) in enumerate(selected, start=1):
        original = np.asarray(data[index], dtype=np.float32)
        sample_name = names[index]
        for factor in factors:
            pair_counter += 1
            pair_id = "pair_{:04d}".format(pair_counter)
            warped, _ = legacy_warp(original, factor)
            pair_arrays[pair_id + "_original"] = original
            pair_arrays[pair_id + "_warped"] = warped
            animation_rel = Path("pairs") / (pair_id + (".gif" if args.render_gifs else ".mp4"))
            action_name = class_names.get(label, "class_{}".format(label))
            if args.render_gifs:
                gif_path = output / Path("pairs") / (pair_id + ".gif")
                make_animation(original, warped, gif_path, action_name=action_name, max_frames=args.max_gif_frames)
                if not gif_path.exists():
                    animation_rel = animation_rel.with_suffix(".mp4")

            selected_rows.append({
                "pair_id": pair_id,
                "animation": str(animation_rel),
                "action_category": action_name,
                "annotator_1": "",
                "annotator_2": "",
                "notes": "",
            })
            key_rows.append({
                "pair_id": pair_id,
                "sample_number": sample_number,
                "dataset": "NTU-60",
                "split": split_id,
                "index": index,
                "class_label": label,
                "class_name": class_names.get(label, "class_{}".format(label)),
                "sample_name": sample_name,
                "speed_factor": factor,
                "animation": str(animation_rel),
            })

    np.savez_compressed(output / "audit_pairs.npz", **pair_arrays)
    write_csv(output / "annotation_sheet.csv", selected_rows[0].keys(), selected_rows)
    write_csv(output / "pair_key.csv", key_rows[0].keys(), key_rows)
    write_csv(output / "boundary_statistics.csv", boundary_rows[0].keys(), boundary_rows)

    summary = {
        "dataset": "NTU-60",
        "protocol": "legacy fixed-window temporal intervention",
        "data": str(Path(args.data).resolve()),
        "label_pkl": str(Path(args.label_pkl).resolve()),
        "shape": list(data.shape),
        "seed": args.seed,
        "selected_original_samples": len(selected),
        "pairs": pair_counter,
        "factors": list(factors),
        "split_counts": {str(split): sum(row["split"] == split for row in key_rows[::len(factors)]) for split in DEFAULT_SPLITS},
        "annotation_values": ["Yes", "No", "Uncertain"],
        "retention_formula": "Yes / (Yes + No + Uncertain)",
        "note": "Retention is intentionally left for independent annotators; no model or checkpoint is used.",
    }
    with (output / "README.txt").open("w", encoding="utf-8") as handle:
        handle.write("NTU-60 action-identity audit package\n")
        handle.write("===================================\n\n")
        handle.write("Open the files under pairs/. The left panel is Original and the right panel is Transformed.\n")
        handle.write("The speed factor is hidden from the annotation sheet; the action category is shown in each video.\n")
        handle.write("Fill annotation_sheet.csv with Yes, No, or Uncertain in annotator_1 and annotator_2.\n")
        handle.write("Use pair_key.csv only after annotation to recover the factor and class.\n")
        handle.write("Retention = Yes / (Yes + No + Uncertain).\n")
        handle.write("\nGeneration summary:\n")
        for key, value in summary.items():
            handle.write("{}: {}\n".format(key, value))

    print("Generated {} original samples and {} blinded pairs in {}".format(len(selected), pair_counter, output))
    print("Annotation sheet: {}".format(output / "annotation_sheet.csv"))
    print("Boundary statistics: {}".format(output / "boundary_statistics.csv"))


if __name__ == "__main__":
    main()
