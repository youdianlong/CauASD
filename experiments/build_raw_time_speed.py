#!/usr/bin/env python3
"""Build a nominal-time-aware speed proxy from raw NTU skeleton files.

This script is an analysis-only utility.  It does not modify the CauASD
training target and does not resize a sequence to 50 frames.

The preprocessing follows the relevant NTU/Neuron data-processing steps:
  1. read raw NTU ``.skeleton`` files while retaining original frame indices;
  2. remove very short and highly abnormal body tracks;
  3. retain at most the two highest-motion bodies;
  4. translate all selected bodies by the first-frame joint-2 origin;
  5. compute trajectory length on the remaining variable-length sequence.

The reported ``speed`` is

    mean-joint trajectory length / original duration / body scale,

where original duration is ``(last_frame-first_frame)/fps``.  NTU skeleton
files do not contain per-sample timestamps, so the default 30 fps is a
nominal frame rate and must be described as a nominal-time-aware proxy.

Example:
  python3 build_raw_time_speed.py \
    --raw-dir /path/to/nturgb+d_skeletons \
    --label-pkl /path/to/train_label.pkl \
    --label-pkl /path/to/val_label.pkl \
    --output analysis/ntu60_raw_time_speed.npz

The label pickle files must contain the usual ``(sample_names, labels)``
pair used by the AimCLR/SMIE preprocessing.
"""

import argparse
import concurrent.futures
import json
import os
import pickle

import numpy as np


NUM_JOINTS = 25
MAX_SELECTED_BODIES = 2
TRANSLATION_JOINT = 1       # NTU joint-2, matching Neuron seq_translation.py
BODY_SCALE_ROOT = 0         # NTU joint-1, matching the current Table V/XI proxy

# Thresholds used by Neuron's NTU denoising code.
NOISE_LENGTH_THRESHOLD = 11
NOISE_SPREAD_THRESHOLD = 0.69754
SPREAD_RATIO_THRESHOLD = 0.8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", action="append", required=True,
                        help="Directory containing raw NTU .skeleton files. Pass once per directory.")
    parser.add_argument(
        "--label-pkl", action="append", required=True,
        help="AimCLR/SMIE pickle containing (sample_names, labels). Pass once per split.",
    )
    parser.add_argument("--output", required=True,
                        help="Output .npz file.")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Nominal NTU frame rate; default: 30.0.")
    parser.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 1) * 2),
                        help="Number of threaded raw-file readers.")
    return parser.parse_args()


def load_samples(paths):
    names, labels = [], []
    seen = set()
    for path in paths:
        with open(path, "rb") as handle:
            part_names, part_labels = pickle.load(handle)
        if len(part_names) != len(part_labels):
            raise ValueError("Mismatched names and labels in {}".format(path))
        for name, label in zip(part_names, part_labels):
            name = str(name)
            if name in seen:
                raise ValueError("Duplicate sample name across label files: {}".format(name))
            seen.add(name)
            names.append(name)
            labels.append(int(label))
    if not names:
        raise ValueError("No samples found in label pickle files.")
    return names, np.asarray(labels, dtype=np.int64)


def skeleton_path(raw_dirs, sample_name):
    filename = sample_name if sample_name.endswith(".skeleton") else sample_name + ".skeleton"
    candidates = [os.path.join(raw_dir, filename) for raw_dir in raw_dirs]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def read_raw_skeleton(path):
    """Read one raw NTU file as ``[(frame_index, {body_id: joints})]``."""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        try:
            frame_count = int(handle.readline().strip())
        except ValueError as exc:
            raise ValueError("Invalid NTU skeleton header: {}".format(path)) from exc

        frames = []
        for frame_index in range(frame_count):
            line = handle.readline()
            if not line:
                raise ValueError("Unexpected end of file in {}".format(path))
            try:
                body_count = int(line.strip())
            except ValueError as exc:
                raise ValueError("Invalid body count at frame {} in {}".format(frame_index, path)) from exc

            bodies = {}
            for _ in range(body_count):
                metadata = handle.readline().split()
                if not metadata:
                    raise ValueError("Missing body metadata in {}".format(path))
                body_id = metadata[0]
                try:
                    joint_count = int(handle.readline().strip())
                except ValueError as exc:
                    raise ValueError("Invalid joint count in {}".format(path)) from exc

                joints = np.zeros((NUM_JOINTS, 3), dtype=np.float32)
                for joint_index in range(joint_count):
                    values = handle.readline().split()
                    if len(values) < 3:
                        raise ValueError("Invalid joint row in {}".format(path))
                    if joint_index < NUM_JOINTS:
                        joints[joint_index] = np.asarray(values[:3], dtype=np.float32)
                if joint_count >= NUM_JOINTS and np.isfinite(joints).all():
                    bodies[body_id] = joints
            frames.append((frame_index, bodies))
    return frames


def body_tracks(frames):
    tracks = {}
    for frame_index, bodies in frames:
        for body_id, joints in bodies.items():
            tracks.setdefault(body_id, []).append((frame_index, joints))
    return tracks


def spread_noise_ratio(track):
    """Return the Neuron-style fraction of frames failing the X/Y spread test."""
    if not track:
        return 1.0
    noisy = 0
    for _frame_index, joints in track:
        x_span = float(joints[:, 0].max() - joints[:, 0].min())
        y_span = float(joints[:, 1].max() - joints[:, 1].min())
        if x_span > SPREAD_RATIO_THRESHOLD * y_span:
            noisy += 1
    return noisy / float(len(track))


def motion_amount(track):
    points = np.stack([joints for _frame_index, joints in track], axis=0)
    return float(np.var(points, axis=0).sum())


def select_neuron_bodies(tracks):
    """Apply Neuron's short-track/spread filtering and motion ranking."""
    candidates = []
    for body_id, track in tracks.items():
        # Neuron removes tracks whose length is <= 11.
        if len(track) <= NOISE_LENGTH_THRESHOLD:
            continue
        if spread_noise_ratio(track) >= NOISE_SPREAD_THRESHOLD:
            continue
        candidates.append((body_id, track, motion_amount(track)))

    # Neuron ranks actors by motion and retains the main actors for two-body
    # actions.  Keeping at most two also matches the model tensor convention.
    candidates.sort(key=lambda item: item[2], reverse=True)
    return {body_id: track for body_id, track, _motion in candidates[:MAX_SELECTED_BODIES]}


def translate_tracks(selected):
    """Apply Neuron's sequence-level joint-2 translation."""
    if not selected:
        return selected
    main_id = max(selected, key=lambda body_id: motion_amount(selected[body_id]))
    main_track = selected[main_id]
    origin = main_track[0][1][TRANSLATION_JOINT].copy()
    translated = {}
    for body_id, track in selected.items():
        translated[body_id] = [(frame_index, joints - origin) for frame_index, joints in track]
    return translated


def estimate_body_scale(selected):
    """Median root-relative body extent, matching the Table V/XI analysis."""
    extents = []
    for track in selected.values():
        for _frame_index, joints in track:
            root = joints[BODY_SCALE_ROOT]
            valid = np.isfinite(joints).all(axis=1) & (np.abs(joints).sum(axis=1) > 1e-8)
            if not valid.any():
                continue
            values = np.linalg.norm(joints - root[None, :], axis=1)
            values = values[valid]
            values = values[values > 1e-8]
            if values.size:
                extents.extend(values.tolist())
    if not extents:
        return 1.0
    scale = float(np.median(np.asarray(extents, dtype=np.float64)))
    return scale if np.isfinite(scale) and scale > 1e-8 else 1.0


def compute_speed(selected, fps):
    """Return speed_time and diagnostic values for one selected sequence."""
    all_frames = [frame_index for track in selected.values() for frame_index, _ in track]
    first_frame = min(all_frames)
    last_frame = max(all_frames)
    duration = (last_frame - first_frame) / float(fps)
    if duration <= 0:
        raise ValueError("Selected sequence has no positive duration.")

    # Lookup by original frame index.  Only adjacent original frames are used
    # for trajectory increments; missing raw frames are not silently treated as
    # consecutive frames.
    lookup = {
        body_id: {frame_index: joints for frame_index, joints in track}
        for body_id, track in selected.items()
    }
    path_length = 0.0
    observed_intervals = 0
    for frame_index in range(first_frame, last_frame):
        step_values = []
        for body_id, frames_for_body in lookup.items():
            if frame_index not in frames_for_body or frame_index + 1 not in frames_for_body:
                continue
            previous = frames_for_body[frame_index]
            current = frames_for_body[frame_index + 1]
            valid = np.isfinite(previous).all(axis=1) & np.isfinite(current).all(axis=1)
            if valid.any():
                step_values.append(float(np.linalg.norm(current[valid] - previous[valid], axis=1).mean()))
        if step_values:
            # Average active bodies so a two-person sequence is not given an
            # artificial 2x speed merely because it has two actors.
            path_length += float(np.mean(step_values))
            observed_intervals += 1

    if observed_intervals == 0:
        raise ValueError("Selected sequence has no adjacent valid transition.")

    scale = estimate_body_scale(selected)
    speed_time = path_length / duration / scale
    speed_raw_frame = path_length / observed_intervals / scale
    return {
        "speed_time": float(speed_time),
        "speed_raw_frame": float(speed_raw_frame),
        "duration_sec": float(duration),
        "body_scale": float(scale),
        "raw_frame_count": int(last_frame - first_frame + 1),
        "first_valid_frame": int(first_frame),
        "last_valid_frame": int(last_frame),
        "observed_intervals": int(observed_intervals),
        "selected_body_count": int(len(selected)),
    }


def process_one(path, fps):
    frames = read_raw_skeleton(path)
    tracks = body_tracks(frames)
    selected = select_neuron_bodies(tracks)
    if not selected:
        # A small number of valid NTU files can be rejected by Neuron's
        # strict spread filter.  For this auxiliary raw-time measurement,
        # fall back to the longest available body track rather than aborting
        # the entire dataset.  The fallback still uses the same translation,
        # body-scale normalization, and trajectory-speed calculation.
        fallback = [
            (body_id, track) for body_id, track in tracks.items()
            if len(track) > 1
        ]
        fallback.sort(
            key=lambda item: (len(item[1]), motion_amount(item[1])),
            reverse=True,
        )
        selected = dict(fallback[:MAX_SELECTED_BODIES])
    if not selected:
        raise ValueError("No usable body track: {}".format(path))
    selected = translate_tracks(selected)
    return compute_speed(selected, fps)


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.workers < 1:
        raise ValueError("--workers must be positive.")
    for raw_dir in args.raw_dir:
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(raw_dir)

    names, labels = load_samples(args.label_pkl)
    paths = [skeleton_path(args.raw_dir, name) for name in names]
    missing = next((path for path in paths if not os.path.isfile(path)), None)
    if missing:
        raise FileNotFoundError("Raw skeleton not found: {}".format(missing))

    results = [None] * len(names)
    if args.workers == 1:
        iterator = ((index, process_one(path, args.fps)) for index, path in enumerate(paths))
        executor = None
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        futures = {executor.submit(process_one, path, args.fps): index for index, path in enumerate(paths)}
        iterator = ((index, future.result()) for future, index in futures.items())

    try:
        for done, (index, result) in enumerate(iterator, start=1):
            results[index] = result
            if done == 1 or done % 500 == 0 or done == len(names):
                print("raw-time speed: {}/{}".format(done, len(names)), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    speed_time = np.asarray([item["speed_time"] for item in results], dtype=np.float32)
    speed_raw_frame = np.asarray([item["speed_raw_frame"] for item in results], dtype=np.float32)
    duration_sec = np.asarray([item["duration_sec"] for item in results], dtype=np.float32)
    body_scale = np.asarray([item["body_scale"] for item in results], dtype=np.float32)
    raw_frame_count = np.asarray([item["raw_frame_count"] for item in results], dtype=np.int32)
    first_valid_frame = np.asarray([item["first_valid_frame"] for item in results], dtype=np.int32)
    last_valid_frame = np.asarray([item["last_valid_frame"] for item in results], dtype=np.int32)
    observed_intervals = np.asarray([item["observed_intervals"] for item in results], dtype=np.int32)
    selected_body_count = np.asarray([item["selected_body_count"] for item in results], dtype=np.int8)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.output,
        names=np.asarray(names),
        labels=labels,
        # ``speed`` is provided for direct compatibility with linear_speed_probe.py.
        speed=speed_time,
        speed_time=speed_time,
        speed_raw_frame=speed_raw_frame,
        duration_sec=duration_sec,
        body_scale=body_scale,
        raw_frame_count=raw_frame_count,
        first_valid_frame=first_valid_frame,
        last_valid_frame=last_valid_frame,
        observed_intervals=observed_intervals,
        selected_body_count=selected_body_count,
        fps=np.asarray(args.fps, dtype=np.float32),
        metric=np.asarray("mean-joint raw trajectory length / nominal duration / median root-relative body extent"),
    )

    summary = {
        "n_samples": int(len(names)),
        "fps": float(args.fps),
        "fps_interpretation": "nominal frame rate; raw NTU files do not provide per-sample timestamps",
        "preprocessing": "Neuron-style body filtering, motion ranking, joint-2 sequence translation; no 50-frame resize",
        "metric": "mean-joint trajectory length / original nominal duration / median joint-1-relative body extent",
        "speed_mean": float(speed_time.mean()),
        "speed_std": float(speed_time.std()),
        "duration_mean_sec": float(duration_sec.mean()),
        "selected_body_count_mean": float(selected_body_count.mean()),
    }
    with open(args.output + ".json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print("saved raw-time speed labels -> {}".format(args.output))


if __name__ == "__main__":
    main()
