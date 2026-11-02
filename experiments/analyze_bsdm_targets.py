#!/usr/bin/env python3
"""Diagnose BSDM displacement targets under fixed-length temporal warping.

Run from the project root: python experiments/analyze_bsdm_targets.py --split 1
For a differently named training file, add --training-script TRAINING_FILE.py.
Training-source text is decoded as UTF-8 first, then GB18030 (including GBK).
Only NumPy is required. No checkpoints, text embeddings, or raw-time files
are used. This is a float32 NumPy equivalent of RandomTemporalIntervention
and compute_speed_from_skeleton in main.py, NOT the body-normalized metric
used for natural-speed stratification. Zero slots and repeated endpoint
frames remain in the training target's denominator intentionally.

Outputs: a summary CSV, a per-sample CSV, and reproducibility metadata JSON.
All summary statistics are sample-level within one split; splits are NOT
pooled or averaged. Std is the population standard deviation (ddof=0).
"""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SPLIT_DATASET = {i: "ntu60" if i <= 3 else "ntu120" if i <= 6 else "pku"
                 for i in range(1, 10)}


def positive_factor(value):
    factor = float(value)
    if not np.isfinite(factor) or factor <= 0 or not np.isfinite(np.float32(factor)) or np.float32(factor) == 0:
        raise argparse.ArgumentTypeError("Factors must be finite, positive float32 values.")
    return factor


def project_root():
    # Works both in experiments/ and when copied to the server project root.
    candidates = [Path.cwd()] + list(Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "data" / "zeroshot").is_dir():
            return candidate
    for candidate in candidates:
        if (candidate / "main.py").is_file() and (candidate / "config.py").is_file():
            return candidate
    return Path.cwd()


def validate_batch(data):
    if data.ndim != 5 or data.shape[1] != 3 or min(data.shape[0], data.shape[3], data.shape[4]) < 1 or data.shape[2] < 2:
        raise ValueError("Expected nonempty (N,3,T,J,M) input with T >= 2; got {}".format(data.shape))
    if not np.isfinite(data).all():
        raise ValueError("Non-finite coordinates found; no samples are silently excluded.")


def training_target(data):
    """Same raw displacement target as training, including all person slots."""
    x = np.asarray(data, dtype=np.float32)
    validate_batch(x)
    with np.errstate(over="ignore", invalid="ignore"):
        diff = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        magnitude = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float32))
        target = magnitude.mean(axis=(1, 2, 3), dtype=np.float32)
    if not np.isfinite(target).all():
        raise ValueError("Target overflow: coordinate magnitudes exceed the float32 training calculation.")
    return target


def temporal_warp(data, factor):
    """tau=clip(t/factor,0,T-1), gather/interpolate, preserving output T."""
    x = np.asarray(data, dtype=np.float32)
    validate_batch(x)
    factor32 = np.float32(factor)
    if not np.isfinite(factor32) or factor32 <= 0:
        raise ValueError("Factor must be finite and positive.")
    n, c, t, joints, bodies = x.shape
    with np.errstate(over="ignore"):
        source = np.clip(np.arange(t, dtype=np.float32) / factor32, 0, t - 1)
    left = np.floor(source).astype(np.int64)
    right = np.ceil(source).astype(np.int64)
    weight = (source - left.astype(np.float32))[None, None, :]
    flat = x.transpose(0, 1, 3, 4, 2).reshape(n, -1, t)
    left_values = np.take_along_axis(flat, left[None, None, :], axis=2)
    right_values = np.take_along_axis(flat, right[None, None, :], axis=2)
    warped = (np.float32(1) - weight) * left_values + weight * right_values
    return warped.reshape(n, c, joints, bodies, t).transpose(0, 1, 4, 2, 3)


def read_training_source(path):
    """Read legacy Chinese source without changing bytes or ignoring errors."""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot decode training source {} as UTF-8 or GB18030/GBK. "
                     "The training file is unchanged; inspect its encoding.".format(path))


def load_training_definitions(root, training_script=None):
    """Resolve an explicit training file, or find supported project entrypoints."""
    names = {"RandomTemporalIntervention", "compute_speed_from_skeleton"}
    explicit = training_script is not None
    candidates = ([Path(training_script).resolve()] if explicit else
                  [root / "main2_gu_speed.py", root / "main.py"])
    rejected = []
    for candidate in candidates:
        if not candidate.is_file():
            if explicit:
                raise FileNotFoundError("Training script not found: {}".format(candidate))
            continue
        source_text, encoding = read_training_source(candidate)
        tree = ast.parse(source_text, filename=str(candidate))
        nodes = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        missing = sorted(names - {node.name for node in nodes})
        if not missing:
            return candidate, nodes, encoding
        rejected.append("{} (missing: {})".format(candidate, ", ".join(missing)))
    if rejected:
        raise ValueError("No matching training definitions in {}. Specify the actual CauASD "
                         "training file with --training-script FILE.py; do not replace your "
                         "training code to satisfy this check.".format("; ".join(rejected)))
    return None, [], None


def run_self_test(root, training_script=None):
    """Analytic tests; additionally compare with actual training code if possible."""
    line = np.zeros((1, 3, 50, 1, 1), dtype=np.float32)
    line[0, 0, :, 0, 0] = np.arange(50)
    for factor, expected in [(0.5, 1.0), (1.0, 1.0), (2.0, 0.5)]:
        np.testing.assert_allclose(training_target(temporal_warp(line, factor)), [expected], rtol=1e-6)
    static = np.ones((2, 3, 50, 4, 2), dtype=np.float32)
    np.testing.assert_array_equal(training_target(static), [0, 0])
    padded = np.concatenate([line, np.zeros_like(line)], axis=4)
    np.testing.assert_allclose(training_target(padded), [0.5])
    np.testing.assert_allclose(training_target(2 * padded), [1.0])
    rng = np.random.default_rng(17)
    random_data = rng.standard_normal((4, 3, 50, 5, 2)).astype(np.float32)
    np.testing.assert_array_equal(temporal_warp(random_data, 1), random_data)
    for factor in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
        assert np.all(training_target(temporal_warp(random_data, factor)) <= training_target(random_data) + 2e-6)
    for bad in [0, -1, float("nan"), float("inf")]:
        try:
            temporal_warp(line, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid factor was accepted")
    invalid = line.copy()
    invalid[0, 0, 0, 0, 0] = np.nan
    try:
        training_target(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite coordinates were accepted")
    print("Analytic checks passed (linear targets: sf=0.5 -> 1, sf=1 -> 1, sf=2 -> 0.5).")

    source_file, nodes, source_encoding = load_training_definitions(root, training_script)
    if source_file is None:
        print("Training-code parity check skipped: no training source found. "
              "Use --training-script FILE.py to verify the actual implementation.")
        return dict(status="analytic_only_training_source_not_found")
    check = dict(status="analytic_only_torch_unavailable", training_script=str(source_file),
                 training_script_encoding=source_encoding,
                 training_script_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest())
    print("Training source: {} (encoding: {})".format(source_file, source_encoding))
    try:
        import torch
    except ImportError:
        print("Training-code parity check skipped: PyTorch not installed locally.")
        return check
    # Extract ONLY these definitions: do not import the training entrypoint, initialize
    # Sacred, load data, allocate CUDA models, or trigger a training run.
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_file), "exec"), namespace)
    with torch.no_grad():
        tensor = torch.from_numpy(random_data.copy())
        np.testing.assert_allclose(training_target(random_data), namespace["compute_speed_from_skeleton"](tensor).cpu().numpy(), rtol=3e-6, atol=2e-7)
        for factor in [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
            warped, _ = namespace["RandomTemporalIntervention"](factor, factor)(tensor)
            np.testing.assert_allclose(temporal_warp(random_data, factor), warped.cpu().numpy(), rtol=3e-6, atol=2e-6)
            np.testing.assert_allclose(training_target(temporal_warp(random_data, factor)), namespace["compute_speed_from_skeleton"](warped).cpu().numpy(), rtol=3e-6, atol=2e-7)
    print("Parity with {} training definitions passed (CPU float32).".format(source_file.name))
    check["status"] = "analytic_and_training_code_parity_passed"
    return check


def analyze_split(data_root, split, factors, max_samples, seed, batch_size, sample_writer):
    dataset = SPLIT_DATASET[split]
    path = data_root / dataset / ("split_" + str(split)) / "seen_train_data.npy"
    if not path.is_file():
        raise FileNotFoundError("Missing training data: {}. Run from the project root or set --data-root.".format(path))
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 5 or data.shape[1] != 3 or data.shape[2] != 50 or min(data.shape[0], data.shape[3], data.shape[4]) < 1:
        raise ValueError("{}: expected nonempty (N,3,50,J,M), got {}. Inputs are not automatically resampled.".format(path, data.shape))
    n = len(data)
    count = n if max_samples == 0 else min(n, max_samples)
    indices = (np.arange(n) if count == n else
               np.sort(np.random.default_rng(seed).choice(n, count, replace=False)))
    targets = np.empty((count, len(factors)), dtype=np.float32)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = np.array(data[indices[start:stop]], dtype=np.float32, copy=True)
        try:
            for column, factor in enumerate(factors):
                targets[start:stop, column] = training_target(temporal_warp(batch, factor))
        except ValueError as error:
            raise ValueError("{}; dataset={}, split={}, input rows={}".format(error, dataset, split, indices[start:stop].tolist())) from error
    baseline = targets[:, factors.index(1.0)].astype(np.float64)
    baseline_mean = float(baseline.mean())
    rows = []
    for column, factor in enumerate(factors):
        values = targets[:, column].astype(np.float64)
        mean = float(values.mean())
        row = dict(dataset=dataset, split=split, sf=factor, n_total=n, n_used=count,
                   seed=seed, target_mean=mean, target_std=float(values.std(ddof=0)),
                   mean_ratio_to_original=mean / baseline_mean if baseline_mean > 0 else None,
                   mean_paired_change=float((values - baseline).mean()),
                   n_zero_original=int(np.count_nonzero(baseline == 0)),
                   output_frames=50)
        rows.append(row)
        for position, source_index in enumerate(indices):
            sample_writer.writerow(dict(dataset=dataset, split=split, sample_index=int(source_index),
                                        sf=factor, original_target=float(baseline[position]),
                                        transformed_target=float(values[position]),
                                        paired_change=float(values[position] - baseline[position])))
    metadata = dict(dataset=dataset, split=split, input=str(path), shape=list(data.shape),
                    selected_sample_indices=indices.tolist())
    return rows, metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=int, nargs="+", choices=sorted(SPLIT_DATASET), default=[1])
    parser.add_argument("--factors", type=positive_factor, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--max-samples", type=int, default=1000, help="Uniform random subset per split; 0 means all training samples.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", type=Path, help="Directory containing ntu60/, ntu120/, pku/; defaults to project data/zeroshot.")
    parser.add_argument("--training-script", type=Path, help="Actual training file for parity checks; otherwise try project main2_gu_speed.py, then main.py.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to project analysis/bsdm_targets.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic/parity checks only; no data needed and no output files written.")
    args = parser.parse_args()
    if args.max_samples < 0 or args.seed < 0 or args.batch_size < 1:
        parser.error("--max-samples and --seed must be nonnegative; --batch-size must be positive.")
    return args


def main():
    args = parse_args()
    root = project_root()
    check_status = run_self_test(root, args.training_script)
    if args.self_test:
        return
    factors = sorted(set([1.0] + args.factors))
    splits = list(dict.fromkeys(args.split))
    data_root = (args.data_root or root / "data" / "zeroshot").resolve()
    output_dir = (args.output_dir or root / "analysis" / "bsdm_targets").resolve()
    # Factor/seed/subset settings are recorded below; split tags keep runs
    # for different requested split sets separate.
    tag = "splits_" + "_".join(map(str, splits))
    summary_path = output_dir / ("bsdm_target_summary_" + tag + ".csv")
    samples_path = output_dir / ("bsdm_target_samples_" + tag + ".csv")
    metadata_path = output_dir / ("bsdm_target_metadata_" + tag + ".json")
    for output in (summary_path, samples_path, metadata_path):
        if output.exists():
            raise FileExistsError("Refusing to overwrite {}. Choose a new --output-dir for a new run.".format(output))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(status="running", target="raw_mean_inter_frame_joint_displacement",
                    transform="tau_t=clip(t/sf,0,T-1); linear interpolation; fixed T=50",
                    backend="numpy_float32", self_test=check_status, body_scale_normalized=False,
                    person_mask=False, endpoint_repetition_in_denominator=True,
                    sampling="uniform without replacement; same rows for every factor; static/zero samples retained",
                    max_samples=args.max_samples, seed=args.seed, factors=factors, splits=splits,
                    std_ddof=0, ratio_definition="mean(transformed_target)/mean(original_target), NOT mean of per-sample ratios",
                    aggregation="sample statistics per split; no averaging across classes, batches, or splits",
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), inputs=[])
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    all_rows = []
    try:
        with samples_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "sample_index", "sf", "original_target", "transformed_target", "paired_change"])
            writer.writeheader()
            for split in splits:
                rows, info = analyze_split(data_root, split, factors, args.max_samples, args.seed, args.batch_size, writer)
                all_rows.extend(rows)
                metadata["inputs"].append(info)
                print("\n{} split {}: {} / {} training samples".format(SPLIT_DATASET[split], split, rows[0]["n_used"], rows[0]["n_total"]))
                print("sf       mean           std            ratio_to_original")
                for row in rows:
                    ratio = "NA" if row["mean_ratio_to_original"] is None else "{:.6f}".format(row["mean_ratio_to_original"])
                    print("{:<8g} {:.8g}     {:.8g}     {}".format(row["sf"], row["target_mean"], row["target_std"], ratio))
        with summary_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        metadata["status"] = "complete"
        metadata["summary"] = all_rows
        metadata["outputs"] = dict(summary=str(summary_path), samples=str(samples_path))
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nSummary: {}\nPer-sample targets: {}\nMetadata: {}".format(summary_path, samples_path, metadata_path))


if __name__ == "__main__":
    main()
