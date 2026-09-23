#!/usr/bin/env python3
"""Frozen-feature speed probe for DVTA using the common fixed-50-frame split.

This standalone script loads the DVTA skeleton encoder and DVTA checkpoint,
extracts the learned skeleton-side visual representation from the same
fixed-50-frame data used by CauASD, and runs Linear Ridge and nonlinear MLP
speed probes.  The default protocol uses the within-action-class residual of
the fixed-50-frame speed proxy and subject-grouped folds.  With
``--features-npz``, an existing DVTA feature archive is reused and no feature
re-extraction is performed.

Example:

    python experiments/speed_probe/dvta_speed_probe.py \
      --split 1 \
      --repo-root /path/to/DVTA-main \
      --checkpoint /path/to/dvta_best_model.pt \
      --encoder-checkpoint /path/to/dvta_stgcn.pt
"""

import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SPLIT_DATASET = {1: "ntu60", 2: "ntu60", 3: "ntu60", 4: "ntu120", 5: "ntu120", 6: "ntu120"}
UNSEEN_LABELS = {
    1: [4, 19, 31, 47, 51], 2: [12, 29, 32, 44, 59], 3: [7, 20, 28, 39, 58],
    4: [3, 18, 26, 38, 41, 60, 87, 99, 102, 110],
    5: [5, 12, 14, 15, 17, 42, 67, 82, 100, 119],
    6: [6, 20, 27, 33, 42, 55, 71, 97, 104, 118],
}


def detect_project_root():
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        if os.path.isdir(os.path.join(path, "data", "zeroshot")):
            return path
    return os.path.dirname(os.path.dirname(here))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=int, required=True, choices=tuple(SPLIT_DATASET))
    parser.add_argument("--repo-root", default=None, help="DVTA-main directory; required when extracting features.")
    parser.add_argument("--model-checkpoint", "--checkpoint", dest="model_checkpoint", default=None,
                        help="DVTA model checkpoint; required when extracting features.")
    parser.add_argument("--encoder-checkpoint", default=None, help="DVTA ST-GCN encoder checkpoint; required when extracting features.")
    parser.add_argument("--project-root", default=detect_project_root())
    parser.add_argument("--features-npz", default=None,
                        help="Reuse an existing DVTA feature archive; skips model loading and feature extraction.")
    parser.add_argument("--target", choices=("global", "within-class"), default="within-class",
                        help="Probe target: within-class residual (default) or global 50-frame speed.")
    parser.add_argument("--fold-mode", choices=("subject", "class"), default="subject",
                        help="Fold split: subject-grouped (default) or class-stratified random.")
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--leaky-sigmoid-alpha", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save-oof", action="store_true")
    return parser.parse_args()


def resolve_device(args):
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(args.gpu)
        return torch.device("cuda:{}".format(args.gpu))
    return torch.device("cpu")


def state_dict_from_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise ValueError("Unsupported checkpoint format: {}".format(type(state).__name__))
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def load_module(module, path, strict):
    result = module.load_state_dict(state_dict_from_file(path), strict=strict)
    if strict:
        return
    print("loaded {}: missing={}, unexpected={}".format(path, len(result.missing_keys), len(result.unexpected_keys)), flush=True)


def load_split(project_root, split):
    dataset = SPLIT_DATASET[split]
    root = os.path.join(project_root, "data", "zeroshot", dataset, "split_{}".format(split))
    paths = {name: os.path.join(root, name) for name in ("unseen_data.npy", "unseen_label.npy", "unseen_sample_names.npy")}
    missing = [path for path in paths.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing fixed-50-frame file(s):\n{}".format("\n".join(missing)))
    data = np.load(paths["unseen_data.npy"], mmap_mode="r")
    labels = np.asarray(np.load(paths["unseen_label.npy"])).reshape(-1).astype(np.int64)
    names = np.asarray(np.load(paths["unseen_sample_names.npy"]).astype(str))
    if data.ndim != 5 or data.shape[1:3] != (3, 50):
        raise ValueError("Expected (N,3,50,V,M), got {}".format(data.shape))
    if not (len(data) == len(labels) == len(names)):
        raise ValueError("Data, labels, and names have different lengths")
    if not np.array_equal(np.sort(np.unique(labels)), np.sort(np.asarray(UNSEEN_LABELS[split]))):
        raise ValueError("Labels do not match configured split {}".format(split))
    subjects = np.asarray([subject_from_name(name) for name in names])
    return dataset, data, labels, names, subjects


def load_feature_archive(path, split):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"features_raw", "speed_model_input", "labels"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError("{} is missing fields {}".format(path, sorted(missing)))
        features = np.asarray(archive["features_raw"], dtype=np.float64)
        speed = np.asarray(archive["speed_model_input"], dtype=np.float64).reshape(-1)
        labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)
        names = np.asarray(archive["names"].astype(str)) if "names" in archive.files else None
    if features.ndim != 2 or len(features) != len(speed) or len(features) != len(labels):
        raise ValueError("Invalid feature archive shapes: features={}, speed={}, labels={}".format(
            features.shape, speed.shape, labels.shape))
    if not np.isfinite(features).all() or not np.isfinite(speed).all():
        raise ValueError("Feature archive contains NaN/Inf values")
    if not np.array_equal(np.sort(np.unique(labels)), np.sort(np.asarray(UNSEEN_LABELS[split]))):
        raise ValueError("Feature archive labels do not match configured split {}".format(split))
    if names is not None and len(names) != len(labels):
        raise ValueError("Feature archive names and labels have different lengths")
    return features, speed, labels, names


def subject_from_name(name):
    match = re.search(r"P(\d{3})", str(name))
    if match is None:
        raise ValueError("Cannot extract subject from sample name: {}".format(name))
    return "P" + match.group(1)


def extract_features(data, repo_root, encoder_path, model_path, args, device, feature_path, labels, names):
    repo_root = os.path.abspath(repo_root)
    sys.path.insert(0, repo_root)
    from encoders.gcn.st_gcn import Model
    from model import DVTA

    encoder = Model(
        in_channels=3, hidden_channels=16, hidden_dim=256, dropout=0.5,
        graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        edge_importance_weighting=True,
    ).to(device)
    load_module(encoder, encoder_path, strict=False)
    encoder.eval()
    model = DVTA(
        skeleton_dim=256, text_dim=768, temperature=args.temperature,
        leaky_sigmoid_alpha=args.leaky_sigmoid_alpha,
    ).to(device)
    load_module(model, model_path, strict=True)
    model.eval()

    raw_features, speeds = [], []
    with torch.inference_mode():
        for start in range(0, len(data), args.batch_size):
            stop = min(start + args.batch_size, len(data))
            batch = torch.as_tensor(np.asarray(data[start:stop]), dtype=torch.float32, device=device)
            gcn_feature = encoder(batch)
            raw = model.direct_alignment.visual_projector(F.normalize(gcn_feature, dim=-1))
            speed = torch.norm(batch[:, :, 1:] - batch[:, :, :-1], dim=1).mean(dim=(1, 2, 3))
            raw_features.append(raw.cpu().numpy().astype(np.float32))
            speeds.append(speed.cpu().numpy().astype(np.float32))
            print("DVTA feature export: {}/{}".format(stop, len(data)), flush=True)
    raw = np.concatenate(raw_features)
    speed = np.concatenate(speeds)
    norm = F.normalize(torch.from_numpy(raw), dim=1).numpy().astype(np.float32)
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    np.savez_compressed(feature_path, names=names, labels=labels, features_raw=raw,
                        features=norm, speed_model_input=speed)
    return raw.astype(np.float64), speed.astype(np.float64)


def subject_folds(subjects, folds, seed):
    unique, counts = np.unique(subjects, return_counts=True)
    if len(unique) < folds:
        raise ValueError("{} subjects are insufficient for {} folds".format(len(unique), folds))
    rng = np.random.default_rng(seed)
    order = np.arange(len(unique))
    rng.shuffle(order)
    order = sorted(order.tolist(), key=lambda i: -int(counts[i]))
    assigned, loads = [[] for _ in range(folds)], np.zeros(folds, dtype=np.int64)
    for index in order:
        destination = int(np.argmin(loads))
        assigned[destination].append(unique[index])
        loads[destination] += counts[index]
    return [np.flatnonzero(np.isin(subjects, np.asarray(group))) for group in assigned]


def class_stratified_folds(labels, folds, seed):
    if folds < 2:
        raise ValueError("folds must be >= 2")
    rng = np.random.default_rng(seed)
    pieces = [[] for _ in range(folds)]
    for label in np.unique(labels):
        index = np.flatnonzero(labels == label)
        if len(index) < folds:
            raise ValueError("Class {} has only {} samples, fewer than {} folds".format(
                int(label), len(index), folds))
        rng.shuffle(index)
        for fold, part in enumerate(np.array_split(index, folds)):
            pieces[fold].append(part)
    return [np.concatenate(parts) for parts in pieces]


def residual_target(train_y, train_labels, query_y, query_labels):
    means = {int(label): float(train_y[train_labels == label].mean()) for label in np.unique(train_labels)}
    missing = sorted(set(np.unique(query_labels).tolist()) - set(means))
    if missing:
        raise ValueError("Classes absent from training fold: {}".format(missing))
    train_r = train_y - np.asarray([means[int(label)] for label in train_labels])
    query_r = query_y - np.asarray([means[int(label)] for label in query_labels])
    return train_r, query_r


def standardise(train, test):
    mean, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def fit_ridge(train_x, train_y, test_x, alpha):
    train_x, test_x = standardise(train_x, test_x)
    y_mean, y_scale = float(train_y.mean()), max(float(train_y.std()), 1e-12)
    train_y = (train_y - y_mean) / y_scale
    gram = train_x.T @ train_x
    gram.flat[::gram.shape[0] + 1] += alpha
    try:
        weight = np.linalg.solve(gram, train_x.T @ train_y)
    except np.linalg.LinAlgError:
        weight = np.linalg.lstsq(gram, train_x.T @ train_y, rcond=None)[0]
    return (test_x @ weight) * y_scale + y_mean


class SpeedMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, value):
        return self.net(value).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fit_mlp(train_x, train_y, test_x, args, seed, fold, device):
    train_x, test_x = standardise(train_x, test_x)
    y_mean, y_scale = float(train_y.mean()), max(float(train_y.std()), 1e-12)
    train_y = (train_y - y_mean) / y_scale
    set_seed(seed * 1000 + fold)
    model = SpeedMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.as_tensor(train_x, dtype=torch.float32),
            torch.as_tensor(train_y, dtype=torch.float32),
        ), batch_size=args.batch_size, shuffle=True,
    )
    model.train()
    for _ in range(args.epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
            batch_y = batch_y.to(device, non_blocking=device.type == "cuda")
            loss = torch.mean(torch.square(model(batch_x) - batch_y))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.as_tensor(test_x, dtype=torch.float32, device=device))
    return prediction.cpu().numpy().astype(np.float64) * y_scale + y_mean


def metrics(target, prediction):
    error = prediction - target
    sse, sst = float(np.square(error).sum()), float(np.square(target - target.mean()).sum())
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "r2": float(1.0 - sse / sst) if sst > 0 else None}


def run_probe(kind, features, speed, labels, subjects, args, seed, device, prefix):
    prediction = np.empty(len(speed), dtype=np.float64)
    target_oof = np.empty(len(speed), dtype=np.float64)
    fold_id = np.empty(len(speed), dtype=np.int64)
    all_index = np.arange(len(speed))
    if args.fold_mode == "subject":
        folds = subject_folds(subjects, args.folds, seed)
    else:
        folds = class_stratified_folds(labels, args.folds, seed)
    for fold, test_index in enumerate(folds):
        train_mask = np.ones(len(speed), dtype=bool)
        train_mask[test_index] = False
        train_index = all_index[train_mask]
        if args.target == "within-class":
            train_y, test_y = residual_target(
                speed[train_index], labels[train_index], speed[test_index], labels[test_index]
            )
        else:
            train_y, test_y = speed[train_index], speed[test_index]
        # Keep only the target defined for this sample's held-out fold.
        # Do not write train_y into target_oof: samples from earlier
        # test folds would otherwise be overwritten in later iterations.
        target_oof[test_index] = test_y
        if kind == "linear":
            prediction[test_index] = fit_ridge(features[train_index], train_y, features[test_index], args.ridge_alpha)
        else:
            prediction[test_index] = fit_mlp(features[train_index], train_y, features[test_index], args, seed, fold, device)
        fold_id[test_index] = fold
    if not np.isfinite(target_oof).all():
        raise RuntimeError("OOF residual targets were not assigned for every sample")
    result = metrics(target_oof, prediction)
    if args.save_oof:
        np.savez_compressed("{}_{}_seed{}.npz".format(prefix, kind, seed),
                            target_original=speed, target_oof=target_oof,
                            prediction=prediction, labels=labels, subjects=subjects, fold=fold_id)
    return result


def main():
    args = parse_args()
    if args.folds < 2 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("folds, epochs, and batch-size must be positive")
    if args.ridge_alpha < 0 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid probe hyper-parameter")
    if args.repo_root:
        args.repo_root = os.path.abspath(args.repo_root)
    args.project_root = os.path.abspath(args.project_root)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    device = resolve_device(args)
    output_dir = os.path.join(args.project_root, "analysis", "speed_probe")
    os.makedirs(output_dir, exist_ok=True)
    dataset = SPLIT_DATASET[args.split]
    tag = "{}_split{}_DVTA".format(dataset, args.split)
    feature_path, prefix = os.path.join(output_dir, tag + "_features.npz"), os.path.join(output_dir, tag + "_probe")
    if args.features_npz:
        feature_path = os.path.abspath(args.features_npz)
        features, speed, labels, archive_names = load_feature_archive(feature_path, args.split)
        if args.fold_mode == "subject":
            _, _, split_labels, names, subjects = load_split(args.project_root, args.split)
            if len(split_labels) != len(labels) or not np.array_equal(split_labels, labels):
                raise ValueError("Feature archive labels do not match project split order; cannot use subject folds")
        else:
            names = archive_names if archive_names is not None else np.asarray(
                ["row_{:08d}".format(i) for i in range(len(labels))]
            )
            subjects = np.asarray(["not_used"] * len(labels))
        model_path, encoder_path = None, None
    else:
        if not args.repo_root or not args.model_checkpoint or not args.encoder_checkpoint:
            raise ValueError("--repo-root, --model-checkpoint, and --encoder-checkpoint are required unless --features-npz is used")
        model_path = args.model_checkpoint if os.path.isabs(args.model_checkpoint) else os.path.join(args.repo_root, args.model_checkpoint)
        encoder_path = args.encoder_checkpoint if os.path.isabs(args.encoder_checkpoint) else os.path.join(args.repo_root, args.encoder_checkpoint)
        model_path, encoder_path = os.path.abspath(model_path), os.path.abspath(encoder_path)
        dataset, data, labels, names, subjects = load_split(args.project_root, args.split)
        features, speed = extract_features(data, args.repo_root, encoder_path, model_path, args, device, feature_path, labels, names)

    result = {
        "protocol": "frozen DVTA representation; {} {}-fold OOF; train-fold-only standardisation{}".format(
            args.fold_mode, args.folds, "; reused feature archive" if args.features_npz else ""),
        "method": "DVTA", "model_checkpoint": model_path, "encoder_checkpoint": encoder_path,
        "dataset": dataset, "split": args.split, "feature_key": "features_raw", "feature_dim": int(features.shape[1]),
        "target": ("within-action-class residual of 50-frame speed_model_input"
                   if args.target == "within-class" else "global 50-frame speed_model_input"),
        "fold_mode": args.fold_mode, "n": int(len(speed)),
        "n_classes": int(len(np.unique(labels))),
        "n_subjects": int(len(np.unique(subjects))) if args.fold_mode == "subject" else None,
        "normalisation": "feature and target mean/std fitted on each training fold only",
        "mlp": {"architecture": "Linear({}->256)-ReLU-Linear(256->128)-ReLU-Linear(128->1)".format(features.shape[1]),
                "optimizer": "AdamW", "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "weight_decay": args.weight_decay},
        "linear": {"ridge_alpha": args.ridge_alpha}, "seeds": seeds, "feature_archive": feature_path, "probes": {},
    }
    for kind in ("linear", "mlp"):
        runs = []
        for seed in seeds:
            value = run_probe(kind, features, speed, labels, subjects, args, seed, device, prefix)
            runs.append({"seed": seed, "metrics": value})
            print("{} seed={} R2={:.6f}".format(kind, seed, value["r2"]), flush=True)
        result["probes"][kind] = {"runs": runs, "summary": {
            key + suffix: float(np.mean([run["metrics"][key] for run in runs])) if suffix == "_mean" else float(np.std([run["metrics"][key] for run in runs], ddof=1))
            for key in ("mae", "rmse", "r2") for suffix in ("_mean", "_std")
        }}
    with open(prefix + ".json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print("wrote -> {}".format(prefix + ".json"))


if __name__ == "__main__":
    main()
