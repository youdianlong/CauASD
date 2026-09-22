#!/usr/bin/env python3
"""Standalone frozen-feature speed-probe experiment.

This single file loads a trained CauASD checkpoint, reads the fixed-50-frame
split data, exports frozen features, and runs both a linear ridge probe and a
nonlinear MLP probe.  The default protocol uses the within-action-class
residual of the fixed-50-frame speed target and subject-grouped folds.  Class
means are computed from the training fold only, preventing target leakage.
It does not need the raw NTU files, a separate exporter, or model retraining.

Run from the project root:

    python experiments/speed_probe/cauasd_speed_probe.py \
      --split 1 \
      --checkpoint output/model/split_1_kl_DA_des_support_factor1.0_lr0.05.pt

The checkpoint must contain the ``encoder`` and ``adapter`` state dictionaries
saved by the CauASD training entry point.
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

SPLIT_DATASET = {
    1: "ntu60", 2: "ntu60", 3: "ntu60",
    4: "ntu120", 5: "ntu120", 6: "ntu120",
}

UNSEEN_LABELS = {
    1: "4,19,31,47,51",
    2: "12,29,32,44,59",
    3: "7,20,28,39,58",
    4: "3,18,26,38,41,60,87,99,102,110",
    5: "5,12,14,15,17,42,67,82,100,119",
    6: "6,20,27,33,42,55,71,97,104,118",
}

NTU_BONE_PAIRS = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (21, 21), (22, 23),
    (23, 8), (24, 25), (25, 12),
)


def detect_project_root(script_path):
    script_dir = os.path.dirname(os.path.abspath(script_path))
    candidates = [
        script_dir,
        os.path.abspath(os.path.join(script_dir, os.pardir)),
        os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir)),
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "module")) and os.path.isdir(os.path.join(path, "data")):
            return path
    return candidates[-1]


PROJECT_ROOT = detect_project_root(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=int, required=True, choices=tuple(SPLIT_DATASET),
                        help="1--3=NTU60; 4--6=NTU120.")
    parser.add_argument("--checkpoint", required=False, default=None,
                        help="Complete CauASD checkpoint containing encoder and adapter.")
    parser.add_argument("--features-npz", default=None,
                        help="Reuse an existing CauASD feature archive; skips checkpoint loading and feature extraction.")
    parser.add_argument("--method-name", default="CauASD")
    parser.add_argument("--backbone", choices=("stgcn", "shiftgcn-1s", "shiftgcn-4s"), default="stgcn")
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--target", choices=("global", "within-class"), default="within-class",
                        help="Probe target: within-class residual (default) or global 50-frame speed.")
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save-oof", action="store_true")
    return parser.parse_args()


def parse_seeds(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer values")
    return seeds


def resolve_device(args):
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(args.gpu)
        return torch.device("cuda:{}".format(args.gpu))
    return torch.device("cpu")


def remove_module_prefix(state):
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def load_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "encoder" not in checkpoint or "adapter" not in checkpoint:
        keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise ValueError(
            "Expected a complete CauASD checkpoint with keys ['encoder', 'adapter']; found {}".format(keys)
        )
    return checkpoint


class FourStreamShiftGCN(nn.Module):
    def __init__(self, model_cls):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: model_cls() for name in ("joint", "bone", "joint_motion", "bone_motion")
        })

    @staticmethod
    def derive_streams(joint):
        bone = torch.zeros_like(joint)
        for child, parent in NTU_BONE_PAIRS:
            bone[:, :, :, child - 1, :] = joint[:, :, :, child - 1, :] - joint[:, :, :, parent - 1, :]
        joint_motion = torch.zeros_like(joint)
        joint_motion[:, :, :-1, :, :] = joint[:, :, 1:, :, :] - joint[:, :, :-1, :, :]
        bone_motion = torch.zeros_like(bone)
        bone_motion[:, :, :-1, :, :] = bone[:, :, 1:, :, :] - bone[:, :, :-1, :, :]
        return {"joint": joint, "bone": bone, "joint_motion": joint_motion, "bone_motion": bone_motion}

    def forward(self, joint):
        streams = self.derive_streams(joint)
        features = [F.normalize(self.encoders[name](streams[name]), dim=1) for name in streams]
        return F.normalize(torch.stack(features, dim=0).mean(dim=0), dim=1)


def build_encoder(backbone):
    if backbone == "stgcn":
        from module.gcn.st_gcn import Model
        return Model(
            in_channels=3,
            hidden_channels=16,
            hidden_dim=256,
            dropout=0.5,
            graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
            edge_importance_weighting=True,
        )
    from module.shift_gcn import Model
    return Model() if backbone == "shiftgcn-1s" else FourStreamShiftGCN(Model)


def load_models(checkpoint, backbone, device):
    encoder = build_encoder(backbone)
    encoder_state = remove_module_prefix(checkpoint["encoder"])
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    allowed_missing = {"fc.weight", "fc.bias"} if backbone.startswith("shiftgcn") else set()
    missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError("Encoder/checkpoint mismatch: missing={}, unexpected={}".format(
            missing, incompatible.unexpected_keys
        ))

    adapter = nn.Linear(256, 768)
    adapter_state = remove_module_prefix(checkpoint["adapter"])
    # The CauASD adapter uses module.adapter.Linear, whose state dictionary
    # may retain the extra ``adapter.`` prefix.  The standalone probe uses an
    # equivalent nn.Linear, so accept both checkpoint naming conventions.
    if "weight" not in adapter_state and "adapter.weight" in adapter_state:
        adapter_state = {
            (key[8:] if key.startswith("adapter.") else key): value
            for key, value in adapter_state.items()
        }
    incompatible = adapter.load_state_dict(adapter_state, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if key not in {"logit_scale", "logit_scale_v2"}]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError("Adapter/checkpoint mismatch: missing={}, unexpected={}".format(
            incompatible.missing_keys, unexpected
        ))
    return encoder.to(device).eval(), adapter.to(device).eval()


def normalise_name(value):
    value = str(value)
    return value[:-9] if value.endswith(".skeleton") else value


def subject_from_name(name):
    match = re.search(r"P(\d{3})", normalise_name(name))
    if match is None:
        raise ValueError("Cannot extract subject Pxxx from sample name: {}".format(name))
    return "P" + match.group(1)


def load_fixed_split(project_root, split):
    dataset = SPLIT_DATASET[split]
    split_dir = os.path.join(project_root, "data", "zeroshot", dataset, "split_{}".format(split))
    data_path = os.path.join(split_dir, "unseen_data.npy")
    label_path = os.path.join(split_dir, "unseen_label.npy")
    names_path = os.path.join(split_dir, "unseen_sample_names.npy")
    missing = [path for path in (data_path, label_path, names_path) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing fixed-50-frame split file(s):\n{}".format("\n".join(missing)))

    data = np.load(data_path, mmap_mode="r")
    labels = np.asarray(np.load(label_path)).reshape(-1).astype(np.int64)
    names = np.asarray([normalise_name(item) for item in np.load(names_path).astype(str)])
    if data.ndim != 5 or data.shape[1:3] != (3, 50):
        raise ValueError("Expected input shape (N,3,50,V,M), got {}".format(data.shape))
    if not (len(data) == len(labels) == len(names)):
        raise ValueError("Data, labels, and sample names have different lengths")
    configured = np.asarray([int(item) for item in UNSEEN_LABELS[split].split(",")], dtype=np.int64)
    if not np.array_equal(np.sort(np.unique(labels)), np.sort(configured)):
        raise ValueError("Labels do not match configured unseen labels for split {}".format(split))
    subjects = np.asarray([subject_from_name(name) for name in names])
    return dataset, data, labels, names, subjects


def load_feature_archive(path, split, project_root):
    """Load features exported by export_cauasd_speed_features.py.

    The older exporter may store placeholder row names when no sample-name
    sidecar was supplied.  In that case, the fixed split names are used after
    verifying that the archive labels have exactly the same row order.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"features_raw", "speed_model_input", "labels"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError("{} is missing fields {}".format(path, sorted(missing)))
        features = np.asarray(archive["features_raw"], dtype=np.float64)
        target = np.asarray(archive["speed_model_input"], dtype=np.float64).reshape(-1)
        archive_labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)
        archive_names = (np.asarray(archive["names"]).astype(str)
                         if "names" in archive.files else None)
    if features.ndim != 2 or len(features) != len(target) or len(features) != len(archive_labels):
        raise ValueError("Invalid feature archive shapes: features={}, speed={}, labels={}".format(
            features.shape, target.shape, archive_labels.shape))
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Feature archive contains NaN/Inf values")

    _, _, split_labels, split_names, split_subjects = load_fixed_split(project_root, split)
    if len(split_labels) != len(archive_labels) or not np.array_equal(split_labels, archive_labels):
        raise ValueError("Feature archive labels do not match the fixed split row order")

    # Use real sample IDs from the fixed split whenever the archive has only
    # exporter-generated row IDs.  This enables subject-grouped CV safely.
    use_archive_names = archive_names is not None and len(archive_names) == len(archive_labels)
    if use_archive_names:
        use_archive_names = not all(name.startswith("row_") for name in archive_names)
    if use_archive_names:
        names = np.asarray([normalise_name(name) for name in archive_names])
        if not np.array_equal(names, split_names):
            raise ValueError("Feature archive sample names do not match fixed split order")
    else:
        names, subjects = split_names, split_subjects
    if use_archive_names:
        subjects = np.asarray([subject_from_name(name) for name in names])
    return features, target, archive_labels, names, subjects


def export_features(data, labels, names, checkpoint, backbone, device, batch_size, output_path):
    encoder, adapter = load_models(checkpoint, backbone, device)
    raw_features = []
    speeds = []
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            batch = torch.as_tensor(np.asarray(data[start:stop]), dtype=torch.float32, device=device)
            raw_feature = adapter(encoder(batch))
            speed = torch.norm(batch[:, :, 1:] - batch[:, :, :-1], dim=1).mean(dim=(1, 2, 3))
            raw_features.append(raw_feature.cpu().numpy().astype(np.float32))
            speeds.append(speed.cpu().numpy().astype(np.float32))
            print("feature export: {}/{}".format(stop, len(data)), flush=True)
    raw = np.concatenate(raw_features, axis=0)
    speed = np.concatenate(speeds, axis=0)
    features = F.normalize(torch.from_numpy(raw), dim=1).numpy().astype(np.float32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path, names=names, labels=labels, features_raw=raw,
        features=features, speed_model_input=speed,
    )
    return raw.astype(np.float64), speed.astype(np.float64)


def subject_grouped_folds(subjects, folds, seed):
    unique, counts = np.unique(subjects, return_counts=True)
    if len(unique) < folds:
        raise ValueError("{} subjects are insufficient for {} folds".format(len(unique), folds))
    rng = np.random.default_rng(seed)
    order = np.arange(len(unique))
    rng.shuffle(order)
    order = sorted(order.tolist(), key=lambda index: -int(counts[index]))
    assigned = [[] for _ in range(folds)]
    loads = np.zeros(folds, dtype=np.int64)
    for index in order:
        destination = int(np.argmin(loads))
        assigned[destination].append(unique[index])
        loads[destination] += counts[index]
    return [np.flatnonzero(np.isin(subjects, np.asarray(group))) for group in assigned]


def standardise_train_test(train, test):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def target_stats(train):
    mean = float(train.mean())
    scale = float(train.std())
    return mean, max(scale, 1e-12)


def fit_ridge(train_x, train_y, test_x, alpha):
    train_x, test_x = standardise_train_test(train_x, test_x)
    mean, scale = target_stats(train_y)
    train_y = (train_y - mean) / scale
    gram = train_x.T @ train_x
    gram.flat[::gram.shape[0] + 1] += alpha
    try:
        weight = np.linalg.solve(gram, train_x.T @ train_y)
    except np.linalg.LinAlgError:
        weight = np.linalg.lstsq(gram, train_x.T @ train_y, rcond=None)[0]
    return (test_x @ weight) * scale + mean


class SpeedProbeMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
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
    train_x, test_x = standardise_train_test(train_x, test_x)
    mean, scale = target_stats(train_y)
    train_y = (train_y - mean) / scale
    set_seed(seed * 1000 + fold)
    model = SpeedProbeMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(train_x, dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
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
    return prediction.cpu().numpy().astype(np.float64) * scale + mean


def metrics(target, prediction):
    error = prediction - target
    sse = float(np.square(error).sum())
    sst = float(np.square(target - target.mean()).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - sse / sst) if sst > 0 else None,
    }


def class_residual(train_target, train_labels, query_target, query_labels):
    """Remove action-class means estimated from the training fold only."""
    means = {}
    for label in np.unique(train_labels):
        values = train_target[train_labels == label]
        means[int(label)] = float(values.mean())
    missing = sorted(set(np.unique(query_labels).tolist()) - set(means))
    if missing:
        raise ValueError("Classes {} are absent from the training fold".format(missing))
    train_residual = train_target - np.asarray([means[int(label)] for label in train_labels])
    query_residual = query_target - np.asarray([means[int(label)] for label in query_labels])
    return train_residual, query_residual


def run_probe(kind, features, target, labels, subjects, args, seed, device, output_prefix):
    folds = subject_grouped_folds(subjects, args.folds, seed)
    prediction = np.empty(len(target), dtype=np.float64)
    probe_target = np.empty(len(target), dtype=np.float64)
    fold_id = np.empty(len(target), dtype=np.int64)
    all_index = np.arange(len(target))
    for fold, test_index in enumerate(folds):
        train_mask = np.ones(len(target), dtype=bool)
        train_mask[test_index] = False
        train_index = all_index[train_mask]
        if args.target == "within-class":
            train_target, test_target = class_residual(
                target[train_index], labels[train_index], target[test_index], labels[test_index]
            )
        else:
            train_target, test_target = target[train_index], target[test_index]
        probe_target[test_index] = test_target
        if kind == "linear":
            prediction[test_index] = fit_ridge(
                features[train_index], train_target, features[test_index], args.ridge_alpha
            )
        else:
            prediction[test_index] = fit_mlp(
                features[train_index], train_target, features[test_index],
                args, seed, fold, device
            )
        fold_id[test_index] = fold
    result = metrics(probe_target, prediction)
    if args.save_oof:
        np.savez_compressed(
            "{}_{}_seed{}.npz".format(output_prefix, kind, seed),
            target_original=target, target_probe=probe_target,
            prediction=prediction, labels=labels, fold=fold_id, subjects=subjects,
        )
    return result


def summarize(runs):
    result = {}
    for key in ("mae", "rmse", "r2"):
        values = [run["metrics"][key] for run in runs]
        result[key + "_mean"] = float(np.mean(values))
        result[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def main():
    args = parse_args()
    if args.folds < 2 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("folds, epochs, and batch-size must be positive")
    if args.ridge_alpha < 0 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid probe hyper-parameter")
    args.project_root = os.path.abspath(args.project_root)
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(args.project_root, args.checkpoint)
        checkpoint_path = os.path.abspath(checkpoint_path)
    seeds = parse_seeds(args.seeds)
    device = resolve_device(args)
    dataset = SPLIT_DATASET[args.split]
    if args.features_npz:
        feature_path = os.path.abspath(args.features_npz)
        features, target, labels, names, subjects = load_feature_archive(
            feature_path, args.split, args.project_root
        )
        checkpoint = None
    else:
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required unless --features-npz is used")
        dataset, data, labels, names, subjects = load_fixed_split(args.project_root, args.split)
        checkpoint = load_checkpoint(checkpoint_path)

    output_dir = os.path.join(args.project_root, "analysis", "speed_probe")
    os.makedirs(output_dir, exist_ok=True)
    tag = "{}_split{}_{}".format(dataset, args.split, args.method_name)
    if not args.features_npz:
        feature_path = os.path.join(output_dir, tag + "_features.npz")
    target_tag = "global" if args.target == "global" else "within_class"
    output_prefix = os.path.join(output_dir, tag + "_" + target_tag + "_probe")
    if not args.features_npz:
        features, target = export_features(
            data, labels, names, checkpoint, args.backbone, device, args.batch_size, feature_path
        )

    result = {
        "protocol": "frozen representation; subject-grouped {}-fold OOF; train-fold-only standardisation".format(args.folds),
        "method_name": args.method_name,
        "checkpoint": checkpoint_path,
        "dataset": dataset,
        "split": args.split,
        "feature_key": "features_raw",
        "target": ("within-action-class residual of speed_model_input; class means fitted on each training fold only"
                   if args.target == "within-class" else "global 50-frame speed_model_input"),
        "n": int(len(target)),
        "feature_dim": int(features.shape[1]),
        "n_classes": int(len(np.unique(labels))),
        "n_subjects": int(len(np.unique(subjects))),
        "normalisation": "feature and target mean/std fitted on each training fold only",
        "mlp": {
            "architecture": "Linear({}->256)-ReLU-Linear(256->128)-ReLU-Linear(128->1)".format(features.shape[1]),
            "optimizer": "AdamW", "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
        },
        "linear": {"ridge_alpha": args.ridge_alpha},
        "seeds": seeds,
        "feature_archive": feature_path,
        "probes": {},
    }

    for kind in ("linear", "mlp"):
        runs = []
        for seed in seeds:
            value = run_probe(kind, features, target, labels, subjects, args, seed, device, output_prefix)
            runs.append({"seed": seed, "metrics": value})
            print("{} seed={} R2={:.6f}".format(kind, seed, value["r2"]), flush=True)
        result["probes"][kind] = {"runs": runs, "summary": summarize(runs)}

    with open(output_prefix + ".json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print("wrote -> {}".format(output_prefix + ".json"))


if __name__ == "__main__":
    main()
