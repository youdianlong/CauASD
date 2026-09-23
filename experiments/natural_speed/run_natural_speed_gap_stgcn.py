#!/usr/bin/env python3
"""One-command natural slow/fast test for two ST-GCN ZSL checkpoints.

Example:
  python run_natural_speed_gap_stgcn.py --split 1 --device 2 \
    --baseline output/model/pgfa.pt --cauasd output/model/cauasd.pt
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
    ("ntu60", "1"): [4, 19, 31, 47, 51], ("ntu60", "2"): [12, 29, 32, 44, 59],
    ("ntu60", "3"): [7, 20, 28, 39, 58],
    ("ntu120", "4"): [3, 18, 26, 38, 41, 60, 87, 99, 102, 110],
    ("ntu120", "5"): [5, 12, 14, 15, 17, 42, 67, 82, 100, 119],
    ("ntu120", "6"): [6, 20, 27, 33, 42, 55, 71, 97, 104, 118],
}


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True)
    p.add_argument("--device", type=int, required=True)
    p.add_argument("--baseline", required=True, help="PGFA checkpoint with encoder/adapter keys.")
    p.add_argument("--pgfaaug", required=True, help="PGFA augmented checkpoint with encoder/adapter keys.")
    p.add_argument("--cauasd", required=True, help="CauASD checkpoint with encoder/adapter keys.")
    p.add_argument("--output", default="", help="Optional CSV path.")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def speed_proxy(data):
    """Per-sample speed on original 50-frame input; excludes zero padding."""
    x = data.cpu().numpy()
    active = np.any(np.abs(x) > 1e-6, axis=(1, 3, 4))
    delta = np.linalg.norm(x[:, :, 1:] - x[:, :, :-1], axis=1)
    transition = active[:, 1:] & active[:, :-1]
    total = (delta * transition[:, :, None, None]).sum(axis=(1, 2, 3), dtype=np.float64)
    count = transition.sum(axis=1) * x.shape[3] * x.shape[4]
    speed = total / np.maximum(count, 1)
    root = x[:, :, :, :1, :]
    extent = np.linalg.norm(x - root, axis=1)
    extent = np.where(active[:, :, None, None], extent, np.nan)
    invalid = ~active.any(axis=1)
    extent[invalid, 0, 0, 0] = 1.0
    scale = np.nanmedian(extent.reshape(len(x), -1), axis=1)
    scale = np.where(
        np.isfinite(scale) & (scale > 1e-6),
        scale,
        1.0,
    )
    return speed / scale


def predict(checkpoint, loader, device, text, unseen):
    model_args = SimpleNamespace(
        backbone="stgcn", checkpoint=checkpoint, stgcn_hidden_channels=16,
        stgcn_hidden_dim=256, stgcn_dropout=0.5, stgcn_layout="ntu-rgb+d",
        stgcn_strategy="spatial",
    )
    encoder, adapter = load_models(model_args, device)
    prediction, speeds, targets = [], [], []
    with torch.no_grad():
        for x, y, _ in loader:
            speeds.append(speed_proxy(x))
            targets.append(y.numpy())
            feature = F.normalize(adapter(encoder(x.to(device, non_blocking=True))), dim=1)
            prediction.append(unseen[(feature @ text.T).argmax(dim=1).cpu().numpy()])
    return np.concatenate(prediction), np.concatenate(speeds), np.concatenate(targets)


def groups(speed, labels):
    slow = np.zeros(len(labels), bool); fast = np.zeros(len(labels), bool)
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        n = int(np.floor(.30 * len(idx)))
        if n:
            order = idx[np.argsort(speed[idx], kind="stable")]
            slow[order[:n]] = True; fast[order[-n:]] = True
    return {"Naturally slower": slow, "Middle": ~(slow | fast), "Naturally faster": fast}


def accuracy(labels, prediction, conditions):
    result = {}
    for name, mask in conditions.items():
        per_class = [(prediction[(labels == c) & mask] == c).mean() for c in np.unique(labels) if ((labels == c) & mask).any()]
        result[name] = float(np.mean(per_class))
    result["Slow-fast gap"] = abs(result["Naturally faster"] - result["Naturally slower"])
    return result


def main():
    a = args(); split = str(a.split)
    dataset = "ntu60" if split in {"1", "2", "3"} else "ntu120" if split in {"4", "5", "6"} else ""
    key = (dataset, split)
    if key not in UNSEEN: raise ValueError("Supported pairs: {}".format(sorted(UNSEEN)))
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required.")
    torch.cuda.set_device(a.device); device = torch.device("cuda:{}".format(a.device))
    base = os.path.join("data", "zeroshot", dataset, "split_" + split)
    data_path, label_path = os.path.join(base, "unseen_data.npy"), os.path.join(base, "unseen_label.npy")
    labels = np.load(label_path).reshape(-1).astype(np.int64)
    unseen = np.asarray(UNSEEN[key], dtype=np.int64)
    if not np.isin(labels, unseen).all(): raise ValueError("Unexpected labels in {}.".format(label_path))
    language = torch.as_tensor(np.load(os.path.join("data", "language", dataset + "_des_embeddings.npy")), dtype=torch.float32, device=device)
    text = F.normalize(language[torch.as_tensor(unseen, device=device)], dim=1)
    loader = DataLoader(NpyDataset(data_path, labels), batch_size=a.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    base_pred, speed, returned_labels = predict(a.baseline, loader, device, text, unseen)
    aug_pred, speed_aug, returned_labels_aug = predict(a.pgfaaug, loader, device, text, unseen)
    cau_pred, speed_2, returned_labels_2 = predict(a.cauasd, loader, device, text, unseen)
    if (
        not np.array_equal(labels, returned_labels)
        or not np.array_equal(labels, returned_labels_aug)
        or not np.array_equal(labels, returned_labels_2)
        or not np.allclose(speed, speed_aug)
        or not np.allclose(speed, speed_2)
    ):
        raise RuntimeError("Prediction/sample alignment failed.")
    condition = groups(np.log(np.maximum(speed, 1e-12)), labels)
    pgfa = accuracy(labels, base_pred, condition)
    pgfaaug = accuracy(labels, aug_pred, condition)
    cauasd = accuracy(labels, cau_pred, condition)
    output = a.output or os.path.join("analysis", "natural_speed", "{}_split{}_natural_speed.csv".format(dataset, split))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Method", "Naturally slower", "Middle", "Naturally faster", "Slow-fast gap"])
        for name, value in (("PGFA", pgfa), ("PGFAaug", pgfaaug), ("CauASD", cauasd)):
            w.writerow([name] + ["{:.2f}".format(100 * value[k]) for k in ("Naturally slower", "Middle", "Naturally faster", "Slow-fast gap")])
    print(json.dumps({
        "dataset": dataset,
        "split": split,
        "samples": len(labels),
        "groups": {k: int(v.sum()) for k, v in condition.items()},
        "PGFA": pgfa,
        "PGFAaug": pgfaaug,
        "CauASD": cauasd,
        "csv": output,
    }, indent=2))


if __name__ == "__main__": main()
