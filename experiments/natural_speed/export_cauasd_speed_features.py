#!/usr/bin/env python3
"""Export CauASD/baseline ZSL features, predictions, and sample IDs.

This module is imported by the natural-speed evaluation scripts.  It only
loads an already trained checkpoint and performs frozen inference; it does not
train or update model parameters.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# The natural-speed scripts live two levels below the project root, while the
# model implementations are stored in ``CauASD/module``.  Resolve that path
# from this file so the scripts work from any current working directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


NTU_BONE_PAIRS = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (21, 21), (22, 23),
    (23, 8), (24, 25), (25, 12),
)
STREAMS = ("joint", "bone", "joint_motion", "bone_motion")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        choices=("stgcn", "shiftgcn-1s", "shiftgcn-4s"),
        required=True,
        help="Backbone used by the checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Skeleton .npy, shape (N,C,T,V,M).")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--sample-names", default=None)
    parser.add_argument("--language", required=True, help="All-class text embedding .npy.")
    parser.add_argument("--unseen-labels", required=True, help="Comma-separated zero-indexed class IDs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stgcn-hidden-channels", type=int, default=16)
    parser.add_argument("--stgcn-hidden-dim", type=int, default=256)
    parser.add_argument("--stgcn-dropout", type=float, default=0.5)
    parser.add_argument("--stgcn-layout", default="ntu-rgb+d")
    parser.add_argument("--stgcn-strategy", default="spatial")
    return parser.parse_args()


class NpyDataset(Dataset):
    def __init__(self, data_path, labels):
        self.data = np.load(data_path, mmap_mode="r")
        self.labels = labels
        if self.data.shape[0] != len(labels):
            raise ValueError(
                "Data/label row mismatch: {} vs {}".format(
                    self.data.shape[0], len(labels)
                )
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = torch.from_numpy(np.asarray(self.data[index], dtype=np.float32))
        return data, int(self.labels[index]), index


class LinearAdapter(nn.Module):
    """Checkpoint-compatible projection used by the ZSL classifier."""

    def __init__(self):
        super().__init__()
        self.adapter = nn.Linear(256, 768)

    def forward(self, x):
        return self.adapter(x)


def import_shift_gcn():
    from module.shift_gcn import Model

    return Model


def import_st_gcn():
    from module.gcn.st_gcn import Model

    return Model


class FourStreamShiftGCN(nn.Module):
    def __init__(self, model_cls):
        super().__init__()
        self.encoders = nn.ModuleDict({name: model_cls() for name in STREAMS})

    @staticmethod
    def derive_streams(joint):
        bone = torch.zeros_like(joint)
        for child, parent in NTU_BONE_PAIRS:
            bone[:, :, :, child - 1, :] = (
                joint[:, :, :, child - 1, :] - joint[:, :, :, parent - 1, :]
            )

        joint_motion = torch.zeros_like(joint)
        joint_motion[:, :, :-1, :, :] = joint[:, :, 1:, :, :] - joint[:, :, :-1, :, :]

        bone_motion = torch.zeros_like(bone)
        bone_motion[:, :, :-1, :, :] = bone[:, :, 1:, :, :] - bone[:, :, :-1, :, :]
        return {
            "joint": joint,
            "bone": bone,
            "joint_motion": joint_motion,
            "bone_motion": bone_motion,
        }

    def forward(self, joint):
        streams = self.derive_streams(joint)
        features = [
            F.normalize(self.encoders[name](streams[name]), dim=1)
            for name in STREAMS
        ]
        return F.normalize(torch.stack(features, dim=0).mean(dim=0), dim=1)


def remove_module_prefix(state):
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def load_models(args, device):
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or "encoder" not in checkpoint or "adapter" not in checkpoint:
        raise ValueError(
            "Expected checkpoint with 'encoder' and 'adapter' keys: {}".format(
                args.checkpoint
            )
        )

    if args.backbone == "stgcn":
        model_cls = import_st_gcn()
        encoder = model_cls(
            in_channels=3,
            hidden_channels=args.stgcn_hidden_channels,
            hidden_dim=args.stgcn_hidden_dim,
            dropout=args.stgcn_dropout,
            graph_args={"layout": args.stgcn_layout, "strategy": args.stgcn_strategy},
            edge_importance_weighting=True,
        )
    else:
        model_cls = import_shift_gcn()
        encoder = (
            model_cls()
            if args.backbone == "shiftgcn-1s"
            else FourStreamShiftGCN(model_cls)
        )

    encoder_state = remove_module_prefix(checkpoint["encoder"])
    incompatible = encoder.load_state_dict(encoder_state, strict=False)
    allowed_missing = {"fc.weight", "fc.bias"} if args.backbone.startswith("shiftgcn") else set()
    missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/backbone mismatch. Missing={}, unexpected={}".format(
                missing, incompatible.unexpected_keys
            )
        )

    adapter = LinearAdapter()
    adapter_state = remove_module_prefix(checkpoint["adapter"])
    incompatible = adapter.load_state_dict(adapter_state, strict=False)
    allowed_unexpected = {"logit_scale", "logit_scale_v2"}
    unexpected = [
        key for key in incompatible.unexpected_keys if key not in allowed_unexpected
    ]
    if (
        "adapter.weight" in incompatible.missing_keys
        or "adapter.bias" in incompatible.missing_keys
        or unexpected
    ):
        raise RuntimeError(
            "Invalid adapter state. Missing={}, unexpected={}".format(
                incompatible.missing_keys, unexpected
            )
        )

    return encoder.to(device).eval(), adapter.to(device).eval()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Feature export requires a CUDA GPU.")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda:{}".format(args.device))

    labels = np.asarray(np.load(args.labels)).reshape(-1).astype(np.int64)
    if args.sample_names:
        names = np.asarray(np.load(args.sample_names)).astype(str)
        if len(names) != len(labels):
            raise ValueError(
                "Sample-name/label row mismatch: {} vs {}".format(
                    len(names), len(labels)
                )
            )
    else:
        names = np.asarray(["row_{:08d}".format(index) for index in range(len(labels))])

    if len(set(names.tolist())) != len(names):
        raise ValueError("--sample-names contains duplicates.")

    unseen = np.asarray(
        [int(value) for value in args.unseen_labels.split(",") if value.strip()],
        dtype=np.int64,
    )
    if unseen.size == 0:
        raise ValueError("--unseen-labels must contain at least one class ID.")
    if not np.isin(labels, unseen).all():
        raise ValueError("Evaluation labels contain IDs outside --unseen-labels.")

    language = torch.as_tensor(np.load(args.language), dtype=torch.float32, device=device)
    if unseen.max() >= language.shape[0]:
        raise ValueError("An unseen label exceeds the text-embedding class count.")
    text = F.normalize(language[torch.as_tensor(unseen, device=device)], dim=1)

    loader = DataLoader(
        NpyDataset(args.data, labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    encoder, adapter = load_models(args, device)

    features, raw_features = [], []
    input_speeds, predictions, rows = [], [], []
    with torch.no_grad():
        for data, _label, row in loader:
            data = data.to(device, non_blocking=True)
            raw_feature = adapter(encoder(data))
            feature = F.normalize(raw_feature, dim=1)
            displacement = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
            input_speed = torch.norm(displacement, dim=1).mean(dim=(1, 2, 3))
            pred = unseen[(feature @ text.T).argmax(dim=1).detach().cpu().numpy()]

            features.append(feature.cpu().numpy().astype(np.float32))
            raw_features.append(raw_feature.cpu().numpy().astype(np.float32))
            input_speeds.append(input_speed.cpu().numpy().astype(np.float32))
            predictions.append(pred.astype(np.int64))
            rows.append(row.numpy())

    rows = np.concatenate(rows)
    if not np.array_equal(rows, np.arange(len(labels))):
        raise RuntimeError("DataLoader changed sample order; aborting to prevent misalignment.")

    features = np.concatenate(features)
    raw_features = np.concatenate(raw_features)
    input_speeds = np.concatenate(input_speeds)
    predictions = np.concatenate(predictions)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.output,
        names=names,
        labels=labels,
        predictions=predictions,
        features=features,
        features_raw=raw_features,
        speed_model_input=input_speeds,
    )

    info = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "backbone": args.backbone,
        "n": int(len(labels)),
        "feature_dim": int(features.shape[1]),
        "feature_keys": ["features", "features_raw"],
        "speed_target": "mean joint displacement per transition in the 50-frame model input",
        "zsl_accuracy": float((predictions == labels).mean()),
        "unseen_labels": unseen.tolist(),
    }
    with open(args.output + ".json", "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
