#!/usr/bin/env python3
"""Test-only variable-length temporal-speed stress test for CauASD.

This script does not train a model and does not modify a checkpoint. It loads
the standard 50-frame unseen split, creates a full-trajectory-preserving
variable-length version for each speed factor, and evaluates the existing
checkpoint.

For the speed-factor convention used by the CauASD temporal intervention:

    sf < 1  -> shorter sequence / acceleration
    sf > 1  -> longer sequence / deceleration

The source trajectory is sampled over its complete [0, T-1] interval:

    T_out = round(T * sf)
    tau_j = j * (T - 1) / (T_out - 1)

There is no clipping and no endpoint repetition in this test. The source
data are still the standard 50-frame model inputs, so this should be reported
as a variable-length test-time stress test, not as a raw-time physical-speed
experiment.

Example:

    python test_variable_length_speed.py \
      --project-root /path/to/CauASD \
      --split 1 \
      --checkpoint /path/to/checkpoint.pt \
      --speed-factors 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
      --device cuda --gpu 0

The same transformed samples must be evaluated with every comparison model
when reporting a cross-method table.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SPLIT_DATASET = {
    1: "ntu60", 2: "ntu60", 3: "ntu60",
    4: "ntu120", 5: "ntu120", 6: "ntu120",
}

UNSEEN_LABELS = {
    1: [4, 19, 31, 47, 51],
    2: [12, 29, 32, 44, 59],
    3: [7, 20, 28, 39, 58],
    4: [3, 18, 26, 38, 41, 60, 87, 99, 102, 110],
    5: [5, 12, 14, 15, 17, 42, 67, 82, 100, 119],
    6: [6, 20, 27, 33, 42, 55, 71, 97, 104, 118],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", required=True,
        help="CauASD project root containing data/ and module/.",
    )
    parser.add_argument(
        "--split", type=int, required=True, choices=tuple(SPLIT_DATASET),
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Existing checkpoint with encoder and adapter state dictionaries.",
    )
    parser.add_argument(
        "--speed-factors", default="0.5,0.75,1.0,1.25,1.5,1.75,2.0",
        help="Comma-separated positive factors.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--da", action="store_true",
        help="Also report prototype-guided DA accuracy using the same test-only rule.",
    )
    parser.add_argument("--support-factor", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_device(args):
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        torch.cuda.set_device(args.gpu)
        return torch.device("cuda:{}".format(args.gpu))
    return torch.device("cpu")


def parse_factors(value):
    factors = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not factors or any(not np.isfinite(item) or item <= 0 for item in factors):
        raise ValueError("--speed-factors must contain positive finite numbers.")
    return factors


def add_project_to_path(project_root):
    project_root = os.path.abspath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from module.gcn.st_gcn import Model
    return Model


class CheckpointAdapter(nn.Module):
    """CPU-safe copy of the adapter used by the saved CauASD checkpoint."""

    def __init__(self, hidden_size=256, output_size=768):
        super().__init__()
        self.adapter = nn.Linear(hidden_size, output_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))
        self.logit_scale_v2 = nn.Parameter(torch.ones([]) * np.log(1.0 / 0.07))

    def forward(self, value):
        return self.adapter(value)


def strip_module_prefix(state):
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def load_models(project_root, checkpoint_path, device):
    Model = add_project_to_path(project_root)
    encoder = Model(
        in_channels=3,
        hidden_channels=16,
        hidden_dim=256,
        dropout=0.5,
        graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        edge_importance_weighting=True,
    ).to(device)
    adapter = CheckpointAdapter().to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")
    if "encoder" not in checkpoint or "adapter" not in checkpoint:
        raise ValueError(
            "Expected checkpoint keys ['encoder', 'adapter']; got {}".format(
                list(checkpoint.keys())
            )
        )

    encoder_state = strip_module_prefix(checkpoint["encoder"])
    adapter_state = strip_module_prefix(checkpoint["adapter"])

    encoder_result = encoder.load_state_dict(encoder_state, strict=False)
    if encoder_result.missing_keys or encoder_result.unexpected_keys:
        raise RuntimeError(
            "Encoder/checkpoint mismatch: missing={}, unexpected={}".format(
                encoder_result.missing_keys, encoder_result.unexpected_keys
            )
        )

    # Some saved checkpoints use weight/bias; others use adapter.weight/bias.
    if "weight" in adapter_state and "adapter.weight" not in adapter_state:
        adapter_state["adapter.weight"] = adapter_state.pop("weight")
    if "bias" in adapter_state and "adapter.bias" not in adapter_state:
        adapter_state["adapter.bias"] = adapter_state.pop("bias")

    adapter_result = adapter.load_state_dict(adapter_state, strict=False)
    optional_missing = {"logit_scale", "logit_scale_v2"}
    required_missing = [
        key for key in adapter_result.missing_keys if key not in optional_missing
    ]
    if required_missing or adapter_result.unexpected_keys:
        raise RuntimeError(
            "Adapter/checkpoint mismatch: missing={}, unexpected={}".format(
                required_missing, adapter_result.unexpected_keys
            )
        )

    encoder.eval()
    adapter.eval()
    return encoder, adapter


def load_split(project_root, split):
    dataset = SPLIT_DATASET[split]
    split_root = os.path.join(
        project_root, "data", "zeroshot", dataset, "split_{}".format(split)
    )
    data_path = os.path.join(split_root, "unseen_data.npy")
    label_path = os.path.join(split_root, "unseen_label.npy")
    missing = [path for path in (data_path, label_path) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing split file(s): {}".format(missing))

    data = np.load(data_path, mmap_mode="r")
    labels = np.asarray(np.load(label_path)).reshape(-1).astype(np.int64)
    if data.ndim != 5 or data.shape[1:3] != (3, 50):
        raise ValueError("Expected unseen data shape (N,3,50,V,M), got {}".format(data.shape))
    if len(data) != len(labels):
        raise ValueError("Data/label length mismatch.")
    return dataset, data, labels


def load_language(project_root, dataset, split, device):
    path = os.path.join(
        project_root, "data", "language", dataset + "_des_embeddings.npy"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    language = torch.as_tensor(np.load(path), dtype=torch.float32, device=device)
    unseen = torch.as_tensor(UNSEEN_LABELS[split], dtype=torch.long, device=device)
    if language.ndim != 2 or language.shape[1] != 768:
        raise ValueError(
            "Expected language shape (num_classes,768), got {}".format(tuple(language.shape))
        )
    return F.normalize(language[unseen], dim=-1), unseen


def variable_length_warp(x, speed_factor):
    """Resample the complete source trajectory to a variable number of frames."""
    if x.ndim != 5:
        raise ValueError("Expected (N,C,T,V,M), got {}".format(tuple(x.shape)))
    if speed_factor <= 0:
        raise ValueError("speed_factor must be positive")

    n, c, t, v, m = x.shape
    if t < 2:
        return x
    target_t = max(2, int(round(t * float(speed_factor))))

    # The source coordinate runs from exactly 0 to exactly T-1. Therefore
    # both endpoints are retained and no clipping or endpoint holding occurs.
    flat = x.permute(0, 1, 3, 4, 2).reshape(n * c * v * m, 1, t)
    warped = F.interpolate(
        flat, size=target_t, mode="linear", align_corners=True
    )
    return warped.reshape(n, c, v, m, target_t).permute(0, 1, 4, 2, 3).contiguous()


def da_accuracy(features, labels, language, unseen_labels, support_factor):
    """Match the test-only prototype-guided evaluation used by the project."""
    logits = features @ language.T
    prediction_index = logits.argmax(dim=1)
    probability = logits.softmax(dim=1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1)

    prototypes = []
    for class_index in range(language.shape[0]):
        mask = prediction_index == class_index
        class_features = features[mask]
        class_entropy = entropy[mask]
        support_num = int(class_entropy.numel() * support_factor)
        if support_num < 1:
            prototype = language[class_index:class_index + 1]
        else:
            _, indices = torch.topk(-class_entropy, support_num)
            prototype = class_features[indices].mean(dim=0, keepdim=True)
        prototypes.append(prototype)

    prototypes = F.normalize(torch.cat(prototypes, dim=0), dim=-1)
    da_index = (features @ prototypes.T).argmax(dim=1)
    da_prediction = unseen_labels[da_index]
    return float((da_prediction == labels).float().mean().item())


@torch.inference_mode()
def evaluate_factor(
    encoder, adapter, language, unseen_labels, data, labels,
    factor, device, batch_size, use_da, support_factor
):
    predictions = []
    feature_blocks = []
    label_blocks = []
    target_t = max(2, int(round(data.shape[2] * float(factor))))

    for start in tqdm(
        range(0, len(data), batch_size),
        desc="variable sf={:g}, T={}".format(factor, target_t),
    ):
        stop = min(start + batch_size, len(data))
        batch = torch.as_tensor(
            np.array(data[start:stop], copy=True),
            dtype=torch.float32,
            device=device,
        )
        target = torch.as_tensor(labels[start:stop], dtype=torch.long, device=device)
        batch = variable_length_warp(batch, factor)

        feature = adapter(encoder(batch)).view(batch.shape[0], -1)
        feature = F.normalize(feature, dim=-1)
        batch_prediction = unseen_labels[(feature @ language.T).argmax(dim=1)]
        predictions.append(batch_prediction.cpu())
        feature_blocks.append(feature.cpu())
        label_blocks.append(target.cpu())

    if not predictions:
        raise RuntimeError("The evaluation split is empty.")

    predictions = torch.cat(predictions)
    all_features = torch.cat(feature_blocks).to(device)
    all_labels = torch.cat(label_blocks).to(device)
    result = {
        "output_frames": target_t,
        "trajectory_coverage": 1.0,
        "endpoint_repeat_ratio": 0.0,
        "out_of_range_ratio": 0.0,
        "accuracy": float(
            (predictions == all_labels.cpu()).float().mean().item()
        ),
    }
    if use_da:
        result["da_accuracy"] = da_accuracy(
            all_features, all_labels, language, unseen_labels, support_factor
        )
    return result


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not 0.0 <= args.support_factor <= 1.0:
        raise ValueError("--support-factor must be in [0,1]")

    args.project_root = os.path.abspath(args.project_root)
    args.checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    factors = parse_factors(args.speed_factors)
    device = resolve_device(args)
    dataset, data, labels = load_split(args.project_root, args.split)
    language, unseen_labels = load_language(
        args.project_root, dataset, args.split, device
    )
    encoder, adapter = load_models(args.project_root, args.checkpoint, device)

    records = []
    for factor in factors:
        metrics = evaluate_factor(
            encoder, adapter, language, unseen_labels, data, labels,
            factor, device, args.batch_size, args.da, args.support_factor,
        )
        records.append({"speed_factor": factor, **metrics})
        print(
            "sf={:g}, output_frames={}, accuracy={:.2f}%{}".format(
                factor,
                metrics["output_frames"],
                100.0 * metrics["accuracy"],
                " , DA={:.2f}%".format(100.0 * metrics["da_accuracy"])
                if "da_accuracy" in metrics else "",
            )
        )

    output = args.output
    if output is None:
        output = os.path.join(
            args.project_root,
            "analysis",
            "variable_length_speed",
            "{}_split{}_variable_length.json".format(dataset, args.split),
        )
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    result = {
        "protocol": "test-only endpoint-preserving variable-length temporal resampling",
        "source_input": "standard fixed 50-frame unseen input",
        "checkpoint": args.checkpoint,
        "dataset": dataset,
        "split": args.split,
        "n_samples": int(len(data)),
        "source_shape": list(data.shape),
        "speed_factor_convention": "sf<1 shorter/accelerated; sf>1 longer/decelerated",
        "target_length_rule": "round(50 * sf), minimum 2",
        "resampling": "linear interpolation over the complete source interval [0,49] with align_corners=True",
        "prototype_guided_da": bool(args.da),
        "support_factor": float(args.support_factor) if args.da else None,
        "device": str(device),
        "records": records,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print("wrote -> {}".format(output))


if __name__ == "__main__":
    main()
