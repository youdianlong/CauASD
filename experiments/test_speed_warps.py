#!/usr/bin/env python3
"""Unified test-time speed evaluation for one saved checkpoint.

Usage on the server (run from the CauASD project root)::

    python experiments/test_speed_warps.py \
      --project-root /path/to/CauASD \
      --split 1 \
      --checkpoint /path/to/checkpoint.pt \
      --warp all \
      --speed-factors 0.5,2.0 \
      --device cuda \
      --gpu 0

Run only the ordinary fixed-50-frame test::

    python experiments/test_speed_warps.py \
      --project-root /path/to/CauASD \
      --split 1 \
      --checkpoint /path/to/checkpoint.pt \
      --warp linear \
      --speed-factors 0.5,2.0 \
      --device cuda \
      --gpu 0

The script is standalone with respect to the other test scripts: it does not
import or call test1.py, test_unlinear.py, test_sinusoidal.py, or
test_Quadratic.py. It does require the project-root data/, module/, and
language-embedding files, plus the explicitly supplied checkpoint.

The script keeps the two protocols explicit:

1. linear_legacy is the original fixed-50-frame transform
       tau_t = clip(t / sf, 0, T - 1)
   used by the ordinary speed table.
2. random, sinusoidal, and quadratic are endpoint-preserving,
   full-trajectory nonlinear stress tests. They start from the same fixed
   50-frame input, use T_out = round(50 * sf), and interpolate over the
   complete source interval [0, 49].

Only one checkpoint is needed per run. To compare another method, rerun the
same command with its checkpoint and the same split, factors, and seed.
"""

import argparse
import json
import os
import random
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
    parser.add_argument("--project-root", required=True,
                        help="CauASD project root containing data/ and module/.")
    parser.add_argument("--split", type=int, required=True,
                        choices=tuple(SPLIT_DATASET),
                        help="1-3=NTU60; 4-6=NTU120.")
    parser.add_argument("--checkpoint", required=True,
                        help="One checkpoint containing encoder and adapter states.")
    parser.add_argument("--warp",
                        choices=("all", "linear", "random", "sinusoidal", "quadratic"),
                        default="all",
                        help="Warp type; default: all.")
    parser.add_argument("--speed-factors", default="0.5,2.0",
                        help="Comma-separated positive factors; default: 0.5,2.0.")
    parser.add_argument("--nonlinear-strength", type=float, default=0.5,
                        help="Strength of random/sinusoidal local tempo variation.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output JSON path.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    """Adapter shape used by the saved CauASD checkpoints."""

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
    if not isinstance(checkpoint, dict) or "encoder" not in checkpoint or "adapter" not in checkpoint:
        raise ValueError(
            "Expected checkpoint keys ['encoder', 'adapter']; got {}".format(
                list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)
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
    if not os.path.isfile(data_path) or not os.path.isfile(label_path):
        raise FileNotFoundError("Missing {} or {}".format(data_path, label_path))
    data = np.load(data_path, mmap_mode="r")
    labels = np.asarray(np.load(label_path)).reshape(-1).astype(np.int64)
    if data.ndim != 5 or tuple(data.shape[1:3]) != (3, 50):
        raise ValueError("Expected unseen data shape (N,3,50,V,M), got {}".format(data.shape))
    if len(data) != len(labels):
        raise ValueError("Data/label length mismatch.")
    return dataset, data, labels


def load_language(project_root, dataset, split, device):
    path = os.path.join(project_root, "data", "language", dataset + "_des_embeddings.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    language = torch.as_tensor(np.load(path), dtype=torch.float32, device=device)
    unseen = torch.as_tensor(UNSEEN_LABELS[split], dtype=torch.long, device=device)
    if language.ndim != 2 or language.shape[1] != 768:
        raise ValueError(
            "Expected language shape (num_classes,768), got {}".format(tuple(language.shape))
        )
    return F.normalize(language[unseen], dim=-1), unseen


def interpolate_source(x, source_positions):
    """Interpolate a batch using source positions shaped (N,T_out)."""
    n, c, t, v, m = x.shape
    flat = x.permute(0, 1, 3, 4, 2).reshape(n, -1, t)
    left = source_positions.floor().long()
    right = source_positions.ceil().long()
    weight = (source_positions - left.to(source_positions.dtype)).unsqueeze(1)
    channels = flat.size(1)
    left_value = torch.gather(flat, 2, left.unsqueeze(1).expand(-1, channels, -1))
    right_value = torch.gather(flat, 2, right.unsqueeze(1).expand(-1, channels, -1))
    warped = (1.0 - weight) * left_value + weight * right_value
    return warped.reshape(
        n, c, v, m, source_positions.shape[1]
    ).permute(0, 1, 4, 2, 3).contiguous()


def legacy_linear_warp(x, speed_factor):
    """Original fixed-50 temporal-index transform, including its boundaries."""
    n, _, t, _, _ = x.shape
    raw = torch.arange(t, dtype=x.dtype, device=x.device).view(1, t) / float(speed_factor)
    source = raw.clamp(0, t - 1).expand(n, -1)
    warped = interpolate_source(x, source)
    diagnostic = {
        "protocol": "legacy fixed-length temporal-index warping",
        "output_frames": int(t),
        "trajectory_coverage": float(
            (source.max() - source.min()).item() / max(t - 1, 1)
        ),
        "endpoint_repeat_ratio": float(
            (source[:, 1:] == source[:, :-1]).float().mean().item()
        ),
        "out_of_range_ratio": float(
            ((raw < 0) | (raw > t - 1)).float().mean().item()
        ),
    }
    return warped, diagnostic


def endpoint_positions(n, source_t, output_t, kind, strength, seed, device):
    """Create monotone endpoint-preserving normalized source positions."""
    u = torch.linspace(0.0, 1.0, output_t, dtype=torch.float32, device=device)
    if kind == "linear":
        q = u.view(1, -1).expand(n, -1)
    elif kind == "sinusoidal":
        amplitude = min(max(float(strength), 0.0), 0.8)
        q_one = u + amplitude * torch.sin(2.0 * torch.pi * u) / (2.0 * torch.pi)
        q = q_one.view(1, -1).expand(n, -1)
    elif kind == "quadratic":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        ease_in = torch.rand(n, generator=generator) >= 0.5
        q_in = u.pow(2.0).view(1, -1).expand(n, -1)
        q_out = (1.0 - (1.0 - u).pow(2.0)).view(1, -1).expand(n, -1)
        q = torch.where(ease_in.to(device).view(-1, 1), q_in, q_out)
    elif kind == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        noise = torch.randn(
            (n, max(output_t - 1, 1)), generator=generator
        ) * float(strength)
        local_rate = torch.exp(noise).to(device)
        cumulative = torch.cat(
            [torch.zeros((n, 1), device=device), torch.cumsum(local_rate, dim=1)],
            dim=1,
        )
        q = cumulative / cumulative[:, -1:].clamp_min(1e-8)
    else:
        raise ValueError("Unknown nonlinear warp: {}".format(kind))
    return q * float(source_t - 1)


def endpoint_variable_warp(x, speed_factor, kind, strength, seed):
    """Resample the complete 50-frame source trajectory to round(50*sf)."""
    n, _, source_t, _, _ = x.shape
    output_t = max(2, int(round(source_t * float(speed_factor))))
    source = endpoint_positions(n, source_t, output_t, kind, strength, seed, x.device)
    warped = interpolate_source(x, source)
    diagnostic = {
        "protocol": "endpoint-preserving variable-length temporal resampling",
        "output_frames": int(output_t),
        "trajectory_coverage": 1.0,
        "endpoint_repeat_ratio": 0.0,
        "out_of_range_ratio": 0.0,
    }
    return warped, diagnostic


@torch.inference_mode()
def evaluate_factor(encoder, adapter, language, unseen_labels, data, labels,
                    warp, speed_factor, strength, seed, device, batch_size):
    correct = 0
    total = 0
    first_diagnostic = None
    for start in tqdm(
        range(0, len(data), batch_size),
        desc="{} sf={:g}".format(warp, speed_factor),
    ):
        stop = min(start + batch_size, len(data))
        batch = torch.as_tensor(
            np.array(data[start:stop], copy=True),
            dtype=torch.float32,
            device=device,
        )
        target = torch.as_tensor(labels[start:stop], dtype=torch.long, device=device)
        if warp == "linear_legacy":
            transformed, diagnostic = legacy_linear_warp(batch, speed_factor)
        else:
            transformed, diagnostic = endpoint_variable_warp(
                batch, speed_factor, warp, strength, seed + start
            )
        if first_diagnostic is None:
            first_diagnostic = diagnostic
        feature = adapter(encoder(transformed)).view(transformed.shape[0], -1)
        feature = F.normalize(feature, dim=-1)
        prediction = unseen_labels[(feature @ language.T).argmax(dim=1)]
        correct += int((prediction == target).sum().item())
        total += int(target.numel())
    if total == 0:
        raise RuntimeError("The evaluation split is empty.")
    return {
        **first_diagnostic,
        "speed_factor": float(speed_factor),
        "accuracy": float(correct / total),
        "n_samples": int(total),
    }


def selected_warps(value):
    if value == "all":
        return ["linear_legacy", "random", "sinusoidal", "quadratic"]
    return ["linear_legacy" if value == "linear" else value]


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.nonlinear_strength < 0:
        raise ValueError("--nonlinear-strength must be non-negative")
    args.project_root = os.path.abspath(args.project_root)
    args.checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    set_seed(args.seed)
    factors = parse_factors(args.speed_factors)
    device = resolve_device(args)
    dataset, data, labels = load_split(args.project_root, args.split)
    language, unseen_labels = load_language(args.project_root, dataset, args.split, device)
    encoder, adapter = load_models(args.project_root, args.checkpoint, device)

    records = []
    for warp in selected_warps(args.warp):
        for factor_index, factor in enumerate(factors):
            record = evaluate_factor(
                encoder, adapter, language, unseen_labels, data, labels,
                warp, factor, args.nonlinear_strength,
                args.seed + 1009 * factor_index, device, args.batch_size,
            )
            records.append(record)
            print(
                "{} sf={:g}, T_out={}, accuracy={:.2f}%".format(
                    warp, factor, record["output_frames"], 100.0 * record["accuracy"]
                ),
                flush=True,
            )

    output = args.output
    if output is None:
        output = os.path.join(
            args.project_root, "analysis", "nonlinear_speed",
            "{}_split{}_speed_warps.json".format(dataset, args.split),
        )
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    result = {
        "protocol": "unified test-only speed evaluation",
        "checkpoint": args.checkpoint,
        "dataset": dataset,
        "split": args.split,
        "source_input": "standard fixed 50-frame unseen input",
        "speed_factors": factors,
        "warp_types": selected_warps(args.warp),
        "seed": args.seed,
        "nonlinear_strength": args.nonlinear_strength,
        "device": str(device),
        "records": records,
        "interpretation": {
            "linear_legacy": "ordinary fixed-50-frame test; exact t/sf clipping protocol",
            "random": "full-trajectory endpoint-preserving variable-length nonlinear test",
            "sinusoidal": "full-trajectory endpoint-preserving sinusoidal tempo test",
            "quadratic": "full-trajectory endpoint-preserving quadratic tempo test",
        },
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print("wrote -> {}".format(output))


if __name__ == "__main__":
    main()
