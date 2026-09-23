"""Command-line t-SNE visualization for skeleton features."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.gcn.st_gcn import Model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize skeleton features with t-SNE."
    )
    parser.add_argument("--data", required=True, help="Path to unseen_data.npy")
    parser.add_argument("--labels", required=True, help="Path to unseen_label.npy")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument(
        "--feature-space",
        choices=("projected", "encoder"),
        default="projected",
        help="Use the 768-D projected feature or the 256-D encoder feature.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning-rate", type=float, default=200.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument(
        "--save-coordinates",
        default=None,
        help="Optional CSV path for the 2-D coordinates and labels.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_state_dict(state):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state must be a dictionary.")
    normalized = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        normalized[key] = value
    return normalized


def get_checkpoint_state(checkpoint, name: str):
    if isinstance(checkpoint, dict) and name in checkpoint:
        return checkpoint[name]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        prefix = name + "."
        return {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state.items()
        }
    return checkpoint


def load_encoder_and_projection(checkpoint_path: Path, device: torch.device):
    encoder = Model(
        in_channels=3,
        hidden_channels=16,
        hidden_dim=256,
        graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        edge_importance_weighting=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder_state = normalize_state_dict(get_checkpoint_state(checkpoint, "encoder"))
    encoder_state = {
        key[len("encoder."): ] if key.startswith("encoder.") else key: value
        for key, value in encoder_state.items()
    }
    encoder.load_state_dict(encoder_state, strict=True)

    projection = None
    projection_state = get_checkpoint_state(checkpoint, "adapter")
    if projection_state is not None:
        projection_state = normalize_state_dict(projection_state)
        weight = projection_state.get("adapter.weight")
        bias = projection_state.get("adapter.bias")
        if weight is None:
            weight = projection_state.get("weight")
        if bias is None:
            bias = projection_state.get("bias")
        if weight is not None and bias is not None:
            projection = nn.Linear(weight.shape[1], weight.shape[0]).to(device)
            projection.load_state_dict({"weight": weight, "bias": bias})

    encoder.eval()
    if projection is not None:
        projection.eval()
    return encoder, projection


def choose_indices(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    n = len(labels)
    if max_samples <= 0 or n <= max_samples:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    selected = []
    for class_id in classes:
        class_indices = np.flatnonzero(labels == class_id)
        count = max(1, round(max_samples * len(class_indices) / n))
        count = min(count, len(class_indices))
        selected.extend(rng.choice(class_indices, size=count, replace=False).tolist())

    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) > max_samples:
        selected = rng.choice(selected, size=max_samples, replace=False)
    elif len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(n), selected)
        extra_count = min(max_samples - len(selected), len(remaining))
        if extra_count:
            selected = np.concatenate(
                [selected, rng.choice(remaining, size=extra_count, replace=False)]
            )
    return np.sort(selected)


def extract_features(args: argparse.Namespace, device: torch.device):
    data_path = resolve_path(args.data)
    labels_path = resolve_path(args.labels)
    checkpoint_path = resolve_path(args.checkpoint)

    data = np.load(data_path, mmap_mode="r")
    labels = np.asarray(np.load(labels_path)).reshape(-1)
    if data.ndim != 5:
        raise ValueError(f"Expected data with shape (N,C,T,V,M), got {data.shape}.")
    if len(data) != len(labels):
        raise ValueError("The number of samples and labels does not match.")

    indices = choose_indices(labels, args.max_samples, args.seed)
    tensor_data = torch.from_numpy(np.asarray(data[indices], dtype=np.float32))
    tensor_labels = torch.from_numpy(labels[indices].astype(np.int64))
    loader = DataLoader(
        TensorDataset(tensor_data, tensor_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    encoder, projection = load_encoder_and_projection(checkpoint_path, device)
    if args.feature_space == "projected" and projection is None:
        raise ValueError(
            "The checkpoint has no adapter weights. Use --feature-space encoder."
        )

    features = []
    output_labels = []
    with torch.inference_mode():
        for batch_data, batch_labels in loader:
            batch_features = encoder(batch_data.to(device))
            if args.feature_space == "projected":
                batch_features = projection(batch_features)
            features.append(batch_features.cpu().numpy())
            output_labels.append(batch_labels.numpy())

    return np.concatenate(features, axis=0), np.concatenate(output_labels, axis=0)


def run_tsne(features: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if len(features) < 4:
        raise ValueError("At least four samples are required for t-SNE.")
    perplexity = min(args.perplexity, float(len(features) - 1))
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            init="pca",
            random_state=args.seed,
        )
    except TypeError:  # compatibility with older scikit-learn versions
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=args.learning_rate,
            n_iter=args.max_iter,
            init="pca",
            random_state=args.seed,
        )
    return tsne.fit_transform(features)


def save_plot(coordinates: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classes = np.unique(labels)
    cmap = plt.get_cmap("tab20", max(len(classes), 1))
    class_to_index = {class_id: i for i, class_id in enumerate(classes)}
    colors = np.asarray([class_to_index[class_id] for class_id in labels])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=colors,
        cmap=cmap,
        s=8,
        alpha=0.7,
        vmin=0,
        vmax=max(len(classes) - 1, 1),
    )
    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", markersize=6,
            markerfacecolor=cmap(i), markeredgecolor=cmap(i),
            label=f"Class {class_id}"
        )
        for i, class_id in enumerate(classes)
    ]
    ax.legend(handles=handles, title="Classes", bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE of Skeleton Features")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    features, labels = extract_features(args, device)
    coordinates = run_tsne(features, args)
    output_path = resolve_path(args.output)
    save_plot(coordinates, labels, output_path)

    if args.save_coordinates:
        coordinates_path = resolve_path(args.save_coordinates)
        coordinates_path.parent.mkdir(parents=True, exist_ok=True)
        table = np.column_stack((coordinates, labels))
        np.savetxt(
            coordinates_path,
            table,
            delimiter=",",
            header="tsne_x,tsne_y,label",
            comments="",
        )

    print(f"Saved t-SNE figure to: {output_path}")
    print(f"Samples: {len(labels)}, feature dimension: {features.shape[1]}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
