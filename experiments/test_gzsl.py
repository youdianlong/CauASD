"""GZSL fixed-speed evaluation for a saved CauASD checkpoint.

The script evaluates both seen and unseen test sets with all class text
prototypes as candidates, and reports seen accuracy (S), unseen accuracy (U),
and their harmonic mean (H).  It also supports calibrated stacking, i.e.
subtracting a scalar gamma from every seen-class score before prediction.

Standard one-line use:
    cd /path/to/CauASD
    python experiments/test_gzsl.py with \
        split='1' \
        dataset='ntu60' \
        checkpoint_path=/path/to/cauasd_checkpoint.pt \
        speed_factor=1.0 \
        test_gpu=0 \
        calibration_mode='scan'

Use ``split='2'`` or ``split='3'`` for the other NTU-60 splits.  Use
``dataset='ntu120'`` with ``split='4'``, ``'5'``, or ``'6'`` for NTU-120.
The default ``speed_factor=1.0`` evaluates the original, unwarped inputs.
"""
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import *
from dataset import DataSet

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from module.gcn.st_gcn import Model
from module.adapter import Linear


@ex.config
def gzsl_speed_test_config():
    checkpoint_path = ""
    # Table-VII GZSL evaluation uses the original, unwarped test inputs.
    speed_factor = 1.0
    test_gpu = 0
    test_batch_size = 64
    test_num_workers = 16
    # Leave blank to infer the standard names from test_list/test_label:
    # seen_test_data.npy and seen_test_label.npy.
    seen_test_list = ""
    seen_test_label = ""
    # ``none`` reports ordinary all-class GZSL. ``fixed`` uses
    # calibration_gamma. ``scan`` scans the specified range and reports the
    # gamma with the highest H on the supplied GZSL test data (the common
    # paper-style/oracle calibrated-stacking protocol).
    calibration_mode = "scan"
    calibration_gamma = 0.0
    calibration_min = -1.0
    calibration_max = 1.0
    calibration_steps = 401


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def linear_time_scale(x, speed_factor):
    """The original main2 fixed-length linear temporal-index transform."""
    if x.ndim != 5:
        raise ValueError("Expected (N,C,T,V,M), got {}.".format(tuple(x.shape)))
    if speed_factor <= 0:
        raise ValueError("speed_factor must be positive.")

    n, c, t, v, m = x.shape
    x_flat = x.permute(0, 1, 3, 4, 2).reshape(n, -1, t)
    source_t = torch.arange(t, device=x.device, dtype=x.dtype).view(1, t)
    source_t = (source_t / float(speed_factor)).clamp_(0, t - 1).expand(n, -1)
    left = source_t.floor().long()
    right = source_t.ceil().long()
    weight = (source_t - left.to(source_t.dtype)).unsqueeze(1)
    channels = x_flat.size(1)
    left_value = torch.gather(x_flat, 2, left.unsqueeze(1).expand(-1, channels, -1))
    right_value = torch.gather(x_flat, 2, right.unsqueeze(1).expand(-1, channels, -1))
    warped = (1.0 - weight) * left_value + weight * right_value
    return warped.reshape(n, c, v, m, t).permute(0, 1, 4, 2, 3)


def resolve_seen_paths(unseen_data_path, unseen_label_path, seen_data_path, seen_label_path):
    """Infer the names already documented in the project's config.py."""
    if not seen_data_path:
        seen_data_path = unseen_data_path.replace("unseen_data.npy", "seen_test_data.npy")
    if not seen_label_path:
        seen_label_path = unseen_label_path.replace("unseen_label.npy", "seen_test_label.npy")
    missing = [path for path in (seen_data_path, seen_label_path) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Cannot find seen GZSL test files: {}. Supply their exact paths with "
            "seen_test_list=/... seen_test_label=/...".format(missing)
        )
    return seen_data_path, seen_label_path


def load_checkpoint(path):
    if not path:
        raise ValueError("Set checkpoint_path=/path/to/cauasd_checkpoint.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "encoder" not in checkpoint or "adapter" not in checkpoint:
        raise KeyError(
            "Expected a checkpoint containing 'encoder' and 'adapter'. Received: {}".format(
                list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)
            )
        )
    return checkpoint


class Main2GUGZSLSpeedTester:
    @ex.capture
    def __init__(
        self,
        test_list,
        test_label,
        language_path,
        unseen_label,
        in_channels,
        hidden_channels,
        hidden_dim,
        dropout,
        graph_args,
        edge_importance_weighting,
        checkpoint_path,
        speed_factor,
        test_gpu,
        test_batch_size,
        test_num_workers,
        seen_test_list,
        seen_test_label,
        calibration_mode,
        calibration_gamma,
        calibration_min,
        calibration_max,
        calibration_steps,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("This GZSL test requires CUDA.")
        torch.cuda.set_device(test_gpu)
        self.device = torch.device("cuda:{}".format(test_gpu))
        self.speed_factor = float(speed_factor)
        if self.speed_factor <= 0:
            raise ValueError("speed_factor must be positive.")
        self.calibration_mode = str(calibration_mode).lower()
        self.calibration_gamma = float(calibration_gamma)
        self.calibration_min = float(calibration_min)
        self.calibration_max = float(calibration_max)
        self.calibration_steps = int(calibration_steps)
        if self.calibration_mode not in {"none", "fixed", "scan"}:
            raise ValueError("calibration_mode must be one of none, fixed, scan.")
        if self.calibration_mode == "scan" and (
            self.calibration_steps < 2 or self.calibration_max <= self.calibration_min
        ):
            raise ValueError("scan needs calibration_max > calibration_min and calibration_steps >= 2.")

        seen_test_list, seen_test_label = resolve_seen_paths(
            test_list, test_label, seen_test_list, seen_test_label
        )
        language = torch.tensor(np.load(language_path), dtype=torch.float32)
        # Match CauASD test preprocessing: no extra normalization is added.
        self.full_language = language.to(self.device)
        self.unseen_label = list(unseen_label)
        unseen_set = set(self.unseen_label)
        self.seen_label = [idx for idx in range(self.full_language.size(0)) if idx not in unseen_set]
        if not self.seen_label or not self.unseen_label:
            raise ValueError("Both seen_label and unseen_label must be non-empty for GZSL.")

        self.unseen_loader = torch.utils.data.DataLoader(
            DataSet(test_list, test_label), batch_size=test_batch_size,
            num_workers=test_num_workers, shuffle=False,
        )
        self.seen_loader = torch.utils.data.DataLoader(
            DataSet(seen_test_list, seen_test_label), batch_size=test_batch_size,
            num_workers=test_num_workers, shuffle=False,
        )

        self.encoder = Model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting,
        ).to(self.device)
        self.adapter = Linear().to(self.device)
        checkpoint = load_checkpoint(checkpoint_path)
        self.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        self.adapter.load_state_dict(checkpoint["adapter"], strict=True)
        self.encoder.eval()
        self.adapter.eval()

        # Candidate label order exactly matches the candidate text feature order.
        self.all_label = self.seen_label + self.unseen_label
        self.all_language = self.full_language[self.all_label]
        self.all_label_tensor = torch.as_tensor(self.all_label, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def collect_scores(self, loader, partition_name):
        """Return CPU cosine-score matrix and global NTU labels for a split."""
        all_scores = []
        all_targets = []
        for data, label in tqdm(
            loader,
            desc="GZSL {} sf={:g}".format(partition_name, self.speed_factor),
        ):
            data = data.to(self.device, dtype=torch.float32)
            label = label.to(self.device, dtype=torch.long)
            feature = self.adapter(self.encoder(linear_time_scale(data, self.speed_factor)))
            # Explicit global-label prediction avoids depending on whether a
            # helper returns candidate indices or original NTU class IDs.
            logits = F.normalize(feature, dim=1) @ F.normalize(self.all_language, dim=1).T
            all_scores.append(logits.cpu())
            all_targets.append(label.cpu())
        if not all_scores:
            raise RuntimeError("{} GZSL loader is empty.".format(partition_name))
        return torch.cat(all_scores, dim=0), torch.cat(all_targets, dim=0)

    def metrics_at_gamma(self, scores, target, gamma):
        """Metrics after calibrated stacking: seen logits are reduced by gamma."""
        adjusted = scores.clone()
        adjusted[:, :len(self.seen_label)] -= float(gamma)
        predicted_index = adjusted.argmax(dim=1)
        prediction = self.all_label_tensor.cpu()[predicted_index]
        correct = (prediction == target).float().mean().item()
        routed_seen = (predicted_index < len(self.seen_label)).float().mean().item()
        return correct, routed_seen, 1.0 - routed_seen

    def gamma_candidates(self):
        if self.calibration_mode == "none":
            return [0.0]
        if self.calibration_mode == "fixed":
            return [self.calibration_gamma]
        return np.linspace(
            self.calibration_min, self.calibration_max, self.calibration_steps, dtype=np.float64
        ).tolist()

    def evaluate(self):
        unseen_scores, unseen_target = self.collect_scores(self.unseen_loader, "unseen")
        seen_scores, seen_target = self.collect_scores(self.seen_loader, "seen")

        records = []
        for gamma in self.gamma_candidates():
            unseen_acc, unseen_to_seen, unseen_to_unseen = self.metrics_at_gamma(
                unseen_scores, unseen_target, gamma
            )
            seen_acc, seen_to_seen, seen_to_unseen = self.metrics_at_gamma(
                seen_scores, seen_target, gamma
            )
            harmonic = 0.0 if seen_acc + unseen_acc == 0 else (
                2.0 * seen_acc * unseen_acc / (seen_acc + unseen_acc)
            )
            records.append({
                "gamma": gamma,
                "seen_acc": seen_acc,
                "unseen_acc": unseen_acc,
                "harmonic": harmonic,
                "unseen_to_seen": unseen_to_seen,
                "unseen_to_unseen": unseen_to_unseen,
                "seen_to_seen": seen_to_seen,
                "seen_to_unseen": seen_to_unseen,
            })

        best = max(records, key=lambda item: item["harmonic"])

        print("=" * 60)
        print("Linear temporal-index speed factor: {:g}".format(self.speed_factor))
        print("Calibration mode: {}".format(self.calibration_mode))
        if self.calibration_mode == "scan":
            print("WARNING: gamma selected on these GZSL test sets (paper-style/oracle calibration).")
        print("Selected seen-class gamma: {:.6f}".format(best["gamma"]))
        print("Seen samples: {}, accuracy S: {:.2f}%".format(seen_target.numel(), best["seen_acc"] * 100.0))
        print("Unseen samples: {}, accuracy U: {:.2f}%".format(unseen_target.numel(), best["unseen_acc"] * 100.0))
        print("Unseen inputs predicted as seen / unseen: {:.2f}% / {:.2f}%".format(
            best["unseen_to_seen"] * 100.0, best["unseen_to_unseen"] * 100.0
        ))
        print("Seen inputs predicted as seen / unseen: {:.2f}% / {:.2f}%".format(
            best["seen_to_seen"] * 100.0, best["seen_to_unseen"] * 100.0
        ))
        print("GZSL harmonic mean H: {:.2f}%".format(best["harmonic"] * 100.0))
        print("=" * 60)
        return best["seen_acc"], best["unseen_acc"], best["harmonic"]


@ex.automain
def main():
    setup_seed(0)
    Main2GUGZSLSpeedTester().evaluate()
