"""
Deep learning utilities for player action prediction from Atari gameplay frames.
"""
import io
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_sessions(
    zip_path: Path,
    img_size: int = 84,
    ale_to_class: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict]]:
    """Load all Atari-HEAD sessions from a zip archive into CPU RAM.

    Parses each session's label file to obtain frames in game-sequence order,
    extracts and resizes each frame from the paired tar.bz2 archive, and maps
    ALE action codes to simplified class labels. All frames are stored as uint8
    to minimise memory usage; normalisation to float32 is deferred to the Dataset.

    Parameters
    ----------
    zip_path : Path
        Path to the game zip file (e.g. montezuma_revenge.zip).
    img_size : int
        Target width and height in pixels after resizing. Default 84.
    ale_to_class : dict, optional
        Mapping from ALE action code (int) to class label (int).
        Unknown codes default to class 0.

    Returns
    -------
    frames : np.ndarray, shape (N, img_size, img_size), dtype uint8
    labels : np.ndarray, shape (N,), dtype int64
    session_lengths : list[int]
        Number of valid frames per session, in load order.
    session_meta : list[dict]
        Per-session metadata: keys 'name', 'type' ('standard'|'highscore'), 'n_frames'.
    """
    if ale_to_class is None:
        ale_to_class = {}

    all_frames: list[np.ndarray] = []
    all_labels: list[int] = []
    session_lengths: list[int] = []
    session_meta: list[dict] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = zf.namelist()
        txt_names = sorted(e for e in entries if e.endswith(".txt") and "meta" not in e.lower())
        tar_names = sorted(e for e in entries if e.endswith(".tar.bz2"))

        for txt_name, tar_name in zip(txt_names, tar_names):
            session_name = Path(tar_name).stem.replace(".tar", "")
            session_type = "highscore" if "highscore" in tar_name else "standard"
            print(f"  {session_type:<10} {session_name} ...", end=" ", flush=True)

            # Parse label file in game-sequence order
            frame_order: list[tuple[str, int]] = []
            with zf.open(txt_name) as raw:
                raw.readline()  # skip header
                for line in raw:
                    parts = line.decode("utf-8", errors="replace").strip().split(",")
                    if len(parts) < 6:
                        continue
                    frame_id = parts[0].strip()
                    try:
                        ale_action = int(parts[5].strip())
                    except ValueError:
                        continue
                    frame_order.append((frame_id, ale_action))

            target_ids = {fid for fid, _ in frame_order}

            # Load tar.bz2 archive into memory once, then extract needed frames
            with zf.open(tar_name) as raw:
                tar_bytes = io.BytesIO(raw.read())

            frames_dict: dict[str, np.ndarray] = {}
            with tarfile.open(fileobj=tar_bytes, mode="r:bz2") as tar:
                for member in tar:
                    if member.isdir():
                        continue
                    stem = Path(member.name).stem
                    if stem not in target_ids:
                        continue
                    try:
                        raw_img = tar.extractfile(member)
                        img = Image.open(raw_img).convert("L")
                        img = img.resize((img_size, img_size), Image.BILINEAR)
                        frames_dict[stem] = np.array(img, dtype=np.uint8)
                    except Exception:
                        continue

            # Assemble in game-sequence order
            sess_frames: list[np.ndarray] = []
            sess_labels: list[int] = []
            for fid, ale in frame_order:
                if fid not in frames_dict:
                    continue
                sess_frames.append(frames_dict[fid])
                sess_labels.append(ale_to_class.get(ale, 0))

            all_frames.extend(sess_frames)
            all_labels.extend(sess_labels)
            session_lengths.append(len(sess_frames))
            session_meta.append({
                "name": session_name,
                "type": session_type,
                "n_frames": len(sess_frames),
            })
            print(f"{len(sess_frames):,} frames")

    frames = np.array(all_frames, dtype=np.uint8)
    labels = np.array(all_labels, dtype=np.int64)
    return frames, labels, session_lengths, session_meta


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class GameFrameDataset(Dataset):
    """Single-frame dataset for baseline action prediction.

    Parameters
    ----------
    frames : np.ndarray, shape (N, H, W), uint8
    labels : np.ndarray, shape (N,), int
    """

    def __init__(self, frames: np.ndarray, labels: np.ndarray):
        self.frames = frames  # kept as uint8; conversion deferred to __getitem__
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame = torch.from_numpy(self.frames[idx]).unsqueeze(0).float() / 255.0
        return frame, self.labels[idx]


class SessionAwareStackedDataset(Dataset):
    """Four-frame stacked dataset that respects session boundaries.

    Consecutive frames are stacked along the channel dimension to provide
    temporal context. Stacks are never built across session boundaries,
    preventing the model from seeing a frame from the end of one session
    concatenated with frames from the start of the next.

    Parameters
    ----------
    frames : np.ndarray, shape (N, H, W), uint8
    labels : np.ndarray, shape (N,), int
    session_lengths : list[int]
        Number of frames per session in the same order as the concatenated arrays.
    stack_size : int
        Number of consecutive frames to stack. Default 4.
    """

    def __init__(
        self,
        frames: np.ndarray,
        labels: np.ndarray,
        session_lengths: list[int],
        stack_size: int = 4,
    ):
        self.frames = frames  # kept as uint8; conversion deferred to __getitem__
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.k = stack_size

        # Build index of valid sample start positions within each session
        valid: list[int] = []
        pos = 0
        for length in session_lengths:
            for i in range(pos, pos + length - stack_size + 1):
                valid.append(i)
            pos += length
        self.valid_indices = np.array(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.valid_indices[idx])
        stack = torch.from_numpy(self.frames[start: start + self.k]).float() / 255.0  # (k, H, W)
        label = self.labels[start + self.k - 1]
        return stack, label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ActionCNN(nn.Module):
    """DQN-inspired CNN for player action prediction from Atari frames.

    Architecture: three convolutional layers followed by two fully connected
    layers. The flattened feature size after the conv stack is computed
    dynamically, so the same class works for any input size or channel count.

    Parameters
    ----------
    in_channels : int
        1 for single-frame baseline; 4 for four-frame stacked experimental model.
    num_classes : int
        Number of action classes to predict.
    img_size : int
        Spatial input dimension (frames are assumed square). Default 84.
    """

    def __init__(self, in_channels: int, num_classes: int, img_size: int = 84):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            flat_size = self.features(dummy).view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: conv features then classification head."""
        return self.classifier(self.features(x).view(x.size(0), -1))


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return (mean loss, accuracy).

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    optimizer : torch.optim.Optimizer
    criterion : nn.Module
        Loss function (e.g. weighted CrossEntropyLoss).
    device : torch.device

    Returns
    -------
    tuple[float, float]
        Mean loss per sample and accuracy over the epoch.
    """
    model.train()
    total_loss = correct = total = 0
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(frames)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (out.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a DataLoader and return (mean loss, accuracy).

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : nn.Module
    device : torch.device

    Returns
    -------
    tuple[float, float]
        Mean loss per sample and accuracy.
    """
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            out = model(frames)
            loss = criterion(out, labels)
            total_loss += loss.item() * len(labels)
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_names: list[str],
) -> dict:
    """Run full evaluation and return a metrics dictionary.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    loader : DataLoader
        Test set DataLoader.
    device : torch.device
    action_names : list[str]
        Human-readable class names in label order.

    Returns
    -------
    dict
        Keys: accuracy, f1_macro, precision_per_class, recall_per_class,
        f1_per_class, confusion_matrix, y_pred, y_true, report.
    """
    model.eval()
    all_preds: list[int] = []
    all_true: list[int] = []
    with torch.no_grad():
        for frames, labels in loader:
            preds = model(frames.to(device)).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(labels.tolist())

    n = len(action_names)
    p, r, f1, _ = precision_recall_fscore_support(
        all_true, all_preds, labels=list(range(n)), zero_division=0
    )
    return {
        "accuracy":            accuracy_score(all_true, all_preds),
        "f1_macro":            f1_score(all_true, all_preds, average="macro", zero_division=0),
        "precision_per_class": p,
        "recall_per_class":    r,
        "f1_per_class":        f1,
        "confusion_matrix":    confusion_matrix(all_true, all_preds, labels=list(range(n))),
        "y_pred":              np.array(all_preds),
        "y_true":              np.array(all_true),
        "report":              classification_report(all_true, all_preds, target_names=action_names, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Save model state dict to disk.

    Parameters
    ----------
    model : nn.Module
    path : Path
        Destination file path (parent directories are created if needed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device = None) -> nn.Module:
    """Load model state dict from disk.

    Parameters
    ----------
    model : nn.Module
        Model instance with matching architecture.
    path : Path
    device : torch.device, optional
        Target device for map_location. Defaults to CPU.

    Returns
    -------
    nn.Module
        Model with loaded weights.
    """
    map_loc = device if device is not None else torch.device("cpu")
    model.load_state_dict(torch.load(path, map_location=map_loc))
    return model
