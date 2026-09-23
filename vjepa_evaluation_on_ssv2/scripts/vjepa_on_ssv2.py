"""
Evaluate V-JEPA 2 (facebook/vjepa2-vitg-fpc64-384-ssv2) on the SSv2 validation set.

Computes:
  - top1_accuracy
  - top1_balanced_accuracy  (macro-averaged recall across 174 classes)
  - top5_accuracy
  - top5_balanced_accuracy

Results are written as a JSON file under the --results_dir directory.

Usage example:
  python scripts/vjepa_on_ssv2.py \
      --csv_path data/ssv2_val_filtered.csv \
      --video_dir /path/to/ssv2/videos \
      --mapping_path data/template2id_mapping.txt \
      --results_dir results \
      --batch_size 4 \
      --num_workers 4 \
      --device cuda
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVideoClassification, AutoVideoProcessor
from sklearn.metrics import balanced_accuracy_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_template_mapping(mapping_path: str) -> dict[str, int]:
    """Parse template2id_mapping.txt into {template_string: class_id}."""
    mapping = {}
    with open(mapping_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                class_id = int(parts[0])
                template = parts[1].strip()
                mapping[template] = class_id
    return mapping


def sample_frame_indices(total_frames: int, num_frames: int = 64) -> np.ndarray:
    """
    Uniformly sample `num_frames` frame indices from a video with `total_frames`.
    If the video has fewer frames than requested, indices are repeated (with wrap-around).
    """
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=int)
    if total_frames <= num_frames:
        # Repeat frames if the video is too short
        indices = np.arange(total_frames)
        indices = np.resize(indices, num_frames)  # tile-repeat
        return np.sort(indices)
    # Uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    return indices


def decode_video_frames(video_path: str, num_frames: int = 64) -> np.ndarray | None:
    """
    Decode `num_frames` uniformly sampled frames from `video_path`.

    Returns:
        np.ndarray of shape (T, H, W, C) uint8, or None on failure.

    Tries backends in order: torchcodec → decord → PyAV.
    """
    # --- Backend 1: torchcodec ---
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: PLC0415

        vd = VideoDecoder(video_path)
        total = len(vd)
        indices = sample_frame_indices(total, num_frames)
        frames_tensor = vd.get_frames_at(indices=indices.tolist()).data  # (T, C, H, W) uint8
        # Convert to (T, H, W, C)
        frames = frames_tensor.permute(0, 2, 3, 1).numpy()
        return frames
    except Exception:
        pass

    # --- Backend 2: decord ---
    try:
        import decord  # noqa: PLC0415
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path, num_threads=1)
        total = len(vr)
        indices = sample_frame_indices(total, num_frames)
        frames = vr.get_batch(indices.tolist()).asnumpy()  # (T, H, W, C) uint8
        return frames
    except Exception:
        pass

    # --- Backend 3: PyAV ---
    try:
        import av  # noqa: PLC0415

        container = av.open(video_path)
        stream = container.streams.video[0]
        total = stream.frames if stream.frames else 0

        # If total unknown, decode all and sample afterwards
        if total == 0:
            raw_frames = []
            for frame in container.decode(video=0):
                raw_frames.append(frame.to_ndarray(format="rgb24"))
            container.close()
            if not raw_frames:
                return None
            total = len(raw_frames)
            indices = sample_frame_indices(total, num_frames)
            frames = np.stack([raw_frames[i] for i in indices], axis=0)
            return frames

        indices = sample_frame_indices(total, num_frames)
        index_set = set(indices.tolist())
        raw_frames = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in index_set:
                raw_frames[i] = frame.to_ndarray(format="rgb24")
            if i > indices[-1]:
                break
        container.close()

        frames = np.stack([raw_frames[i] for i in indices if i in raw_frames], axis=0)
        # Pad if not enough frames
        if len(frames) < num_frames:
            pad = np.repeat(frames[-1:], num_frames - len(frames), axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        return frames
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SSv2VideoDataset(Dataset):
    """Loads SSv2 videos and returns raw numpy frames + ground-truth label."""

    def __init__(
        self,
        df: pd.DataFrame,
        video_dir: str,
        template_to_id: dict[str, int],
        num_frames: int = 64,
        video_ext: str = ".webm",
    ):
        self.df = df.reset_index(drop=True)
        self.video_dir = Path(video_dir)
        self.template_to_id = template_to_id
        self.num_frames = num_frames
        self.video_ext = video_ext

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        video_id = str(row["id"])
        template = row["template"]
        label = self.template_to_id.get(template, -1)

        video_path = str(self.video_dir / f"{video_id}{self.video_ext}")
        frames = decode_video_frames(video_path, self.num_frames)

        return {
            "video_id": video_id,
            "frames": frames,  # np.ndarray (T, H, W, C) or None
            "label": label,
            "video_path": video_path,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate a list of dataset items.
    Filters out samples whose frames failed to decode.
    Returns raw list of frames arrays so the processor handles resizing/normalization.
    """
    valid = [item for item in batch if item["frames"] is not None]
    if not valid:
        return {}
    return {
        "video_ids": [item["video_id"] for item in valid],
        "frames_list": [item["frames"] for item in valid],  # list of (T, H, W, C)
        "labels": [item["label"] for item in valid],
        "video_paths": [item["video_path"] for item in valid],
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def top_k_correct(logits: torch.Tensor, labels: list[int], k: int) -> list[bool]:
    """Return a boolean list: True if ground-truth is in the top-k predictions."""
    topk_indices = logits.topk(k, dim=-1).indices  # (B, k)
    correct = []
    for i, gt in enumerate(labels):
        correct.append(gt in topk_indices[i].tolist())
    return correct


def balanced_top_k_accuracy(
    all_preds_top1: list[int],
    all_preds_top5: list[list[int]],
    all_labels: list[int],
    num_classes: int,
) -> tuple[float, float]:
    """
    Compute macro-averaged recall for top-1 and top-5 predictions.

    For top-1: standard balanced accuracy.
    For top-5: a sample is "correct" if the true label is among the top-5;
               then balanced accuracy is computed using top-1 prediction from
               the top-5 set (i.e., the best possible label within top-5 that
               matches; we treat it as: predicted = true label if in top-5,
               else predicted = top-1 prediction).
    """
    labels_arr = np.array(all_labels)

    # Top-1 balanced accuracy
    preds_top1_arr = np.array(all_preds_top1)
    ba_top1 = balanced_accuracy_score(labels_arr, preds_top1_arr)

    # Top-5 balanced accuracy: re-map predictions so that if true label is
    # in top-5 we credit it as correct (i.e., set pred = true label).
    preds_top5_adjusted = []
    for i, gt in enumerate(all_labels):
        if gt in all_preds_top5[i]:
            preds_top5_adjusted.append(gt)
        else:
            preds_top5_adjusted.append(all_preds_top5[i][0])  # top-1 prediction
    preds_top5_arr = np.array(preds_top5_adjusted)
    ba_top5 = balanced_accuracy_score(labels_arr, preds_top5_arr)

    return ba_top1, ba_top5


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    # ---- Setup ----
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    # ---- Load mapping ----
    print(f"Loading template-to-id mapping from: {args.mapping_path}")
    template_to_id = load_template_mapping(args.mapping_path)
    print(f"  Loaded {len(template_to_id)} class templates.")

    # ---- Load CSV ----
    print(f"Loading CSV from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"  Total samples: {len(df)}")

    # Validate that all templates are in the mapping
    unknown = df["template"][~df["template"].isin(template_to_id)].unique()
    if len(unknown) > 0:
        print(f"  WARNING: {len(unknown)} templates not found in mapping — they will get label -1.")
        for t in unknown[:5]:
            print(f"    '{t}'")

    # ---- Load model + processor ----
    print(f"Loading model: {args.model}")
    processor = AutoVideoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForVideoClassification.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if (device.type == "cuda" and args.fp16) else torch.float32,
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    num_frames = getattr(model.config, "frames_per_clip", 64)
    print(f"  Model frames_per_clip: {num_frames}")

    # ---- Dataset / DataLoader ----
    dataset = SSv2VideoDataset(
        df=df,
        video_dir=args.video_dir,
        template_to_id=template_to_id,
        num_frames=num_frames,
        video_ext=args.video_ext,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Inference ----
    all_labels: list[int] = []
    all_top1_preds: list[int] = []
    all_top5_preds: list[list[int]] = []
    all_top1_correct: list[bool] = []
    all_top5_correct: list[bool] = []
    skipped = 0

    print(f"\nRunning inference on {len(dataset)} videos (batch_size={args.batch_size})...")
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        if not batch:
            skipped += args.batch_size
            continue

        frames_list = batch["frames_list"]  # list of (T, H, W, C) np.ndarray
        labels = batch["labels"]

        # Filter out invalid labels
        valid_mask = [lbl >= 0 for lbl in labels]
        if not any(valid_mask):
            skipped += len(labels)
            continue

        valid_frames = [frames_list[i] for i, ok in enumerate(valid_mask) if ok]
        valid_labels = [labels[i] for i, ok in enumerate(valid_mask) if ok]

        # Pre-process: the HF processor expects a list of video arrays.
        # Each array should be (T, H, W, C) uint8.
        try:
            inputs = processor(valid_frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # (B, num_classes)  float
        except Exception as e:
            print(f"  Batch {batch_idx}: inference error — {e}")
            skipped += len(valid_labels)
            continue

        # Collect top-1 and top-5 predictions
        top5 = logits.topk(5, dim=-1).indices.cpu().tolist()  # (B, 5)
        top1 = [t[0] for t in top5]

        all_labels.extend(valid_labels)
        all_top1_preds.extend(top1)
        all_top5_preds.extend(top5)
        all_top1_correct.extend([p == g for p, g in zip(top1, valid_labels)])
        all_top5_correct.extend([g in preds for g, preds in zip(valid_labels, top5)])

        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            elapsed = time.time() - t0
            done = len(all_labels)
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"  [{batch_idx + 1}/{len(loader)}] processed={done}, "
                f"skipped={skipped}, speed={rate:.1f} vid/s"
            )

    elapsed = time.time() - t0
    total_evaluated = len(all_labels)
    print(f"\nDone in {elapsed:.1f}s. Evaluated={total_evaluated}, Skipped={skipped}.")

    if total_evaluated == 0:
        print("ERROR: No samples were successfully evaluated. Check video paths and decoder.")
        return

    # ---- Compute metrics ----
    top1_acc = sum(all_top1_correct) / total_evaluated
    top5_acc = sum(all_top5_correct) / total_evaluated
    ba_top1, ba_top5 = balanced_top_k_accuracy(
        all_top1_preds, all_top5_preds, all_labels, num_classes=174
    )

    results = {
        "model": args.model,
        "csv_path": args.csv_path,
        "video_dir": args.video_dir,
        "total_evaluated": total_evaluated,
        "total_skipped": skipped,
        "top1_accuracy": round(top1_acc, 6),
        "top1_balanced_accuracy": round(ba_top1, 6),
        "top5_accuracy": round(top5_acc, 6),
        "top5_balanced_accuracy": round(ba_top5, 6),
        "elapsed_seconds": round(elapsed, 1),
    }

    print("\n===== Results =====")
    for k, v in results.items():
        print(f"  {k}: {v}")

    # ---- Save results ----
    os.makedirs(args.results_dir, exist_ok=True)
    model_slug = args.model.replace("/", "_")
    out_path = os.path.join(args.results_dir, f"vjepa_ssv2_{model_slug}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V-JEPA 2 on SSv2 validation split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/ssv2_val_filtered.csv",
        help="Path to the SSv2 validation CSV file.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing SSv2 video files (e.g. /path/to/ssv2/videos).",
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default="data/template2id_mapping.txt",
        help="Path to the template-to-id mapping text file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/vjepa2-vitg-fpc64-384-ssv2",
        help="HuggingFace model repo (must be a VideoClassification checkpoint).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where the JSON results file will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of videos per inference batch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes for video decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use float16 for model weights (CUDA only, reduces VRAM).",
    )
    parser.add_argument(
        "--video_ext",
        type=str,
        default=".webm",
        help="Video file extension (e.g. .webm, .mp4).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
