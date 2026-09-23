"""
Evaluate V-JEPA 2 (facebook/vjepa2-vitg-fpc64-384-ssv2) on the SSv2 validation set,
optionally constrained to a subset of classes (e.g. 10 classes).

When constrained (e.g. using data/template2id_mapping_only10classes.txt):
  - Model outputs full 174 logits.
  - Phase 2 extracts the relevant class logits corresponding to the target mapping.
  - Predictions and confidences are computed over the constrained class subspace.

Results are written to:
  - Parquet file containing all input metadata + inference predictions & confidences.
  - JSON file summarizing overall top-1/top-5 and balanced accuracy metrics.

Usage example:
  python scripts/vjepa_on_ssv2.py \
      --csv_path data/ssv2_val_filtered.csv \
      --video_dir /user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/data/public/ssv2/clips/ \
      --mapping_path data/template2id_mapping_only10classes.txt \
      --full_mapping_path data/template2id_mapping.txt \
      --results_dir results_ssv2_filtered \
      --batch_size 8 \
      --num_workers 16 \
      --bfloat16

"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVideoClassification, AutoVideoProcessor
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_template_mapping(mapping_path: str) -> tuple[dict[str, int], list[str]]:
    """
    Parse a template mapping file (tab or whitespace separated) into:
      - template_to_id: {template_string: class_id}
      - class_names: list of template strings indexed by class_id
    """
    template_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}

    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(" ", 1)
            if len(parts) == 2:
                class_id = int(parts[0].strip())
                template = parts[1].strip()
                template_to_id[template] = class_id
                id_to_name[class_id] = template

    num_classes = max(id_to_name.keys()) + 1 if id_to_name else 0
    class_names = [id_to_name.get(i, f"UNKNOWN_{i}") for i in range(num_classes)]
    return template_to_id, class_names


def sample_frame_indices(total_frames: int, num_frames: int = 64) -> np.ndarray:
    """
    Uniformly sample `num_frames` frame indices from a video with `total_frames`.
    If the video has fewer frames than requested, indices are repeated (with wrap-around).
    """
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=int)
    if total_frames <= num_frames:
        indices = np.arange(total_frames)
        indices = np.resize(indices, num_frames)  # tile-repeat
        return np.sort(indices)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


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

        if total == 0:
            raw_frames = []
            for frame in container.decode(video=0):
                raw_frames.append(frame.to_ndarray(format="rgb24"))
            container.close()
            if not raw_frames:
                return None
            total = len(raw_frames)
            indices = sample_frame_indices(total, num_frames)
            return np.stack([raw_frames[i] for i in indices], axis=0)

        indices = sample_frame_indices(total, num_frames)
        index_set = set(indices.tolist())
        raw_frames: dict[int, np.ndarray] = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in index_set:
                raw_frames[i] = frame.to_ndarray(format="rgb24")
            if i > indices[-1]:
                break
        container.close()

        frames = np.stack([raw_frames[i] for i in indices if i in raw_frames], axis=0)
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
    """Loads SSv2 videos and returns raw numpy frames + ground-truth label + metadata."""

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
        template = str(row["template"])
        label = self.template_to_id.get(template, -1)
        text_label = str(row["label"]) if "label" in row else ""
        placeholders = str(row["placeholders"]) if "placeholders" in row else ""
        split = str(row["split"]) if "split" in row else ""

        video_path = str(self.video_dir / f"{video_id}{self.video_ext}")
        frames = decode_video_frames(video_path, self.num_frames)

        return {
            "video_id": video_id,
            "text_label": text_label,
            "template": template,
            "placeholders": placeholders,
            "split": split,
            "label": label,
            "frames": frames,  # np.ndarray (T, H, W, C) or None
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
        "text_labels": [item["text_label"] for item in valid],
        "templates": [item["template"] for item in valid],
        "placeholders": [item["placeholders"] for item in valid],
        "splits": [item["split"] for item in valid],
        "labels": [item["label"] for item in valid],
        "frames_list": [item["frames"] for item in valid],  # list of (T, H, W, C)
        "video_paths": [item["video_path"] for item in valid],
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def balanced_top_k_accuracy(
    all_preds_top1: list[int],
    all_preds_top5: list[list[int]],
    all_labels: list[int],
    num_classes: int,
) -> tuple[float, float]:
    """
    Compute macro-averaged recall for top-1 and top-5 predictions.
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
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # ---- Load full 174 mapping ----
    print(f"Loading full mapping from: {args.full_mapping_path}")
    template_to_id_174, class_names_174 = load_template_mapping(args.full_mapping_path)
    print(f"  Loaded {len(template_to_id_174)} full class templates.")

    # ---- Load target mapping (e.g. 10 classes) ----
    print(f"Loading target mapping from: {args.mapping_path}")
    template_to_id, class_names = load_template_mapping(args.mapping_path)
    num_classes = len(class_names)
    print(f"  Loaded {len(template_to_id)} target class templates (subspace size = {num_classes}).")

    # Prepare extraction indices if target subspace is a subset of 174 classes
    target_indices_tensor = None
    if num_classes < len(class_names_174):
        target_indices_in_174 = [
            template_to_id_174[class_names[i]] for i in range(num_classes)
        ]
        target_indices_tensor = torch.tensor(target_indices_in_174, dtype=torch.long, device=device)
        print(f"  Constraining model logits: 174 -> {num_classes} classes.")
        for i, idx174 in enumerate(target_indices_in_174):
            print(f"    Target ID {i} -> Model Logit Index {idx174}: '{class_names[i]}'")

    # ---- Load CSV ----
    print(f"\nLoading CSV from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"  Total samples: {len(df)}")

    # Validate that all templates in CSV are present in target mapping
    unknown = df["template"][~df["template"].isin(template_to_id)].unique()
    if len(unknown) > 0:
        print(f"  WARNING: {len(unknown)} templates not found in target mapping — will get label -1.")
        for t in unknown[:5]:
            print(f"    '{t}'")

    # ---- Load model + processor ----
    print(f"\nLoading model: {args.model} (local_files_only=True)")
    t_load = time.time()
    processor = AutoVideoProcessor.from_pretrained(args.model, local_files_only=True)

    if device.type == "cuda" and args.bfloat16:
        model_dtype = torch.bfloat16
    elif device.type == "cuda" and args.fp16:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = AutoModelForVideoClassification.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    num_frames = getattr(model.config, "frames_per_clip", 64)
    print(f"  Model loaded in {time.time() - t_load:.1f}s (dtype={model_dtype}, frames_per_clip={num_frames})")

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
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=(args.num_workers > 0),
    )

    # ---- Inference accumulators ----
    all_video_ids: list[str] = []
    all_text_labels: list[str] = []
    all_templates: list[str] = []
    all_placeholders: list[str] = []
    all_splits: list[str] = []
    all_labels: list[int] = []
    all_top1_preds: list[int] = []
    all_top5_preds: list[list[int]] = []
    all_top1_confs: list[float] = []
    all_top1_correct: list[bool] = []
    all_top5_correct: list[bool] = []
    skipped = 0

    print(f"\nRunning inference on {len(dataset)} videos (batch_size={args.batch_size}, num_workers={args.num_workers})...\n")
    t0 = time.time()

    autocast_dtype = torch.bfloat16 if args.bfloat16 else (torch.float16 if args.fp16 else None)

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating SSv2", dynamic_ncols=True)):
        if not batch or "frames_list" not in batch:
            skipped += args.batch_size
            continue

        frames_list = batch["frames_list"]  # list of (T, H, W, C) np.ndarray
        labels = batch["labels"]

        # Filter out invalid labels (-1)
        valid_mask = [lbl >= 0 for lbl in labels]
        if not any(valid_mask):
            skipped += len(labels)
            continue

        valid_frames = [frames_list[i] for i, ok in enumerate(valid_mask) if ok]
        valid_labels = [labels[i] for i, ok in enumerate(valid_mask) if ok]
        valid_video_ids = [batch["video_ids"][i] for i, ok in enumerate(valid_mask) if ok]
        valid_text_labels = [batch["text_labels"][i] for i, ok in enumerate(valid_mask) if ok]
        valid_templates = [batch["templates"][i] for i, ok in enumerate(valid_mask) if ok]
        valid_placeholders = [batch["placeholders"][i] for i, ok in enumerate(valid_mask) if ok]
        valid_splits = [batch["splits"][i] for i, ok in enumerate(valid_mask) if ok]

        try:
            inputs = processor(valid_frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                if autocast_dtype is not None and device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=autocast_dtype):
                        outputs = model(**inputs)
                        logits = outputs.logits  # (B, 174)
                else:
                    outputs = model(**inputs)
                    logits = outputs.logits  # (B, 174)

                # Phase 2: Constrain predictions to the target class subspace
                if target_indices_tensor is not None:
                    logits = logits.index_select(dim=-1, index=target_indices_tensor)  # (B, num_classes)

                logits_f32 = logits.float().cpu()
                probs_f32 = F.softmax(logits_f32, dim=-1)

        except Exception as e:
            tqdm.write(f"  Batch {batch_idx}: inference error — {e}")
            skipped += len(valid_labels)
            continue

        # Collect predictions within the constrained subspace
        k = min(5, num_classes)
        topk_arr = logits_f32.topk(k, dim=-1).indices.numpy()  # (B, k)
        top1 = topk_arr[:, 0].tolist()
        top5 = topk_arr.tolist()
        top1_conf = probs_f32.numpy()[np.arange(len(top1)), top1].tolist()

        all_video_ids.extend(valid_video_ids)
        all_text_labels.extend(valid_text_labels)
        all_templates.extend(valid_templates)
        all_placeholders.extend(valid_placeholders)
        all_splits.extend(valid_splits)
        all_labels.extend(valid_labels)
        all_top1_preds.extend(top1)
        all_top5_preds.extend(top5)
        all_top1_confs.extend(top1_conf)
        all_top1_correct.extend([p == g for p, g in zip(top1, valid_labels)])
        all_top5_correct.extend([g in preds for g, preds in zip(valid_labels, top5)])

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
        all_top1_preds, all_top5_preds, all_labels, num_classes=num_classes
    )

    # ---- Save Parquet Results DataFrame ----
    os.makedirs(args.results_dir, exist_ok=True)
    model_slug = args.model.replace("/", "_")

    meta_df = pd.DataFrame({
        "id":           all_video_ids,
        "video_id":     all_video_ids,
        "text_label":   all_text_labels,
        "template":     all_templates,
        "placeholders": all_placeholders,
        "split":        all_splits,
        "label":        all_labels,
        "top1_pred":    all_top1_preds,
        "top1_class":   [class_names[i] for i in all_top1_preds],
        "top1_conf":    all_top1_confs,
        "top5_preds":   all_top5_preds,
        "top5_classes": [
            [class_names[j] for j in row] for row in all_top5_preds
        ],
        "correct_top1": all_top1_correct,
        "correct_top5": all_top5_correct,
    })

    if args.parquet_path:
        parquet_path = args.parquet_path
    else:
        parquet_path = os.path.join(args.results_dir, f"vjepa_ssv2_{model_slug}_meta.parquet")
    meta_df.to_parquet(parquet_path, index=False)
    print(f"\nParquet results saved to: {parquet_path}")

    # ---- Save JSON Results ----
    results = {
        "model": args.model,
        "csv_path": args.csv_path,
        "video_dir": args.video_dir,
        "mapping_path": args.mapping_path,
        "num_classes": num_classes,
        "total_evaluated": total_evaluated,
        "total_skipped": skipped,
        "top1_accuracy": round(top1_acc, 6),
        "top1_balanced_accuracy": round(ba_top1, 6),
        "top5_accuracy": round(top5_acc, 6),
        "top5_balanced_accuracy": round(ba_top5, 6),
        "elapsed_seconds": round(elapsed, 1),
        "parquet_path": parquet_path,
    }

    print("\n===== Results =====")
    for k, v in results.items():
        print(f"  {k}: {v}")

    out_path = os.path.join(args.results_dir, f"vjepa_ssv2_{model_slug}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V-JEPA 2 on SSv2 validation split (full or constrained classes).",
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
        default="data/template2id_mapping_only10classes.txt",
        help="Path to the target template-to-id mapping text file (e.g., 10 classes).",
    )
    parser.add_argument(
        "--full_mapping_path",
        type=str,
        default="data/template2id_mapping.txt",
        help="Path to full 174-class template-to-id mapping text file.",
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
        help="Directory where results files will be saved.",
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        default=None,
        help="Optional custom path to save the predictions parquet file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of videos per inference batch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
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
        "--bfloat16",
        action="store_true",
        default=True,
        help="Use bfloat16 for model weights (CUDA only, recommended for A100).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use float16 for model weights (CUDA only, alternative to bfloat16).",
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
