"""
Run V-JEPA 2 (facebook/vjepa2-vitg-fpc64-384-ssv2) inference on custom .avi videos,
optionally constrained to a subset of classes (e.g. 10 classes).

Videos are indexed by a CSV with columns: id, template, placeholders, session, cam.
For each row, the video file is located at:
    {video_dir}/{id}_{cam}.avi
  e.g. /path/to/videos/260922-134214_cam0.avi

When constrained (e.g. using data/template2id_mapping_only10classes.txt):
  - Model outputs full 174 logits.
  - Phase 2 extracts the relevant class logits corresponding to the target mapping.
  - Softmax probabilities and predictions are computed over the constrained class subspace.

Outputs (saved to --output_dir):
  - logits_output.npz          : compressed NumPy archive with all logits + metadata
  - logits_output_meta.parquet : tabular metadata (video_id, cam, session, template, label,
                                  top1_pred, top1_class, top1_conf, top5_preds, etc.)
  - skipped_videos.txt         : list of videos that failed to decode (if any)
  - logits_manifest.json       : run manifest with configuration and summary metrics

The .npz format stores:
  logits      (N, num_classes) float32  — logits for the target classes
  probs       (N, num_classes) float32  — softmax probabilities
  labels      (N,)            int16    — ground-truth class IDs (-1 if template unknown)
  video_ids   (N,)            object   — recording IDs (e.g. "260922-134214")
  cams        (N,)            object   — camera views (e.g. "cam0")
  templates   (N,)            object   — SSv2 template string
  class_names (num_classes,)  object   — ordered class name strings (index = class ID)

Usage:
  python scripts/vjepa_on_myvideos.py \
      --csv_path           data/index_df.csv \
      --video_dir          /user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/data/hamza_vgd_data/raw \
      --mapping_path       data/template2id_mapping_only10classes.txt \
      --full_mapping_path  data/template2id_mapping.txt \
      --output_dir         results_myvideos \
      --batch_size         8 \
      --num_workers        16


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
from tqdm import tqdm
from transformers import AutoModelForVideoClassification, AutoVideoProcessor
from sklearn.metrics import balanced_accuracy_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_template_mapping(mapping_path: str) -> tuple[dict[str, int], list[str]]:
    """
    Parse a template mapping file (tab or whitespace separated) into:
      - template_to_id: {template_string: class_id}
      - class_names: list of template strings indexed by class_id
    """
    mapping: dict[str, int] = {}
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
                mapping[template] = class_id
                id_to_name[class_id] = template

    num_classes = max(id_to_name.keys()) + 1 if id_to_name else 0
    class_names = [id_to_name.get(i, f"UNKNOWN_{i}") for i in range(num_classes)]
    return mapping, class_names


def sample_frame_indices(total_frames: int, num_frames: int = 64) -> np.ndarray:
    """
    Uniformly sample `num_frames` indices from a video with `total_frames`.
    If the video has fewer frames than requested, indices are tiled (wrap-around).
    """
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=int)
    if total_frames <= num_frames:
        indices = np.arange(total_frames)
        indices = np.resize(indices, num_frames)  # tile-repeat to fill
        return np.sort(indices)
    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


def decode_video_frames(video_path: str, num_frames: int = 64) -> np.ndarray | None:
    """
    Decode `num_frames` uniformly sampled frames from `video_path`.

    Returns:
        np.ndarray of shape (T, H, W, C) uint8, or None on failure.

    Tries backends in order: decord → PyAV.
    """
    # --- Backend 1: decord (fast, recommended) ---
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

    # --- Backend 2: PyAV (fallback) ---
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
# Dataset
# ---------------------------------------------------------------------------

class MyVideosDataset(Dataset):
    """
    PyTorch Dataset for custom .avi videos indexed by index_df.csv.

    Each CSV row is one video:  {video_dir}/{id}_{cam}.avi
    """

    def __init__(
        self,
        df: pd.DataFrame,
        video_dir: str,
        template_to_id: dict[str, int],
        num_frames: int = 64,
    ):
        self.df = df.reset_index(drop=True)
        self.video_dir = Path(video_dir)
        self.template_to_id = template_to_id
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        video_id = str(row["id"])
        cam = str(row["cam"])
        template = str(row["template"])
        session = str(row["session"]) if "session" in row else ""
        label = self.template_to_id.get(template, -1)

        video_path = self.video_dir / f"{video_id}_{cam}.avi"
        frames = decode_video_frames(str(video_path), self.num_frames)

        return {
            "video_id":   video_id,
            "cam":        cam,
            "session":    session,
            "template":   template,
            "label":      label,
            "frames":     frames,        # (T, H, W, C) uint8 or None if decode failed
            "video_path": str(video_path),
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate a list of dataset items.
    Items where `frames` is None (decode failure) are filtered out and reported.
    Returns raw frame arrays; the HF processor handles resize + normalization.
    """
    valid = [item for item in batch if item["frames"] is not None]
    failed = [item["video_path"] for item in batch if item["frames"] is None]

    if not valid:
        return {"failed_paths": failed}

    return {
        "video_ids":    [item["video_id"]   for item in valid],
        "cams":         [item["cam"]        for item in valid],
        "sessions":     [item["session"]    for item in valid],
        "templates":    [item["template"]   for item in valid],
        "labels":       [item["label"]      for item in valid],
        "frames_list":  [item["frames"]     for item in valid],  # list of (T, H, W, C)
        "failed_paths": failed,
    }


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:

    # ---- Device ----
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
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
    print(f"  Loaded {len(template_to_id)} target templates (subspace size = {num_classes}).")

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

    # ---- CSV ----
    print(f"\nLoading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"  Rows: {len(df)}  |  Unique IDs: {df['id'].nunique()}  |  Cams: {df['cam'].unique().tolist()}")

    # Warn about unknown templates
    unknown_templates = df["template"][~df["template"].isin(template_to_id)].unique()
    if len(unknown_templates) > 0:
        print(f"  WARNING: {len(unknown_templates)} template(s) not found in mapping → label will be -1.")
        for t in unknown_templates[:5]:
            print(f"    '{t}'")

    # ---- Model + processor ----
    print(f"\nLoading model: {args.model}  (local_files_only=True)")
    t_load = time.time()
    processor = AutoVideoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForVideoClassification.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    print(f"  Model loaded in {time.time() - t_load:.1f}s")

    # Honour model's native frames_per_clip if different from user flag
    model_fpc = getattr(model.config, "frames_per_clip", args.num_frames)
    if model_fpc != args.num_frames:
        print(f"  NOTE: model.config.frames_per_clip={model_fpc} differs from "
              f"--num_frames={args.num_frames}. Using model value: {model_fpc}.")
    num_frames = model_fpc

    # ---- Dataset / DataLoader ----
    dataset = MyVideosDataset(
        df=df,
        video_dir=args.video_dir,
        template_to_id=template_to_id,
        num_frames=num_frames,
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

    # ---- Accumulators ----
    all_video_ids: list[str]        = []
    all_cams:      list[str]        = []
    all_sessions:  list[str]        = []
    all_templates: list[str]        = []
    all_labels:    list[int]        = []
    all_logits:    list[np.ndarray] = []   # each: (B, num_classes) float32
    all_probs:     list[np.ndarray] = []   # each: (B, num_classes) float32
    failed_paths:  list[str]        = []

    # ---- Inference loop ----
    print(f"\nRunning inference on {len(dataset)} videos "
          f"(batch_size={args.batch_size}, num_workers={args.num_workers})...\n")
    t0 = time.time()

    for batch in tqdm(loader, desc="Inferring", unit="batch", dynamic_ncols=True):

        # Collect any decode failures from this batch
        if batch.get("failed_paths"):
            failed_paths.extend(batch["failed_paths"])

        # Skip fully-empty batches
        if "frames_list" not in batch:
            continue

        frames_list = batch["frames_list"]   # list of (T, H, W, C) np.ndarray
        labels      = batch["labels"]
        video_ids   = batch["video_ids"]
        cams        = batch["cams"]
        sessions    = batch["sessions"]
        templates   = batch["templates"]

        # Filter out invalid labels (-1) if any
        valid_mask = [lbl >= 0 for lbl in labels]
        if not any(valid_mask):
            continue

        valid_frames    = [frames_list[i] for i, ok in enumerate(valid_mask) if ok]
        valid_labels    = [labels[i]      for i, ok in enumerate(valid_mask) if ok]
        valid_video_ids = [video_ids[i]   for i, ok in enumerate(valid_mask) if ok]
        valid_cams      = [cams[i]        for i, ok in enumerate(valid_mask) if ok]
        valid_sessions  = [sessions[i]    for i, ok in enumerate(valid_mask) if ok]
        valid_templates = [templates[i]   for i, ok in enumerate(valid_mask) if ok]

        try:
            # AutoVideoProcessor: center-crop to 384×384 + SSv2 normalization
            inputs = processor(valid_frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**inputs)
                logits = outputs.logits  # (B, 174) bfloat16

            # Phase 2: Constrain predictions to the target class subspace
            if target_indices_tensor is not None:
                logits = logits.index_select(dim=-1, index=target_indices_tensor)  # (B, num_classes)

            # Cast to float32 on CPU before storing
            logits_f32 = logits.float().cpu()
            probs_f32  = F.softmax(logits_f32, dim=-1)

            all_logits.append(logits_f32.numpy())
            all_probs.append(probs_f32.numpy())
            all_video_ids.extend(valid_video_ids)
            all_cams.extend(valid_cams)
            all_sessions.extend(valid_sessions)
            all_templates.extend(valid_templates)
            all_labels.extend(valid_labels)

        except Exception as e:
            tqdm.write(f"  [WARN] Batch inference error: {e}")
            tqdm.write(f"         Skipping {len(valid_video_ids)} sample(s): {valid_video_ids}")
            failed_paths.extend(
                [f"{vid}_{cam}" for vid, cam in zip(valid_video_ids, valid_cams)]
            )

    elapsed = time.time() - t0
    n_ok = len(all_labels)
    n_fail = len(failed_paths)

    # ---- Stack arrays ----
    logits_arr    = np.concatenate(all_logits, axis=0).astype(np.float32)   # (N, num_classes)
    probs_arr     = np.concatenate(all_probs,  axis=0).astype(np.float32)   # (N, num_classes)
    labels_arr    = np.array(all_labels,    dtype=np.int16)                  # (N,)
    video_ids_arr = np.array(all_video_ids, dtype=object)                    # (N,)
    cams_arr      = np.array(all_cams,      dtype=object)                    # (N,)
    sessions_arr  = np.array(all_sessions,  dtype=object)                    # (N,)
    templates_arr = np.array(all_templates, dtype=object)                    # (N,)
    class_names_arr = np.array(class_names, dtype=object)                    # (num_classes,)

    # ---- Save outputs ----
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. NPZ — primary output (fast load, structured)
    npz_path = os.path.join(args.output_dir, "logits_output.npz")
    np.savez_compressed(
        npz_path,
        logits      = logits_arr,
        probs       = probs_arr,
        labels      = labels_arr,
        video_ids   = video_ids_arr,
        cams        = cams_arr,
        sessions    = sessions_arr,
        templates   = templates_arr,
        class_names = class_names_arr,
    )
    print(f"\nNPZ saved: {npz_path}  (shape: {logits_arr.shape})")

    # 2. Parquet — metadata table for easy per-class / per-cam analysis
    k_top = min(5, num_classes)
    top5_indices = np.argsort(logits_arr, axis=1)[:, -k_top:][:, ::-1]   # (N, k_top) descending
    top1_indices = top5_indices[:, 0]                                    # (N,)
    top1_conf    = probs_arr[np.arange(n_ok), top1_indices]             # (N,) float32

    meta_df = pd.DataFrame({
        "video_id":    all_video_ids,
        "cam":         all_cams,
        "session":     all_sessions,
        "template":    all_templates,
        "label":       all_labels,
        "top1_pred":   top1_indices.tolist(),
        "top1_class":  [class_names[i] for i in top1_indices],
        "top1_conf":   top1_conf.tolist(),
        "top5_preds":  top5_indices.tolist(),
        "top5_classes": [
            [class_names[j] for j in row] for row in top5_indices
        ],
        "correct_top1": (top1_indices == labels_arr).tolist(),
        "correct_top5": [
            int(labels_arr[i]) in top5_indices[i].tolist()
            for i in range(n_ok)
        ],
    })
    parquet_path = os.path.join(args.output_dir, "logits_output_meta.parquet")
    meta_df.to_parquet(parquet_path, index=False)
    print(f"Parquet saved: {parquet_path}")

    # 3. Skipped videos log
    if failed_paths:
        skip_path = os.path.join(args.output_dir, "skipped_videos.txt")
        with open(skip_path, "w") as f:
            f.write("\n".join(failed_paths) + "\n")
        print(f"Skipped log: {skip_path}  ({n_fail} entries)")

    # ---- Compute accuracy metrics ----
    top1_acc = (top1_indices == labels_arr).mean() if n_ok > 0 else 0.0
    top5_acc = np.mean([
        int(labels_arr[i]) in top5_indices[i].tolist() for i in range(n_ok)
    ]) if n_ok > 0 else 0.0
    ba_top1, ba_top5 = balanced_top_k_accuracy(
        top1_indices.tolist(), top5_indices.tolist(), all_labels, num_classes=num_classes
    )

    # 4. Run manifest JSON (metadata for reproducibility)
    manifest = {
        "model":                  args.model,
        "csv_path":               args.csv_path,
        "video_dir":              args.video_dir,
        "mapping_path":           args.mapping_path,
        "full_mapping_path":      args.full_mapping_path,
        "output_dir":             args.output_dir,
        "num_frames":             int(num_frames),
        "batch_size":             args.batch_size,
        "num_workers":            args.num_workers,
        "dtype":                  "bfloat16 (autocast)",
        "n_total_csv":            len(df),
        "n_evaluated":            n_ok,
        "n_skipped":              n_fail,
        "logits_shape":           list(logits_arr.shape),
        "num_classes":            num_classes,
        "top1_accuracy":          round(float(top1_acc), 6),
        "top1_balanced_accuracy": round(float(ba_top1), 6),
        "top5_accuracy":          round(float(top5_acc), 6),
        "top5_balanced_accuracy": round(float(ba_top5), 6),
        "elapsed_seconds":        round(elapsed, 1),
        "throughput_vid_per_s":   round(n_ok / elapsed, 2) if elapsed > 0 else 0,
    }
    manifest_path = os.path.join(args.output_dir, "logits_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}")

    print("\n===== Inference Complete =====")
    print(f"  Total videos in CSV       : {len(df)}")
    print(f"  Successfully decoded      : {n_ok}")
    print(f"  Skipped (errors)          : {n_fail}")
    print(f"  Output logits shape       : {logits_arr.shape}")
    print(f"  Top-1 Accuracy            : {top1_acc:.4f}")
    print(f"  Top-1 Balanced Accuracy   : {ba_top1:.4f}")
    print(f"  Top-5 Accuracy            : {top5_acc:.4f}")
    print(f"  Top-5 Balanced Accuracy   : {ba_top5:.4f}")
    print(f"  Elapsed                   : {elapsed:.1f}s ({n_ok / elapsed:.2f} vid/s)")
    print(f"\n  NPZ     → {npz_path}")
    print(f"  Parquet → {parquet_path}")
    print(f"  JSON    → {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run V-JEPA 2 inference on custom .avi videos indexed by a CSV file.\n"
            "Supports constraining predictions to a target class subspace (e.g. 10 classes)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/index_df.csv",
        help="Path to index CSV (columns: id, template, placeholders, session, cam).",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing <id>_<cam>.avi video files.",
    )
    parser.add_argument(
        "--mapping_path",
        type=str,
        default="data/template2id_mapping_only10classes.txt",
        help="Path to target template-to-id mapping text file (e.g. 10 classes).",
    )
    parser.add_argument(
        "--full_mapping_path",
        type=str,
        default="data/template2id_mapping.txt",
        help="Path to full 174-class template-to-id mapping text file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory where output files are saved.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/vjepa2-vitg-fpc64-384-ssv2",
        help="HuggingFace model repo ID (must be cached locally).",
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
        "--num_frames",
        type=int,
        default=64,
        help="Frames to uniformly sample per video. Overridden by model.config if different.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
