"""
Run V-JEPA 2 (facebook/vjepa2-vitg-fpc64-384-ssv2) inference on custom .avi videos.

Videos are indexed by a CSV with columns: id, template, placeholders, session, cam.
For each row, the video file is located at:
    {video_dir}/{id}_{cam}.avi
  e.g. /path/to/videos/260922-134214_cam0.avi

Outputs (saved to --output_dir):
  - logits_output.npz          : compressed NumPy archive with all logits + metadata
  - logits_output_meta.parquet : tabular metadata (video_id, cam, template, label,
                                  top1_pred, top1_class, top1_conf, top5_preds)
  - skipped_videos.txt         : list of videos that failed to decode (if any)

The .npz format stores:
  logits     (N, 174) float32  — raw model logits for all 174 SSv2 classes
  probs      (N, 174) float32  — softmax probabilities
  labels     (N,)    int16    — ground-truth class IDs  (-1 if template unknown)
  video_ids  (N,)    object   — recording IDs  (e.g. "260922-134214")
  cams       (N,)    object   — camera views   (e.g. "cam0")
  templates  (N,)    object   — SSv2 template string
  class_names (174,) object   — ordered class name strings (index = class ID)

Usage:
  python scripts/vjepa_on_myvideos.py \
      --csv_path      data/index_df.csv \
      --video_dir     /user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/data/hamza_vgd_data/raw/ \
      --mapping_path  data/template2id_mapping.txt \
      --output_dir    results \
      --batch_size    8 \
      --num_workers   8
      
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_template_mapping(mapping_path: str) -> tuple[dict[str, int], list[str]]:
    """
    Parse template2id_mapping.txt into:
      - {template_string: class_id}  (dict, for lookup)
      - class_names list             (list[str], index = class_id, length 174)

    File format (tab-separated):
      0\tApproaching [something] with your camera
      1\tAttaching [something] to [something]
      ...
    """
    mapping: dict[str, int] = {}
    id_to_name: dict[int, str] = {}

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
                id_to_name[class_id] = template

    # Build ordered list (index = class_id)
    num_classes = max(id_to_name.keys()) + 1
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
        session = str(row["session"])
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
        "video_ids":   [item["video_id"]   for item in valid],
        "cams":        [item["cam"]        for item in valid],
        "sessions":    [item["session"]    for item in valid],
        "templates":   [item["template"]   for item in valid],
        "labels":      [item["label"]      for item in valid],
        "frames_list": [item["frames"]     for item in valid],  # list of (T, H, W, C)
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

    # ---- Template mapping ----
    print(f"\nLoading template mapping: {args.mapping_path}")
    template_to_id, class_names = load_template_mapping(args.mapping_path)
    num_classes = len(class_names)
    print(f"  {len(template_to_id)} templates loaded, {num_classes} total classes.")

    # ---- CSV ----
    print(f"Loading CSV: {args.csv_path}")
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
    all_logits:    list[np.ndarray] = []   # each: (B, 174) float32
    all_probs:     list[np.ndarray] = []   # each: (B, 174) float32
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

        try:
            # AutoVideoProcessor: center-crop to 384×384 + SSv2 normalization
            inputs = processor(frames_list, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**inputs)
                logits = outputs.logits  # (B, 174) bfloat16

            # Cast to float32 on CPU before storing
            logits_f32 = logits.float().cpu()
            probs_f32  = F.softmax(logits_f32, dim=-1)

            all_logits.append(logits_f32.numpy())
            all_probs.append(probs_f32.numpy())
            all_video_ids.extend(video_ids)
            all_cams.extend(cams)
            all_sessions.extend(sessions)
            all_templates.extend(templates)
            all_labels.extend(labels)

        except Exception as e:
            tqdm.write(f"  [WARN] Batch inference error: {e}")
            tqdm.write(f"         Skipping {len(video_ids)} sample(s): {video_ids}")
            failed_paths.extend(
                [f"{vid}_{cam}" for vid, cam in zip(video_ids, cams)]
            )

    elapsed = time.time() - t0
    n_ok = len(all_labels)
    n_fail = len(failed_paths)

    # ---- Stack arrays ----
    logits_arr    = np.concatenate(all_logits, axis=0).astype(np.float32)   # (N, 174)
    probs_arr     = np.concatenate(all_probs,  axis=0).astype(np.float32)   # (N, 174)
    labels_arr    = np.array(all_labels,    dtype=np.int16)                  # (N,)
    video_ids_arr = np.array(all_video_ids, dtype=object)                    # (N,)
    cams_arr      = np.array(all_cams,      dtype=object)                    # (N,)
    templates_arr = np.array(all_templates, dtype=object)                    # (N,)
    class_names_arr = np.array(class_names, dtype=object)                    # (174,)

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
        templates   = templates_arr,
        class_names = class_names_arr,
    )
    print(f"\nNPZ saved: {npz_path}  (shape: {logits_arr.shape})")

    # 2. Parquet — metadata table for easy per-class / per-cam analysis
    top5_indices = np.argsort(logits_arr, axis=1)[:, -5:][:, ::-1]   # (N, 5) descending
    top1_indices = top5_indices[:, 0]                                  # (N,)
    top1_conf    = probs_arr[np.arange(n_ok), top1_indices]           # (N,) float32

    meta_df = pd.DataFrame({
        "video_id":    all_video_ids,
        "cam":         all_cams,
        "session":     all_sessions,
        "template":    all_templates,
        "label":       all_labels,
        "top1_pred":   top1_indices.tolist(),
        "top1_class":  [class_names[i] for i in top1_indices],
        "top1_conf":   top1_conf.tolist(),
        "top5_preds":  top5_indices.tolist(),               # stored as list-of-list
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

    # 4. Run manifest JSON (metadata for reproducibility)
    manifest = {
        "model":         args.model,
        "csv_path":      args.csv_path,
        "video_dir":     args.video_dir,
        "mapping_path":  args.mapping_path,
        "output_dir":    args.output_dir,
        "num_frames":    int(num_frames),
        "batch_size":    args.batch_size,
        "num_workers":   args.num_workers,
        "dtype":         "bfloat16 (autocast)",
        "n_total_csv":   len(df),
        "n_evaluated":   n_ok,
        "n_skipped":     n_fail,
        "logits_shape":  list(logits_arr.shape),
        "num_classes":   num_classes,
        "elapsed_seconds": round(elapsed, 1),
        "throughput_vid_per_s": round(n_ok / elapsed, 2) if elapsed > 0 else 0,
    }
    manifest_path = os.path.join(args.output_dir, "logits_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}")

    # ---- Summary ----
    top1_acc = (top1_indices == labels_arr).mean() if n_ok > 0 else 0.0
    top5_acc = np.mean([
        int(labels_arr[i]) in top5_indices[i].tolist() for i in range(n_ok)
    ]) if n_ok > 0 else 0.0

    print("\n===== Inference Complete =====")
    print(f"  Total videos in CSV  : {len(df)}")
    print(f"  Successfully decoded : {n_ok}")
    print(f"  Skipped (errors)     : {n_fail}")
    print(f"  Output logits shape  : {logits_arr.shape}")
    print(f"  Quick top-1 accuracy : {top1_acc:.4f}  (only valid if labels are correct)")
    print(f"  Quick top-5 accuracy : {top5_acc:.4f}")
    print(f"  Elapsed              : {elapsed:.1f}s  ({n_ok / elapsed:.2f} vid/s)")
    print(f"\n  NPZ    → {npz_path}")
    print(f"  Parquet→ {parquet_path}")
    print(f"  JSON   → {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run V-JEPA 2 inference on custom .avi videos indexed by a CSV file.\n"
            "Saves raw logits, softmax probabilities, and metadata for all 174 SSv2 classes."
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
        default="data/template2id_mapping.txt",
        help="Path to template2id_mapping.txt (tab-separated: id<TAB>template).",
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
