
"""
apply_dinov2_16frame_attn_probe.py

Usage Example:
--------------
python apply_dinov2_16frame_attn_probe.py \
  --csv_path /user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/vjepa_evaluation_on_ssv2/data/index_df_rec_session_2.csv \
  --video_dir /user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/data/hamza_vgd_data/capture/raw/ \
  --ssv2_label_mapping /user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/vjepa_evaluation_on_ssv2/data/something-something-v2-id2label.json \
  --dino_model vit_base_patch14_reg4_dinov2.lvd142m \
  --model_path /mnt/vast-nhr/projects/nib00021/repos_jan/TrAction/checkpoints/ssv2_rebuttal_dinov2_seed42/best_acc.pt \
  --output_dir ./results/myvideos_2/ \
  --batch_size 12 \
  --num_workers 16 \
  --device cuda


Description:
------------
This script processes video clips listed in a CSV file, extracts 16 uniformly
sampled frames directly in memory, computes DINOv2 patch features, and runs
inference through an AttentiveProbe model checkpoint. Results are saved in both
NPZ and Parquet formats (retaining all original metadata columns from the CSV).
"""

import argparse
import json
import math
import os
# Must be set BEFORE importing timm or huggingface_hub
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import torchvision.io as tv_io
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =====================================================================
# 1. Positional Embeddings & AttentiveProbe Model Architecture
# =====================================================================
def _get_1d_sincos_pos_embed(length, dim):
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _get_3d_sincos_pos_embed(T, H, W, dim):
    pe_t = _get_1d_sincos_pos_embed(T, dim)
    pe_h = _get_1d_sincos_pos_embed(H, dim)
    pe_w = _get_1d_sincos_pos_embed(W, dim)
    pe = (
        pe_t[:, None, None, :] + pe_h[None, :, None, :] + pe_w[None, None, :]
    )
    return pe.reshape(T * H * W, dim)


class AttentiveProbe(nn.Module):
    def __init__(
        self,
        dim_embed,
        num_frames=16,
        spatial_size=16,
        num_heads=16,
        pos_enc=True,
        feature_key="dino_feat",
        num_classes=174,
    ):
        super().__init__()
        self.dim_embed = dim_embed
        self.feature_key = feature_key
        self.num_classes = num_classes
        self.query = nn.Parameter(torch.zeros(1, 1, dim_embed))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.norm_attn = nn.LayerNorm(dim_embed)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_embed,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm_mlp = nn.LayerNorm(dim_embed)
        self.mlp = nn.Sequential(
            nn.Linear(dim_embed, dim_embed * 4),
            nn.GELU(),
            nn.Linear(dim_embed * 4, dim_embed),
        )

        self.linear = nn.Linear(self.dim_embed, self.num_classes)

        if pos_enc:
            pos = _get_3d_sincos_pos_embed(
                num_frames, spatial_size, spatial_size, dim_embed
            )
            self.register_buffer("pos_embed", pos.unsqueeze(0))
        else:
            self.pos_embed = 0

    def forward_feat(self, data):
        assert (
            self.feature_key in data
        ), f'"{self.feature_key}" does not exist in the data.'

        x = data[self.feature_key]  # [B, T, P, D] -> [B, T*P, D]
        if x.dim() == 4:
            x = x.flatten(1, 2)
        B = x.shape[0]
        x = x + self.pos_embed

        query = self.query.expand(B, -1, -1)

        # Cross-attention block
        x = self.norm_attn(x)
        query = query + self.attn(query, x, x)[0]

        # MLP block
        query = query + self.mlp(self.norm_mlp(query))

        return query.squeeze(1)

    def forward(self, data):
        feat = self.forward_feat(data)
        return self.linear(feat)


# =====================================================================
# 2. Dataset for 16-Frame In-Memory Video Sampling
# =====================================================================
class Video16FrameDataset(Dataset):
    def __init__(self, df, video_dir, label_to_id, num_frames=16, image_size=224):
        self.df = df.reset_index(drop=True)
        self.video_dir = Path(video_dir)
        self.label_to_id = label_to_id
        self.num_frames = num_frames
        self.image_size = image_size

        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def _sample_frame_indices(self, total_frames):
        if total_frames <= 0:
            return np.zeros(self.num_frames, dtype=int)
        if total_frames < self.num_frames:
            # Repeat last frame if video is too short
            indices = np.arange(total_frames)
            pad = np.full(self.num_frames - total_frames, total_frames - 1, dtype=int)
            return np.concatenate([indices, pad])
        return np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_filename = f"{row['id']}_{row['cam']}.avi"
        video_path = self.video_dir / video_filename

        template_str = str(row["template"])
        label_id = self.label_to_id.get(template_str, -1)

        # Fallback dummy tensor (16, 3, 224, 224) if file fails to load
        dummy_frames = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)

        if not video_path.exists():
            print(f"\n[Warning] File missing: {video_path}")
            return dummy_frames, label_id, video_filename, idx

        try:
            # Read video directly in memory
            video_pts, _, _ = tv_io.read_video(str(video_path), pts_unit="sec")
            total_frames = video_pts.shape[0]

            if total_frames == 0:
                print(f"\n[Warning] 0 frames read from {video_path}")
                return dummy_frames, label_id, video_filename, idx

            frame_indices = self._sample_frame_indices(total_frames)
            sampled_raw_frames = video_pts[frame_indices]  # [16, H, W, C] (uint8)

            processed_frames = []
            for i in range(self.num_frames):
                # Convert uint8 frame -> PIL Image for transforms
                frame_np = sampled_raw_frames[i].numpy()
                img = Image.fromarray(frame_np)
                processed_frames.append(self.transform(img))

            # Stack into tensor: [16, 3, H, W]
            video_tensor = torch.stack(processed_frames, dim=0)
            return video_tensor, label_id, video_filename, idx

        except Exception as e:
            print(f"\n[Warning] Failed decoding {video_path}: {e}")
            return dummy_frames, label_id, video_filename, idx


# =====================================================================
# 3. Main Processing & Pipeline Execution
# =====================================================================
def forward_features(model, images):
    """Extract patch tokens excluding CLS/register prefix tokens."""
    return model.forward_features(images)[:, model.num_prefix_tokens:]


def main():
    parser = argparse.ArgumentParser(
        description="Extract 16-frame DINOv2 features & run AttentiveProbe inference."
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to input metadata CSV"
    )
    parser.add_argument(
        "--video_dir", type=str, required=True, help="Directory containing raw .avi videos"
    )
    parser.add_argument(
        "--ssv2_label_mapping",
        type=str,
        required=True,
        help="Path to something-something-v2-id2label.json",
    )
    parser.add_argument(
        "--dino_model",
        type=str,
        default="vit_base_patch14_reg4_dinov2.lvd142m",
        help="timm DINOv2 backbone model name",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained AttentiveProbe model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory where results will be stored",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for video evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="PyTorch DataLoader workers"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Execution device (cuda or cpu)"
    )

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load CSV & Label Mapping JSON
    df = pd.read_csv(args.csv_path)
    with open(args.ssv2_label_mapping, "r") as f:
        mapping_data = json.load(f)

    # Support id2label (str -> str) or label2id (str -> int) maps
    if isinstance(next(iter(mapping_data.keys())), str) and isinstance(
        next(iter(mapping_data.values())), str
    ):
        label_to_id = {v: int(k) for k, v in mapping_data.items()}
    else:
        label_to_id = {k: int(v) for k, v in mapping_data.items()}

    # 2. Instantiate Dataset & Loader
    dataset = Video16FrameDataset(df, args.video_dir, label_to_id, num_frames=16) 
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 3. Load DINOv2 Feature Extractor Model
    print(f"Loading DINOv2 Backbone: {args.dino_model} ...")
    dino_backbone = timm.create_model(
                args.dino_model, 
                pretrained=True, 
                num_classes=0, 
                dynamic_img_size=True,
                # pretrained_cfg_overlay=dict(file=""),
    )
    dino_backbone = dino_backbone.to(device).eval()

    # Determine patch and feature dimensions automatically using a dummy image
    with torch.no_grad():
        dummy_img = torch.zeros(1, 3, 224, 224, device=device)
        dummy_feat = forward_features(dino_backbone, dummy_img)
        num_patches, feat_dim = dummy_feat.shape[1], dummy_feat.shape[2]
        spatial_size = int(round(num_patches**0.5))

    # 4. Load AttentiveProbe Checkpoint
    print(f"Loading AttentiveProbe Checkpoint: {args.model_path} ...")
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    num_classes = ckpt_args.get("num_classes", 174)
    num_heads = ckpt_args.get("num_heads", 16)
    pos_enc = not ckpt_args.get("no_pos_enc", False)

    probe_model = AttentiveProbe(
        dim_embed=feat_dim,
        num_frames=16,
        spatial_size=spatial_size,
        num_heads=num_heads,
        pos_enc=pos_enc,
        feature_key="dino_feat",
        num_classes=num_classes,
    )
    probe_model.load_state_dict(state_dict)
    probe_model.to(device).eval()

    # 5. Batch Inference Loop
    all_logits = []
    all_probs = []
    processed_labels = []
    processed_filenames = []
    processed_indices = []

    print("Running 16-Frame Extraction & AttentiveProbe Inference...")
    with torch.no_grad():
        for video_tensors, labels, filenames, idxs in tqdm(loader, desc="Inference"):
            B, T, C, H, W = video_tensors.shape

            # Collapse batch and frame dimensions for DINOv2 backbone: [B * 16, 3, 224, 224]
            flat_images = video_tensors.view(B * T, C, H, W).to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                patch_feats = forward_features(dino_backbone, flat_images)  # [B * 16, P, D]

            # Reshape back to video sequence structure: [B, 16, P, D]
            patch_feats = patch_feats.float().view(B, T, num_patches, feat_dim)

            # Pass sequences through AttentiveProbe
            logits = probe_model({"dino_feat": patch_feats})  # [B, num_classes]
            probs = F.softmax(logits, dim=-1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            processed_labels.extend(labels.numpy())
            processed_filenames.extend(filenames)
            processed_indices.extend(idxs.numpy())

    logits_np = torch.cat(all_logits, dim=0).numpy()
    probs_np = torch.cat(all_probs, dim=0).numpy()
    labels_np = np.array(processed_labels, dtype=np.int64)

    # 6. Compute Accuracy Metrics
    top1_preds = np.argmax(probs_np, axis=1)
    correct_top1 = top1_preds == labels_np
    top1_acc = np.mean(correct_top1) * 100

    top_k = min(5, logits_np.shape[1])
    top5_preds = np.argsort(probs_np, axis=1)[:, -top_k:]
    correct_top5 = np.array(
        [label in top5 for label, top5 in zip(labels_np, top5_preds)]
    )
    top5_acc = np.mean(correct_top5) * 100

    # 7. Merge Original CSV Metadata & Save Output Parquet File
    result_df = df.iloc[processed_indices].copy().reset_index(drop=True)
    result_df["constructed_filename"] = processed_filenames
    result_df["label"] = labels_np
    result_df["top1_pred"] = top1_preds
    result_df["correct_top1"] = correct_top1
    result_df['top5_preds'] = [list(top5) for top5 in top5_preds]
    result_df["correct_top5"] = correct_top5

    parquet_path = output_dir / "predictions.parquet"
    result_df.to_parquet(parquet_path, index=False)

    # 8. Save Compressed NPZ File
    npz_path = output_dir / "predictions.npz"
    np.savez_compressed(
        npz_path,
        logits=logits_np,
        probs=probs_np,
        video_ids=np.array(processed_filenames),
        labels=labels_np,
    )

    # 9. Print Execution Summary
    print("\n" + "=" * 50)
    print("         INFERENCE METRICS (16-FRAME PIPELINE)")
    print("=" * 50)
    print(f"Total Samples Processed: {len(labels_np)}")
    print(f"Top-1 Accuracy:          {top1_acc:.2f}%")
    print(f"Top-5 Accuracy:          {top5_acc:.2f}%")
    print("-" * 50)
    print("ARRAY DIMENSIONS:")
    print(f"  Logits shape:          {logits_np.shape}")
    print(f"  Probs shape:           {probs_np.shape}")
    print("-" * 50)
    print("SAVED FILES:")
    print(f"  Parquet:               {parquet_path.resolve()}")
    print(f"  NPZ:                   {npz_path.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
