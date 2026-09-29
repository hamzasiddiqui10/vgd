"""
python 3_infer.py \
  --h5_file data/extracted_features/myvideos-center-dinov2.h5 \
  --model_path /user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/experiment_ssv2_object_occlusion/stripped_down/runs/baseline_test/best.pt \
  --split test \
  --output_dir ./results/


******
RESULT:
Running inference (test): 100%|███████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 17.36it/s]

==================================================
         INFERENCE METRICS (Split: test)
==================================================
Total Samples: 150
Top-1 Accuracy: 20.67%
Top-5 Accuracy: 51.33%
--------------------------------------------------
ARRAY DIMENSIONS:
  Logits shape: (150, 174)
  Probs shape:  (150, 174)
--------------------------------------------------
SAVED FILES:
  Parquet: /mnt/vast-nhr/projects/nib00021/hamza/repos/vgd/vjepa_evaluation_on_ssv2/apply_dinov2_attn_probe/results/predictions.parquet
  NPZ:     /mnt/vast-nhr/projects/nib00021/hamza/repos/vgd/vjepa_evaluation_on_ssv2/apply_dinov2_attn_probe/results/predictions.npz 
*****
"""

import argparse
import json
import math
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


# =====================================================================
# 1. Positional Embeddings & Model Architecture (from train.py)
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
        pe_t[:, None, None, :] + pe_h[None, :, None, :] + pe_w[None, None, :, :]
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

        x = data[self.feature_key]  # [B, T, D] / [B, T, HW, D]
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
# 2. Data Loader & Inference Engine
# =====================================================================
def load_data_from_h5(h5_path, split=None):
    with h5py.File(h5_path, "r") as f:
        available_splits = list(f.keys())

        if split is None:
            split = available_splits[0]
        elif split not in available_splits:
            raise KeyError(
                f"Split '{split}' not found. Available splits: {available_splits}"
            )

        group = f[split]
        features = group["features"][:]
        labels = group["labels"][:]
        raw_names = group["names"][:]

        names = [
            n.decode("utf-8") if isinstance(n, bytes) else str(n)
            for n in raw_names
        ]

    return features, labels, names, split


def run_inference(h5_file, model_path, output_dir, split=None, batch_size=64):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read features, labels, and video IDs directly from HDF5
    features, labels, video_ids, loaded_split = load_data_from_h5(
        h5_file, split
    )
    labels = np.array(labels, dtype=np.int64)
    x = torch.from_numpy(features).float()  # Shape: (N, T, P, D)

    # Calculate dimension parameters automatically from feature tensor
    feat_shape = x.shape[1:]  # (num_frames, num_patches, dim)
    num_frames, num_patches, dim = feat_shape
    spatial_size = int(round(num_patches**0.5))
    assert (
        spatial_size**2 == num_patches
    ), f"{num_patches} patch tokens are not a square grid"

    # 2. Load Checkpoint & Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        ckpt_args = checkpoint.get("args", {})
    else:
        state_dict = checkpoint
        ckpt_args = {}

    num_classes = ckpt_args.get("num_classes", 174)
    num_heads = ckpt_args.get("num_heads", 16)
    pos_enc = not ckpt_args.get("no_pos_enc", False)

    # 3. Instantiate & Load Model Architecture
    model = AttentiveProbe(
        dim_embed=dim,
        num_frames=num_frames,
        spatial_size=spatial_size,
        num_heads=num_heads,
        pos_enc=pos_enc,
        feature_key="dino_feat",
        num_classes=num_classes,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Batch Inference Loop
    all_logits = []
    all_probs = []

    with torch.no_grad():
        for i in tqdm(
            range(0, len(x), batch_size),
            desc=f"Running inference ({loaded_split})",
        ):
            batch_feats = x[i : i + batch_size].to(device)

            # Forward pass matching train.py signature
            logits = model({"dino_feat": batch_feats})
            probs = F.softmax(logits, dim=-1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())

    logits_np = torch.cat(all_logits, dim=0).numpy()
    probs_np = torch.cat(all_probs, dim=0).numpy()

    # 5. Compute Top-1 & Top-5 Accuracies
    top1_preds = np.argmax(probs_np, axis=1)
    correct_top1 = top1_preds == labels
    top1_acc = np.mean(correct_top1) * 100

    top_k = min(5, logits_np.shape[1])
    top5_preds = np.argsort(probs_np, axis=1)[:, -top_k:]
    correct_top5 = np.array(
        [label in top5 for label, top5 in zip(labels, top5_preds)]
    )
    top5_acc = np.mean(correct_top5) * 100

    # 6. Save Parquet File
    df = pd.DataFrame({
        "video_id": video_ids,
        "label": labels,
        "top1_label": top1_preds,
        "correct_top1": correct_top1,
    })
    parquet_path = output_dir / "predictions.parquet"
    df.to_parquet(parquet_path, index=False)

    # 7. Save NPZ Compressed File
    npz_path = output_dir / "predictions.npz"
    np.savez_compressed(
        npz_path,
        logits=logits_np,
        probs=probs_np,
        video_ids=np.array(video_ids),
        labels=labels,
    )

    # 8. Print Summary
    print("\n" + "=" * 50)
    print(f"         INFERENCE METRICS (Split: {loaded_split})")
    print("=" * 50)
    print(f"Total Samples: {len(labels)}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print("-" * 50)
    print("ARRAY DIMENSIONS:")
    print(f"  Logits shape: {logits_np.shape}")
    print(f"  Probs shape:  {probs_np.shape}")
    print("-" * 50)
    print("SAVED FILES:")
    print(f"  Parquet: {parquet_path.resolve()}")
    print(f"  NPZ:     {npz_path.resolve()}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with AttentiveProbe on extracted H5 features."
    )
    parser.add_argument(
        "--h5_file",
        type=str,
        required=True,
        help="Path to extracted features .h5 file",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved .pt model checkpoint (e.g., best.pt)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="H5 split group to evaluate (e.g., 'test'). Defaults to first available split.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Inference batch size"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save predictions",
    )

    args = parser.parse_args()
    run_inference(
        args.h5_file,
        args.model_path,
        args.output_dir,
        args.split,
        args.batch_size,
    )


if __name__ == "__main__":
    main()