"""Train + evaluate an attentive probe on frozen DINOv2 features (single GPU).

Reads the HDF5 file produced by extract.py, trains the attentive probe with a 
constant learning rate, and evaluates on the val split after every epoch.

Every run gets its own directory `<out_dir>/<run_name>/` containing:
    args.json    the full configuration of the run
    last.pt      checkpoint after the most recent epoch
    best.pt      checkpoint of the epoch with the best val top-1 accuracy

Example:
    python stripped_down/train.py --h5 /path/ssv2-dinov2-large-center.h5 --run_name probe_lr1e-4
"""

import argparse
import json
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

import wandb  # noqa: E402

def _get_1d_sincos_pos_embed(length, dim):
    """Standard 1-D sinusoidal embedding.  Returns (length, dim)."""
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
    """Additive 3-D sinusoidal embedding.  Returns (T*H*W, dim)."""
    pe_t = _get_1d_sincos_pos_embed(T, dim)
    pe_h = _get_1d_sincos_pos_embed(H, dim)
    pe_w = _get_1d_sincos_pos_embed(W, dim)
    pe = (pe_t[:, None, None, :]
          + pe_h[None, :, None, :]
          + pe_w[None, None, :, :])
    return pe.reshape(T * H * W, dim)

class AttentiveProbe(nn.Module):
    """Attentive probe similar to V-JEPA, but with an additional absolute positional encoding for image models
    """
    def __init__(self, dim_embed, num_frames=16, spatial_size=16, num_heads=6, pos_enc=True, feature_key='dino_feat', num_classes=174):
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
            pos = _get_3d_sincos_pos_embed(num_frames, spatial_size,
                                        spatial_size, dim_embed)
            self.register_buffer('pos_embed', pos.unsqueeze(0))  # [1, T*H*W, D]
        else:
            self.pos_embed = 0
    
    def forward_feat(self, data):
        assert self.feature_key in data, f'"{self.feature_key}" does not exist in the data.'

        x = data[self.feature_key]              # [B, T, D] / [B, T, HW, D]
        if x.dim() == 4:
            # x = x[:, torch.randperm(x.shape[1])]  # random temporal order
            # x = x[:,8].unsqueeze(1)               # center frame
            x = x.flatten(1,2)
        B = x.shape[0]                          # [B, T, D]
        x = x + self.pos_embed

        query = self.query.expand(B, -1, -1)    # [B, 1, D]

        # Cross-attention block (pre-norm)
        x = self.norm_attn(x)
        query = query + self.attn(query, x, x)[0]

        # MLP block (pre-norm)
        query = query + self.mlp(self.norm_mlp(query))

        return query.squeeze(1)                  # [B, D]

    def forward(self, data):
        feat = self.forward_feat(data)
        return self.linear(feat)

class FeatureDataset(Dataset):
    """Frozen features from one split of the extraction file: [T, P, D] per sample."""

    def __init__(self, h5_path, split):
        self.h5_path = h5_path
        self.split = split
        self._h5 = None

        with h5py.File(h5_path, "r") as f:
            group = f[split]
            num_written = int(group.attrs.get("num_written", group["features"].shape[0]))
            assert num_written == group["features"].shape[0], \
                f"'{split}' in {h5_path} is incomplete ({num_written}/{group['features'].shape[0]})"
            self.labels = group["labels"][...].astype(np.int64)
            self.feat_shape = group["features"].shape[1:]  # [T, P, D]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self._h5 is None:  # open lazily, once per dataloader worker
            self._h5 = h5py.File(self.h5_path, "r")
        feat = self._h5[self.split]["features"][idx].astype(np.float32)
        return torch.from_numpy(feat), self.labels[idx]


def build_model(args, feat_shape):
    num_frames, num_patches, dim = feat_shape
    spatial_size = int(round(num_patches ** 0.5))
    assert spatial_size ** 2 == num_patches, f"{num_patches} patch tokens are not a square grid"

    # head = ActionHeadLinear(num_classes=args.num_classes, dim_embed=dim, layer_norm=args.head_layer_norm)
    return AttentiveProbe(
        # head=head,
        dim_embed=dim,
        num_frames=num_frames,
        spatial_size=spatial_size,
        num_heads=args.num_heads,
        pos_enc=not args.no_pos_enc,
        feature_key="dino_feat",
        num_classes=args.num_classes
    )


def run_epoch(model, loader, task_metrics, device, optimizer=None, log_prefix="train", epoch=0, log_interval=50):
    train = optimizer is not None
    model.train(train)
    for metric in task_metrics.values():
        metric.reset()

    total_loss, num_batches = 0.0, 0
    with torch.set_grad_enabled(train):
        for step, (feats, labels) in enumerate(tqdm(loader, desc=f"epoch {epoch} [{log_prefix}]")):
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                preds = model({"dino_feat": feats})
                loss = F.cross_entropy(preds, labels)

            if train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step % log_interval == 0:
                    wandb.log({"train/loss_step": loss.item()})

            total_loss += loss.item()
            num_batches += 1
            for metric in task_metrics.values():
                metric.update(preds.detach().float(), labels)

    stats = {f"{log_prefix}/loss": total_loss / max(num_batches, 1)}
    stats.update({f"{log_prefix}/{name}": m.compute().item() for name, m in task_metrics.items()})
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--h5", required=True, help="feature file written by stripped_down/extract.py")
    p.add_argument("--train_split", default="train")
    p.add_argument("--val_split", default="val")
    p.add_argument("--num_classes", type=int, default=174)
    p.add_argument("--num_workers", type=int, default=8)
    # model
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--no_pos_enc", action="store_true", help="disable the sinusoidal positional encoding")
    p.add_argument("--head_layer_norm", action="store_true", help="LayerNorm before the classification layer")
    # optimization (constant LR)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    # bookkeeping
    p.add_argument("--out_dir", default="stripped_down/runs")
    p.add_argument("--run_name", default=None, help="defaults to a timestamp")
    p.add_argument("--wandb_project", default="dinov2-attn-probe")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ds_train = FeatureDataset(args.h5, args.train_split)
    ds_val = FeatureDataset(args.h5, args.val_split)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(args, ds_train.feat_shape).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics = {
        "top1": MulticlassAccuracy(num_classes=args.num_classes, average="micro").to(device),
        "top1_bal": MulticlassAccuracy(num_classes=args.num_classes, average="macro").to(device),
    }

    os.environ["HTTPS_PROXY"] = "http://www-cache.gwdg.de:3128"
    os.environ["HTTP_PROXY"] = "http://www-cache.gwdg.de:3128"
    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args),
               dir=run_dir, mode="disabled" if args.no_wandb else "online")
    print(f"run dir: {run_dir}")
    print(f"features: {ds_train.feat_shape} | train {len(ds_train)} | val {len(ds_val)}")

    best_acc = -1.0
    for epoch in range(args.epochs):
        stats = run_epoch(model, dl_train, metrics, device, optimizer=optimizer, log_prefix="train", epoch=epoch)
        stats.update(run_epoch(model, dl_val, metrics, device, log_prefix="val", epoch=epoch))

        wandb.log({**stats, "epoch": epoch})
        print(f"epoch {epoch}: " + " | ".join(f"{k} {v:.4f}" for k, v in stats.items()), flush=True)

        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "stats": stats, "args": vars(args)}
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))
        if stats["val/top1"] > best_acc:
            best_acc = stats["val/top1"]
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))

    print(f"best val top-1: {best_acc:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
