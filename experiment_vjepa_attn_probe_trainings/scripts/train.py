
"""
V-JEPA Attention Probe Training Script

This script trains an action classification attention probe on pre-extracted 
V-JEPA 2.1 features. It loads sharded HDF5 datasets efficiently, dynamically 
allocates features to float32, and employs a Cosine Annealing learning rate 
scheduler with a linear warmup phase.

Usage Example:
    python scripts/train.py --config configs/config_00.json
    
Required `config.json` structure:
    {
        "wandb_project": "ssv2-vjepa-probe",
        "vjepa_feats_h5_dir": "./data/output_features/",
        "train_h5_files": ["train_shard_00.h5", "train_shard_01.h5"],
        "val_h5_files": ["val_shard_00.h5"],
        "epochs": 10,
        "lr": 1e-3,
        "weight_decay": 0.05,
        "warmup_epochs": 2,
        "eta_min": 1e-8,
        "batch_size": 128,
        "num_workers": 4,
        "prefetch_factor": 2,
        "val_frequency": 1,
        "pin_memory": true
    }
"""

import argparse
import json
import os
import gc
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score
import wandb
from tqdm import tqdm

# ---------------------------------------------------------
# 0. Model Definition
# ---------------------------------------------------------
class VJepaAttentiveProbe(nn.Module):
    """Attentive probe for V-JEPA features loaded directly from HDF5.
    (no positional encoding because vjepa features doesnt require it)
    Expects input tensor of shape (batch_size, seq_len, dim_embed).
    """
    def __init__(self, dim_embed=768, seq_len=4608, num_heads=12, num_classes=174):
        super().__init__()
        self.dim_embed = dim_embed
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # Learnable query token
        self.query = nn.Parameter(torch.zeros(1, 1, dim_embed))
        nn.init.trunc_normal_(self.query, std=0.02)

        # Cross-attention block
        self.norm_attn = nn.LayerNorm(dim_embed)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_embed,
            num_heads=num_heads,
            batch_first=True,
        )

        # MLP block
        self.norm_mlp = nn.LayerNorm(dim_embed)
        self.mlp = nn.Sequential(
            nn.Linear(dim_embed, dim_embed * 4),
            nn.GELU(),
            nn.Linear(dim_embed * 4, dim_embed),
        )

        # Classification head
        self.linear = nn.Linear(self.dim_embed, self.num_classes)

    def forward_feat(self, x):
        # x shape: [B, seq_len, dim_embed]
        B = x.shape[0]                         
        query = self.query.expand(B, -1, -1)    # [B, 1, D]

        # Cross-attention block (pre-norm)
        x = self.norm_attn(x)
        query = query + self.attn(query, x, x)[0]

        # MLP block (pre-norm)
        query = query + self.mlp(self.norm_mlp(query))
        return query.squeeze(1)                  # [B, D]


    def forward(self, x):
        feat = self.forward_feat(x)
        return self.linear(feat)


# ---------------------------------------------------------
# 1. Dataset & HDF5 Schema
# ---------------------------------------------------------
class VJepaH5Dataset(Dataset):
    """
    Custom Dataset for loading pre-extracted V-JEPA 2.1 HDF5 features.
    Handles reading multiple sharded .h5 files, safely managing file 
    pointers for multiprocessing DataLoader environments.
    """
    def __init__(self, h5_dir, h5_files):
        super().__init__()
        self.h5_paths = [os.path.join(h5_dir, f) for f in h5_files]
        self.lengths = []
        
        # Read dataset lengths to build a global index mapping.
        # We immediately close the files to ensure fork-safety for DataLoaders.
        for p in self.h5_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"HDF5 shard not found: {p}")
            with h5py.File(p, 'r') as f:
                self.lengths.append(len(f['labels']))
                
        self.cum_lengths = np.cumsum(self.lengths)
        self.h5_handles = {} # Lazy initialization in __getitem__

    def __len__(self):
        return self.cum_lengths[-1] if len(self.cum_lengths) > 0 else 0

    def __getitem__(self, idx):
        # Determine which shard contains the global index
        file_idx = np.searchsorted(self.cum_lengths, idx, side='right')
        local_idx = idx if file_idx == 0 else idx - self.cum_lengths[file_idx - 1]
        path = self.h5_paths[file_idx]

        # Lazy open per-worker to prevent HDF5 multiprocessing context errors
        if path not in self.h5_handles:
            self.h5_handles[path] = h5py.File(path, 'r', swmr=True)
            
        f = self.h5_handles[path]
        
        # Extract features (Float16) and immediately cast to Float32
        features = torch.from_numpy(f['features'][local_idx]).to(torch.float32)
        
        # Extract target labels and cast to Long
        label = torch.tensor(f['labels'][local_idx], dtype=torch.long)
        
        # We can extract names here if required, but it is omitted from the return 
        # to optimize Dataloader collation overhead, as they are unused for loss.
        return features, label


# ---------------------------------------------------------
# 2. Metrics & Utilities
# ---------------------------------------------------------
def clean_memory():
    """Strategically collects garbage and clears CUDA cache to prevent OOM."""
    gc.collect()
    torch.cuda.empty_cache()

def calculate_metrics(logits, targets):
    """
    Calculates standard and balanced top-1 and top-5 accuracies.
    Returns: top1_acc, top1_bal_acc, top5_acc, top5_bal_acc
    """
    with torch.no_grad():
        _, top5_preds = logits.topk(5, dim=1, largest=True, sorted=True)
        top1_preds = top5_preds[:, 0]
        
        # Standard Accuracies
        top1_acc = (top1_preds == targets).float().mean().item()
        top5_correct_mask = (top5_preds == targets.unsqueeze(1)).any(dim=1)
        top5_acc = top5_correct_mask.float().mean().item()
        
        # Move to CPU / NumPy for balanced metrics
        targets_np = targets.cpu().numpy()
        top1_preds_np = top1_preds.cpu().numpy()
        
        # Top-1 Balanced Accuracy
        top1_bal_acc = balanced_accuracy_score(targets_np, top1_preds_np)
        
        # Top-5 Balanced Accuracy (Mean of per-class Top-5 recall)
        classes = np.unique(targets_np)
        top5_correct_np = top5_correct_mask.cpu().numpy()
        top5_bal_acc = np.mean([
            top5_correct_np[targets_np == c].mean() for c in classes
        ]) if len(classes) > 0 else 0.0
        
        return top1_acc, top1_bal_acc, top5_acc, top5_bal_acc

def save_checkpoint(model, config, metrics_dict, filename):
    """Saves the PyTorch model and an accompanying JSON containing metrics/config."""
    # Save weights
    torch.save(model.state_dict(), filename)
    
    # Save accompanying metadata JSON
    json_filename = filename.replace('.pt', '.json')
    meta_payload = {
        "config": config,
        "metrics": metrics_dict
    }
    with open(json_filename, 'w') as f:
        json.dump(meta_payload, f, indent=4)


# ---------------------------------------------------------
# 3. Main Training Routine
# ---------------------------------------------------------
def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # Generate a unique run name with timestamp
    import pandas as pd
    run_name = f"{config['wandb_project']}_{pd.Timestamp.now().strftime('%y%m%d_%H%M')}"
    CHECKPOINT_BASE_DIR = "checkpoints"
    run_checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, run_name)
    
    if os.path.exists(run_checkpoint_dir):
        # time.sleep(61)
        run_name = f"{config['wandb_project']}_{pd.Timestamp.now().strftime('%y%m%d_%H%M')}"
        run_checkpoint_dir = os.path.join(CHECKPOINT_BASE_DIR, run_name)
        
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    print(f"[*] Checkpoints will be saved to: {run_checkpoint_dir}")
    

    # Initialize W&B
    os.environ["HTTPS_PROXY"] = "http://www-cache.gwdg.de:3128"
    os.environ["HTTP_PROXY"] = "http://www-cache.gwdg.de:3128"
    wandb.init(project=config['wandb_project'], 
               config=config,
               name=run_name)

    # Prepare DataLoaders
    train_dataset = VJepaH5Dataset(config['vjepa_feats_h5_dir'], config['train_h5_files'])
    val_dataset = VJepaH5Dataset(config['vjepa_feats_h5_dir'], config['val_h5_files'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'], 
        prefetch_factor=config['prefetch_factor'], 
        pin_memory=config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'], 
        prefetch_factor=config['prefetch_factor'], 
        pin_memory=config['pin_memory']
    )

    # Initialize Model
    model = VJepaAttentiveProbe(
        dim_embed=768, 
        seq_len=4608, 
        num_heads=12, 
        num_classes=174
    ).to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # Scheduler (Warmup -> Cosine Annealing)
    warmup_epochs = config['warmup_epochs']
    total_epochs = config['epochs']
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(total_epochs - warmup_epochs), eta_min=config['eta_min']
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    best_val_acc = -1.0
    best_val_loss = float('inf')

    # Ensure clean memory context before starting loop
    clean_memory()

    for epoch in range(total_epochs):
        print(f"\n[Epoch {epoch+1}/{total_epochs}] Starting Training Phase...")
        model.train()
        
        train_loss_accum = 0.0
        all_train_logits = []
        all_train_targets = []

        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{total_epochs}", leave=False)
        for batch_idx, (features, labels) in enumerate(train_pbar):
        # for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            
            # Store targets and detach logits for epoch-level metrics computation
            all_train_logits.append(logits.detach().cpu())
            all_train_targets.append(labels.detach().cpu())

        # Update Scheduler (per epoch)
        scheduler.step()
        
        # Calculate Epoch Train Metrics
        train_loss = train_loss_accum / len(train_loader)
        train_logits_tensor = torch.cat(all_train_logits, dim=0)
        train_targets_tensor = torch.cat(all_train_targets, dim=0)
        
        t1, t1_bal, t5, t5_bal = calculate_metrics(train_logits_tensor, train_targets_tensor)
        
        current_lr = optimizer.param_groups[0]['lr']
        log_metrics = {
            "main/epoch": epoch + 1,
            "main/lr": current_lr,
            "train/loss": train_loss,
            "train/top1acc": t1,
            "train/top1balacc": t1_bal,
            "train/top5acc": t5,
            "train/top5balacc": t5_bal
        }
        
        # Free memory before validation
        del all_train_logits, all_train_targets, train_logits_tensor, train_targets_tensor
        clean_memory()
        
        # Trigger Validation
        if (epoch + 1) % config['val_frequency'] == 0:
            print(f"[Epoch {epoch+1}/{total_epochs}] Starting Validation Phase...")
            model.eval()
            
            val_loss_accum = 0.0
            all_val_logits = []
            all_val_targets = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{total_epochs}", leave=False)
                for features, labels in val_pbar:
                # for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    
                    logits = model(features)
                    loss = criterion(logits, labels)
                    
                    val_loss_accum += loss.item()
                    all_val_logits.append(logits.cpu())
                    all_val_targets.append(labels.cpu())

            val_loss = val_loss_accum / len(val_loader)
            val_logits_tensor = torch.cat(all_val_logits, dim=0)
            val_targets_tensor = torch.cat(all_val_targets, dim=0)
            
            v1, v1_bal, v5, v5_bal = calculate_metrics(val_logits_tensor, val_targets_tensor)
            
            val_metrics = {
                "val/loss": val_loss,
                "val/top1acc": v1,
                "val/top1balacc": v1_bal,
                "val/top5acc": v5,
                "val/top5balacc": v5_bal
            }
            log_metrics.update(val_metrics)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Top-1 Acc: {v1:.4f}")
            
            # Checkpointing
            if v1 > best_val_acc:
                best_val_acc = v1
                save_checkpoint(model, config, log_metrics, os.path.join(run_checkpoint_dir, "best_acc.pt"))
                print(f" => Saved new best_acc.pt (Acc: {best_val_acc:.4f})")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, config, log_metrics, os.path.join(run_checkpoint_dir, "best_loss.pt"))
                print(f" => Saved new best_loss.pt (Loss: {best_val_loss:.4f})")

            # Free memory after validation
            del all_val_logits, all_val_targets, val_logits_tensor, val_targets_tensor
            clean_memory()
        
        # Log all metrics to wandb
        wandb.log(log_metrics)
        
    # Save the final model state at the end of the run
    print("[*] Training Complete. Saving last_acc.pt checkpoint.")
    save_checkpoint(model, config, log_metrics, os.path.join(run_checkpoint_dir, "last_acc.pt"))
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train V-JEPA Attention Probe.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json file.")
    args = parser.parse_args()
    
    main(args.config)


