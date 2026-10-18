"""
V-JEPA Attention Probe Inference Script

This script loads a trained V-JEPA attention probe checkpoint and runs inference 
on a single HDF5 dataset containing features, labels, and names. It calculates 
Top-1 and Top-5 standard and balanced accuracies, and exports the results.

Usage Example:
    python scripts/infer.py \
        --vjepa_feats_h5_path ./data/output_features/test_shard_00.h5 \
        --ckpt_path ./checkpoints/ssv2-vjepa-probe_261016_0853/best_acc.pt \
        --results_path ./results/inference_results/ssv2_test_100pc_model/ \
        --save_logits_and_metadata False \
        --batch_size 128 \
        --num_workers 12

"""

import argparse
import json
import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

# ---------------------------------------------------------
# 0. Model Definition (Same as train.py)
# ---------------------------------------------------------
class VJepaAttentiveProbe(nn.Module):
    """Attentive probe for V-JEPA features loaded directly from HDF5."""
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
        B = x.shape[0]                         
        query = self.query.expand(B, -1, -1)    

        x = self.norm_attn(x)
        query = query + self.attn(query, x, x)[0]
        query = query + self.mlp(self.norm_mlp(query))
        return query.squeeze(1)                  

    def forward(self, x):
        feat = self.forward_feat(x)
        return self.linear(feat)


# ---------------------------------------------------------
# 1. Dataset for Inference
# ---------------------------------------------------------
class VJepaH5InferDataset(Dataset):
    """
    Inference Dataset for loading V-JEPA 2.1 HDF5 features.
    Extracts features, labels, and names/IDs for granular tracking.
    """
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
            
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['labels'])
            # Safely identify the correct key for names (usually 'name' or 'names')
            self.name_key = 'name' if 'name' in f else 'names'
            if self.name_key not in f:
                print("Warning: No 'name' or 'names' key found in HDF5. Using numerical indices.")
                
        self.h5_handle = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Lazy load to avoid multiprocessing issues
        if self.h5_handle is None:
            self.h5_handle = h5py.File(self.h5_path, 'r', swmr=True)
            
        f = self.h5_handle
        
        features = torch.from_numpy(f['features'][idx]).to(torch.float32)
        label = torch.tensor(f['labels'][idx], dtype=torch.long)
        
        if self.name_key in f:
            name = f[self.name_key][idx]
            # Handle byte decoding if HDF5 strings are stored as bytes
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            elif isinstance(name, np.bytes_):
                name = name.astype(str)
        else:
            name = str(idx)
            
        return features, label, name


# ---------------------------------------------------------
# 2. Main Inference Routine
# ---------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running inference on device: {device}")
    
    os.makedirs(args.results_path, exist_ok=True)
    
    # 1. Setup Model & Load Checkpoint
    model = VJepaAttentiveProbe(dim_embed=768, seq_len=4608, num_heads=12, num_classes=174)
    
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
        
    print(f"[*] Loading checkpoint: {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 2. Setup Dataloader
    dataset = VJepaH5InferDataset(args.vjepa_feats_h5_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
    
    all_logits = []
    all_labels = []
    all_names = []
    total_loss = 0.0
    
    # 3. Inference Loop
    print(f"[*] Starting inference on {len(dataset)} samples...")
    with torch.no_grad():
        for features, labels, names in tqdm(dataloader, desc="Inference"):
            features, labels = features.to(device), labels.to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            # Multiply by batch size to compute accurate average loss later
            total_loss += loss.item() * features.size(0)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_names.extend(names)
            
    # 4. Concatenate Results & Compute Probabilities
    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    probs_tensor = torch.softmax(logits_tensor, dim=-1)
    
    avg_loss = total_loss / len(dataset)
    
    # 5. Compute Top-1 & Top-5 Predictions
    _, top5_preds = logits_tensor.topk(5, dim=1, largest=True, sorted=True)
    top1_preds = top5_preds[:, 0]
    
    top1_correct_mask = (top1_preds == labels_tensor)
    top5_correct_mask = (top5_preds == labels_tensor.unsqueeze(1)).any(dim=1)
    
    # 6. Calculate Standard Metrics
    top1_acc = top1_correct_mask.float().mean().item()
    top5_acc = top5_correct_mask.float().mean().item()
    
    # 7. Calculate Balanced Metrics
    labels_np = labels_tensor.numpy()
    top1_preds_np = top1_preds.numpy()
    top1_bal_acc = balanced_accuracy_score(labels_np, top1_preds_np)
    
    classes = np.unique(labels_np)
    top5_correct_np = top5_correct_mask.numpy()
    top5_bal_acc = np.mean([
        top5_correct_np[labels_np == c].mean() for c in classes
    ]) if len(classes) > 0 else 0.0
    
    print("\n" + "="*30)
    print("Inference Metrics")
    print("="*30)
    print(f"Loss:           {avg_loss:.4f}")
    print(f"Top-1 Acc:      {top1_acc:.4f}")
    print(f"Top-1 Bal Acc:  {top1_bal_acc:.4f}")
    print(f"Top-5 Acc:      {top5_acc:.4f}")
    print(f"Top-5 Bal Acc:  {top5_bal_acc:.4f}")
    
    # 8. Save Metrics JSON
    metrics_dict = {
        "loss": avg_loss,
        "top1acc": top1_acc,
        "top1balacc": top1_bal_acc,
        "top5acc": top5_acc,
        "top5balacc": top5_bal_acc,
        "inference_args": vars(args)
    }
    json_path = os.path.join(args.results_path, "inference_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"\n[*] Saved metrics to {json_path}")
    
    # 9. Optional: Save Granular Metadata & Logits
    if args.save_logits_and_metadata:
        # Save Parquet Metadata
        df = pd.DataFrame({
            "name": all_names,
            "label": labels_np,
            "top1_pred": top1_preds_np,
            "correct_top1": top1_correct_mask.numpy(),
            "correct_top5": top5_correct_np
        })
        parquet_path = os.path.join(args.results_path, "predictions.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"[*] Saved metadata predictions to {parquet_path}")
        
        # Save NPZ Logits & Probabilities
        npz_path = os.path.join(args.results_path, "predictions.npz")
        np.savez_compressed(
            npz_path, 
            logits=logits_tensor.numpy(), 
            probabilities=probs_tensor.numpy()
        )
        print(f"[*] Saved logits and probabilities to {npz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V-JEPA Attention Probe Inference.")
    parser.add_argument("--vjepa_feats_h5_path", type=str, required=True, help="Path to target HDF5 features file.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained PyTorch checkpoint (.pt).")
    parser.add_argument("--results_path", type=str, required=True, help="Directory to save inference results.")
    parser.add_argument("--save_logits_and_metadata", type=str2bool, default=False, 
                        help="Boolean (True/False): Save granular parquet metadata and compressed npz logits.")
    parser.add_argument("--batch_size", type=int, required=True, help="data loader batch size.")
    parser.add_argument("--num_workers", type=int, required=True, help="data loader num workers.")

    args = parser.parse_args()
    
    main(args)


