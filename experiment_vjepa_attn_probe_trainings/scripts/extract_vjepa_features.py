#!/usr/bin/env python3
"""
Feature Extraction Script for Something-Something V2 (SSv2) using V-JEPA 2.1
Extracts features and stores them into sharded HDF5 files.

python extract_vjepa_features.py \
  --annotations /path/to/train.txt \
  --video_dir /path/to/20bn-something-something-v2 \
  --output_dir /path/to/output_features \
  --shard_prefix train \
  --shard_size 30000 \
  --batch_size 32 \
  --num_workers 8 \
  --device cuda

"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import numpy as np
import h5py
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import av

# Suppress HuggingFace warnings for cleaner CLI output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from transformers import AutoConfig, AutoVideoProcessor, AutoModel

class SSv2Dataset(Dataset):
    """
    Custom Dataset to uniformly load frames from SSv2 videos.
    Returns RGB frames ready to be ingested by the HuggingFace video processor.
    """
    def __init__(self, annotations_file, video_dir, num_frames=16):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.samples = []
        
        # Parse the annotation file
        with open(annotations_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                vid_name = parts[0]
                label_id = int(parts[1])
                self.samples.append((vid_name, label_id))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        vid_name, label_id = self.samples[idx]
        vid_path = os.path.join(self.video_dir, vid_name)
        
        try:
            if not os.path.exists(vid_path):
                raise FileNotFoundError(f"Missing file: {vid_path}")

            # Video Decoding using decord (commented out due to robustness issues with short WebM files)
            # vr = VideoReader(vid_path, ctx=cpu(0))
            # total_frames = len(vr)
            
            # if total_frames == 0:
            #     raise ValueError("Video contains 0 frames.")
                
            # # Frame Sampling Strategy
            # if total_frames >= self.num_frames:
            #     # Uniformly sample across the video duration
            #     indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            # else:
            #     # Wrap-around / repeat indices if the video is too short
            #     indices = np.array([i % total_frames for i in range(self.num_frames)])
                
            # # Extract frames: Shape returns as (16, H, W, 3) in uint8 RGB
            # frames = vr.get_batch(indices).asnumpy()
            # del vr  # explicitly free reader from worker memory
            
            # Video decoding using PyAV for better robustness with short WebM files
            with av.open(vid_path) as container:
                # Sequentially decode all frames (highly robust for short WebM files)
                video_frames = [
                    frame.to_ndarray(format='rgb24') 
                    for frame in container.decode(video=0)
                ]
            
            total_frames = len(video_frames)
            
            if total_frames == 0:
                raise ValueError("Video contains 0 frames or could not be decoded.")
                
            # Frame Sampling Strategy
            if total_frames >= self.num_frames:
                # Uniformly sample across the video duration
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            else:
                # Wrap-around / repeat indices if the video is too short
                indices = np.array([i % total_frames for i in range(self.num_frames)])
                
            # Stack the sampled frames into a single numpy array: Shape (16, H, W, 3)
            frames = np.stack([video_frames[i] for i in indices])



            return {
                "vid_name": vid_name,
                "label_id": label_id,
                "frames": frames,
                "error": None
            }
            
        except Exception as e:
            return {
                "vid_name": vid_name,
                "label_id": label_id,
                "frames": None,
                "error": str(e)
            }


def custom_collate(batch):
    """
    Filters out failed video loads and constructs batched lists for processing.
    """
    valid_batch = [item for item in batch if item['error'] is None]
    error_batch = [item for item in batch if item['error'] is not None]
    
    if len(valid_batch) == 0:
        return None, error_batch
        
    vid_names = [item['vid_name'] for item in valid_batch]
    label_ids = [item['label_id'] for item in valid_batch]
    frames = [item['frames'] for item in valid_batch]
    
    return {
        "vid_names": vid_names,
        "label_ids": np.array(label_ids, dtype=np.int64),
        "frames": frames
    }, error_batch


class HDF5ShardWriter:
    """
    Efficient HDF5 writer that automatically rotates to a new shard file 
    once `max_size` is reached. Pre-allocates datasets for faster chunking I/O.
    """
    def __init__(self, output_dir, shard_prefix, max_size=30000):
        self.output_dir = output_dir
        self.shard_prefix = shard_prefix
        self.max_size = max_size
        self.current_shard_idx = 0
        self.current_count = 0
        
        self.h5_file = None
        self._open_new_shard()
        
    def _open_new_shard(self):
        if self.h5_file is not None:
            self._close_current_shard()
            
        shard_name = f"{self.shard_prefix}_shard_{self.current_shard_idx:02d}.h5"
        h5_path = os.path.join(self.output_dir, shard_name)
        self.h5_file = h5py.File(h5_path, 'w')
        
        chunk_size = min(32, self.max_size)
        
        # Pre-allocate schema
        self.features_ds = self.h5_file.create_dataset(
            'features', 
            shape=(self.max_size, 4608, 768), 
            maxshape=(self.max_size, 4608, 768), 
            dtype=np.float16, 
            chunks=(chunk_size, 4608, 768)
        )
        self.labels_ds = self.h5_file.create_dataset(
            'labels', 
            shape=(self.max_size,), 
            maxshape=(self.max_size,),
            dtype=np.int64
        )
        self.names_ds = self.h5_file.create_dataset(
            'names', 
            shape=(self.max_size,), 
            maxshape=(self.max_size,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        self.current_count = 0
        
    def write_batch(self, features, labels, names):
        batch_size = len(features)
        
        # Check if the incoming batch spills over the shard boundary
        if self.current_count + batch_size > self.max_size:
            remaining = self.max_size - self.current_count
            if remaining > 0:
                self._append_data(features[:remaining], labels[:remaining], names[:remaining])
            
            self.current_shard_idx += 1
            self._open_new_shard()
            
            features = features[remaining:]
            labels = labels[remaining:]
            names = names[remaining:]
            batch_size = len(features)
            
        if batch_size > 0:
            self._append_data(features, labels, names)
            
    def _append_data(self, features, labels, names):
        start = self.current_count
        end = start + len(features)
        
        self.features_ds[start:end] = features
        self.labels_ds[start:end] = labels
        self.names_ds[start:end] = names
        
        self.current_count += len(features)
        
    def _close_current_shard(self):
        if self.h5_file is not None:
            # Downsize the pre-allocated datasets to match exactly what was written
            self.features_ds.resize((self.current_count, 4608, 768))
            self.labels_ds.resize((self.current_count,))
            self.names_ds.resize((self.current_count,))
            self.h5_file.close()

    def close(self):
        self._close_current_shard()


def main():
    parser = argparse.ArgumentParser(description="Extract features from SSv2 using V-JEPA 2.1")
    parser.add_argument('--annotations', type=str, required=True, help="Path to the input .txt annotation file")
    parser.add_argument('--video_dir', type=str, required=True, help="Directory containing .webm videos")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save HDF5 shards")
    parser.add_argument('--shard_prefix', type=str, default="train", help="Base name for shards (e.g., train, val)")
    parser.add_argument('--shard_size', type=int, default=30000, help="Max videos per shard (Default: 30000)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for extraction")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of CPU workers for dataloader")
    parser.add_argument('--device', type=str, default="cuda", choices=['cuda', 'cpu'], help="Target inference device")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    failed_log_path = os.path.join(args.output_dir, "failed_videos.txt")
    
    print(f"Loading V-JEPA 2.1 processor and model to {args.device}...")
    model_id = "apiantonio/vjepa2.1-vit-base-384"
    
    # Load Model and Processor
    processor = AutoVideoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.to(args.device)
    model.eval()

    # Dataloader Setup
    dataset = SSv2Dataset(args.annotations, args.video_dir, num_frames=16)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=custom_collate,
        pin_memory=(args.device == 'cuda')
    )
    
    writer = HDF5ShardWriter(args.output_dir, args.shard_prefix, max_size=args.shard_size)
    
    print(f"Starting extraction for {len(dataset)} videos...")
    
    # Process batches
    with open(failed_log_path, 'w') as f_err:
        pbar = tqdm(total=len(dataset), desc=f"Extracting {args.shard_prefix}", unit="vid")
        
        for valid_batch, error_batch in dataloader:
            
            # Log any videos that threw exceptions (corruption, 0-frames, etc.)
            for err_item in error_batch:
                f_err.write(f"{err_item['vid_name']}\t{err_item['error']}\n")
                f_err.flush()
                
            if valid_batch is not None:
                # Processor builds tensor [B, 3, 16, 384, 384] and normalizes automatically
                inputs = processor(videos=valid_batch['frames'], return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # Utilize mixed-precision acceleration if on CUDA to vastly speed up V-JEPA inference
                    if args.device == 'cuda':
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                        
                # Extract features, cast to float16, and dump to numpy
                features = outputs.last_hidden_state.cpu().numpy().astype(np.float16)
                
                writer.write_batch(
                    features=features,
                    labels=valid_batch['label_ids'],
                    names=valid_batch['vid_names']
                )
                
            # Progress bar update factors in both successfully processed and skipped (failed) videos
            processed_count = len(valid_batch['frames']) if valid_batch else 0
            pbar.update(len(error_batch) + processed_count)
            
        pbar.close()
    
    writer.close()
    print(f"Extraction complete! Shards and error logs successfully saved to {args.output_dir}")

if __name__ == "__main__":
    main()


