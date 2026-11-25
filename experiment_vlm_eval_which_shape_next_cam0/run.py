## conda env vgd has openai
import base64
import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2


vids_list = [
'261125-125558', # --> square -- which_shape_next_cam0_a 
'261125-125633', # --> triangle -- which_shape_next_cam0_a 
'261125-125913', # --> rectangle -- which_shape_next_cam0_a 
'261125-130240', # --> circle -- which_shape_next_cam0_a 
'261125-130305', # --> triangle -- which_shape_next_cam0_a 
'261125-130345', # --> square -- which_shape_next_cam0_a 
'261125-130412', # --> rectangle -- which_shape_next_cam0_b 
'261125-130439', # --> rectangle -- which_shape_next_cam0_b 
'261125-130458', # --> triangle -- which_shape_next_cam0_b 
'261125-130522', # --> circle -- which_shape_next_cam0_b 
'261125-130550', # --> rectangle -- which_shape_next_cam0_b 
'261125-131157', # --> square -- which_shape_next_cam0_a 
'261125-131216', # --> circle -- which_shape_next_cam0_a 
'261125-131241', # --> square -- which_shape_next_cam0_a 
'261125-131306', # --> triangle -- which_shape_next_cam0_a 
'261125-131326', # --> square -- which_shape_next_cam0_a
]

print(len(vids_list))




## compress cam0
import os
import subprocess
from tqdm import tqdm

raw_dir = "/user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/data/hamza_vgd_data/capture/raw"
compressed_dir = "data/compressed_videos"

os.makedirs(compressed_dir, exist_ok=True)

pbar = tqdm(vids_list, desc="Compressing Videos - cam0")
for vid in pbar:
    # Strip extension if vids_list already includes '.avi'
    stem = os.path.splitext(vid)[0]
    
    input_path = os.path.join(raw_dir, f"{stem}_cam0.avi")       # Change to f"{stem}_cam0.avi" if suffix is still used
    output_path = os.path.join(compressed_dir, f"{stem}_cam0.mp4")

    if os.path.exists(output_path):
        print(f"[-] Skipping (already exists): {stem}_cam0.mp4")
        continue

    if not os.path.exists(input_path):
        print(f"[!] Warning (missing input): {input_path}")
        continue

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-i", input_path,
            "-vf", "fps=20,scale=-2:480p",
            "-c:v", "libx264",
            "-crf", "23",
            "-an",
            output_path,
        ],
        check=True,
    )

    orig_mb = os.path.getsize(input_path) / (1024 * 1024)
    new_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[*] Converted {stem}: {orig_mb:.2f} MB -> {new_mb:.2f} MB")




### compress cam1
raw_dir = "/user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/data/hamza_vgd_data/capture/raw"
compressed_dir = "data/compressed_videos"

os.makedirs(compressed_dir, exist_ok=True)

pbar = tqdm(vids_list, desc="Compressing Videos - cam1")
for vid in pbar:
    # Strip extension if vids_list already includes '.avi'
    stem = os.path.splitext(vid)[0]
    
    input_path = os.path.join(raw_dir, f"{stem}_cam1.avi")       # Change to f"{stem}_cam0.avi" if suffix is still used
    output_path = os.path.join(compressed_dir, f"{stem}_cam1.mp4")

    if os.path.exists(output_path):
        print(f"[-] Skipping (already exists): {stem}_cam1.mp4")
        continue

    if not os.path.exists(input_path):
        print(f"[!] Warning (missing input): {input_path}")
        continue

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-i", input_path,
            "-vf", "fps=20,scale=-2:480p",
            "-c:v", "libx264",
            "-crf", "23",
            "-an",
            output_path,
        ],
        check=True,
    )

    orig_mb = os.path.getsize(input_path) / (1024 * 1024)
    new_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[*] Converted {stem}: {orig_mb:.2f} MB -> {new_mb:.2f} MB")








## combine videos

def concatenate_videos_side_by_side(video_path_0, video_path_1, output_path):
    """Stitches two synchronized videos horizontally into a single video file."""
    if os.path.exists(output_path):
        print(f"[-] Skipping concatenation: '{output_path}' already exists.")
        return
    
    cap0 = cv2.VideoCapture(video_path_0)
    cap1 = cv2.VideoCapture(video_path_1)

    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    width0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Match height of video 1 to video 0 if they differ
    target_height = height0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Combined width = width of cam0 + width of cam1
    out_width = width0 + int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH) * (target_height / height1))
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, target_height))

    while cap0.isOpened() and cap1.isOpened():
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        # Resize frame1 if heights don't match exactly
        if frame1.shape[0] != target_height:
            new_w = int(frame1.shape[1] * (target_height / frame1.shape[0]))
            frame1 = cv2.resize(frame1, (new_w, target_height))

        # Concatenate horizontally (Left: View 0 | Right: View 1)
        combined_frame = cv2.hconcat([frame0, frame1])
        out.write(combined_frame)

    cap0.release()
    cap1.release()
    out.release()

    print(f"[*] New file created: {output_path}")




videos_file_path = "data/compressed_videos/"
combined_videos_file_path = 'data/combined_videos/'

pbar = tqdm(vids_list, desc="Combining Videos")
for video_name in pbar:
    path_cam0 = f"{videos_file_path}{video_name}_cam0.mp4"
    path_cam1 = f"{videos_file_path}{video_name}_cam1.mp4"
    combined_path = f"{combined_videos_file_path}{video_name}_combined.mp4"

    # Stitch the videos horizontally
    concatenate_videos_side_by_side(path_cam0, path_cam1, combined_path)

