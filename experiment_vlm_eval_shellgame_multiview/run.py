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
    '261125-104457', # --> C     -- shellgame_mv_a 
    '261125-104554', # --> B     -- shellgame_mv_a 
    '261125-104705', # --> C     -- shellgame_mv_a 
    '261125-104732', # --> A     -- shellgame_mv_a 
    '261125-104759', # --> A     -- shellgame_mv_a 
    '261125-104837', # --> C     -- shellgame_mv_a 
    '261125-104917', # --> A     -- shellgame_mv_a 
    '261125-104952', # --> A     -- shellgame_mv_a 
    '261125-105943', # --> C     -- shellgame_mv_a 
    '261125-110012', # --> C     -- shellgame_mv_a 
    '261125-110038', # --> A     -- shellgame_mv_a 
    '261125-110106', # --> A     -- shellgame_mv_a 
    '261125-110135', # --> C     -- shellgame_mv_a 
    '261125-110711', # --> A     -- shellgame_mv_a 
    '261125-110736', # --> C     -- shellgame_mv_a 
    '261125-110758', # --> C     -- shellgame_mv_a 
    '261125-110830', # --> A     -- shellgame_mv_a 
    '261125-110901', # --> B     -- shellgame_mv_a 
    '261125-111934', # --> A     -- shellgame_mv_b 
    '261125-112001', # --> C     -- shellgame_mv_b 
    '261125-112026', # --> A     -- shellgame_mv_b 
    '261125-112050', # --> B     -- shellgame_mv_b 
    '261125-112116', # --> C     -- shellgame_mv_b 
    '261125-112139', # --> A     -- shellgame_mv_b 
    '261125-112205', # --> B     -- shellgame_mv_b 
    '261125-112306', # --> C     -- shellgame_mv_b 
    '261125-112333', # --> A     -- shellgame_mv_b 
    '261125-112521', # --> C     -- shellgame_mv_b 
    '261125-112540', # --> C     -- shellgame_mv_b 
    '261125-114107', # --> C     -- shellgame_mv_c 
    '261125-114146', # --> A     -- shellgame_mv_c 
    '261125-114209', # --> C     -- shellgame_mv_c 
    '261125-114232', # --> B     -- shellgame_mv_c 
    '261125-114306', # --> C     -- shellgame_mv_c 
    '261125-114327', # --> B     -- shellgame_mv_c 
    '261125-114354', # --> C     -- shellgame_mv_c 
    '261125-114427', # --> A     -- shellgame_mv_c 
    '261125-114451', # --> A     -- shellgame_mv_c 
    '261125-114516', # --> C     -- shellgame_mv_c 
    '261125-114540', # --> B     -- shellgame_mv_c 
    '261125-114606', # --> A     -- shellgame_mv_c 
]
print(len(vids_list))




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