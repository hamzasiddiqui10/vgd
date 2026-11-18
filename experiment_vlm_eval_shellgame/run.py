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



vids_list = [
    '261117-155127', # --> C     -- shellgame-medium 
    '261117-155207', # --> B     -- shellgame-medium 
    '261117-155252', # --> A     -- shellgame-medium 
    '261117-155426', # --> C     -- shellgame-medium 
    '261117-155503', # --> C     -- shellgame-medium 
    '261117-155543', # --> B     -- shellgame-fast 
    '261117-155621', # --> C     -- shellgame-fast 
    '261117-155722', # --> B     -- shellgame-fast 
    '261117-160815', # --> A     -- shellgame-fast 
    '261117-160858', # --> A     -- shellgame-fast 
    '261117-160936', # --> B     -- shellgame-fast 
    '261117-161017', # --> C     -- shellgame-fast 
    '261117-161053', # --> B     -- shellgame-fast 
    '261117-161121', # --> C     -- shellgame-fast 
    '261117-161150', # --> C     -- shellgame-fast 
    '261117-161257', # --> B     -- shellgame-medium 
    '261117-161345', # --> A     -- shellgame-medium 
    '261117-161420', # --> A     -- shellgame-medium 
    '261117-161455', # --> B     -- shellgame-medium 
    '261117-161544', # --> A     -- shellgame-medium 
    '261117-161811', # --> C     -- shellgame-slow 
    '261117-161859', # --> A     -- shellgame-slow 
    '261117-161936', # --> A     -- shellgame-slow 
    '261117-162009', # --> C     -- shellgame-slow 
    '261117-162259', # --> A     -- shellgame-slow 
    '261117-162319', # --> C     -- shellgame-slow 
    '261117-162343', # --> B     -- shellgame-slow 
    '261117-162402', # --> B     -- shellgame-slow 
    '261117-162434', # --> A     -- shellgame-slow 
    '261117-162457', # --> A     -- shellgame-slow 
    '261117-162515', # --> B     -- shellgame-slow 
    '261117-162538', # --> C     -- shellgame-slow 

]
print(len(vids_list))


# Load environment variables from .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GWDG_OPENAI_API_KEY"), 
    base_url="https://chat-ai.academiccloud.de/v1"
)







videos_dir = 'data/compressed_videos/'  # Directory containing the compressed videos








system_prompt = (
    "You are an expert visual object-tracking and spatial-temporal reasoning model. "
    "You are provided with a single bird's-eye view video of a table with three clearly marked positions: A, B, and C, each having an inverted cup. "
    "At the beginning of the video, a puck is revealed under one of the cups, after which the cups are shuffled. "
    "Track the target cup continuously across all movements to determine its final position."
)

user_prompt = f"""
Analyze this bird's-eye view video of a shell game and determine which position (A, B, or C) currently hides the puck at the end of the video.

Valid Positions: {{"A", "B", "C"}}

Instructions:
1. Identify the starting position (A, B, or C) where the puck is initially revealed under a cup at the start of the video.
2. Track the cup with the puck chronologically through every move or swap from start to finish, noting how the target cup's position changes at each step.
3. State your final predicted position ("A", "B", or "C") where the puck is located at the end of the video.
4. Provide a confidence score as a float between 0.0 and 1.0 denoting your certainty in the final predicted position (where 1.0 is completely certain and 0.33 is random guessing).

Output strictly in JSON matching this schema:|
{{
  "initial_position": "<A, B, or C>",
  "swap_sequence": "A -> C -> A -> B", 
  "predicted_position": "<A, B, or C>",
  "confidence": 0.0
}}
"""

print(system_prompt)
print(user_prompt)






results_path = "data/vlm_eval_results/shellgame_predictions.jsonl"





# 1. Load already-completed video names so you never lose progress
completed_videos = set()
if os.path.exists(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                completed_videos.add(json.loads(line)["video_file"])

print(f"[*] Found {len(completed_videos)} completed videos. Resuming from last checkpoint.")
print(f"[*] -- Completed Videos: {completed_videos}")



pbar = tqdm(vids_list[2:], desc="Processing Videos")

for video_name in pbar:
    # Make file path and check against completed checkpoint
    video_file_path = os.path.join(videos_dir, f"{video_name}_cam0.mp4")
    video_file_basename = os.path.basename(video_file_path)

    if video_file_basename in completed_videos:
        continue

    with open(video_file_path, "rb") as f:
        b64_video = base64.b64encode(f.read()).decode("utf-8")

    # 2. Retry up to 3 times with backoff on 504 / timeout errors
    for attempt in range(3):
        try:
            start_time = time.time()

            response = client.chat.completions.create(
                model="glm-5.3-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "video_url",
                                "video_url": {"url": f"data:video/mp4;base64,{b64_video}"},
                            },
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=180.0,
            )

            elapsed_time = round(time.time() - start_time, 2)

            # Extract JSON prediction content
            prediction_json = json.loads(response.choices[0].message.content)

            # Extract token & performance metadata
            usage = response.usage

            run_result = {
                "video_file": video_file_basename,
                "model": response.model,
                "elapsed_seconds": elapsed_time,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
                "initial_position": prediction_json.get("initial_position"),
                "predicted_position": prediction_json.get("predicted_position"),
                "confidence": float(prediction_json.get("confidence", 0.0)),
                "swap_sequence": prediction_json.get("swap_sequence"),
            }

            # Append result as a single JSON line
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_result) + "\n")

            completed_videos.add(video_file_basename)
            break  # Success, move to next video

        except (openai.InternalServerError, openai.APITimeoutError, openai.RateLimitError, json.JSONDecodeError) as e:
            print(f"\n[Attempt {attempt + 1}/3] {video_file_basename} failed ({e}). Retrying in 10s...")
            time.sleep(10)
    else:
        print(f"\nSkipping {video_file_basename} after 3 failed attempts.")