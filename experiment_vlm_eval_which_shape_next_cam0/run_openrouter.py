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


# Load environment variables from .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), # openrouter change
    base_url="https://openrouter.ai/api/v1", # openrouter change
    )



system_prompt = (
    "You are an expert multi-view 3D spatial-temporal reasoning and object-tracking model. "
    "You are provided with a single synchronized split-screen video showing two side-by-side camera views of the exact same physical event: "
    "the LEFT half is 'cam0' (a bird's-eye top-down camera looking straight down at the top face of a 4-sided rectangular paper prism), "
    "and the RIGHT half is 'cam1' (a table-level side camera looking horizontally at one vertical side face of the prism). "
    "Each of the four longitudinal sides of the prism has a unique geometric shape drawn on it. "
    "During the video, hands rotate the prism in consistent 90-degree turns around its longitudinal axis. "
    "You must synthesize spatial and temporal cues across both camera views to determine the 3D orientation of the prism, "
    "the direction of rotation, and which shape will appear on the top face ('cam0') after one additional 90-degree turn in the same direction."
)

user_prompt = """
Analyze this synchronized split-screen video ('cam0' bird's-eye top view on the left, 'cam1' table-level side view on the right) of a 4-sided paper prism being rotated, and predict which shape will appear on the top face in 'cam0' if the prism is rotated one more 90-degree turn in the exact same direction after the video ends.

Instructions:
1. Initial State: Identify the shape visible on the top face in 'cam0' and the shape visible on the side face in 'cam1' at the beginning of the video.
2. Rotation Tracking: Track every 90-degree rotation step chronologically across both views, noting how faces move between 'cam0' (top) and 'cam1' (side) to establish the exact direction of rotation and the prism's 3D face adjacency.
3. Final State at Video End: Identify the shape currently on the top face in 'cam0' and the shape currently on the side face in 'cam1' at the end of the recording.
4. Next-Turn Prediction: Extrapolate one additional 90-degree rotation in the exact same direction from the final state, and state the single lowercase name of the shape (e.g., "triangle", "square", "rectangle", "circle") that will appear on the top face in 'cam0'.
5. Metrics: Provide a confidence score (float between 0.0 and 1.0, where 1.0 is completely certain and 0.25 is random guessing among 4 sides), the exact integer number of video frames sampled and processed.

Output strictly in JSON matching this schema:
{
  "initial_cam0_shape": "<shape>",
  "initial_cam1_shape": "<shape>",
  "rotation_sequence": "start(cam0:square, cam1:rectangle) -> turn1(cam0:rectangle, cam1:triangle)",
  "final_cam0_shape": "<shape>",
  "final_cam1_shape": "<shape>",
  "predicted_next_cam0_shape": "<lowercase shape name>",
  "confidence": 0.0,
  "frames_seen": 60
}
"""

print(system_prompt)
print(user_prompt)




videos_dir = 'data/combined_videos/'
results_path = "data/vlm_eval_results/shellgame_predictions_gemini-3.8-flash-agentic.jsonl"

pbar = tqdm(vids_list[1:], desc="Processing Videos")

for video_name in pbar:
    # Make file path and load video in base64 format
    video_file_path = videos_dir + video_name + '_combined.mp4'
    with open(video_file_path, "rb") as f:
        b64_video = base64.b64encode(f.read()).decode("utf-8")

    # system_prompts and user_prompt exactly same for all videos.
    for attempt in range(3):
        try:
            # API call
            start_time = time.time()

            response = client.chat.completions.create(
                model="google/gemini-3.8-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{b64_video}",
                                    "processing": "agentic",
                                    },
                            },
                            {
                                "type": "text",
                                "text": user_prompt,
                            },

                        ],
                    },
                ],
                extra_body={
                    "reasoning": {"effort": "medium"},
                },
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            elapsed_time = round(time.time() - start_time, 2)

            # Extract JSON prediction content
            prediction_json = json.loads(response.choices[0].message.content)

            # Extract token & performance metadata
            usage = response.usage

            # 3. Update the dictionary to log both video file names
            run_result = {
                "video_file": os.path.basename(video_file_path),
                "model": response.model,
                "elapsed_seconds": elapsed_time,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
                "cost": usage.cost if usage else None,

                'initial_cam0_shape': prediction_json.get("initial_cam0_shape"),
                'initial_cam1_shape': prediction_json.get("initial_cam1_shape"),
                'rotation_sequence': prediction_json.get("rotation_sequence"),
                'final_cam0_shape': prediction_json.get("final_cam0_shape"),
                'final_cam1_shape': prediction_json.get("final_cam1_shape"),
                'predicted_next_cam0_shape': prediction_json.get("predicted_next_cam0_shape"),
                "confidence": prediction_json.get("confidence"),
                "frames_seen": prediction_json.get("frames_seen"),
            }

            # Append result as a single JSON line
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_result) + "\n")

            break

        except Exception as e:
            if attempt < 2:
                print(f"\n[!] [Attempt {attempt + 1}/3] {video_name} failed ({e}). Retrying in 7s...")
                time.sleep(7)
            else:
                print(f"[!!!] Skipping {video_name} after 3 failed attempts.")


            

