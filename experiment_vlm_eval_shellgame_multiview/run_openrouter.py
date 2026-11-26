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
'261125-104952',
 '261125-110758',
 '261125-110830',
 '261125-112050',
 '261125-112333',
 '261125-112521',
 '261125-112540',
 '261125-114107',
 '261125-114146',
 '261125-114209',
 '261125-114232',
 '261125-114354',
 '261125-114427',
 '261125-114451',
 '261125-114540',
 '261125-114606',
]
print(len(vids_list))


# Load environment variables from .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), # openrouter change
    base_url="https://openrouter.ai/api/v1", # openrouter change
    )



system_prompt = (
    "You are an expert multi-view visual object-tracking and spatial-temporal reasoning model. "
    "You are provided with a single synchronized split-screen video showing two side-by-side camera views of the exact same event: "
    "the LEFT half is 'cam0' and the RIGHT half is 'cam1'. "
    "In 'cam1', there are three clearly marked positions: A, B, and C, each starting with an inverted cup. "
    "At the start of the video, a puck is revealed under one of the cups in 'cam1'. "
    "The cups are then moved out of 'cam1' into 'cam0', shuffled inside 'cam0', and finally returned to 'cam1' onto markers A, B, and C. "
    "You must synthesize spatial and temporal cues across both camera views to continuously maintain the identity of the target cup across camera boundaries and determine its final resting position."
)

user_prompt = """
Analyze this synchronized split-screen video ('cam0' on the left, 'cam1' on the right) of a multi-view shell game and determine which position (A, B, or C) in 'cam1' hides the puck at the end of the video.

Valid Positions: {"A", "B", "C"}

Instructions:
1. Initial Reveal ('cam1' - Right): Identify the starting position (A, B, or C) where the puck is initially revealed under a cup at the beginning of the recording.
2. Track the cup with the puck chronologically:
  - Outbound Transition ('cam1' -> 'cam0'): Track the target cup as it leaves 'cam1' and enters 'cam0', noting its relative spatial order (e.g., left, middle, right) upon entering 'cam0'. 
  - Shuffle Tracking ('cam0' - Left): Continuously track the physical target cup through every swap and movement inside 'cam0'. 
  - Return Transition ('cam0' -> 'cam1'): Track the target cup as it exits 'cam0', re-enters 'cam1', and is placed onto one of the final markers (A, B, or C).
3. State your final predicted position ("A", "B", or "C") where the puck is located at the end of the video.
4. Provide a confidence score as a float between 0.0 and 1.0 denoting your certainty in the final predicted position (where 1.0 is completely certain and 0.33 is random guessing).
5. State the exact number of frames you sampled and processed from the given video in integer format.

Output strictly in JSON matching this schema:
{
  "initial_position": "<A, B, or C>",
  "swap_sequence": "cam1:A -> enter_cam0:left -> cam0_swap:right -> return_cam1:B",
  "predicted_position": "<A, B, or C>",
  "confidence": 0.0,
  "frames_seen": 60
}
"""

print(system_prompt)
print(user_prompt)




videos_dir = 'data/combined_videos/'
results_path = "data/vlm_eval_results/shellgame_predictions_gemini-3.8-flash-agentic.jsonl"

pbar = tqdm(vids_list, desc="Processing Videos")

for video_name in pbar:
    # Make file path and load video in base64 format
    video_file_path = videos_dir + video_name + '_combined.mp4'
    with open(video_file_path, "rb") as f:
        b64_video = base64.b64encode(f.read()).decode("utf-8")

    # system_prompts and user_prompt exactly same for all videos.
    for attempt in range(2):
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
                "initial_position": prediction_json.get("initial_position"),
                "predicted_position": prediction_json.get("predicted_position"),
                "confidence": float(prediction_json.get("confidence", 0.0)),
                "swap_sequence": prediction_json.get("swap_sequence"),
                "frames_seen": prediction_json.get("frames_seen"),
                "camera_setup_reasoning": prediction_json.get("camera_setup_reasoning", "disabled"),
            }

            # Append result as a single JSON line
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(run_result) + "\n")

            break

        except Exception as e:
            if attempt < 1:
                print(f"\n[!] [Attempt {attempt + 1}/2] {video_name} failed ({e}). Retrying in 7s...")
                time.sleep(7)
            else:
                print(f"[!!!] Skipping {video_name} after 2 failed attempts.")


            

