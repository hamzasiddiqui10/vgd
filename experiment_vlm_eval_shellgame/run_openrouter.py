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
'261117-155543',
 '261117-155621',
 '261117-160815',
 '261117-160858',
 '261117-161017',
 '261117-161150',
]
print(len(vids_list))


# Load environment variables from .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"), # openrouter change
    base_url="https://openrouter.ai/api/v1", # openrouter change
    )





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




videos_dir = 'data/compressed_videos/'
results_path = "data/vlm_eval_results/shellgame_predictions_gemini-3.8-flash-agentic.jsonl"

pbar = tqdm(vids_list, desc="Processing Videos")

for video_name in pbar:
    # Make file path and load video in base64 format
    video_file_path = videos_dir + video_name + '_cam0.mp4'
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
                                "type": "text",
                                "text": user_prompt,
                            },
                            {
                                "type": "video_url",
                                "video_url": {
                                    "url": f"data:video/mp4;base64,{b64_video}",
                                    "processing": "agentic",
                                    },
                            },
                        ],
                    },
                ],
                extra_body={
                    # "provider": {
                    #     "order": ["Google AI Studio"],  
                    #     "allow_fallbacks": False,
                    # },
                    "reasoning": {"effort": "medium"},
                },
                response_format={"type": "json_object"},
                temperature=0.1,
                # timeout=180,
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
                print(f"[!] Skipping {video_name} after 3 failed attempts.")


            

