## conda env vgd has openai
import base64
import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
from tqdm import tqdm




## this supersets action classes file was made on 20th Oct 2026 manually by hamza.
ssv2_action_class_superset = pd.read_csv("/user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/testing/ssv2_action_classes_supersets.csv")
print(ssv2_action_class_superset.shape)
# ssv2_action_class_superset.sample(3)

## index_df
index_df = pd.read_csv('/user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/data/hamza_vgd_data/capture/index.csv')
index_df = index_df[~index_df['session'].str.startswith('vqa')].reset_index(drop=True)

index_df['cam'] = 'cam1'
index_df = index_df.rename(columns={'id':'video_id'})

# Bring in superset with a left join
index_df = pd.merge(index_df, ssv2_action_class_superset, how = 'left', on = ['template']) # LEFT JOIN

print(f"Total classes:             {ssv2_action_class_superset.template.nunique()}")
print(f"Total superset classes:    {ssv2_action_class_superset.template_superset.nunique()}")
print("-"*30)
print(f"myvideos total classes: {index_df.template.nunique()}")
print(f"myvideos superset classes: {index_df.template_superset.nunique()}")


vids_list_distractors = [
    '261027-125414', # distractors_are_endproducts | plug-inside-electrical-extension --> Putting [something] on a surface 
    '261027-125434', # distractors_are_endproducts | plug-inside-electrical-extension, wallet --> Putting [something] next to [something] 
    '261027-125510', # distractors_are_endproducts | board marker, plug-inside-electrical-extension --> Pushing [something] with [something] 
    '261027-125541', # distractors_are_endproducts | cable-inside-phone --> Putting [something] on a surface 
    '261027-125551', # distractors_are_endproducts | cable-inside-phone, board marker --> Putting [something] next to [something] 
    '261027-125622', # distractors_are_endproducts | cup, cable-inside-phone --> Pushing [something] with [something] 
    '261027-130210', # distractors_are_endproducts | two-torn-pieces-of-paper --> Putting [something] on a surface 
    '261027-130232', # distractors_are_endproducts | two-torn-pieces-of-paper, phone --> Putting [something] next to [something] 
    '261027-130312', # distractors_are_endproducts | two-torn-pieces-of-paper, books --> Pushing [something] with [something] 
    '261027-130336', # distractors_are_endproducts | bent-wire --> Putting [something] on a surface 
    '261027-130357', # distractors_are_endproducts | bent-wire, plate --> Putting [something] next to [something] 
    '261027-130424', # distractors_are_endproducts | bent-wire, phone --> Pushing [something] with [something]
]


# Load environment variables from .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GWDG_OPENAI_API_KEY"), 
    base_url="https://chat-ai.academiccloud.de/v1"
    )


## Define these before for loop once

labels_list = index_df.template_superset.unique().tolist()
labels_list.sort()

system_prompt = (
    "You are an expert video action classification model. "
    "Analyze video clips carefully and classify the primary action into a closed set of categories."
)

user_prompt = f"""
Analyze this short video and classify the action performed.

Candidate Action Classes:
{json.dumps(labels_list, indent=2)}

Instructions:
1. Assign a probability (float between 0.0 and 1.0) to EVERY class. The probabilities MUST sum to 1.0.
2. Select the top predicted class name.
3. Provide a brief 1-sentence reasoning for your visual observation.

Output strictly in JSON matching this schema:
{{
  "reasoning": "<1-sentence visual description>",
  "predicted_class": "<exact class name from list>",
  "probabilities": {{
    "{labels_list[0]}": 0.0,
    "{labels_list[1]}": 0.0,
    ...
  }}
}}
"""

videos_dir = '/user/h.siddiqui/u27673/.project/dir.project/hamza/repos/vgd/data/hamza_vgd_data/capture/raw/'
video_paths = [videos_dir + video + '_cam1.avi' for video in vids_list_distractors]
print(video_paths)


pbar = tqdm(video_paths, desc="Processing Videos")

for video_path in pbar:
    # Load video in b64
    with open(video_path, "rb") as f:
        b64_video = base64.b64encode(f.read()).decode("utf-8")

    # API call
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
                        "video_url": {"url": f"data:video/avi;base64,{b64_video}"},
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,  # Low temperature for deterministic evaluation outputs
    )

    elapsed_time = round(time.time() - start_time, 2)

    # Parse Output & Save data
    prediction_json = json.loads(response.choices[0].message.content)
    usage = response.usage

    run_result = {
        "video_file": os.path.basename(video_path),
        "model": response.model,
        "elapsed_seconds": elapsed_time,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        "predicted_class": prediction_json.get("predicted_class"),
        "reasoning": prediction_json.get("reasoning"),
        "probabilities": prediction_json.get("probabilities", {}),
    }

    with open('data/vlm_eval_results/action_predictions_distractors.jsonl', "a", encoding="utf-8") as f:
        f.write(json.dumps(run_result) + "\n")

