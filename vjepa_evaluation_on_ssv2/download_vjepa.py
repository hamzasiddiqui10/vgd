"""
Download and cache V-JEPA 2 models from HuggingFace.

Run this script once on the login node (which has internet access).
The models are stored in the HuggingFace cache (~/.cache/huggingface by default)
and can then be loaded offline on compute nodes via local_files_only=True.

Models downloaded:
  1. facebook/vjepa2-vitg-fpc64-384       — pretrained backbone (no head)
  2. facebook/vjepa2-vitg-fpc64-384-ssv2  — backbone + SSv2 classification head (174 classes)
"""

from transformers import AutoVideoProcessor, AutoModel, AutoModelForVideoClassification

# ---- 1. Backbone-only model ----
HF_REPO_BACKBONE = "facebook/vjepa2-vitg-fpc64-384"
print(f"Downloading backbone model: {HF_REPO_BACKBONE}")
processor_backbone = AutoVideoProcessor.from_pretrained(HF_REPO_BACKBONE)
model_backbone = AutoModel.from_pretrained(HF_REPO_BACKBONE)
print(f"  Done: {HF_REPO_BACKBONE}\n")

# ---- 2. SSv2 classification model ----
HF_REPO_SSV2 = "facebook/vjepa2-vitg-fpc64-384-ssv2"
print(f"Downloading SSv2 classification model: {HF_REPO_SSV2}")
processor_ssv2 = AutoVideoProcessor.from_pretrained(HF_REPO_SSV2)
model_ssv2 = AutoModelForVideoClassification.from_pretrained(HF_REPO_SSV2)
print(f"  Done: {HF_REPO_SSV2}\n")

print("All models cached to HuggingFace cache directory.")
print("You can now run the evaluation script on compute nodes without internet access.")
