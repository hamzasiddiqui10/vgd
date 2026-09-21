"""
Download and cache the V-JEPA 2 pretrained backbone from HuggingFace.
Model: facebook/vjepa2-vitg-fpc64-384
"""

from transformers import AutoVideoProcessor, AutoModel

HF_REPO = "facebook/vjepa2-vitg-fpc64-384"

print(f"Downloading model: {HF_REPO}")
processor = AutoVideoProcessor.from_pretrained(HF_REPO)
model = AutoModel.from_pretrained(HF_REPO)
print("Done. Model and processor cached to HuggingFace cache directory.")
