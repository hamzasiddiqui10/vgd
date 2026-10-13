"""
Download and cache V-JEPA 2 models from HuggingFace.

Run this script once on the login node (which has internet access).
The models are stored in the HuggingFace cache (~/.cache/huggingface by default)
and can then be loaded offline on compute nodes via local_files_only=True.

Models downloaded:
  1. apiantonio/vjepa2.1-vit-base-384     — pretrained backbone (no head)
"""

from transformers import AutoVideoProcessor, AutoModel

# 1. Update to the repository containing the proper model script
HF_REPO_VJEPA_BACKBONE = "apiantonio/vjepa2.1-vit-base-384"

# 2. Add trust_remote_code=True so transformers runs the custom architecture
processor = AutoVideoProcessor.from_pretrained(HF_REPO_VJEPA_BACKBONE, trust_remote_code=True)
model = AutoModel.from_pretrained(HF_REPO_VJEPA_BACKBONE, trust_remote_code=True)

print(f"  Done: {HF_REPO_VJEPA_BACKBONE}\n")