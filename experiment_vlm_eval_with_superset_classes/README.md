# Video-Language Model (VLM) Object Bias Probing

**Purpose:**  
To evaluate if massive, state-of-the-art Vision-Language Models (VLMs) inherently exhibit appearance or object bias when tasked with zero-shot video action recognition.

**Experimental Setup:**  
- Condensed the 174 SSv2 action classes into smaller semantic supersets (~10 classes per prompt) to stabilize the VLM's output distribution.
- Engineered a pipeline to encode custom distractor videos into base64 format and query a cloud-hosted VLM API.
- Implemented strict system and human prompts forcing structured `json_object` responses to parse action probabilities, token usage, and latency.

**Models & Algorithms:**  
- **glm-5.3-flash (320B Parameters):** Hosted via GWDG Academic Cloud API.
- **OpenAI Python Wrapper:** Used for structuring API calls and JSON schema enforcement.

**Results:**  
Despite the massive parameter count and broad pretraining, the VLM did not exhibit significant object appearance bias on the custom distractor dataset, aligning with the robust performance of the self-supervised V-JEPA models.