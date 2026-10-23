# Object Bias Probing: 16-Frame DINOv2

**Purpose:**  
To determine if expanding the input context from a single center frame to a 16-frame temporal sequence mitigates object bias in spatial-first foundation models.

**Experimental Setup:**  
- Sampled 16 uniform frames per video from both the SSv2 validation set and the custom multi-camera distractor dataset.
- Extracted per-frame features and trained a cross-attention probe over the temporal sequence to classify actions.
- Conducted instance-level probability analysis to identify patterns of distraction.

**Models & Algorithms:**  
- **DINOv2:** Feature extraction across 16 frames per video.
- **Cross-Attention Probe:** Aggregates temporal frame features for final action classification.

**Results:**  
Incorporating 16 frames provided sufficient temporal context to largely overcome object bias. The performance ranked cleanly between the single-frame DINOv2 baseline (lowest) and the natively temporal V-JEPA2-G model (highest).