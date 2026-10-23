# Object Bias Probing: V-JEPA2-G Pretrained Backbone

**Purpose:**  
To evaluate the out-of-domain transfer capabilities and object bias resistance of massive self-supervised video models when exposed to intentional visual distractors.

**Experimental Setup:**  
- Created a custom dataset using a synchronized 3-camera setup, recording SSv2-style hand-object interactions containing intentional distractor objects (designed to spike conditional probability biases).
- Evaluated the pretrained model on both the official SSv2 validation set and the custom distractor dataset.
- Conducted granular, instance-level analysis on class probability scores to detect over-reliance on distractors.

**Models & Algorithms:**  
- **V-JEPA2-G (facebook/vjepa2-g-ssv2):** Pretrained video backbone.
- **Pretrained Cross-Attention Probe:** Applied directly for SSv2 action classification.

**Results:**  
The model demonstrated extremely robust feature representations. The granular probability analysis revealed that the V-JEPA2-G architecture successfully prioritizes temporal action dynamics and is not easily distracted by spatial object biases.