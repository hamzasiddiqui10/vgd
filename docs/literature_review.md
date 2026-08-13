# Literature Review: Video Action Generalization


## Multi-Modal Domain Adaptation for Fine-Grained Action Recognition
EPIC-Kitchens D1/D2/D3: Three kitchens from EPIC-Kitchens-55 treated as separate domains over 8 verb classes, yielding 6 source→target pairs.
Multi-Modal Domain Adaptation for Fine-Grained Action Recognition (MM-SADA), CVPR 2020 https://arxiv.org/abs/2001.09691

### Basics:

* **Base Dataset(s):** EPIC-Kitchens (specifically environments D1, D2, and D3 corresponding to participants P08, P01, and P22).
* **Dataset Scale:** 10,094 video clips. Total hours: Not reported.
* **Classes:** 8 unique classes (Human-Object Interaction).
* **Example Classes:** put, take, open, close, wash.

#### Domain Shift Tested

The paper evaluates cross-environment domain adaptation in egocentric video across three distinct real-world kitchen environments. Models are trained on labeled action segments from a source kitchen and tested on an unlabeled target kitchen. Generalization is evaluated against environmental changes including background geometry, spatial layouts, lighting conditions, camera angles, and visual clutter while maintaining egocentric viewpoints.

#### Non-Object Shift Suitability

This dataset provides a useful framework for evaluating non-object environmental shifts. Keeping the verb label set consistent across different physical kitchens isolates shifts caused by room architecture, background lighting, visual distraction, and participant-specific egocentric motion. Incorporating this setup allows testing model resilience to spatial background shifts without changing the underlying action definitions.

#### Verb / Motion Gap Filled

Although the paper isolates verb classes across kitchens, the actions remain heavily tied to egocentric kitchen settings where co-occurring objects (e.g., taps, fridges, drawers) strongly correlate with specific actions. Our new 1000+ clip dataset addresses this limitation by evaluating temporal motion patterns across diverse non-kitchen environments and broader contexts, prioritizing pure verb dynamics over object appearances to mitigate object bias.

#### Technique to Create Synthetic Dataset

No Strategy
