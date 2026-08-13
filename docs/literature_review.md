# Literature Review: Video Action Generalization

| Paper Title | Base Dataset | Scale | Classes | Domain Shift Tested | Non-Object Shift Suitability | Verb/Motion Gap Filled |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Something-Something v2** (Goyal et al.) | Something-Something v1 | 220,847 clips (approx. 215 hrs) | 174 unique classes (Motion-based / Fine-grained verbs) | Unseen objects, background variations, and daily physical interactions. | **High:** Isolates motion by pairing similar verbs with vastly different everyday objects, preventing models from relying on static scenes. | **Covers core gap:** Provides an excellent template for fine-grained verb control, but lacks explicit cross-view or camera-motion shifts. |





# Literature Review: Video Action Generalization

## Quick Table of Contents
* [1. Something-Something v2](#1-something-something-v2)
* [2. EPIC-KITCHENS-100](#2-epic-kitchens-100)

---

## 1. Something-Something v2

### Metadata
* **Base Dataset:** Something-Something v1
* **Dataset Scale:** ~220,847 video clips (approx. 215 hours)
* **Classes:** 174 unique classes (Motion-based / Fine-grained verbs)
* **Example Classes:** *Pushing something from left to right*, *Tipping something over*, *Pretending to open something*

### Detailed Analysis

#### Domain Shift Tested
The paper evaluates domain generalization across unseen physical objects and diverse background environments. Models are trained on specific object-verb pairings and tested on completely novel object instances performing the same action. The exact domain shift forces the network to decouple the motion representation from object visual features.

#### Non-Object Shift Suitability
**High Suitability.** Because actors perform identical actions with vastly different everyday objects, this dataset is an exceptional benchmark for testing scene and object invariance. However, it relies heavily on crowd-sourced mobile footage, meaning it lacks controlled camera-pose or lighting shifts.

#### Verb / Motion Gap Filled
**Partial Alignment.** While SSv2 is a gold standard for verb-centric learning, it primarily features short, hand-centric manipulations. It leaves a critical gap in full-body human actions, controlled lighting transitions, and precise multi-view video setups that our new 1000+ clip dataset can explicitly address.

---
