# Literature Review: Video Action Generalization


## EPIC-Kitchens D1/D2/D3: Multi-Modal Domain Adaptation for Fine-Grained Action Recognition

- EPIC-Kitchens D1/D2/D3: Three kitchens from EPIC-Kitchens-55 treated as separate domains over 8 verb classes, yielding 6 source→target pairs.

- Multi-Modal Domain Adaptation for Fine-Grained Action Recognition (MM-SADA), CVPR 2020 https://arxiv.org/abs/2001.09691

### Basics:

* **Base Dataset(s):** EPIC-Kitchens (specifically environments D1, D2, and D3 corresponding to participants P08, P01, and P22).
* **Dataset Scale:** 10,094 video clips. Total hours: Not reported.
* **Classes:** 8 unique classes (Human-Object Interaction).
* **Example Classes:** put, take, open, close, wash.

#### Domain Shift Tested

The paper evaluates cross-environment domain adaptation in egocentric video across three distinct real-world kitchen environments. Models are trained on labeled action segments from a source kitchen and tested on an unlabeled target kitchen. Generalization is evaluated against environmental changes including background geometry, spatial layouts, lighting conditions, camera angles, and visual clutter while maintaining egocentric viewpoints.

#### Non-Object Shift Suitability (environment shifts)

This dataset provides a useful framework for evaluating non-object environmental shifts. Keeping the verb label set consistent across different physical kitchens isolates shifts caused by room architecture, background lighting, visual distraction, and participant-specific egocentric motion. Incorporating this setup allows testing model resilience to spatial background shifts without changing the underlying action definitions.

#### Verb / Motion Gap Filled +++

Although the paper isolates verb classes across kitchens, the actions remain heavily tied to egocentric kitchen settings where co-occurring objects (e.g., taps, fridges, drawers) strongly correlate with specific actions. Our new 1000+ clip dataset addresses this limitation by evaluating temporal motion patterns across diverse non-kitchen environments and broader contexts, prioritizing pure verb dynamics over object appearances to mitigate object bias.

#### Technique to Create Synthetic Dataset

No Strategy



The foundational insights of the paper (*Multi-Modal Domain Adaptation for Fine-Grained Action Recognition*, Munro & Damen, CVPR 2020) remain **highly relevant conceptual pillars**, even though the **specific algorithmic tools have evolved** since its publication.

---

## Q. Is it still relevant today?

### YES
- Core Thesis: The paper's core thesis—that fine-grained action recognition models overfit to static environment cues (kitchen counter layouts, lighting, specific appliances) rather than learning pure action dynamics—remains a central challenge in computer vision.

- Multi-Modal Disentanglement (Appearance vs. Motion):
The concept that different modalities carry different levels of domain invariance is still fundamental. a static RGB frame carries high appearance/object bias, whereas temporal/motion signal carries stronger verb representations. Disentangling static "nouns" from temporal "verbs" via multi-modal signals remains the standard approach for fighting object bias.

### 3. Egocentric Cross-Environment Benchmark Protocols

The cross-kitchen transfer benchmark (D1, D2, D3) established in this paper remains a widely recognized evaluation protocol for fine-grained action generalization in egocentric vision.

---

## What Has Evolved or Become Outdated

While the underlying problem remains identical, the modeling paradigms have shifted significantly:

### 1. Adversarial Alignment vs. Contrastive / Self-Supervised Adaptation

* **2020 Approach (MM-SADA):** Relied on domain discriminators and adversarial loss (GRL-style) to force feature alignment between source and target domains.
* **Modern Shift:** Adversarial UDA is largely superseded by **contrastive learning** (e.g., InfoNCE, CLIP-style joint text-video alignment), **masked video autoencoders (VideoMAE)**, and **feature disentanglement networks**. Modern approaches focus on learning domain-invariant representation spaces directly rather than training unstable min-max discriminators.

### 2. Explicit Optical Flow vs. Vision Transformers

* **2020 Approach:** Required dense Optical Flow calculation as a explicit second modality alongside RGB.
* **Modern Shift:** Spatiotemporal Vision Transformers (ViTs, TimeSformer, VideoMAE) extract temporal motion dynamics directly from raw video frames without requiring computationally expensive pre-computed optical flow.

### 3. UDA (Unsupervised Domain Adaptation) vs. DG (Domain Generalization)

* **2020 Approach:** UDA assumes access to unlabeled target-domain videos during training.
* **Modern Shift:** Real-world deployment favors **Domain Generalization (DG)** or **Test-Time Adaptation (TTA)**, where the model must generalize to unseen target environments without needing unlabeled training samples beforehand.

---

## Takeaway for Your Evaluation Suite

| Aspect | Status | Direct Application to Your Framework |
| --- | --- | --- |
| **Problem Definition** | **Still Relevant** | Validates your test suite's goal to isolate verbs/motion from object biases. |
| **Benchmark Protocol** | **Still Relevant** | Using EPIC-Kitchens cross-kitchen environments provides a realistic, non-synthetic baseline for spatial background shift. |
| **Algorithmic Baseline** | **Outdated** | Avoid evaluating modern models solely against 2020-style adversarial discriminators; benchmark instead against modern Zero-Shot Foundation Models or Video-Language Contrastive Models. |
