# Literature Review: Video Action Generalization

## EPIC-Kitchens D1/D2/D3: Multi-Modal Domain Adaptation for Fine-Grained Action Recognition
<details> 
  
- EPIC-Kitchens D1/D2/D3: Three kitchens from EPIC-Kitchens-55 treated as separate domains over 8 verb classes, yielding 6 source→target pairs.

- Multi-Modal Domain Adaptation for Fine-Grained Action Recognition (MM-SADA), CVPR 2020 https://arxiv.org/abs/2001.09691

### Basics:
This paper has a custom cross-domain split based on 3 participants (P08, P01, and P22 as domains D1, D2, and D3) from **EPIC-Kitchens-55** (the original 2018 release), filtering down to 8 shared verb classes across those specific kitchens.

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

### Things in paper that remain relevant today
- Core Thesis: The paper's core thesis—that fine-grained action recognition models overfit to static environment cues (kitchen counter layouts, lighting, specific appliances) rather than learning pure action dynamics—remains a central challenge in computer vision.

- Multi-Modal Disentanglement (Appearance vs. Motion):
The concept that different modalities carry different levels of domain invariance is still fundamental. a static RGB frame carries high appearance/object bias, whereas temporal/motion signal carries stronger verb representations. Disentangling static "nouns" from temporal "verbs" via multi-modal signals remains the standard approach for fighting object bias.

- Benchmark in this paper: The cross-kitchen transfer benchmark (D1, D2, D3) established in this paper remains a widely recognized evaluation protocol for fine-grained action generalization in egocentric vision.

### Things in paper that are outdated

#### Adversarial Alignment vs. Contrastive / Self-Supervised Adaptation

* **2020 Approach (MM-SADA):** Relied on domain discriminators and adversarial loss (GRL-style) to force feature alignment between source and target domains.
* **Modern Shift:** Adversarial UDA is largely superseded by **contrastive learning** (e.g., InfoNCE, CLIP-style joint text-video alignment), **masked video autoencoders (VideoMAE)**, and **feature disentanglement networks**. Modern approaches focus on learning domain-invariant representation spaces directly rather than training unstable min-max discriminators.

#### Explicit Optical Flow vs. Vision Transformers (Modern ViT don't require Optical Flow)

* **2020 Approach:** Required dense Optical Flow calculation as a explicit second modality alongside RGB.
* **Modern Shift:** Spatiotemporal Vision Transformers (ViTs, TimeSformer, VideoMAE) extract temporal motion dynamics directly from raw video frames without requiring computationally expensive pre-computed optical flow.

#### UDA (Unsupervised Domain Adaptation) vs. DG (Domain Generalization)

* **2020 Approach:** UDA assumes access to unlabeled target-domain videos during training.
* **Modern Shift:** Real-world deployment favors **Domain Generalization (DG)** or **Test-Time Adaptation (TTA)**, where the model must generalize to unseen target environments without needing unlabeled training samples beforehand. (This is what we discussed on 12.08.2026)

</details>



## ARGO1M: What can a cook in Italy teach a mechanic in India? Action Recognition Generalisation Over Scenarios and Locations

<details>
  
- "ARGO1M" = Action Recognition Generalisation Over scenarios and locations dataset
- ARGO1M:  1.1M clips from Ego4D spanning 10 scenarios and 13 geographic locations, with 10 test splits that each pair an unseen scenario with an unseen location.

- ICCV 2023:  https://arxiv.org/abs/2306.08713

  
### Basics:

* **Base Dataset(s):** Ego4D (curated into the ARGO1M benchmark).
* **Dataset Scale:** 1.1 million video clips. Total video hours: Not reported.
* **Classes:** 60 unique classes (Human-Object Interaction and Motion-based activity).
* **Example Classes:** Open, wash, take, put, adjust.

#### Domain Shift Tested (cross-environment)

Generalization is defined as the capability of an action recognition model to classify egocentric human actions when performed across dual domain shifts: novel activity scenarios and unseen geographical locations. Models are trained on specific activity domains and evaluated on test splits comprising unobserved activity scenarios recorded in distinct geographical regions (for example, transferring learned actions from daily cooking in Italy all the way to vehicle repair in India) (washing a plate with kitchen background to washing a tyre in repair-shop background).

They introduce Causal Intervention for Action Recognition (CIR), a training method designed to remove contextual and object bias from egocentric action representations which apparently helps quite significantly. (Explore more)

#### Non-Object Shift Suitability (environment shifts)

The dataset provides an effective testbed for non-object domain shifts by evaluating background environments, recording setups, and geographical contexts. Incorporating this benchmark into our test suite allows us to evaluate how robustly models adapt to drastic changes in scene appearance, lighting, and ambient surroundings across different global environments while keeping action definitions constant.

#### Verb / Motion Gap Filled +++

While ARGO1M categorizes videos using verb labels, its real-world egocentric clips from Ego4D still feature strong correlations between actions, background environments, and specific co-occurring objects (such as kitchen utensils or mechanical tools). As a result, models can rely on static visual object shortcuts rather than temporal motion. Our new 1000+ clip dataset addresses this gap by isolating pure temporal motion dynamics, stripping away object cues to directly target object bias and prioritize verb recognition.

#### Technique to Create Synthetic Dataset

No Strategy

#### Their Protocol 
The ARGO1M evaluation protocol tests action recognition generalization across two distinct distribution shifts: **scenario shifts** and **location shifts**.

The benchmark partitions Ego4D egocentric videos into disjoint training and testing sets based on recorded scenario contexts and geographic regions. 
* Under the **Out-of-Scenario (OOS)** protocol, models are trained on specific activity domains (such as cooking) and evaluated on unobserved activity domains (such as auto repair).
* Under the **Out-of-Location (OOL)** protocol, models are evaluated on actions recorded in geographical regions unseen during training.

The protocol uses top-1 accuracy on verb classification to assess whether a model learns domain-agnostic motion representations or merely memorizes static scene background and context cues.

</details>

note: "Code and data" link from abstract page is not working!



## Something-Else: Compositional Action Recognition with Spatial-Temporal Interaction Networks

<details>

- Something-Else: Re-splits Something-Something V2 so that the verb-noun combinations seen during training never appear at test, and adds dense hand and object box annotations
- CVPR 2020: https://arxiv.org/abs/1912.09930

### Basics:

* **Base Dataset(s):** Something-Something V2 (SSv2).
* **Dataset Scale:** 112,795 video clips total (54,919 training clips, 57,876 testing clips; total hours: Not reported).
* **Classes:** 174 unique classes (Human-Object Interaction (HOI)).
* **Example Classes:** "Pushing something so that it falls off the table", "Moving something and something closer to each other", "Covering something with something", "Tearing something into two pieces".


#### A new network introduced by authors using BBox annotated ssv2 dataset
The **Spatial-Temporal Interaction Network (STIN)**, a model designed to focus on *how* objects move and interact rather than *what* those objects look like.

Instead of analyzing an entire video frame at once, the model relies on hand and object bounding box annotations. It tracks these boxes over time and extracts two key streams of information:

1. **Geometric/Spatial Trajectories:** It calculates geometric relationships—such as the relative positions, distances, and trajectory paths between the actor's hands and the objects. This allows the model to recognize motion patterns (e.g., "moving two things closer") regardless of what the objects actually are.
2. **Visual/Feature Representations:** It extracts visual features from inside those object bounding boxes using ROI-pooling to capture physical state changes (e.g., an object breaking or bending).

For actions without clear hand-object interactions, the model falls back on a standard scene-level 3D convolutional network (like 3D ResNet). By combining geometric motion reasoning with localized object features, the network learns the underlying **verb** dynamics, allowing it to correctly identify actions even when performed with completely unfamiliar objects.

#### Domain Shift Tested (Unique to this paper have not seen yet)

Generalization is defined as compositional OOD transfer, evaluating whether a model can decouple action verbs from specific objects. The exact domain shift tested is a compositional split where the training and testing sets share the same set of action verbs and object categories, but contain completely non-overlapping verb-noun combinations. Models are required to classify actions on novel verb-object pairings that were never observed during training, testing compositional reasoning over spatial-temporal interactions.

#### Evaluation Protocol

The protocol evaluates compositional generalization across 174 categories using 55k training clips and 58k testing clips. Performance is measured using Top-1 and Top-5 classification acc. under a zero-shot split. The paper also includes a few-shot compositional evaluation protocol (1-shot and 5-shot) on base and novel categories to test rapid adaptation. Performance is evaluated under two distinct conditions: using human-annotated ground-truth bounding boxes for hands and objects, and using predictions from automated object detectors and trackers.

#### Non-Object Shift Suitability (does work for object bias)

This dataset focuses entirely on compositional action-object shifts rather than explicit environmental or sensor domain shifts. Although crowdsourced clips contain natural variations in background scenes and lighting, Something-Else does not systematically isolate or control these factors. Without explicit splits for testing egocentric versus exocentric viewpoints, lighting variations, or background changes, this benchmark CANNOT directly evaluate non-object environmental domain generalization within our test suite.

#### Verb / Motion Gap Filled

While Something-Else decouples verbs from nouns via compositional splits, visual object appearance context still heavily influence predictions. It leaves a gap in evaluating pure temporal motion dynamics independent of visual appearance. Our new 1000+ clip dataset addresses this gap by capturing controlled, motion-centric actions that minimize object appearance cues, directly targeting object bias through prioritized verb representations.

#### Technique to Create Synthetic Dataset

No Strategy

</details>







## RareAct: A Video Dataset of Unusual Interactions

<details>

- RareAct: Manually annotated clips of unlikely verb-noun compositions such as blend phone, cut keyboard or microwave shoes, with hard negatives that share either the verb or the noun. Targets compositionality for rare human-object interactions that large datasets systematically omit.
- arXiv 2020 https://arxiv.org/abs/2008.01018
  
### Basics:

* **Base Dataset(s):** HowTo100M (used as the large-scale instructional text corpus to identify rarely co-occurring verb-noun pairs).
* **Dataset Scale:** 905 total videos, 7,607 annotated video clips (10-second clip length, containing 1,765 positive clips).
* **Classes:** 122 unique classes (Human-Object Interaction (HOI))
* **Example Classes:** blend phone, cut keyboard, microwave shoes, drill laptop, wash rock. (verb-noun combinations that almost never occured in labels of HowTo100m dataset)

#### Domain Shift Tested

Generalization is defined as zero-shot and few-shot compositional generalization to rare, out-of-distribution verb-noun combinations. The domain shift explicitly evaluates model recognition capabilities when individual action verbs (e.g., blend, cut, microwave) and object nouns (e.g., phone, keyboard, shoes) appear frequently in training data on their own, but are combined into highly unusual, unobserved pairings that rarely or never cooccur in standard video-text pretraining corpus.

Zero-shot compositionality tests whether a model trained on common pairs (like "cut paper", "cut apple", "type on keyboard") can correctly identify an unlikely pair ("cut keyboard") without ever having seen a single video of that specific combination during training. If a model has true compositionality, it can recognize the motion verb anywhere, regardless of how weird/unexpected the object is.


#### Evaluation Protocol

The evaluation protocol measures zero-shot compositionality transfer using mean weighted average precision (mWAP) and mean subsampled average precision (mSAP) across 122 action classes. These specialized metrics compensate for visual redundancy and sample weighting when multiple clips originate from the same web video. Models are evaluated with and without structured hard negative samples, which include video clips where only the verb occurs, only the noun occurs, or both appear without the action being performed.

#### Non-Object Shift Suitability (not focused on this but does this regardless)

RareAct does NOT explicitly annotate physical non-object domain shifts such as controlled lighting variations, systematic viewpoint transformations, egocentric versus exocentric viewpoint shifts, or environmental perturbations. However, because its video clips are harvested from unconstrained web sources, the benchmark naturally incorporates diverse background environments, unscripted camera motions, and real-world lighting conditions. Nevertheless, its primary focus remains compositional semantic shift rather than controlled physical non-object shifts.

#### Verb / Motion Gap Filled (++)

Although RareAct effectively breaks standard semantic co-occurrence priors by pairing actions with unexpected objects, its action classes remain tied to specific physical object nouns. As a result, action recognition models can still heavily rely on static visual object appearance rather than understanding temporal motion patterns. Our new 1000+ clip dataset directly fills this gap by focusing on pure verbs and motion dynamics across varying contexts, isolating temporal trajectories to mitigate spatial object bias.

#### Technique to Create Synthetic Dataset

No Strategy
</details>


