# Literature Review: Video Action Generalization

## ${\color{green} * }$ EPIC-Kitchens D1/D2/D3: Multi-Modal Domain Adaptation for Fine-Grained Action Recognition
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



## ${\color{green} * }$ ARGO1M: What can a cook in Italy teach a mechanic in India? Action Recognition Generalisation Over Scenarios and Locations

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



## ${\color{green} * }$ Something-Else: Compositional Action Recognition with Spatial-Temporal Interaction Networks

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







## ${\color{green} * }$ RareAct: A Video Dataset of Unusual Interactions

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


## ${\color{green} * }$ Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives

<details>

- Ego-Exo4D:  1,286 hours of simultaneously captured egocentric and exocentric video of skilled activities (sports, music, dance, cooking, bike repair) from 740 participants across 123 scenes in 13 cities.
- CVPR 2024: https://arxiv.org/abs/2311.18259

### Basics:

* **Base Dataset(s):** Ego-Exo4D (authors' original)
* **Dataset Scale:** 1,286 hours of combined synchronized video across 740 participants in 123 scene contexts. (truly multimodal: Each video is accompanied by multichannel audio, eye gaze, 3D point clouds, camera poses, IMU, and multiple paired language descriptions, and an "expert commentary")
* **Classes:** 278 unique classes (Human-Object Interaction (HOI) and physical skilled movement activities spanning 8 domain categories - cooking, music, sports, dance, ...).
* **Example Classes:** Changing a bike wheel, playing violin, bouldering climbing move, chopping garlic, dribbling a basketball.

#### Domain Shift Tested

The dataset primarily evaluates the domain shift between first-person egocentric camera perspectives (recorded via head-mounted Aria smart glasses) and multi-angle third-person exocentric viewpoints (recorded via stationary GoPro cameras) captured simultaneously. 

In addition to viewpoint shifts, it evaluates cross-environment and cross-subject domain adaptation across 123 natural real-world scene contexts and 13 cities worldwide. 

It also incorporates skill level variations, recording subjects across a spectrum of expertise ranging from absolute novices to proficient experts performing identical tasks.

#### Evaluation Protocol

Generalization performance is measured using a structured suite of benchmark tasks designed for multimodal and multi-view video understanding. The primary protocols include fine-grained keystep recognition (segmenting and classifying sequential task steps), cross-view relation and translation (mapping egocentric features to exocentric viewpoints), proficiency estimation, and 3D body and hand pose recovery. Evaluation setups use cross-view validation (for instance, training on exocentric streams and testing on egocentric streams) evaluated via top-1 classification accuracy, mean average precision (mAP), edit distance, and 3D joint position error metrics.

#### Non-Object Shift Suitability (multiple shifts can be tested using this)

Ego-Exo4D provides a robust benchmark for evaluating non-object domain shifts within video action recognition systems. It isolates drastic perspective variations between dynamic wearable viewpoints and fixed ambient views, allowing researchers to evaluate how model representations transfer across viewpoints. Furthermore, because identical procedural and physical tasks are performed across diverse geographic locations, real-world indoor and outdoor environments, and varying ambient lighting conditions, the dataset effectively benchmarks model robustness against background clutter and scene distribution shifts.

#### Verb / Motion Gap Filled (+++ Still room for our dataset to contribute)

Ego-Exo4D focuses heavily on domain-specific procedural tasks and physical skills that inherently rely on distinct objects and contextual environments, such as bicycle repair tools, musical instruments, or sports equipment. Because these items strongly correlate with specific actions, models can easily achieve high performance by leveraging static object appearance or background features rather than learning genuine temporal motion dynamics. Our new 1000+ clip dataset directly fills this gap by prioritizing verb actions over nouns, isolating temporal motion patterns across varied object and background contexts to evaluate and mitigate object bias.

#### Technique to Create Synthetic Dataset

No Strategy
  
</details>

- note: Video recording strategy similar to our multi-camera setup
- note: Many generalization flavors can be tested using this dataset but there still seems to be room for our idea to help with object bias.






## ${\color{yellow} * }$ EPIC-Kitchens-100 UDA challenge - Rescaling Egocentric Vision: Collection Pipeline and Challenges for EPIC-KITCHENS-100

<details> 
  
- EPIC-Kitchens-100 UDA challenge: Models are trained on EPIC-55 footage and adapted to the unlabelled EK-100 extension, so environment and time shift together. The same paper defines the “test of time” protocol, measuring whether models degrade on footage collected two years later.
- IJCV 2022: https://arxiv.org/abs/2006.13256

### Basics:

* **Base Dataset(s):** EPIC-KITCHENS-55 (extended to EPIC-KITCHENS-100)
* **Dataset Scale:** 100 hours, 20,000,000 frames, and 89,977 action segments across 700 videos.
* **Classes:** 97 verb classes and 300 noun classes, forming fine-grained action combinations (Human-Object Interaction (HOI)).
* **Example Classes:** Verbs: take, put, wash, cut. Nouns: plate, pan, knife, spoon.

#### Domain Shift Tested (across unseen environments, unseen humans, temporal drifts)

Evaluates domain shift through cross-environment generalization, participant variation, and temporal drift ("test of time" shift between 2018 and 2020 recordings). It also provides an Unsupervised Domain Adaptation benchmark testing models trained on a subset of kitchen environments against unseen target kitchens.

#### Non-Object Shift Suitability

It provides a valuable benchmark for egocentric non-object shifts, specifically changes in personal kitchen environments, background layouts, camera movement, lighting conditions, and mounting positions across 45 distinct unscripted homes. This helps measure how models handle real-world environmental and viewpoint variance from a first-person perspective.

#### Verb / Motion Gap Filled (+++)

Although the dataset tracks 97 verbs, action labels remain heavily coupled with specific kitchen objects (300 noun classes). Consequently, models often exploit static object shortcuts rather than temporal motion dynamics. Our new 1000+ clip dataset fills this gap by decoupling actions from specific objects, isolating pure motion patterns, and forcing models to prioritize temporal kinematic movement over static spatial context.

</details>





## ${\color{yellow} * }$ Sims4Action - Let's Play for Action: Recognizing Activities of Daily Living by Learning from Life Simulation Video Games

<details>

- Sims4Action: Ten hours of activities of daily living recorded inside “Sims 4”. Its Gaming→Real benchmark evaluates on real Toyota Smarthome video over 10 corresponding classes.
- IROS 2021 https://ieeexplore.ieee.org/document/9636381

### Basics:

* **Base Dataset(s):** SIMS4ACTION (generated via The Sims 4) and Toyota Smarthome.
* **Dataset Scale:** 10 hours of video content.
* **Classes:** 10 unique classes (Human-Object Interaction).
* **Example Classes:** Cooking, drinking, reading, using tablet, walking.

#### Domain Shift Tested

The paper targets synthetic-to-real domain generalization by controlling avatar appearance, room layout, and camera viewpoint shifts within the simulation game environment. The core domain shift measures how models trained on synthetic rendering engines translate to real-world smart-home monitoring camera streams.

#### Evaluation Protocol

The benchmark uses two setups: Gaming-to-Gaming for synthetic validation and Gaming-to-Real for cross-domain generalization. Evaluated models (I3D and S3D) are trained purely on SIMS4ACTION data and tested zero-shot on real-world Toyota Smarthome clips using top-1 classification accuracy.

#### Non-Object Shift Suitability

SIMS4ACTION effectively evaluates synthetic visual domain gaps, synthetic render artifacts, subject physical appearance variations, and fixed-versus-moving camera angles. This setup offers a strong template for testing environment and camera viewpoint shifts across virtual and physical domains.

#### Verb / Motion Gap Filled

SIMS4ACTION heavily relies on static object cues (e.g., tablet, cup, stove) inherent to daily activities, which induces high object bias. Our 1000+ clip dataset addresses this by decoupling verbs from specific objects and prioritizing dynamic temporal motion dynamics over static scene context.

#### Technique to Create Synthetic Dataset

The authors execute desired actions in a top-down manner using commercial life-simulation software (The Sims 4). By manually triggering action routines across custom-built virtual home environments, they capture varied avatar demographics, ambient surroundings, and camera trajectories cheaply without human privacy constraints.

</details>
