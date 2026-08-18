# Literature Review: Video Action Generalization


For each paper, you will find:
1. Some basic information like datasets used and action classes etc.
2. Domain Shift Tested: How generalization is defined in the paper (e.g., egocentric vs. exocentric, lighting variations, specific perturbations) and the exact domain shift flavors being tested.
3. Evaluation Protocol: The primary evaluation setup used to measure generalization performance (e.g., zero-shot transfer, cross-domain validation, OOD benchmarks, metrica used).
4. Domain Shifts (Excluding Object Bias): Whether and how this dataset evaluates a useful non-object bias that we should incorporate into our suite.
5. Verb / Motion Gap Filled (or not): How this paper leaves a gap in verb/temporal-motion evaluation that our new 1000+ clip dataset can explicitly address to combat object bias. Basically, how will we motivate the need for our new dataset.
6. Technique to Create Synthetic Dataset: If this paper gives a strategy to create a synthetic dataset it is mentioned.

---

---


## ${\color{green} * }$ EPIC-Kitchens D1/D2/D3: Multi-Modal Domain Adaptation for Fine-Grained Action Recognition
 
  
- EPIC-Kitchens D1/D2/D3: Three kitchens from EPIC-Kitchens-55 treated as separate domains over 8 verb classes, yielding 6 source→target pairs.

- Multi-Modal Domain Adaptation for Fine-Grained Action Recognition (MM-SADA), CVPR 2020 https://arxiv.org/abs/2001.09691

### Basics:
This paper has a custom cross-domain split based on 3 participants (P08, P01, and P22 as domains D1, D2, and D3) from **EPIC-Kitchens-55**, filtering down to 8 shared verb classes across those specific kitchens.

* **Base Dataset(s):** EPIC-Kitchens-55 (only 3 participants).
* **Dataset Scale:** 10,094 video clips. Total hours: Not reported.
* **Classes:** 8 unique classes (Human-Object Interaction).
* **Example Classes:** put, take, open, close, wash.

#### Domain Shift Tested

The paper evaluates cross-environment domain adaptation in egocentric video across three kitchens. Models are trained on labeled action segments from a source kitchen and tested on an unlabeled target kitchen. Generalization is evaluated against environmental changes including background geometry, spatial layouts, lighting conditions, camera angles, and visual clutter while maintaining egocentric viewpoints.

#### Domain Shifts (Excluding Object Bias) (environment shifts)

Non-object environmental shifts.

#### Verb / Motion Gap Filled +++

Although the paper isolates verb classes across kitchens, the actions remain heavily tied to egocentric kitchen settings where co-occurring objects (e.g., taps, fridges, drawers) strongly correlate with specific actions. Our new 1000+ clip dataset addresses this limitation by evaluating temporal motion patterns across diverse non-kitchen environments and broader contexts, prioritizing pure verb dynamics over object appearances to mitigate object bias.

#### Technique to Create Synthetic Dataset

No Strategy

#### Things in paper that remain relevant today
note: This section was done only for this paper.

- Core Thesis: The paper's core thesis—that fine-grained action recognition models overfit to static environment cues (kitchen counter layouts, appliances) rather than learning pure action dynamics—remains a central challenge in CV.

- Multi-Modal Disentanglement (Appearance vs. Motion):
The concept that different modalities carry different levels of domain invariance is still fundamental. a static RGB frame carries high appearance/object bias, whereas temporal/motion signal carries stronger verb representations. Disentangling static "nouns" from temporal "verbs" via multi-modal signals remains the standard approach for fighting object bias.

- Benchmark in this paper: The cross-kitchen transfer benchmark (D1, D2, D3) established in this paper remains a widely recognized evaluation protocol for fine-grained action generalization in egocentric vision.

#### Things in paper that are outdated

- **Adversarial Alignment vs. Contrastive / Self-Supervised Adaptation:** This paper's approach relied on domain discriminators and adversarial loss (GRL-style) to force feature alignment between source and target domains.
Today, Adversarial UDA is largely superseded by contrastive learning (InfoNCE for example), VideoMAE, and feature disentanglement networks. Modern approaches focus on learning domain-invariant representation spaces directly rather than training unstable min-max discriminators.

- **Explicit Optical Flow vs. Vision Transformers (Modern ViT don't require Optical Flow)**: This paper required dense Optical Flow calculation as a explicit second modality alongside RGB. Nowadays, Spatiotemporal Vision Transformers (ViTs, TimeSformer, VideoMAE) extract temporal motion dynamics directly from raw video frames.

- **UDA (Unsupervised Domain Adaptation) vs. DG (Domain Generalization)** In this paper, UDA assumes access to unlabeled target-domain videos during training.
Nowadays,  Real-world deployment favors Domain Generalization (DG) or Test-Time Adaptation (TTA), where the model must generalize to unseen target environments without needing unlabeled training samples beforehand. (This is what we discussed on 12.08.2026)









## ${\color{green} * }$ ARGO1M: What can a cook in Italy teach a mechanic in India? Action Recognition Generalisation Over Scenarios and Locations

note: "Code and data" link from abstract page is not working!


  
- "ARGO1M" = Action Recognition Generalisation Over scenarios and locations dataset
- ARGO1M:  1.1M clips from Ego4D spanning 10 scenarios and 13 geographic locations, with 10 test splits that each pair an unseen scenario with an unseen location.

- ICCV 2023:  https://arxiv.org/abs/2306.08713

  
### Basics:

* **Base Dataset(s):** Ego4D (curated into the ARGO1M benchmark).
* **Dataset Scale:** 1.1 million video clips. Total video hours: Not reported.
* **Classes:** 60 unique classes (Human-Object Interaction and Motion-based activity).
* **Example Classes:** Open, wash, take, put, adjust.

#### Domain Shift Tested (cross-environment)

Generalization is defined as the capability of an AR model to classify egocentric human actions when performed across dual domain shifts: novel activity scenarios AND unseen geographical locations. Models are trained on specific activity domains and evaluated on test splits comprising unobserved activity scenarios recorded in distinct geographical regions (for example, transferring learned actions from daily cooking in Italy all the way to vehicle repair in India) ('washing' a plate with kitchen background to 'washing' a tyre in repair-shop background).

They introduce Causal Intervention for Action Recognition (CIR), a training method designed to remove contextual and object bias from egocentric action representations which apparently helps quite significantly. (note: Can explore technique more if we are interested.)

#### Evaluation Protocol

**Scenario shifts** and **location shifts**.

The benchmark partitions Ego4D egocentric videos into disjoint training and testing sets based on recorded scenario contexts and geographic regions. 
* Under the **Out-of-Scenario (OOS)** protocol, models are trained on specific activity domains (such as cooking) and evaluated on unobserved activity domains (such as auto repair).
* Under the **Out-of-Location (OOL)** protocol, models are evaluated on actions recorded in geographical regions unseen during training. (e.g. Italy and India)

The protocol uses top-1 accuracy on verb classification to assess whether a model learns domain-agnostic motion representations or merely memorizes static scene background and context cues.


#### Domain Shifts (Excluding Object Bias)

Background environments, recording setups, and geographical contexts. 

Incorporating this benchmark into our test suite will kill multiple birds with one stone.

#### Verb / Motion Gap Filled +++

While ARGO1M categorizes videos using verb labels, its real-world egocentric clips from Ego4D still feature strong correlations between actions, background environments, and specific co-occurring objects (such as kitchen utensils or mechanical tools). As a result, models can lern shortcuts rather than temporal motion. Our new 1000+ clip dataset addresses this gap by directly targeting object bias and prioritizing verb recognition.

#### Technique to Create Synthetic Dataset

No Strategy










## ${\color{green} * }$ Something-Else: Compositional Action Recognition with Spatial-Temporal Interaction Networks


- Something-Else: Re-splits Something-Something V2 so that the verb-noun combinations seen during training never appear at test, and adds dense hand and object box annotations
- CVPR 2020: https://arxiv.org/abs/1912.09930

### Basics:

* **Base Dataset(s):** Something-Something V2 (SSv2).
* **Dataset Scale:** 112,795 video clips total (54,919 training clips, 57,876 testing clips;
* **Classes:** 174 unique classes (Human-Object Interaction (HOI)).
* **Example Classes:** "Pushing something so that it falls off the table", "Moving something and something closer to each other", "Covering something with something", "Tearing something into two pieces".


#### A new network introduced by authors using BBox annotated ssv2 dataset
The **Spatial-Temporal Interaction Network (STIN)**, a model designed to focus on *how* objects move and interact rather than *what* those objects look like.

Instead of analyzing an entire video frame at once, the model relies on hand and object bounding box annotations. It tracks these boxes over time and extracts two key streams of information:

1. **Geometric/Spatial Trajectories:** It calculates geometric relationships—such as the relative positions, distances, and trajectory paths between the actor's hands and the objects. This allows the model to recognize motion patterns (e.g., "moving two things closer") regardless of what the objects actually are.
2. **Visual/Feature Representations:** It extracts visual features from inside those object bounding boxes using ROI-pooling to capture physical state changes (e.g., an object breaking or bending).

For actions without clear hand-object interactions, the model falls back on a standard scene-level 3D convolutional network (like 3D ResNet). By combining geometric motion reasoning with localized object features, the network learns the underlying verb dynamics, allowing it to correctly identify actions even when performed with completely unfamiliar objects.

#### Domain Shift Tested (object bias is targeted)

note: Unique to this paper have not seen yet

Generalization is defined as compositional OOD transfer, evaluating whether a model can decouple action verbs from specific objects. The exact domain shift tested is a compositional split where the training and testing sets share the same set of action verbs and object categories, but contain completely non-overlapping verb-noun combinations. Models are required to classify actions on novel verb-object pairings that were never observed during training, testing compositional reasoning over spatial-temporal interactions. 

#### Evaluation Protocol

The protocol evaluates compositional generalization across 174 categories using 55k training clips and 58k testing clips. Performance is measured using Top-1 and Top-5 classification acc. under a zero-shot split. 

The paper also includes a few-shot compositional evaluation protocol (1-shot and 5-shot) on base and novel categories to test rapid adaptation. Performance is evaluated under two distinct conditions: using human-annotated ground-truth bounding boxes for hands and objects, and using predictions from automated object detectors and trackers.

#### Domain Shifts (Excluding Object Bias)

Although crowdsourced clips contain natural variations in background scenes and lighting, this Something-Else paper does not systematically control these factors.

#### Verb / Motion Gap Filled

Something-Else decouples verbs from nouns via compositional splits, visual object appearance context can still influence verb predictions. 

We can take inspiration from their results and explanations to see exactly which protocols in our recordings will benefit us the most.

#### Technique to Create Synthetic Dataset

No Strategy









## ${\color{green} * }$ RareAct: A Video Dataset of Unusual Interactions


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

#### Domain Shifts (Excluding Object Bias) (not focused on this but does this regardless)

RareAct does NOT explicitly annotate physical non-object domain shifts such as controlled lighting variations, systematic viewpoint transformations, egocentric versus exocentric viewpoint shifts, or environmental perturbations. However, because its video clips are harvested from unconstrained web sources, the benchmark naturally incorporates diverse background environments, unscripted camera motions, and real-world lighting conditions. Nevertheless, its primary focus remains compositional semantic shift rather than controlled physical non-object shifts.

#### Verb / Motion Gap Filled (++)

Although RareAct effectively breaks standard semantic co-occurrence priors by pairing actions with unexpected objects, its action classes remain tied to specific physical object nouns. As a result, action recognition models can still heavily rely on static visual object appearance rather than understanding temporal motion patterns. Our new 1000+ clip dataset directly fills this gap by focusing on pure verbs and motion dynamics across varying contexts, isolating temporal trajectories to mitigate spatial object bias.

#### Technique to Create Synthetic Dataset

No Strategy








## ${\color{green} * }$ Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives



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

#### Domain Shifts (Excluding Object Bias) (multiple shifts can be tested using this)

Ego-Exo4D provides a robust benchmark for evaluating non-object domain shifts within video action recognition systems. It isolates drastic perspective variations between dynamic wearable viewpoints and fixed ambient views, allowing researchers to evaluate how model representations transfer across viewpoints. Furthermore, because identical procedural and physical tasks are performed across diverse geographic locations, real-world indoor and outdoor environments, and varying ambient lighting conditions, the dataset effectively benchmarks model robustness against background clutter and scene distribution shifts.

#### Verb / Motion Gap Filled (+++ Still room for our dataset to contribute)

Ego-Exo4D focuses heavily on domain-specific procedural tasks and physical skills that inherently rely on distinct objects and contextual environments, such as bicycle repair tools, musical instruments, or sports equipment. Because these items strongly correlate with specific actions, models can easily achieve high performance by leveraging static object appearance or background features rather than learning genuine temporal motion dynamics. Our new 1000+ clip dataset directly fills this gap by prioritizing verb actions over nouns, isolating temporal motion patterns across varied object and background contexts to evaluate and mitigate object bias.

#### Technique to Create Synthetic Dataset

No Strategy
  


- note: Video recording strategy similar to our multi-camera setup
- note: Many generalization flavors can be tested using this dataset but there still seems to be room for our idea to help with object bias.






## ${\color{yellow} * }$ EPIC-Kitchens-100 UDA challenge - Rescaling Egocentric Vision: Collection Pipeline and Challenges for EPIC-KITCHENS-100

 
  
- EPIC-Kitchens-100 UDA challenge: Models are trained on EPIC-55 footage and adapted to the unlabelled EK-100 extension, so environment and time shift together. The same paper defines the “test of time” protocol, measuring whether models degrade on footage collected two years later.
- IJCV 2022: https://arxiv.org/abs/2006.13256

### Basics:

* **Base Dataset(s):** EPIC-KITCHENS-55 (extended to EPIC-KITCHENS-100)
* **Dataset Scale:** 100 hours, 20,000,000 frames, and 89,977 action segments across 700 videos.
* **Classes:** 97 verb classes and 300 noun classes, forming fine-grained action combinations (Human-Object Interaction (HOI)).
* **Example Classes:** Verbs: take, put, wash, cut. Nouns: plate, pan, knife, spoon.

#### Domain Shift Tested (across unseen environments, unseen humans, temporal drifts)

Evaluates domain shift through cross-environment generalization, participant variation, and temporal drift ("test of time" shift between 2018 and 2020 recordings). It also provides an Unsupervised Domain Adaptation benchmark testing models trained on a subset of kitchen environments against unseen target kitchens.

#### Domain Shifts (Excluding Object Bias)

It provides a valuable benchmark for egocentric non-object shifts, specifically changes in personal kitchen environments, background layouts, camera movement, lighting conditions, and mounting positions across 45 distinct unscripted homes. This helps measure how models handle real-world environmental and viewpoint variance from a first-person perspective.

#### Verb / Motion Gap Filled (+++)

Although the dataset tracks 97 verbs, action labels remain heavily coupled with specific kitchen objects (300 noun classes). Consequently, models often exploit static object shortcuts rather than temporal motion dynamics. Our new 1000+ clip dataset fills this gap by decoupling actions from specific objects, isolating pure motion patterns, and forcing models to prioritize temporal kinematic movement over static spatial context.







## ${\color{yellow} * }$ Sims4Action - Let's Play for Action: Recognizing Activities of Daily Living by Learning from Life Simulation Video Games



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

#### Domain Shifts (Excluding Object Bias)

SIMS4ACTION effectively evaluates synthetic visual domain gaps, synthetic render artifacts, subject physical appearance variations, and fixed-versus-moving camera angles. This setup offers a strong template for testing environment and camera viewpoint shifts across virtual and physical domains.

#### Verb / Motion Gap Filled

SIMS4ACTION heavily relies on static object cues (e.g., tablet, cup, stove) inherent to daily activities, which induces high object bias. Our 1000+ clip dataset addresses this by decoupling verbs from specific objects and prioritizing dynamic temporal motion dynamics over static scene context.

#### Technique to Create Synthetic Dataset

The authors execute desired actions in a top-down manner using commercial life-simulation software (The Sims 4). By manually triggering action routines across custom-built virtual home environments, they capture varied avatar demographics, ambient surroundings, and camera trajectories cheaply without human privacy constraints.










## ${\color{yellow} * }$  ActorShift: Audio-Adaptive Activity Recognition Across Video Domains



- ActorShift: Human actors from Kinetics-700 as source and animal actors collected from YouTube as target, over 7 classes.
- CVPR 2022: https://arxiv.org/abs/2203.14240

### Basics:

* **Base Dataset(s):** ActorShift (introduced), EPIC-Kitchens, and CharadesEgo.
* **Dataset Scale:** ActorShift contains 1,505 video clips (1,305 source clips from Kinetics-700 and 200 target clips from YouTube). Total hours are Not reported.
* **Classes:** 7 unique classes (Generic activity).
* **Example Classes:** Sleeping, watching TV, eating, drinking, swimming.

#### Domain Shift Tested

Domain generalization and adaptation are evaluated under actor shift, environment/scenery shift, and camera viewpoint shift. Specifically, the paper introduces actor shift where the source domain contains human actors performing activities and the target domain consists of animals performing the same activities, creating significant appearance and motion variance across domains.

#### Evaluation Protocol

The evaluation uses unsupervised domain adaptation across audio-visual benchmarks. Models are trained on a labeled source domain and an unlabeled target domain using paired audio-visual streams. Generalization performance is primarily measured via top-1 classification accuracy on the target domain splits across EPIC-Kitchens, CharadesEgo, and ActorShift.

#### Domain Shifts (Excluding Object Bias)

The paper evaluates actor species shift (human to animal), egocentric versus exocentric viewpoint shifts, and environment shifts across distinct kitchens or home environments. Incorporating these audio-visual actor and viewpoint shifts provides a non-object benchmark for evaluating dynamic motion and cross-modal invariance across distinct recording settings.

#### Verb / Motion Gap Filled

While this paper leverages audio modalities to mitigate visual domain shifts, it relies on broad activity classes where objects and ambient audio heavily anchor predictions. It leaves a gap in purely visual temporal motion dynamics where actions occur silently or share identical objects. Our new 1000+ clip dataset explicitly isolates fine-grained motion trajectories and verb distinctions without reliance on sound cues.

#### Technique to Create Synthetic Dataset

No Strategy










## ${\color{yellow} * }$ Human-Animal-Cartoon (HAC) - SimMMDG: A Simple and Effective Framework for Multi-modal Domain Generalization



- Human-Animal-Cartoon (HAC): Seven actions performed by humans, animals and cartoon characters.
- NeurIPS 2023: https://arxiv.org/abs/2310.19795

### Basics:

* **Base Dataset(s):** EPIC-Kitchens and Human-Animal-Cartoon (HAC).
* **Dataset Scale:** 3,381 video clips for HAC (around 1,000 clips per domain); hours for HAC are Not reported. EPIC-Kitchens contains 55 hours across 39,596 video clips.
* **Classes:** 7 unique classes for HAC (Generic activity and Human-Animal Interaction) and 8 unique classes for EPIC-Kitchens (Human-Object Interaction).
* **Example Classes:** sleeping, eating, running, drinking, opening door (HAC); put, take, wash, cut, pour (EPIC-Kitchens).

#### Domain Shift Tested

Generalization is defined as maintaining classification performance on unseen target domains without modifying target distributions during training. The primary domain shifts tested include cross-environment variations in egocentric scenes (three distinct kitchen environments D1, D2, and D3 in EPIC-Kitchens) and radical cross-species/style visual and acoustic distribution shifts across three domains (humans H, animals A, and cartoon figures C in HAC).

#### Evaluation Protocol

Evaluated using a multi-source leave-one-domain-out cross-domain validation protocol under a zero-shot domain generalization setup. Models train on two source domains and are evaluated directly on the unseen third target domain across combinations of video, audio, and optical flow modalities, using top-1 classification accuracy as the standard metric.

#### Domain Shifts (Excluding Object Bias)

The paper evaluates style, appearance, and actor distribution shifts across human, animal, and cartoon domains in HAC, as well as distinct real-world background and acoustic environments across kitchen setups in EPIC-Kitchens. These shifts capture non-object multi-modal domain variations across visual context, auditory profiles, and motion rendering styles.

#### Verb / Motion Gap Filled

While HAC includes general actions, its action space is constrained to seven coarse classes and depends heavily on stylistic differences across actors rather than fine-grained temporal action dynamics. Our new 1000+ clip dataset addresses this gap by isolating temporal motion structures and prioritizing fine-grained verbs over nouns to explicitly mitigate object bias.

#### Technique to Create Synthetic Dataset

No Strategy










## ${\color{yellow} * }$ SomethingSomething-DG - VideoDG: Generalizing Temporal Relations in Videos to Novel Domains





- SomethingSomething-DG: 20 Something-Something categories, each split into sub-categories with different action consequences (e.g. doing something vs. pretending to do it), so the shift is purely temporal rather than in appearance. The same paper also builds DG splits divided by source dataset and by camera view.

- VideoDG: Generalizing Temporal Relations in Videos to Novel Domains, TPAMI 2021



https://arxiv.org/abs/1912.03716

### Basics:

* **Base Dataset(s):** UCF-HMDB (UCF101 and HMDB51), Something-Something, PKU-MMD, and NTU RGB+D.
* **Dataset Scale:** Not reported
* **Classes:** 12 unique classes across UCF-HMDB; varied across benchmarks (Generic activity and Human-Object Interaction)
* **Example Classes:** Run, jump, shoot ball, hug, wave hands

#### Domain Shift Tested

Generalization is evaluated by training on source domains and testing on completely unseen target domains without target access. The paper evaluates domain shifts categorized by different dataset sources, varying consequences or stages of actions, and distinct camera viewpoints.

#### Evaluation Protocol

The evaluation setup uses cross-domain validation and out-of-distribution domain generalization benchmarks. Models are trained on source domains using adversarial data augmentation and tested on unseen target domains, measuring performance via top-1 classification accuracy.

#### Domain Shifts (Excluding Object Bias)

The paper evaluates non-object domain shifts by testing cross-viewpoint variations via multi-camera setups in NTU RGB+D, as well as structural temporal variations and environment changes across disparate video sources like UCF101, HMDB51, and PKU-MMD.

#### Verb / Motion Gap Filled

While the paper addresses spatial and temporal feature misalignment, its source datasets rely heavily on static appearance and object context. Our new 1000+ clip dataset explicitly isolates pure temporal motion and verb dynamics to prevent models from relying on object or background shortcuts.

#### Technique to Create Synthetic Dataset

No Strategy















## ${\color{yellow} * }$ Human-centric Analysis Toolkit (HAT) - Enabling Detailed Action Recognition Evaluation Through Video Dataset Augmentation



- Human-centric Analysis Toolkit (HAT): Generates manipulated videos from existing datasets (actor masked out, background inpainted, and similar variants) using segmentation and video inpainting, so background bias can be quantified without new manual annotation.

- NeurIPS 2022 Datasets & Benchmarks: https://openreview.net/forum?id=eOnQ2etkxto

### Basics

* **Base Dataset(s):** Kinetics-400
* **Dataset Scale:** Not reported
* **Classes:** 400 unique classes (Generic activity)
* **Example Classes:** Bending back, Capoeira, Snowmobiling, Eating, Drinking coffee

### Domain Shift Tested

Generalization is defined by isolating human motion from scene context. The framework tests domain shifts across counterfactual synthetic variants: background removal via video inpainting (Background-Only), human figure extraction against gray background (Human-Only), and splicing human actions onto incongruous or cross-class background environments (Action-Swap).

### Evaluation Protocol

The primary setup relies on out-of-distribution counterfactual evaluation on synthetically manipulated video sets without fine-tuning. Models are evaluated using top-1 accuracy, relative accuracy retention, and top-1 background-class prediction rates to quantify model sensitivity to scene context versus human motion cues.

### Domain Shifts (Excluding Object Bias)

The dataset evaluates synthetic background swapping and scene removal, isolating pure human motion from environmental cues like landscapes or indoor scenes. This background-isolation framework provides a usable paradigm for testing cross-environment domain shifts.

### Verb / Motion Gap Filled

The paper relies on automated segmentation that treats interacted objects as background, creating potential visual artifacts or removing essential functional cues. Our 1000+ clip dataset can explicitly capture clean, unmanipulated real-world videos of purely temporal, motion-centric verbs to evaluate dynamic actions without reliance on inpainting artifacts.

### Technique to Create Synthetic Dataset

The toolkit segments the temporal human figure using SeMask, applies video inpainting (E2FGVI) to remove human pixels and fill the background, and overlays extracted human masks onto alternative background videos to construct Action-Swap, Human-Only, and Background-Only counterfactual datasets.




Note: Abstract of this paper looks EXACTLY like what we want to achieve in our research.







## ${\color{yellow} * }$ Charades-Ego - Actor and Observer: Joint Modeling of First and Third-Person Videos



- Charades-Ego: Paired first- and third-person recordings of the same scripted daily activities, covering 157 classes over roughly 7.8K videos.
- Actor and Observer: Joint Modeling of First and Third-Person Videos, CVPR 2018: https://arxiv.org/abs/1804.09627

### Basics:

* **Base Dataset(s):** Charades-Ego (introduced alongside Charades).
* **Dataset Scale:** 7,860 paired videos (68.8 total hours; 34.4 hours first-person, 34.4 hours third-person).
* **Classes:** 157 unique classes (Human-Object Interaction (HOI)).
* **Example Classes:** Holding a laptop, sitting in a chair, opening a door, drinking from a cup, putting on shoes.

#### Domain Shift Tested

Generalization is defined as cross-view perspective transfer between third-person (observer) and first-person (actor/egocentric) views. The domain shift tests how representations learned from widely available third-person video data adapt to first-person egocentric execution of daily indoor human activities.

#### Evaluation Protocol

The evaluation setup measures zero-shot and weakly supervised domain adaptation from third-person to first-person views. Models are trained on third-person or weakly paired data and evaluated on egocentric action recognition using mean Average Precision (mAP) across action categories.

#### Domain Shifts (Excluding Object Bias)

This paper evaluates viewpoint shift (first-person egocentric versus third-person exocentric views) across identical indoor scripts, environments, and actor identities. Incorporating this cross-view viewpoint shift allows evaluation of camera pose, body visibility, and motion perspective shifts independent of ambient environment changes.

#### Verb / Motion Gap Filled

Charades-Ego relies heavily on static object cues within Human-Object Interactions (such as laptops or cups) to identify activity labels, making models susceptible to object bias. Our new 1000+ clip dataset addresses this by isolating fine-grained temporal motion and verb-centric actions without relying on distinct object shortcuts.

#### Technique to Create Synthetic Dataset

No Strategy











## ${\color{yellow} * }$ Charades-Ego - A Large-Scale Dataset of Paired Third and First Person Videos

(Dataset-Report)



Dataset report: Charades-Ego: A Large-Scale Dataset of Paired Third and First Person Videos, arXiv 2018 https://arxiv.org/abs/1804.09626

### Basics:

* **Base Dataset(s):** Charades-Ego (built upon the original Charades framework).
* **Dataset Scale:** 7,860 paired video clips totaling 68.8 hours of video (34.4 hours third-person and 34.4 hours first-person).
* **Classes:** 157 unique classes (Human-Object Interaction (HOI) and daily indoor activities).
* **Example Classes:** Pouring into a cup, Closing a door, Holding a phone, Sitting in a chair, Washing a dish.

#### Domain Shift Tested

Generalization is defined as cross-view translation between an observer viewpoint and an actor viewpoint. The exact domain shift being tested is the shift from exocentric (third-person) to egocentric (first-person) video perspectives for daily household activities.

#### Evaluation Protocol

The evaluation protocol measures zero-shot and cross-view activity classification and localization performance. Models are evaluated using mean Average Precision (mAP) on egocentric clips after training on exocentric clips or joint representations.

#### Domain Shifts (Excluding Object Bias)

The dataset evaluates a cross-view domain shift, specifically comparing exocentric (third-person) versus egocentric (first-person) viewpoints. It tests how models handle severe visual perspective changes, self-occlusions, and camera motion without changing the underlying human-object activity.

#### Verb / Motion Gap Filled

While Charades-Ego pairs viewpoints, its actions are tightly bound to static indoor objects (e.g., cups, doors). Models often recognize the object rather than the temporal motion. Our 1000+ clip dataset addresses this by isolating pure verbs and object-agnostic motion dynamics to directly evaluate temporal feature learning without static object cues.

#### Technique to Create Synthetic Dataset

No Strategy



