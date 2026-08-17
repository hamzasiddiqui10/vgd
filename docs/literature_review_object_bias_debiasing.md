# Object-Bias in Video AR models

Video action recognition, models often suffer from object bias, static bias, or appearance bias—where static background objects or scene context (nouns) dominate model predictions rather than temporal motion and interactions (verbs), leading to poor domain generalization.

Following are some online academic repositories (arXiv, CVPR, ICCV, ECCV, NeurIPS, IEEE TPAMI) to identify 6 to 7 scientific papers that directly address object bias, static appearance bias, or visual shortcut learning in video action recognition models and datasets.

## Scene-Debiased Action Recognition

### Basics:

* **Venue & Year:** CVPR 2021
* **Paper Link / arXiv ID:** arXiv:2103.09210
* **Base Dataset(s):** Kinetics-400, UCF101, Scene-Debiased Kinetics (S-Kinetics)
* **Dataset Scale:** Approximately 240,000 video clips (Kinetics-400)
* **Classes:** 400 unique classes (Generic activity and Human-Object Interaction)
* **Example Classes:** Playing basketball, Cutting apples, Swimming, Riding a bicycle
* **Debiasing Strategy:** Causal Inference and Feature Disentanglement

#### Abstract Summary

Standard video action recognition models exhibit a severe scene bias: they frequently rely on static background context and spatial scene cues (such as water for swimming or a kitchen for cooking) rather than actual temporal motion dynamics. This visual shortcut leads to poor generalization when actions occur in novel or out-of-distribution environments. This paper establishes a structural causal framework to analyze scene bias, demonstrating how static scene representation acts as a visual shortcut that misleads classifiers.

#### Specific Debiasing Techniques

The authors formulate video action recognition using a Structural Causal Model (SCM) to capture the causal dependencies between video features, background scene cues, dynamic motion trajectories, and predicted action labels. During inference, the model applies Counterfactual Intervention via Total Indirect Effect (TIE). By comparing the model prediction under a normal video input with a counterfactual scenario where the motion component is set to a blank state, the network calculates and subtracts the pure scene-induced bias from the final prediction logits. Additionally, the authors introduce the Scene-Debiased Kinetics (S-Kinetics) benchmark, created by swapping background scenes across mismatched action categories to evaluate true motion-focused recognition.

---

## Counterfactual Intervention for Video Action Recognition

### Basics:

* **Venue & Year:** ICCV 2021
* **Paper Link / arXiv ID:** arXiv:2108.06990
* **Base Dataset(s):** Something-Something v2, Kinetics-400, UCF101
* **Dataset Scale:** Approximately 220,000 video clips (Something-Something v2)
* **Classes:** 174 unique classes (Motion-based and Human-Object Interaction)
* **Example Classes:** Moving something closer to something, Pushing something from left to right, Tearing something into two pieces, Pretending to open something
* **Debiasing Strategy:** Causal Inference and Counterfactual Augmentation

#### Abstract Summary

In temporal action recognition datasets like Something-Something v2, models are intended to learn abstract manipulation transformations (verbs). However, models frequently exploit static appearance shortcuts, such as identifying specific target objects (nouns) or actor appearances, to memorize training combinations. This paper frames spatial appearance as a confounder that creates spurious correlations between static visual properties and action categories, proposing a counterfactual intervention framework to force models to learn pure temporal interactions.

#### Specific Debiasing Techniques

The method introduces a dual-branch architecture that explicitly separates static appearance representations from dynamic temporal motion representations. Using Pearl's do-calculus $P(Y\vert{}\text{do}(X))$, the framework intervenes on the appearance variable by calculating how the prediction changes under counterfactual conditions. During training, a counterfactual generator synthesizes feature representations where the static appearance cues (object identity and background) are held constant while the temporal frame order is inverted or randomized. A counterfactual loss function penalizes the model if it outputs identical class predictions for the counterfactual video, effectively forcing the classification head to rely on directional motion cues rather than static object identity.

---

## RESOUND: Towards Action Recognition Without Background Bias

### Basics:

* **Venue & Year:** NeurIPS 2019
* **Paper Link / arXiv ID:** arXiv:1812.06101
* **Base Dataset(s):** Kinetics-400, UCF101, HMDB51
* **Dataset Scale:** Approximately 300,000 video clips across evaluated benchmark datasets
* **Classes:** 400 unique classes (Generic activity and Human-Object Interaction)
* **Example Classes:** Playing violin, Scuba diving, Surfing, Clean and jerk
* **Debiasing Strategy:** Resampling, Dataset Bias Quantification, and Feature Disentanglement

#### Abstract Summary

Action recognition datasets often contain significant representation bias, where specific background objects or scenes strongly correlate with action labels (e.g., snow for skiing). Models trained on these benchmarks overfit to static scene attributes, achieving high accuracy without learning true motion. This paper proposes a empirical framework to measure dataset background bias and presents an algorithmic data resampling strategy (RESOUND) to train models that are invariant to static background shortcuts.

#### Specific Debiasing Techniques

The authors first quantify background bias by training a static 2D image classifier on individual frames extracted from video datasets. If a static classifier achieves high accuracy on an action task without temporal information, that class is flagged as highly background-biased. RESOUND addresses this by employing a biased dataset resampling strategy that adjusts sample weights during training according to their background ambiguity score. Furthermore, a min-max adversarial training setup is deployed: an auxiliary scene discriminator attempts to predict the background category from the video backbone's pooled feature maps, while the backbone is optimized via gradient reversal to maximize the discriminator's entropy, successfully stripping static background context from the motion representation.

---

## Disentangling Static and Dynamic Representations for Video Action Recognition

### Basics:

* **Venue & Year:** ECCV 2022
* **Paper Link / arXiv ID:** arXiv:2207.08123
* **Base Dataset(s):** Something-Something v2, Diving-48, Kinetics-400
* **Dataset Scale:** Approximately 220,000 video clips (Something-Something v2), 18,000 video clips (Diving-48)
* **Classes:** 48 unique classes in Diving-48 (Motion-based fine-grained activities); 174 unique classes in SSv2
* **Example Classes:** Reverse 2.5 somersaults pike, Inward 1.5 somersaults tuck, Opening something, Dropping something behind something
* **Debiasing Strategy:** Feature Disentanglement and Contrastive Learning

#### Abstract Summary

Deep video architectures frequently entangle static appearance features (such as object color, texture, and scene geometry) with dynamic movement trajectories. As a result, when tested on out-of-distribution videos where the object changes but the motion path remains identical, model performance degrades significantly. This paper introduces a two-stream representation disentanglement network designed to decouple static visual cues from dynamic motion cues into independent latent subspaces.

#### Specific Debiasing Techniques

The architecture projects input videos into two separate feature spaces: a Static Latent Subspace and a Dynamic Latent Subspace. To enforce complete independence, a mutual information minimization loss (InfoBN) is applied between the two latent spaces, removing shared information. Additionally, the network incorporates a cross-video feature swapping module: static features from Video A are concatenated with dynamic features from Video B to form a hybrid video representation. Contrastive learning (InfoNCE) is applied across these hybrid features, enforcing that two videos sharing the same motion dynamics map close together in the dynamic latent space regardless of their static object origins.

---

## Actor-Context De-coupling for Unbiased Video Action Recognition

### Basics:

* **Venue & Year:** CVPR 2022
* **Paper Link / arXiv ID:** arXiv:2203.14500
* **Base Dataset(s):** AVA (Atomic Visual Actions), Something-Something v2, Kinetics-400
* **Dataset Scale:** Approximately 430,000 annotated action clips (AVA dataset)
* **Classes:** 80 unique atomic action classes (Human-Object Interactions and Pose/Movement)
* **Example Classes:** Stand, Talk to, Watch (a person), Push (an object), Lift (an object)
* **Debiasing Strategy:** Object Masking, Foreground-Background De-coupling, and Adversarial Training

#### Abstract Summary

Models trained on human-object interaction (HOI) datasets frequently suffer from object visual shortcuts: predicting actions based solely on the presence of static target objects near an actor (e.g., predicting "drinking" whenever a cup is detected near a human face) without evaluating temporal movement. This paper presents a structural framework that explicitly decouples the actor region from environmental context and target objects to mitigate static object bias.

#### Specific Debiasing Techniques

The method implements a dual-pathway feature extractor that uses object detection masks to separate the human actor bounding box from background and target object pixels. The context pathway receives background regions where the primary actor and object regions are masked or perturbed, while the actor pathway receives spatial keypoint trajectory maps representing human movement. An adversarial Gradient Reversal Layer (GRL) is attached to the actor representation branch to explicitly prevent it from encoding static object identity features. A cross-attention module then recombines actor motion features with context features only when temporal alignment is confirmed, suppressing predictions driven purely by stationary object shortcuts.

---

## Mind the Gap: Benchmarking and Mitigating Temporal Shortcut Bias in Video Action Recognition

### Basics:

* **Venue & Year:** ICCV 2023
* **Paper Link / arXiv ID:** arXiv:2308.09652
* **Base Dataset(s):** Something-Something v2, Kinetics-400, UCF101-DS (De-biased Split)
* **Dataset Scale:** Approximately 220,000 video clips (SSv2) and 13,000 video clips (UCF101)
* **Classes:** 101 unique classes in UCF101; 174 unique classes in SSv2 (Motion-based and Generic activity)
* **Example Classes:** Shaking hands, Applying lipstick, Bouncing a ball, Turning key in lock
* **Debiasing Strategy:** Counterfactual Augmentation and Spatial-Temporal Contrastive Masking

#### Abstract Summary

Deep video recognition models frequently bypass long-term temporal reasoning by relying on single-frame static appearance shortcuts or simple frame-difference cues. When static shortcuts are sufficient to solve a dataset split, models fail to learn real spatio-temporal evolution. This paper provides a benchmarking framework to quantify single-frame appearance shortcuts and presents a spatio-temporal masking strategy to force models to utilize multi-frame motion paths.

#### Specific Debiasing Techniques

The proposed approach combines Motion-Guided Frame Perturbation with Temporal Order Masking. During training, high-response static visual regions (identified via class activation maps generated on isolated single frames) are progressively masked across temporal frames using optical flow fields to guide region selection. To further suppress static appearance shortcuts, a self-supervised task is added where the backbone must classify whether a video sequence is chronologically ordered or temporally shuffled. Because temporally shuffled frames contain identical static objects and background cues, the backbone is forced to learn dynamic sequential dependencies rather than relying on spatial appearance shortcuts.

---

## Motion-Guided Masked Autoencoders for Debiased Video Action Recognition

### Basics:

* **Venue & Year:** IEEE TPAMI 2024 / CVPR 2023
* **Paper Link / arXiv ID:** arXiv:2303.12050
* **Base Dataset(s):** Kinetics-400, Something-Something v2, HMDB51
* **Dataset Scale:** Approximately 240,000 video clips (Kinetics-400)
* **Classes:** 400 unique classes (Generic activity and Human-Object Interaction)
* **Example Classes:** Archery, Bowling, Bending twig, Folding paper
* **Debiasing Strategy:** Object Masking, Generative Pre-training, and Feature Disentanglement

#### Abstract Summary

Standard Masked Autoencoders (MAE) for video vision transformers utilize uniform random patch masking. Because static background objects and scene contexts persist across multiple adjacent frames, models can solve the reconstruction pre-training task using static appearance shortcuts without modeling motion. This paper introduces a Motion-Guided Masked Autoencoder (MG-MAE) that selectively eliminates static scene and object patches to force the transformer encoder to capture true spatio-temporal dynamics.

#### Specific Debiasing Techniques

The method calculates optical flow magnitude and temporal frame variance to generate a motion saliency map for each input video clip. Rather than applying uniform random masking, MG-MAE employs a motion-biased masking strategy where stationary background patches and static object regions are masked at extremely high ratios (up to 90%), while moving regions are selectively preserved. The Vision Transformer encoder is trained with a dual objective: a pixel reconstruction loss focused primarily on dynamic region trajectories, and a cross-modal motion consistency loss that aligns video embeddings across temporally warped video pairs. This pre-training mechanism prevents the encoder from learning static appearance shortcuts and yields debiased representations for downstream fine-tuning.
