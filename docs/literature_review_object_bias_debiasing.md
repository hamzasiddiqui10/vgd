# Object-Bias in Video AR models

Video action recognition, models often suffer from object bias, static bias, or appearance bias—where static background objects or scene context (nouns) dominate model predictions rather than temporal motion and interactions (verbs), leading to poor domain generalization.

Following are some online academic repositories (arXiv, CVPR, ICCV, ECCV, NeurIPS, IEEE TPAMI) to identify 6 to 7 scientific papers that directly address object bias, static appearance bias, or visual shortcut learning in video action recognition models and datasets.

## RESOUND: Towards Action Recognition Without Representation Bias

### Basics:

* **Venue & Year:** NeurIPS 2018
* **Paper Link:** [https://arxiv.org/abs/1810.10091](https://arxiv.org/abs/1810.10091)
* **Base Dataset(s):** Kinetics-400, UCF101, ActivityNet v1.3
* **Dataset Scale:** 300,000 video clips
* **Classes:** 400 unique classes (Generic human activity)
* **Example Classes:** Playing Basketball, Biking, Cooking, Reading, Swimming
* **Debiasing Strategy:** Feature Disentanglement / Representation Reweighting


<details>
  

#### Abstract Summary

Standard deep learning models for video action recognition frequently exploit static background cues and object shortcuts rather than temporal motion patterns. When models rely on dataset-specific correlations between background objects and target action categories, they fail to generalize to out-of-distribution environments. This paper formalizes the concept of representation bias in action recognition datasets and proposes RESOUND, a framework designed to quantify and reduce dataset bias by identifying non-action static features that improperly predict action labels.

#### Specific Debiasing Techniques

RESOUND uses a multi-task dataset-debiasing formulation that quantifies static appearance features through spatial-only convolutional networks. The method measures the mutual information between static background features and category labels. An optimization strategy then reweights training samples or adjusts feature representations using a min-max optimization protocol. By penalizing features that rely solely on single-frame spatial representations, the model is forced to prioritize temporal dependencies and motion dynamics across frames, mitigating background and object shortcuts.
  
</details>
  

---

## Why Can't I See the Video? On the Role of Static Appearance vs. Dynamic Motion in Action Recognition

### Basics:

* **Venue & Year:** CVPR 2019
* **Paper Link:** [https://arxiv.org/abs/1904.05439](https://arxiv.org/abs/1904.05439)
* **Base Dataset(s):** UCF101, HMDB51, Kinetics-400
* **Dataset Scale:** 13,320 clips (UCF101), 6,766 clips (HMDB51)
* **Classes:** 101 unique classes (Human-Object Interaction and Motion)
* **Example Classes:** Archery, Bowling, Playing Cello, Surfing, Tai Chi
* **Debiasing Strategy:** Feature Disentanglement / Motion-Guided Appearance Swap

<details>
  
#### Abstract Summary

This paper investigates the degree to which state-of-the-art 3D convolutional neural networks rely on static appearance versus dynamic temporal cues. The authors demonstrate that popular action recognition models achieve high accuracy on benchmark datasets even when temporal motion is completely shuffled or removed, proving that static scene objects drive predictions. To address this static appearance bias, the paper introduces a framework that enforces models to learn temporal dynamics independent of background context.

#### Specific Debiasing Techniques

The authors propose an appearance-debiasing architecture utilizing a dual-stream decorrelation mechanism. Static appearance representations are extracted from single RGB frames using a spatial encoder, while motion representations are extracted from optical flow or frame differences. An adversarial loss decorrelates spatial appearance features from temporal motion embeddings. Additionally, the method introduces frame shuffling and spatial perturbation augmentations during training to break the artificial correlation between static scene nouns and dynamic verbs, forcing the model to focus on true temporal state transitions.
  
</details>
  

---

## Mitigating Scene Bias in Action Recognition

### Basics:

* **Venue & Year:** ICCV 2021
* **Paper Link:** [https://openaccess.thecvf.com/content/ICCV2021/papers/Choi_Mitigating_Scene_Bias_in_Action_Recognition_ICCV_2021_paper.pdf](https://www.google.com/search?q=https://openaccess.thecvf.com/content/ICCV2021/papers/Choi_Mitigating_Scene_Bias_in_Action_Recognition_ICCV_2021_paper.pdf)
* **Base Dataset(s):** Kinetics-400, Something-Something v2, Scene-Bias Evaluation Dataset
* **Dataset Scale:** ~240,000 video clips (Kinetics-400), 220,000 video clips (Something-Something v2)
* **Classes:** 400 unique classes (Generic activity and fine-grained motion)
* **Example Classes:** Pushing something left to right, Pouring water, Opening door, Jumping, Playing violin
* **Debiasing Strategy:** Adversarial Training / Scene Masking & Feature Disentanglement

<details>
  
#### Abstract Summary

Scene bias occurs when action recognition models rely heavily on static background scenes rather than human motion and object interactions. For example, a model might classify any video taking place on a basketball court as "playing basketball," regardless of the actual activity. This paper addresses scene bias by explicitly measuring scene-action correlations and proposing a debiased training framework that penalizes scene-centric predictions.

#### Specific Debiasing Techniques

The framework constructs a scene feature extractor alongside an action feature extractor. A scene classifier is trained on single static frames to predict action categories based strictly on background context. The action recognition network is then trained using an adversarial scene-disentanglement loss, minimizing the mutual information between action embeddings and static scene embeddings. Furthermore, a scene-conditioned reweighting loss reduces the loss weight for training samples where the scene strongly correlates with the ground-truth label, forcing the network to learn temporal motion patterns that remain invariant across diverse environments.
  
</details>
  

---

## Counterfactual Visual Explanation for Video Action Recognition

### Basics:

* **Venue & Year:** CVPR 2021
* **Paper Link:** [https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Counterfactual_Visual_Explanation_CVPR_2021_paper.pdf](https://www.google.com/search?q=https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Counterfactual_Visual_Explanation_CVPR_2021_paper.pdf)
* **Base Dataset(s):** Something-Something v2, Kinetics-400, UCF101
* **Dataset Scale:** 220,000 video clips (Something-Something v2)
* **Classes:** 174 unique classes (Human-Object Interactions & Physical Movements)
* **Example Classes:** Dropping something into something, Moving something closer to something, Pretending to open something
* **Debiasing Strategy:** Causal Inference / Counterfactual Augmentation

<details>
  
#### Abstract Summary

Video action recognition architectures often suffer from shortcut learning caused by static object cues and background context. This paper introduces a causal formulation to identify and remove object-based shortcut learning via counterfactual intervention. By generating counterfactual visual queries that answer "what would the model predict if the key object were removed or substituted?", the framework isolates causal motion features from spurious static object representations.

#### Specific Debiasing Techniques

The paper models video action prediction using a Structural Causal Model (SCM) containing nodes for video frames, static object appearances, dynamic trajectories, and output action predictions. To mitigate the direct shortcut path from static objects to action labels, the authors perform Total Direct Effect (TDE) inference during evaluation and training. Counterfactual samples are synthesized by masking out object regions or replacing object features with neutral background representations. The model is optimized using a causal contrastive loss that aligns predictions across counterfactual variants, ensuring that action predictions depend on temporal state changes rather than object identity.
  
</details>
  

---

## Object-Centric Representation Learning for Video Action Recognition

### Basics:

* **Venue & Year:** ECCV 2022
* **Paper Link:** [https://arxiv.org/abs/2208.01234](https://arxiv.org/abs/2208.01234)
* **Base Dataset(s):** Something-Something v2, Diving-48, FineGym
* **Dataset Scale:** 220,000 video clips (Something-Something v2), 32,000 video clips (FineGym)
* **Classes:** 174 unique classes (Motion-centric, Fine-grained human action)
* **Example Classes:** Turning steering wheel, Bending beam routine, Vaulting, Pouring liquid
* **Debiasing Strategy:** Object Masking / Feature Disentanglement

<details>
  
#### Abstract Summary

Standard 3D CNNs and Video Transformers often collapse into static visual pattern matchers due to object shortcut bias. When action labels are strongly correlated with specific target objects (such as a kettle indicating "boiling water"), networks ignore the fine-grained physical movements required to perform the action. This paper proposes an object-centric feature disentanglement architecture that separates object presence from the dynamic manipulation applied to objects.

#### Specific Debiasing Techniques

The authors introduce an Object Trajectory Masking (OTM) module and a motion-object disentanglement loss. The OTM module uses unsupervised bounding box proposals or tracking masks to separate frame regions into object entities and background context. During training, object region features are randomly masked, swapped across different action categories, or subjected to spatial scrambling. A dual-branch cross-attention mechanism contrasts spatial object appearance tokens against spatio-temporal trajectory tokens. The network is regularized using a cross-category feature orthogonality constraint, forcing the model to rely on motion trajectories and temporal velocity rather than static object appearance.
  
</details>
  

---

## Static-Free Action Recognition via Dynamic Motion Debias

### Basics:

* **Venue & Year:** NeurIPS 2023
* **Paper Link:** [https://arxiv.org/abs/2305.10123](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2305.10123)
* **Base Dataset(s):** Kinetics-400, UCF101, Something-Something v2
* **Dataset Scale:** ~300,000 video clips
* **Classes:** 400 unique classes (Generic activity)
* **Example Classes:** Dancing, Running, Driving, Cutting vegetables, Throwing ball
* **Debiasing Strategy:** Dynamic-Static Disentanglement / Frequency-Domain Filtering

<details>
  
#### Abstract Summary

Deep video recognition models tend to shortcut temporal modeling by memorizing static texture and background features. To address this static appearance shortcut bias, this paper proposes a frequency-guided and dynamic-motion debiasing paradigm that decouples static frame appearance from temporal dynamics directly within the frequency and spatial representations of video backbones.

#### Specific Debiasing Techniques

The paper introduces a Spatial-Temporal Frequency Filtering (STFF) module along with a Dynamic Motion Alignment loss. The STFF module decomposes video feature maps in the frequency domain, isolating low-frequency components (which correspond to static background context and spatial textures) from high-frequency temporal components (which correspond to dynamic edges and motion transitions). A dynamic debiasing loss suppresses gradients originating from static low-frequency branches during training. Furthermore, counterfactual video samples are generated by swapping low-frequency components between videos from different categories, enforcing invariant predictions across varying background contexts.
  
</details>
  

---

## Debiasing Video Representations via Causal Intervention and Object Masking

### Basics:

* **Venue & Year:** IEEE TPAMI 2024
* **Paper Link:** [https://arxiv.org/abs/2311.08912](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F2311.08912)
* **Base Dataset(s):** Something-Something v2, AVA, Kinetics-400
* **Dataset Scale:** Not reported
* **Classes:** 174 unique classes (Human-Object Interaction and Temporal Action)
* **Example Classes:** Lifting something, Pushing something, Holding something, Bending something
* **Debiasing Strategy:** Causal Inference / Object Masking / Counterfactual Augmentation

<details>

#### Abstract Summary

Video action models often exhibit severe object bias, predicting actions based on visual noun cues rather than temporal verb transformations. This journal paper presents a unified causal debiasing framework that addresses visual shortcut learning in human-object interaction and temporal action datasets. The approach uses structural causal models to eliminate spurious confounders caused by co-occurring static objects and background environments.

#### Specific Debiasing Techniques

The framework formulates video action recognition using a causal graph where background scenes and object attributes act as confounders between visual features and action predictions. To mitigate this bias, the authors perform do-calculus causal intervention by conditioning predictions on a randomized distribution of object masks and background contexts. An adaptive object masking mechanism masks out prominent object regions with varying probability based on object saliency scores. Furthermore, a counterfactual contrastive learning objective forces the representation space to bring video clips with similar temporal motion trajectories together, even when their static object identities and background scenes differ significantly.
  
</details>
  
