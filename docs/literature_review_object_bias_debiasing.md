# Object-Bias in Video AR models

Video action recognition, models often suffer from object bias, static bias, or appearance bias—where static background objects or scene context (nouns) dominate model predictions rather than temporal motion and interactions (verbs), leading to poor domain generalization.

Following are 7 online academic papers from some repositories (arXiv, CVPR, ICCV, ECCV, NeurIPS, IEEE TPAMI) that directly address object bias, static appearance bias, or visual shortcut learning in video action recognition models and datasets.



---
---
### Some additional information

SCUBA and SCUFO are test benchmarks and evaluation protocols used in video AR. They measure "static bias," or shortcut learning, where a model incorrectly relies on fixed background elements or stationary foreground objects instead of actual movement.
- **SCUBA** evaluates bg bias by replacing or altering the bg behind a moving action clip.
- **SCUFO** evaluates fg bias by stacking a single static frame to strip away motion cues and test reliance on non-moving elements.

Both benchmarks were introduced alongside debiasing methods like StillMix to help vision models learn true dynamic motion instead of taking shortcuts.

## Mitigating and Evaluating Static Bias of Action Representations in the Background and the Foreground

https://doi.org/10.48550/arxiv.2209.09393 (can't access full paper)

### Basics:

* **Venue & Year:** ICCV 2023  (Li et al., 2022)
* **Base Dataset(s):** HMDB51, UCF101, Kinetics-400, plus the bias evaluation benchmarks SCUBA and SCUFO  (Li et al., 2022; Fioresi et al., 2025)
* **Classes:** Not reported in the supplied evidence for the full experimental suite; the paper studies generic action recognition with explicit background and foreground static-cue stress tests.
* **Example Classes:** golf swinging, walking, and other action classes drawn from standard action benchmarks  (Li et al., 2022)
* **Debiasing Strategy:** Bias-inducing frame detection and mixed-video augmentation, termed **StillMix**  (Li et al., 2022)


#### Abstract Summary

In video AR, shortcut static features can interfere with the learning of motion features, resulting in poor (OOD) generalization. 
Sources of static bias:
1. The video background,
2. The video foreground, such as the clothing of the actor, can also provide static bias. 

In this paper, we empirically verify the existence of foreground static bias by creating test videos with conflicting signals from the static and moving portions of the video. To tackle this issue, we propose the StillMix technique, to learn robust action representations. 



#### Specific Debiasing Techniques

Specifically, StillMix identifies bias-inducing video frames using a 2D reference network and mixes them with videos for training, serving as effective bias suppression even when we cannot explicitly extract the source of bias within each video frame or enumerate types of bias. Finally, to precisely evaluate static bias, we synthesize two new benchmarks, SCUBA for static cues in the background, and SCUFO for static cues in the foreground. With extensive experiments, we demonstrate that StillMix mitigates both types of static bias and improves video representations for downstream applications.
 

---

## Why Can't I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition

https://arxiv.org/abs/1912.05534

### Basics:

* **Venue & Year:** arXiv 2019  (Choi et al., 2019)
* **Base Dataset(s):** Kinetics for pre-training, with transfer to downstream action classification, temporal localization, and spatio-temporal action detection tasks  (Choi et al., 2019)
* **Dataset Scale:** Not reported in the supplied evidence  (Choi et al., 2019)
* **Classes:** the focus is generic activity recognition with scene context bias  (Choi et al., 2019)
* **Example Classes:** playing basketball on a basketball court is the motivating example; additional class names are not reported in the supplied evidence  (Choi et al., 2019)
* **Debiasing Strategy:** Scene-adversarial training plus human-mask confusion loss  (Choi et al., 2019)


#### Abstract Summary

This paper focuses specifically on **scene bias**, the tendency of action models to exploit recurring scene context instead of action evidence. It argues that datasets such as Kinetics contain substantial scene representation bias, so features learned during pre-training may transfer poorly to new classes and new tasks. 

The central claim is that suppressing scene predictability during representation learning yields more generalizable action features for downstream tasks.

#### Specific Debiasing Techniques

The training objective augments standard action cross-entropy with two extra losses: 
1. An **adversarial scene-class loss** that pushes the representation to be uninformative about scene type; because ground-truth scene labels are unavailable, scene labels are approximated using predictions from a pre-trained scene classifier. 
2. An **entropy-based confusion loss** on videos where humans are masked out, training the network to fail at correct action prediction when the actor evidence is removed. 

The method is explicitly model-agnostic and was shown to improve both a 3D-ResNet-18 and a VGG-16 based pipeline, although the authors note that the final systems still trail stronger state-of-the-art baselines because the underlying backbones are relatively weak (Choi et al., 2019).
 
---

## ALBAR: Adversarial Learning approach to mitigate Biases in Action Recognition

https://arxiv.org/abs/2502.00156

### Basics:

* **Venue & Year:** ICLR 2025  (Fioresi et al., 2025)
* **Base Dataset(s):** Kinetics-400, UCF101, HMDB51, evaluated with SCUBA and SCUFO bias protocols
* **Dataset Scale:** Not reported in the supplied evidence 
* **Classes:** the tasks are generic action recognition with separate background- and foreground-bias evaluation  
* **Example Classes:** Not reported in the supplied evidence
* **Debiasing Strategy:** End-to-end adversarial debiasing on static clips with entropy maximization and gradient penalty  (Fioresi et al., 2025)

#### Abstract Summary

ALBAR addresses both **background bias** and **foreground bias**, arguing that the literature has emphasized background shortcuts while leaving appearance-related foreground bias less comprehensively studied. The method is positioned against approaches that need scene or object labels, separate critic models, or off-the-shelf saliency detectors, all of which depend on auxiliary supervision quality. 

Its main claim is that static-bias mitigation can be achieved with a single 3D encoder and no attribute annotations, while also improving the evaluation protocol itself.

#### Specific Debiasing Techniques

ALBAR samples a frame from each video and repeats it to create a **static clip**, then applies an adversarial cross-entropy loss through negative gradients so the network is penalized for making action predictions from static evidence alone. 
- An **entropy maximization loss** forces the static-clip predictions toward a uniform class distribution, explicitly discouraging reliance on static cues, and 
- a **gradient penalty** regularizes the debiasing process. 

The paper also identifies a shortcut in a prior UCF101 bias protocol, where bounding boxes leaked background information around the actor, and replaces that with segmentation-based boundaries for cleaner fg/bg separation. It reports state-of-the-art results on SCUBA and SCUFO, including over 12 percent absolute improvement on combined debiasing performance on HMDB51 (Fioresi et al., 2025).
 
---

## Counterfactual Debiasing Inference for Compositional Action Recognition

https://dl.acm.org/doi/10.1145/3474085.3475472

### Basics:

* **Venue & Year:** ACM Multimedia 2021  (Sun et al., 2021)
* **Base Dataset(s):** Something-Else (Sun et al., 2021)
* **Dataset Scale:** Not reported in the supplied evidence  (Sun et al., 2021)
* **Classes:** the task is **compositional action recognition** with verb-noun combinations  (Sun et al., 2021)
* **Example Classes:** tear something, move something.
* **Debiasing Strategy:** Causal inference with factual and counterfactual branches, called **Counterfactual Debiasing Network**  (Sun et al., 2021)



#### Abstract Summary

This paper targets **appearance bias** in compositional AR, where models must generalize to unseen verb-noun combinations rather than memorizing frequentt object-appearance patterns. It argues that simply dropping appearance information fails to generalize well to unseen actions. 

The proposed Counterfactual Debiasing Network explicitly models appearance information and then removes its effect through a causal subtraction mechanism, aiming to improve recognition of unseen action instances. (Like the Panaf-fgbg paper)

#### Specific Debiasing Techniques

The framework defines 
- a **factual inference** path that uses both appearance information and structural information derived from tracklets and video content. And 
- a **counterfactual inference** path that uses video content with appearance information, isolating the contribution of appearance cues. 

A causal graph then removes appearance bias by subtracting the counterfactual prediction from the factual prediction, thereby estimating action evidence that is less contaminated by object appearance shortcuts. The reported result is improved generalization on the Something-Else benchmark relative to prior state-of-the-art methods for unseen compositional actions.
 
    Tracklet definition:

    An instance tracklet is a short, linked sequence of object masks or bounding boxes representing the exact same individual object across multiple consecutive frames in a video. It forms the core output of Video Instance Segmentation (VIS) tasks, merging spatial object detection and temporal tracking into a unified trajectory. 

    A tracklet is a continuous space-time path (T frames long) for a single unique object instance.

---

## RESOUND: Towards Action Recognition Without Representation Bias



### Basics:

* **Venue & Year:** 2018, ECCV  (Li & Vasconcelos, 2018)
* **Base Dataset(s):** Existing action datasets sampled through Explicit RESOUND, plus the newly guided dataset Diving48  (Li & Vasconcelos, 2018)
* **Dataset Scale:** Diving48 contains over 18,000 video clips  (Li & Vasconcelos, 2018)
* **Classes:** 48 unique classes, fine-grained diving actions, which are motion-based rather than generic scene categories  (Li & Vasconcelos, 2018)
* **Example Classes:** Not reported in the supplied evidence
* **Debiasing Strategy:** Dataset calibration and representation-bias minimization through Explicit and Implicit **RESOUND**  (Li & Vasconcelos, 2018)



#### Abstract Summary

RESOUND is primarily a **dataset and evaluation** paper that formalizes representation bias in action recognition and argues that current benchmarks allow non-ground-truth representations to achieve strong performance  (Li & Vasconcelos, 2018). In the action domain, it concludes that major datasets are biased toward static representations such as objects, scenes, and people, which undermines the goal of learning motion-grounded action models  (Li & Vasconcelos, 2018). The paper frames **dataset calibration** as necessary if benchmark evaluation is to converge toward the intended action representation rather than shortcut cues  (Li & Vasconcelos, 2018).

#### Specific Debiasing Techniques

RESOUND proposes a procedure to both **quantify** and **minimize** representation bias  (Li & Vasconcelos, 2018). The **Explicit RESOUND** variant assembles improved datasets by sampling from existing datasets to reduce biasing factors  (Li & Vasconcelos, 2018). The **Implicit RESOUND** variant guides new dataset construction, and its main demonstration is Diving48, a fine-grained competitive diving benchmark with over 18,000 clips spanning 48 classes  (Li & Vasconcelos, 2018). The central debiasing mechanism is therefore not a training loss but a calibration procedure that constrains dataset composition so that static cues such as scene, object, or person appearance are less predictive than the motion pattern itself  (Li & Vasconcelos, 2018). Experimental evaluation confirms that this strategy reduces static bias in action recognition datasets  (Li & Vasconcelos, 2018).
 

---

## Background no more: Action recognition across domains by causal interventions

### Basics:

* **Venue & Year:** Computer Vision and Image Understanding 2024  (Rastegar et al., 2024)
* **Base Dataset(s):** Two video domain generalization benchmarks, including Kinetics2Mimetics  (Rastegar et al., 2024)
* **Dataset Scale:** Not reported in the supplied evidence  (Rastegar et al., 2024)
* **Classes:** the focus is generic action recognition under domain shift  (Rastegar et al., 2024)
* **Example Classes:** Not reported in the supplied evidence
* **Debiasing Strategy:** Causal intervention on action and background with source-time and test-time interventions  (Rastegar et al., 2024)



#### Abstract Summary

This paper addresses **background confounding** under appearance distribution shift between source and target domains, making it a direct fit for static scene-bias mitigation in video action recognition  (Rastegar et al., 2024). Its core claim is that background acts as a confounder for the class label, so the action predictor should intervene on the action representation to separate it from domain-specific appearance  (Rastegar et al., 2024). The problem formulation is explicitly about **video domain generalization**, where shortcut reliance becomes most visible when the target domain differs in appearance from the training domain  (Rastegar et al., 2024).

#### Specific Debiasing Techniques

The method learns a **causally debiased model** on the source domain using three possible do-operator interventions that separate action from background  (Rastegar et al., 2024). It then adds a **test-time action intervention** to better align source and target distributions, extending debiasing beyond training-time regularization alone  (Rastegar et al., 2024). This is a causal alternative to adversarial or augmentation-based approaches: rather than only making static cues harder to exploit, it explicitly models the confounding path between background and label and intervenes on that path  (Rastegar et al., 2024). On two domain generalization benchmarks, it reports state-of-the-art results on Kinetics2Mimetics, which the paper describes as the benchmark with the largest domain shift  (Rastegar et al., 2024).
 

---

## Mitigating Representation Bias in Action Recognition: Algorithms and Benchmarks

### Basics:

* **Venue & Year:** 2022, venue not reported in the supplied evidence  (Duan et al., 2022)
* **Base Dataset(s):** Kinetics-400 re-distributed splits, ARAS, downstream transfer datasets, and external web media for OmniDebias  (Duan et al., 2022)
* **Dataset Scale:** Not reported in the supplied evidence  (Duan et al., 2022)
* **Classes:** the work targets generic action recognition with rare-scene and rare-object shifts  (Duan et al., 2022)
* **Example Classes:** Not reported in the supplied evidence
* **Debiasing Strategy:** Spatial-aware Multi-Aspect Debiasing, plus selective web-data augmentation via **OmniDebias**  (Duan et al., 2022)



#### Abstract Summary

This paper treats representation bias as a broader problem than scene bias alone, emphasizing that action datasets often fail on videos with **rare scenes or objects** because models learn non-action facets that are spuriously predictive in the training distribution  (Duan et al., 2022). The work tackles bias from both the **algorithmic** side and the **dataset** side, combining a generic debiasing framework with selective external-data augmentation  (Duan et al., 2022). It also builds evaluation protocols, including facet-based train-test redistribution on Kinetics-400 and the ARAS dataset focused on actions with rare scenes, to measure out-of-distribution robustness more faithfully  (Duan et al., 2022).

#### Specific Debiasing Techniques

The proposed **SMAD** framework combines **multi-aspect adversarial training** with **Spatial-Aware Actionness Reweighting**  (Duan et al., 2022). Multi-aspect adversarial training extends prior single-factor debiasing methods by enforcing invariance to multiple non-action facets, including background scene and interacted object, rather than assuming one known bias source  (Duan et al., 2022)<paper_cite sid="S





 ---
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Choi, J., Gao, C., Messou, J. C. E., & Huang, J.-B. (2019). Why Can't I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition. *ArXiv, abs/1912.05534*. https://doi.org/10.48550/arxiv.1912.05534
 
Duan, H., Zhao, Y., Chen, K., Xiong, Y., & Lin, D. (2022). Mitigating Representation Bias in Action Recognition: Algorithms and Benchmarks. 557-575. https://doi.org/10.48550/arxiv.2209.09393
 
Fioresi, J., Dave, I., & Shah, M. (2025). ALBAR: Adversarial Learning approach to mitigate Biases in Action Recognition. *ArXiv, abs/2502.00156*. https://doi.org/10.48550/arxiv.2502.00156
 
Li, H., Liu, Y., Zhang, H., & Li, B. A. (2022). Mitigating and Evaluating Static Bias of Action Representations in the Background and the Foreground. *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, 19854-19866. https://doi.org/10.1109/iccv51070.2023.01823
 
Li, Y., & Vasconcelos, N. (2018). RESOUND: Towards Action Recognition Without Representation Bias. 520-535. https://doi.org/10.1007/978-3-030-01231-1_32
 
Rastegar, S., Doughty, H., & Snoek, C. G. M. (2024). Background no more: Action recognition across domains by causal interventions. *Comput. Vis. Image Underst., 242*, 103975. https://doi.org/10.1016/j.cviu.2024.103975
 
Sun, P., Wu, B., Li, X., Li, W., Duan, L., & Gan, C. (2021). Counterfactual Debiasing Inference for Compositional Action Recognition. *Proceedings of the 29th ACM International Conference on Multimedia*. https://doi.org/10.1145/3474085.3475472
 
