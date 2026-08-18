# Object-Bias in Video AR models

Video action recognition, models often suffer from object bias, static bias, or appearance bias—where static background objects or scene context (nouns) dominate model predictions rather than temporal motion and interactions (verbs), leading to poor domain generalization.

Following are 7 online academic papers from some repositories (arXiv, CVPR, ICCV, ECCV, NeurIPS, IEEE TPAMI) that directly address object bias, static appearance bias, or visual shortcut learning in video action recognition models and datasets.



---
---


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
 

    SCUBA and SCUFO are test benchmarks and evaluation protocols used in video AR. They measure "static bias," or shortcut learning, where a model incorrectly relies on fixed background elements or stationary foreground objects instead of actual movement.
    - **SCUBA** evaluates bg bias by replacing or altering the bg behind a moving action clip.
    - **SCUFO** evaluates fg bias by stacking a single static frame to strip away motion cues and test reliance on non-moving elements.

    Both benchmarks were introduced alongside debiasing methods like StillMix to help vision models learn true dynamic motion instead of taking shortcuts.


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

https://link.springer.com/chapter/10.1007/978-3-030-01231-1_32

### Basics:

* **Venue & Year:** 2018, ECCV  (Li & Vasconcelos, 2018)
* **Base Dataset(s):** Existing action datasets sampled through Explicit RESOUND, plus the newly guided dataset Diving48 which has videos of divers performing actions.  (Li & Vasconcelos, 2018)
* **Dataset Scale:** Diving48 contains over 18,000 video clips  (Li & Vasconcelos, 2018)
* **Classes:** 48 unique classes, fine-grained diving actions, which are motion-based rather than generic scene categories  (Li & Vasconcelos, 2018)
* **Debiasing Strategy:** Dataset calibration and representation-bias minimization through Explicit and Implicit **RESOUND**  (Li & Vasconcelos, 2018)



#### Abstract Summary

RESOUND is primarily a **dataset and evaluation** paper that formalizes representation bias in AR and argues that current benchmarks allow non-ground-truth representations to achieve strong performance. In the action domain, it concludes that major datasets are biased toward static representations such as objects, scenes, and people. 

The paper frames **dataset calibration** as necessary if benchmark evaluation is to converge toward the intended action representation rather than shortcut cues.

#### Specific Debiasing Techniques

RESOUND proposes a procedure to both **quantify** and **minimize** representation bias. 

- The **Explicit RESOUND** variant assembles improved datasets by sampling from existing datasets to reduce biasing factors. 
- The **Implicit RESOUND** variant guides new dataset construction (and its main demonstration is Diving48, a fine-grained competitive diving benchmark with over 18,000 clips spanning 48 classes). 

The central debiasing mechanism is therefore not a training loss but a calibration procedure that constrains dataset composition so that static cues such as scene, object, or person appearance are less predictive than the motion pattern itself. Experimental evaluation confirms that this strategy reduces static bias in AR datasets (Li & Vasconcelos, 2018).
 
---

## Background no more: Action recognition across domains by causal interventions

note: only bg bias tackled in this paper. Doesnt do anything about object bias.

https://www.sciencedirect.com/science/article/abs/pii/S1077314224000560?via%3Dihub (cant access full paper)

### Basics:

* **Venue & Year:** Computer Vision and Image Understanding 2024  (Rastegar et al., 2024)
* **Base Dataset(s):** Two video domain generalization benchmarks, including Kinetics2Mimetics  (Rastegar et al., 2024)
* **Classes:** the focus is generic action recognition under domain shift  (Rastegar et al., 2024)
* **Debiasing Strategy:** Causal intervention on action and background with source-time and test-time interventions  (Rastegar et al., 2024)


#### Abstract Summary

This paper addresses **background confounding** under appearance distribution shift between source and target domains, making it a direct fit for static scene-bias mitigation in video action recognition. Its core claim is that background acts as a confounder for the class label, so the action predictor should intervene on the action representation to separate it from domain-specific appearance. 

The problem formulation is explicitly about **video domain generalization**, where shortcut reliance becomes most visible when the target domain differs in appearance from the training domain  (Rastegar et al., 2024).

#### Specific Debiasing Techniques

The method learns a **causally debiased model** on the source domain using three possible do-operator interventions that separate action from background  (Rastegar et al., 2024). It then adds a **test-time action intervention** to better align source and target distributions, extending debiasing beyond training-time regularization alone. This is a causal alternative to adversarial or augmentation-based approaches: rather than only making static cues harder to exploit, it explicitly models the confounding path between background and label and intervenes on that path. On two domain generalization benchmarks, it reports state-of-the-art results on Kinetics2Mimetics, which the paper describes as the benchmark with the largest domain shift (Rastegar et al., 2024).
 

---

## Mitigating Representation Bias in Action Recognition: Algorithms and Benchmarks

https://arxiv.org/abs/2209.09393

### Basics:

* **Venue & Year:** ECCVW 2022 / arXiv 2022
* **Base Dataset(s):** Kinetics-400 (re-distributed splits), UCF101, Something-Something v2, and Action with RAre Scene (ARAS)
* **Dataset Scale:** Kinetics-400 contains approximately 240,000 video clips (around 650 hours). The newly introduced ARAS evaluation set contains 1,038 video clips sampled from 920 untrimmed YouTube videos.
* **Classes:** 400 unique classes in Kinetics-400; 104 evaluation classes in ARAS (Generic activities, human-object interactions, and scene-associated human actions).
* **Example Classes:** Playing basketball, swimming, playing accordion, skateboarding, riding a bike.
* **Debiasing Strategy:** Spatial-aware Multi-Aspect Debiasing (SMAD) combining explicit multi-aspect adversarial training and implicit spatial actionness reweighting, alongside OmniDebias web data augmentation.

#### Abstract Summary

AR models often rely on static bg scenes and non-action object cues rather than dynamic human motion. This causes poor generalization. To address this issue, the paper proposes a joint framework tackling bias from both algorithmic and dataset perspectives. 

- Algorithmic perspective: it introduces Spatial-aware Multi-Aspect Debiasing (SMAD), which incorporates explicit multi-aspect adversarial training alongside an implicit spatial actionness reweighting module to force representations to be invariant to non-action visual attributes. 
- Dataset perspective: it proposes OmniDebias, a strategy to selectively leverage web data to neutralize intrinsic dataset biases with high data efficiency. 

The authors establish re-distributed evaluation splits on existing benchmarks as well as a new OOD evaluation dataset called 'Action with RAre Scene' (ARAS) to systematically evaluate scene and appearance debiasing.

#### Specific Debiasing Techniques

The paper implements a multi-pronged strategy to disentangle dynamic action signals from static background shortcuts:

* **Explicit Multi-Aspect Adversarial Training:** Auxiliary classification heads are trained to predict non-action attributes such as background scene category. Adversarial learning via gradient reversal layers forces the primary feature extractor to remove scene-discriminative information, producing representations that are invariant to static background context.
* **Implicit Spatial Actionness Reweighting:** A spatial attention mechanism estimates local actionness scores across feature maps to highlight regions containing active human movement. The module reweights spatial feature maps, suppressing background clutter and prioritizing visual regions directly involved in human motion and interactions.
* **OmniDebias Web-Data Selection:** To address dataset-level co-occurrence bias where specific actions only appear in standard settings, OmniDebias mines web video data. A teacher filter automatically extracts supplementary training clips containing rare action-scene combinations, balancing the dataset distribution without requiring manual frame-level annotation.
* **OOD Benchmark Protocol:** The paper introduces the Action with RAre Scene (ARAS) dataset (104 classes, 1,038 clips) containing human actions occurring in atypical settings (such as playing basketball in a bedroom). Combined with re-distributed scene splits of Kinetics-400, this setup provides a benchmark for evaluating model invariance to static appearance shortcuts.




---
---

## References
 
Choi, J., Gao, C., Messou, J. C. E., & Huang, J.-B. (2019). Why Can't I Dance in the Mall? Learning to Mitigate Scene Bias in Action Recognition. *ArXiv, abs/1912.05534*. https://doi.org/10.48550/arxiv.1912.05534
 
Duan, H., Zhao, Y., Chen, K., Xiong, Y., & Lin, D. (2022). Mitigating Representation Bias in Action Recognition: Algorithms and Benchmarks. 557-575. https://doi.org/10.48550/arxiv.2209.09393
 
Fioresi, J., Dave, I., & Shah, M. (2025). ALBAR: Adversarial Learning approach to mitigate Biases in Action Recognition. *ArXiv, abs/2502.00156*. https://doi.org/10.48550/arxiv.2502.00156
 
Li, H., Liu, Y., Zhang, H., & Li, B. A. (2022). Mitigating and Evaluating Static Bias of Action Representations in the Background and the Foreground. *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*, 19854-19866. https://doi.org/10.1109/iccv51070.2023.01823
 
Li, Y., & Vasconcelos, N. (2018). RESOUND: Towards Action Recognition Without Representation Bias. 520-535. https://doi.org/10.1007/978-3-030-01231-1_32
 
Rastegar, S., Doughty, H., & Snoek, C. G. M. (2024). Background no more: Action recognition across domains by causal interventions. *Comput. Vis. Image Underst., 242*, 103975. https://doi.org/10.1016/j.cviu.2024.103975
 
Sun, P., Wu, B., Li, X., Li, W., Duan, L., & Gan, C. (2021). Counterfactual Debiasing Inference for Compositional Action Recognition. *Proceedings of the 29th ACM International Conference on Multimedia*. https://doi.org/10.1145/3474085.3475472
 
