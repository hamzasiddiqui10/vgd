# Video Generalization Dataset (VGD) 🎬
**Ecker Lab Research Project** | **Start Date:** August 2026

## Project Overview
State-of-the-art video action recognition models frequently suffer from **object appearance bias** (visual shortcut learning). Instead of learning complex temporal action dynamics, models often "cheat" by correlating static background objects with specific action classes (e.g., predicting "drinking" just because a cup is in the frame).

The **Video Generalization Dataset (VGD)** project aims to systematically quantify and expose this object bias. Using the Something-Something v2 (SSv2) dataset as a baseline, we curated a custom, controlled, multi-camera dataset featuring intentional distractor objects. Through a series of rigorous probing experiments across various foundation models (DINOv2, V-JEPA2, and VLMs), we evaluate how well these models generalize to out-of-domain data without relying on spatial shortcuts.

## Tech Stack & Tools
* **Vision Models:** V-JEPA2 (Base & Giant), DINOv2, SAM3 (Segment Anything 3), GLM-5.3-Flash (320B VLM)
* **Data & Video Processing:** PyAV, Decord, HDF5 (h5py), OpenCV
* **Infrastructure & Training:** PyTorch, Slurm (HPC), WandB, OpenAI API Wrapper

---

## Experiments & Analysis

Below is the directory of experiments conducted during this project. Each section links to a dedicated folder containing the codebase, specific implementation details, and granular results.

### 1. Center Frame Object Bias Probing (DINOv2 & SAM3)
[➡️ View Experiment Details & Code](./experiment_ssv2_object_occlusion/)

An investigation into spatial object shortcuts using single center frames from the SSv2 dataset. We utilized a frozen DINOv2 backbone and built an automated object occlusion pipeline using SAM3 to compare model performance on uncensored vs. object-masked frames. 

### 2. Object Bias Probing (V-JEPA2-G Pretrained)
[➡️ View Experiment Details & Code](./vjepa_evaluation_on_ssv2)

Evaluating the massive self-supervised V-JEPA2-G model against our custom 3-camera distractor dataset. This experiment involved instance-level probability analysis to determine if the model falls for intentional visual distractors or successfully prioritizes temporal dynamics.

### 3. Temporal Context Probing (16-Frame DINOv2)
[➡️ View Experiment Details & Code](./apply_dinov2_16frames_attn_probe)

Testing if expanding input context mitigates appearance bias. We extracted dense features from 16 uniform frames per video using DINOv2 and trained a cross-attention probe to aggregate temporal data, comparing its robustness against the single-frame baseline.

### 4. V-JEPA2-Base Scaling Experiments & Custom Probes
[➡️ View Experiment Details & Code](./experiment_vjepa_attn_probe_trainings)

A large-scale training pipeline deployed on Slurm HPC clusters using HDF5 feature sharding to prevent I/O bottlenecks. This experiment maps how attentive probe performance scales across varying fractions of the training dataset (5% to 100%) using the highly efficient V-JEPA2-Base encoder.

### 5. Vision-Language Model (VLM) Zero-Shot Bias Probing
[➡️ View Experiment Details & Code](./experiment_vlm_eval_with_superset_classes)

Pushing the boundaries of object bias evaluation by testing the 320B-parameter `glm-5.3-flash` VLM. We engineered a cloud-API pipeline with strict JSON schema enforcement to parse zero-shot action probabilities over condensed semantic supersets.

---
*Created by [Hamza Ahmed Siddiqui](https://github.com/hamzasiddiqui10)*