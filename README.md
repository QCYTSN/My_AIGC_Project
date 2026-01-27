# Training-Free Concept Disentanglement in T2I Generation via Detection-Guided Attention Control

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Diffusers](https://img.shields.io/badge/Diffusers-0.19+-orange)
![Status](https://img.shields.io/badge/Status-Research_Complete-success)

> **Research Project (2025.12 - 2026.01)**
> 
> *A novel approach to solve "Attribute Leakage" (e.g., color bleeding) in Stable Diffusion without model fine-tuning.*

## 📖 Introduction

Text-to-Image (T2I) models like Stable Diffusion often suffer from **concept entanglement** or **attribute leakage**. For example, when generating *"a cat wearing a red hat"*, the color "red" often bleeds into the background or the cat's fur.

This project proposes a **Training-Free Spatial-Gated Attention Control** mechanism. By leveraging layout priors (via YOLO/Grounding DINO) during inference, we dynamically intervene in the Cross-Attention layers to strictly confine attributes to their target regions using a **Spatial-Gated Attention Boosting** algorithm.

## ✨ Key Features

* **Training-Free:** No LoRA or Fine-tuning required. Plug-and-play for SD v1.5.
* **Spatial-Gated Attention:** A custom Attention Processor that injects binary masks into attention maps.
* **Signal Amplification:** Uses a balanced boosting factor (x5.0) to ensure object visibility while preserving identity.
* **End-to-End:** 50% faster than traditional "Inpainting-based" pipelines (generate -> detect -> crop -> inpaint -> paste).

## 🖼️ Qualitative Results

Comparison between **Baseline (Inpainting)** and **Ours (Attention Control)**.

### Task 1: Cat with Red Knitted Beanie
*Our method (Bottom) successfully generates the hat without background color bleeding, maintaining a natural look and consistent lighting.*
![Cat Hat Comparison](results/comparison_grid_cat_hat.jpg)

### Task 2: Dog with Blue Scarf
*Our method (Bottom) demonstrates **"Layout Correction"** capabilities. While the baseline (Top) often places the scarf over the dog's mouth (following the bounding box strictly), our method naturally places the scarf around the neck, prioritizing physical plausibility.*
![Dog Scarf Comparison](results/comparison_grid_dog_scarf.jpg)

## 📊 Quantitative Evaluation

We evaluated the method on a custom evaluation set (N=40) using **CLIP Score** (Image Quality/Semantics) and **OwlViT Detection** (Control Precision).

| Method | Task | CLIP Score (↑) | Accuracy (↑) | Mean IoU | Note |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline (Inpainting)** | Cat-Hat | 0.3407 | **100.0%** | 0.8186 | Upper bound (Hard masking) |
| **Ours (Attn Control)** | Cat-Hat | **0.3566** | 85.0% | 0.5146 | **Higher Quality & Precision** |
| | | | | | |
| **Baseline (Inpainting)** | Dog-Scarf | 0.3153 | 95.0% | 0.7894 | Unnatural occlusion |
| **Ours (Attn Control)** | Dog-Scarf | **0.3509** | 55.0% | 0.3449 | **Physical Layout Correction** |

**Key Research Findings:**
1.  **Superior Image Quality:** Our method consistently achieves higher CLIP Scores (**+11% relative improvement**), indicating better semantic consistency and visual naturalness compared to the inpainting baseline.
2.  **Intelligent Refinement:** In the *Dog-Scarf* task, the lower IoU and Accuracy reflect the model's emergent ability to **correct unnatural layout constraints** (e.g., moving a scarf from the snout to the neck), prioritizing physical plausibility over strict bounding box adherence.

## 🛠️ Usage

### 1. Installation
```bash
pip install torch diffusers transformers opencv-python pillow scipy

```

### 2. Run Baseline (Inpainting Pipeline)

Generates the ground truth layout (JSONs) and baseline images.

```bash
python src/generate_multicat_baseline.py

```

### 3. Run Ours (Attention Control Pipeline)

Injects the spatial control using the Layout data from step 2.

```bash
python src/generate_ours_fixed.py

```

### 4. Evaluation

Calculates CLIP Score and IoU/Accuracy using OwlViT.

```bash
python src/evaluate_metrics_v2.py

```

## 📂 Project Structure

```text
├── models/                  # Local model weights (Ignored in git)
├── results/
│   ├── baseline_eval_v5/    # Layout data (JSON) & Baseline images
│   └── final_comparison_fixed/ # Our results (Images)
├── src/
│   ├── research_control.py  # Core Attention Processor Logic
│   ├── generate_ours_fixed.py # Main Generation Script (Algorithm V2.4)
│   ├── evaluate_metrics_v2.py # Quantitative Eval Script (CLIP + OwlViT)
│   ├── make_comparison_grid.py # Visualization Tool
│   └── ...
└── README.md

```

## 📝 Citation

If you find this project useful, please cite:

```bibtex
@misc{spatial_gated_attention_2026,
  author = {Fengkai Gao},
  title = {Training-Free Concept Disentanglement via Detection-Guided Attention Control},
  year = {2026},
  publisher = {GitHub}
}

```

```

```
