# physio_g — Medical VLM for Radiology Image QA

Fine-tuning **Qwen2-VL-7B-Instruct** on radiology image question answering using LoRA, Chain-of-Thought supervision, cross-dataset training, and two-stage output alignment.


## Demo

**[▶ Watch Demo on Loom](https://www.loom.com/share/e4a159f73af24836942f85916a491280)**

> Upload a radiology image and ask a clinical question. The model returns a concise answer based on two-stage LoRA fine-tuning (CoT reasoning + output alignment).

**Demo cases shown:**
| Image Type | Question | Answer |
|------------|----------|--------|
| Chest X-ray | Is there any intraparenchymal abnormalities in the lung fields? | no |
| Abdominal X-ray | Is there evidence of small bowel obstruction on this image? | yes |
| Chest X-ray | Is this a CT image? | no |
| Brain CT | What is the plane of this image? | axial |
| Chest CT | What is the location of the cavitary lesion? | right upper lobe |
| Brain MRI | What lobe of the brain is the lesion located in? | right frontal lobe |

## Overview

This project systematically evaluates training strategies for medical VQA on VQA-RAD and SLAKE. The core research question is: **can CoT supervision improve reasoning quality without sacrificing exact match performance?**

The answer is yes — through two-stage training (G6), which combines CoT reasoning from Stage 1 with output format alignment in Stage 2, achieving 49.7% overall exact match and outperforming both the plain SFT baseline (G2: 49.0%) and the CoT-only model (G3: 32.6%).

---

## Experimental Groups

| Group | Description | Train Data | LoRA r | Epochs |
|-------|-------------|-----------|--------|--------|
| G1 | Zero-shot baseline | — | — | — |
| G2 | LoRA SFT | VQA-RAD (1,793) | 32 | 3 |
| G3 | CoT-LoRA | VQA-RAD (1,793) | 64 | 3 |
| G4 | CoT-LoRA ablation | VQA-RAD (1,793) | 16 | 3 |
| G5 | LoRA SFT + CoT merged | VQA-RAD + SLAKE (6,712) | 32 | 3 |
| G5b | LoRA SFT merged | VQA-RAD + SLAKE (6,712) | 32 | 3 |
| G5c | LoRA SFT merged | VQA-RAD + SLAKE (6,712) | 32 | 1 |
| G6 | **Two-stage: CoT → SFT** | VQA-RAD (1,793) | 64→16 | 3+1 |

---

## Results

### VQA-RAD Test Set (n=451)

| Group | Overall Exact | YN Exact | Open Exact | Contains |
|-------|:---:|:---:|:---:|:---:|
| G1 Zero-shot | 1.8% | 3.2% | 0.0% | 54.5% |
| G2 LoRA r=32 | 49.0% | 73.7% | 18.0% | 57.4% |
| G3 CoT-LoRA r=64 | 32.6% | 58.6% | 0.0% | 53.7% |
| G4 CoT-LoRA r=16 | 32.4% | 58.2% | 0.0% | 53.4% |
| G5 LoRA merged (CoT) | 32.6% | 58.2% | 0.5% | 42.8% |
| G5b LoRA merged 3ep | 12.0% | 19.5% | 2.5% | 62.5% |
| G5c LoRA merged 1ep | 24.8% | 41.8% | 3.5% | 61.9% |
| **G6 Two-stage ★** | **49.7%** | **74.5%** | **18.5%** | 55.4% |

### SLAKE English Test Set (n=1,061) — Cross-Dataset Generalization

| Group | Overall Exact | CLOSED Exact | Open Exact | Contains |
|-------|:---:|:---:|:---:|:---:|
| G2 LoRA r=32 | **37.3%** | **67.1%** | **18.1%** | 46.5% |
| G5 LoRA merged (CoT) | 34.7% | 68.8% | 12.7% | **86.6%** |
| G5c LoRA merged 1ep | 23.9% | 53.1% | 5.1% | 84.4% |

---

## Key Findings

**1. Two-stage training resolves the CoT-accuracy tradeoff.**
G6 achieves 49.7% exact match — outperforming both the SFT baseline (G2: 49.0%) and the CoT model (G3: 32.6%). Stage 1 instills reasoning ability via CoT supervision; Stage 2 aligns output format with concise SFT answers at a 10x lower learning rate to preserve reasoning without catastrophic forgetting.

**2. CoT supervision alone hurts exact match due to verbose output.**
G3/G4 achieve lower training loss than G2 (0.529 vs 1.067) but score lower on exact match because the model generates full reasoning chains instead of short answers. Contains match (53–54%) remains competitive, confirming the model knows the answer but cannot express it concisely.

**3. Cross-dataset merging is sensitive to dataset balance.**
G5b/G5c train on 6,712 examples (73% SLAKE, 27% VQA-RAD) but underperform G2 on VQA-RAD. The SLAKE-dominated training shifts the output distribution toward SLAKE's verbose answer style. G5b (3 epochs) overfits severely (loss 0.21, VQA-RAD exact 12%); G5c (1 epoch) partially recovers (24.8%).

**4. Merged models show strong cross-dataset contains match.**
G5 and G5c achieve 84–87% contains match on SLAKE vs G2's 46.5%, suggesting merged models have internalized cross-domain knowledge but need output format alignment (as in G6) to convert that into exact match gains.

**5. Data format consistency matters more than data volume.**
Mixing datasets with inconsistent annotation styles can hurt performance even when the domain is similar. A balanced merge with format-aligned labels is a promising direction for future work.

---

## Two-Stage Training Design
```
Stage 1 (G3): CoT-LoRA fine-tuning
  - Data: VQA-RAD + GPT-4o-mini CoT rationales
  - LoRA r=64, lr=2e-4, 3 epochs
  - Goal: learn to reason through radiology questions

        ↓ merge_and_unload()

Stage 2 (G6): Output format alignment
  - Data: VQA-RAD (short answers only)
  - New LoRA r=16 on [q_proj, v_proj, o_proj], lr=2e-5, 1 epoch
  - Goal: produce concise answers while retaining Stage 1 reasoning
```

The 10x lower learning rate in Stage 2 prevents catastrophic forgetting of the CoT reasoning capability learned in Stage 1.

---

## Setup
```bash
pip install transformers accelerate peft trl einops qwen-vl-utils pillow datasets
```

**Hardware:** NVIDIA H100 80GB (RunPod On-Demand)
**Base model:** `Qwen/Qwen2-VL-7B-Instruct`
**Datasets:**
- [`flaviagiammarino/vqa-rad`](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) — 2,244 QA pairs, 315 radiology images
- [`BoKelvin/SLAKE`](https://huggingface.co/datasets/BoKelvin/SLAKE) — 14,028 QA pairs, medical images
- CoT rationales generated via GPT-4o-mini (~$0.37 for 1,793 examples)

---

## Project Structure
```
physio_g/
├── data/
│   ├── train.json
│   ├── test.json
│   ├── train_cot.json
│   ├── train_combined.json
│   └── train_combined_sft.json
├── train/
│   ├── group2_lora_sft/
│   ├── group3_cot_lora_r64/
│   ├── group4_cot_lora_r16/
│   ├── group5_combined/
│   ├── group5b_combined_sft/
│   ├── group5c_combined_sft_e1/
│   └── group6_twostage/
├── eval/
│   ├── summary_metrics.json
│   ├── error_analysis.json
│   └── error_analysis_g5.json
└── notebooks/
```

---

## Training Loss Summary

| Group | Final Loss | Notes |
|-------|-----------|-------|
| G2 | 1.067 | Direct SFT |
| G3 | 0.529 | CoT SFT |
| G4 | 0.529 | CoT SFT, smaller r |
| G5 | 0.516 | Merged, CoT |
| G5b | 0.208 | Merged, overfit |
| G5c | 0.687 | Merged, 1 epoch |
| G6 | ~0.97 | Stage 2 only |

---

## Limitations

- Exact match underestimates CoT model performance (G3/G4/G5); contains match is more informative for these groups
- Cross-dataset generalization for G6 not yet evaluated on SLAKE
- Models evaluated on English questions only
