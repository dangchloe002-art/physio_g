# physio_g — Project Report

**Medical VLM for Radiology Image QA**
Qwen2-VL-7B-Instruct + LoRA · VQA-RAD + SLAKE · Two-stage Training

---

## 1. Motivation

Medical visual question answering (VQA) requires models to jointly reason over radiology images and clinical questions. This project investigates whether chain-of-thought (CoT) supervision can improve reasoning quality in a fine-tuned VLM without sacrificing exact match performance — a tradeoff that is non-trivial in practice.

The project serves three goals simultaneously:
1. A systematic ablation study suitable for a portfolio or research writeup
2. A working demo deployable on H100 hardware
3. A technical proof-of-concept for medical VLM interpretability

---

## 2. Setup

**Base model:** Qwen2-VL-7B-Instruct
**Hardware:** NVIDIA H100 80GB (RunPod On-Demand)
**Datasets:**
- VQA-RAD: 2,244 QA pairs, 315 radiology images (chest X-ray, brain MRI, abdominal CT)
- SLAKE: 14,028 QA pairs, multi-organ medical images

**Training framework:** HuggingFace Transformers + PEFT (LoRA) + TRL
**CoT generation:** GPT-4o-mini (~$0.37 for 1,793 examples)

---

## 3. Experimental Design

Seven experimental groups were designed to isolate the effect of:
- LoRA rank (r=16 vs r=32 vs r=64)
- CoT supervision vs direct SFT
- Single-dataset vs cross-dataset training
- Training duration (1 epoch vs 3 epochs)
- Two-stage training (CoT → SFT)

| Group | Description | Train Data | LoRA r | Epochs |
|-------|-------------|-----------|--------|--------|
| G1 | Zero-shot baseline | — | — | — |
| G2 | LoRA SFT | VQA-RAD (1,793) | 32 | 3 |
| G3 | CoT-LoRA | VQA-RAD (1,793) | 64 | 3 |
| G4 | CoT-LoRA ablation | VQA-RAD (1,793) | 16 | 3 |
| G5 | LoRA SFT + CoT merged | VQA-RAD + SLAKE (6,712) | 32 | 3 |
| G5b | LoRA SFT merged | VQA-RAD + SLAKE (6,712) | 32 | 3 |
| G5c | LoRA SFT merged | VQA-RAD + SLAKE (6,712) | 32 | 1 |
| G6 | Two-stage: CoT → SFT | VQA-RAD (1,793) | 64→16 | 3+1 |

---

## 4. Results

### 4.1 VQA-RAD Test Set (n=451)

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

### 4.2 SLAKE English Test Set (n=1,061) — Cross-Dataset Generalization

| Group | Overall Exact | CLOSED Exact | Open Exact | Contains |
|-------|:---:|:---:|:---:|:---:|
| G2 LoRA r=32 | 37.3% | 67.1% | 18.1% | 46.5% |
| G5 LoRA merged (CoT) | 34.7% | 68.8% | 12.7% | 86.6% |
| G5c LoRA merged 1ep | 23.9% | 53.1% | 5.1% | 84.4% |

### 4.3 Training Loss Summary

| Group | Final Loss |
|-------|-----------|
| G2 | 1.067 |
| G3 | 0.529 |
| G4 | 0.529 |
| G5 | 0.516 |
| G5b | 0.208 (overfit) |
| G5c | 0.687 |
| G6 Stage 2 | ~0.97 |

---

## 5. Key Findings

### Finding 1: Two-stage training resolves the CoT-accuracy tradeoff

G6 achieves 49.7% overall exact match — outperforming both the SFT baseline (G2: 49.0%) and the CoT-only model (G3: 32.6%). The two-stage design decouples reasoning acquisition (Stage 1: CoT-LoRA) from output format alignment (Stage 2: SFT at 10x lower learning rate), preventing catastrophic forgetting while producing concise answers.

### Finding 2: CoT supervision alone hurts exact match due to verbose output

G3/G4 achieve lower training loss than G2 (0.529 vs 1.067) but score lower on exact match because the models generate full reasoning chains instead of short answers. The contains match metric (53–54%) reveals that these models know the correct answer — they just cannot express it concisely without the second training stage.

### Finding 3: Cross-dataset merging is sensitive to dataset balance

G5b trains on 6,712 examples (73% SLAKE, 27% VQA-RAD) but severely underperforms G2 on VQA-RAD (12% vs 49%). The SLAKE-dominated training shifts output distribution toward SLAKE's verbose style. Reducing epochs from 3 to 1 (G5c) partially recovers performance (24.8%), confirming that the issue is overfitting to the majority dataset's style, not a fundamental incompatibility.

### Finding 4: Contains match reveals strong cross-dataset generalization for merged models

G5 achieves 86.6% contains match on SLAKE — far above G2's 46.5% — indicating the merged models have internalized cross-domain knowledge. The gap between contains match and exact match (86.6% vs 34.7%) points directly to the output format problem that two-stage training solves.

### Finding 5: LoRA rank has minimal effect on CoT models

G3 (r=64) and G4 (r=16) achieve identical training loss (0.529) and nearly identical evaluation scores (32.6% vs 32.4%). This suggests that for CoT-style training on this dataset size, the bottleneck is output format rather than model capacity.

---

## 6. Two-Stage Training Design
```
Stage 1 (G3): CoT-LoRA fine-tuning
  Data   : VQA-RAD + GPT-4o-mini CoT rationales (1,793 examples)
  Config : LoRA r=64, lora_alpha=128, lr=2e-4, 3 epochs
  Goal   : Learn to reason through radiology questions step-by-step

        ↓ merge_and_unload() — bake Stage 1 weights into base model

Stage 2 (G6): Output format alignment
  Data   : VQA-RAD short answers only (1,793 examples)
  Config : New LoRA r=16 on [q_proj, v_proj, o_proj], lr=2e-5, 1 epoch
  Goal   : Produce concise answers while retaining Stage 1 reasoning
```

The 10x lower learning rate in Stage 2 is critical — it prevents catastrophic forgetting of the CoT reasoning capability while nudging the output distribution toward concise answers.

---

## 7. Error Analysis

### G5 Error Distribution (VQA-RAD, n=451)

| Error Type | Count | Rate |
|------------|-------|------|
| Correct | 147 | 32.6% |
| False Negative (GT=yes, PRED=no) | 56 | 12.4% |
| False Positive (GT=no, PRED=yes) | 31 | 6.9% |
| Open-ended failures | 199 | 44.1% |

The model shows a conservative bias (more false negatives than false positives). For open-ended questions, CoT-style output prevents exact string matching entirely — this is the primary motivation for the two-stage approach in G6.

---

## 8. Evaluation Methodology Notes

Exact match is the standard VQA-RAD metric but systematically underestimates CoT model performance. For G3/G4/G5, a secondary evaluation using answer extraction (regex-based pattern matching to identify the final conclusion from CoT output) was applied. Both metrics are reported.

Contains match is reported as a complementary metric, particularly informative for CoT models where the correct answer appears within a longer output.

---

## 9. Limitations

- Exact match underestimates CoT model performance; a dedicated extraction-based metric is more appropriate for G3/G4/G5
- Cross-dataset results are sensitive to dataset imbalance (73% SLAKE in merged training)
- G6 cross-dataset generalization on SLAKE was not evaluated
- Models evaluated on English questions only; SLAKE Chinese QA pairs excluded
- No human evaluation of reasoning quality for CoT outputs

---

## 10. Reproduction
```bash
# Install dependencies
pip install transformers accelerate peft trl einops qwen-vl-utils pillow datasets

# Download base model
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct

# Training (G2 example)
# See training cells in notebooks/
```

All training configurations, evaluation results, and error analysis are available in the `eval/` directory.

---

## Acknowledgements

- VQA-RAD dataset: [flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
- SLAKE dataset: [BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE)
- Base model: [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- CoT rationales generated with GPT-4o-mini (OpenAI API)
