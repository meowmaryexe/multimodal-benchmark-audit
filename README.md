# When Do Models Look? Understanding Visual Reliance in Chart Reasoning

## Overview

This project studies how vision-language models (VLMs) use visual input when answering chart-based questions.

Multimodal benchmarks such as ChartQA are often used as evidence of visual reasoning ability. However, it is unclear whether models truly rely on visual information or partially exploit learned textual patterns.

This work evaluates **when visual input actually matters**, and how that dependence varies across different types of questions.

---

## Core Idea

We evaluate models under two conditions:

- **With image** — standard multimodal input  
- **Without image** — text-only input (image removed)

Performance is analyzed by question type:

- **Lookup** — direct value retrieval from the chart  
- **Compositional** — multi-step reasoning (difference, ratio, average, etc.)  
- **Yes/No** — binary comparisons  

We then further decompose compositional questions into finer subtypes to test whether performance varies across reasoning operations.

---

## Main Findings

### 1. Visual reliance is task-dependent

- Lookup questions depend heavily on visual input  
- Compositional questions show weaker dependence  

This pattern is consistent across model scale (Qwen2-VL 2B → 7B).

---

### 2. Compositional reasoning is uniformly limited

Full subtype analysis shows that compositional performance is relatively consistent across operations:

- All compositional subtypes benefit significantly from visual input  
- Performance remains moderate across operations (~0.32–0.44 with image)  
- Large drops occur without image input across all subtypes  
- Yes/no questions show much weaker dependence on visual input  

This suggests that compositional reasoning is not highly subtype-dependent, but is uniformly constrained across different operations.

---

## Results (ChartQA)

### Qwen2-VL-2B (full test set)

| Condition | Lookup | Compositional |
|----------|--------|---------------|
| With Image | 0.740 | 0.375 |
| No Image   | 0.033 | 0.082 |

---

### Qwen2-VL-7B (first 500 samples)

| Condition | Lookup | Compositional |
|----------|--------|---------------|
| With Image | 0.731 | 0.491 |
| No Image   | 0.041 | 0.202 |

---

### Accuracy Drop (Image → No Image)

| Model | Lookup Drop | Compositional Drop |
|------|------------|--------------------|
| 2B   | −0.707     | −0.293             |
| 7B   | −0.690     | −0.289             |

---

## Interpretation

These results suggest that benchmark performance may overstate multimodal reasoning:

- Visual grounding is critical for direct retrieval  
- Compositional reasoning improves with visual input but remains uniformly limited across operations  
- Yes/no questions can often be answered without strong visual grounding  
- Aggregate accuracy masks these structural differences  

The consistency across model scale indicates this is a **structural property of model behavior**, not a small-model artifact.

---

## Repository Structure

- `scripts/` — evaluation pipelines and analysis scripts  
- `results/` — experiment outputs and summaries  
- `notes/` — logs and research direction  

---

## Status

- Core counterfactual experiments complete (2B, 7B)  
- Full subtype analysis complete (Qwen2-VL 2B)  
- Cross-model validation in progress  