# When Do Models Look? Understanding Visual Reliance in Chart Reasoning

## Overview

This project studies how vision-language models (VLMs) use visual input when answering chart-based questions.

Multimodal benchmarks such as ChartQA are often used as evidence of visual reasoning ability. However, it is unclear whether models truly rely on visual information or partially exploit learned textual patterns.

This work evaluates **how much visual input actually matters**, and whether that dependence varies across different types of questions.

---

## Core Idea

We evaluate models under two conditions:

- **With image** — standard multimodal input  
- **Without image** — text-only input (image removed)

Performance is analyzed by question type:

- **Lookup** — direct value retrieval from the chart  
- **Compositional** — multi-step reasoning (differences, ratios, averages)  
- **Yes/No** — binary comparisons  

This setup isolates **when visual input is necessary vs. when models can rely on language alone**.

---

## Main Observation

Visual reliance is **not uniform across tasks**:

- Lookup questions depend heavily on visual input  
- Compositional questions show substantially weaker dependence  

This pattern is consistent across model scale (Qwen2-VL 2B → 7B).

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
- Multi-step reasoning is partially supported by language patterns  
- Aggregate accuracy masks this difference  

The consistency across model scale indicates this is a **structural property of model behavior**, not a small-model artifact.

---

## Repository Structure

- `scripts/` — evaluation pipelines (image / no-image)
- `results/` — aggregated outputs and summaries
- `notes/` — experiment logs and research direction

---

## Status

Core counterfactual experiments are complete for Qwen2-VL 2B and 7B.  
Additional robustness checks are planned.