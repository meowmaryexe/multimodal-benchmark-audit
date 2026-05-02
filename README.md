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

We then further decompose compositional questions into finer subtypes to analyze failure modes.

---

## Main Findings

### 1. Visual reliance is task-dependent

- Lookup questions depend heavily on visual input  
- Compositional questions show weaker dependence  

This pattern is consistent across model scale (Qwen2-VL 2B → 7B).

---

### 2. Compositional reasoning is not uniform

Subtype analysis reveals that compositional performance varies significantly by operation:

- **Difference** questions show meaningful visual dependence  
- **Average** and **ratio** questions remain weak even with image input  
- **Sum/compare** questions fall in between  
- **Yes/no** questions show minimal dependence on visual input  

This indicates that compositional reasoning is not a single capability, but a collection of distinct behaviors.

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
- Compositional reasoning contains multiple distinct failure modes  
- Some operations (e.g., averages, ratios) are poorly supported even with visual input  
- Aggregate accuracy masks these differences  

The consistency across model scale indicates this is a **structural property of model behavior**, not a small-model artifact.

---

## Repository Structure

- `scripts/` — evaluation pipelines and analysis scripts  
- `results/` — experiment outputs and summaries  
- `notes/` — logs and research direction  

---

## Status

- Core counterfactual experiments complete (2B, 7B)  
- Subtype analysis validated on 500-sample pilot  
- Full subtype analysis in progress  