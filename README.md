# When Do Models Look? Understanding Visual Reliance in Chart Reasoning

## Overview

This project studies how vision-language models (VLMs) use visual input when answering chart-based questions.

Multimodal benchmarks such as ChartQA are widely used as evidence of visual reasoning ability. However, aggregate accuracy does not distinguish between visually grounded behavior and reliance on non-visual patterns.

This work evaluates **when visual input actually matters**, and shows that benchmark performance reflects a mixture of distinct behaviors rather than a single notion of “visual reasoning.”

---

## Core Idea

We evaluate models under two conditions:

- **With image** — standard multimodal input  
- **Without image** — text-only input (image removed)

Performance is analyzed by question type:

- **Lookup** — direct value retrieval from the chart  
- **Compositional** — multi-step reasoning (difference, ratio, average, etc.)  
- **Yes/No** — binary comparisons  

We further decompose compositional questions into subtypes to analyze whether reasoning performance varies across operations.

---

## Main Findings

### 1. Visual reliance is strongly task-dependent

- Lookup questions depend heavily on visual input  
- Compositional questions show weaker dependence  
- Yes/no questions exhibit the weakest and least consistent dependence  

This pattern holds across model scale (Qwen2-VL 2B → 7B) and across model families (Qwen, InternVL).

---

### 2. Compositional reasoning remains limited

- Visual input improves compositional performance  
- However, accuracy remains moderate across operations (~0.32–0.44)  
- Large drops occur without image input across all compositional subtypes  

This suggests that visual input is necessary but not sufficient for strong reasoning performance.

---

### 3. Benchmark accuracy conflates distinct capabilities

ChartQA performance reflects a mixture of:

- visually grounded retrieval  
- partially visual compositional reasoning  
- weakly visual or non-visual binary judgment  

As a result, aggregate benchmark scores do not cleanly measure visual grounding.

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

### InternVL2-4B (first 500 samples)

| Condition | Lookup | Compositional |
|----------|--------|---------------|
| With Image | 0.628 | 0.400 |
| No Image   | 0.029 | 0.124 |

---

### Accuracy Drop (Image → No Image)

| Model | Lookup Drop | Compositional Drop |
|------|------------|--------------------|
| Qwen2B | −0.707 | −0.293 |
| Qwen7B | −0.690 | −0.289 |
| InternVL2-4B | −0.599 | −0.276 |

---

## Interpretation

These results show that benchmark performance may overstate multimodal reasoning ability:

- Visual grounding is critical for retrieval tasks  
- Compositional reasoning benefits from visual input but remains limited  
- Yes/no questions are often partially answerable without strong visual grounding  

The consistency across model scale and model family suggests this is a **structural property of current VLM behavior**, not a model-specific artifact.

---

## Repository Structure

- `scripts/` — evaluation pipelines and analysis scripts  
- `results/` — experiment outputs and summaries  
- `notes/` — experiment logs and research direction  

---

## Status

- Image vs no-image evaluation complete (Qwen2-VL 2B full, 7B subset)  
- Subtype analysis complete (Qwen2-VL 2B)  
- Cross-model validation complete (InternVL2-4B)  
- Statistical validation (confidence intervals) complete  
- Final figure and analysis ready for submission  