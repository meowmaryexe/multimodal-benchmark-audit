# When Do Models Look? Task-Dependent Visual Reliance in Chart Reasoning

## Overview

This project studies how vision-language models (VLMs) use visual input when answering chart-based questions.

Multimodal benchmarks such as ChartQA are often used as evidence of visual reasoning ability. However, aggregate accuracy does not distinguish between visually grounded behavior and reliance on non-visual patterns.

This work investigates **when visual input actually matters**, using a simple counterfactual evaluation setup.

---

## Core Idea

We evaluate models under two conditions:

- **With image** — standard multimodal input  
- **Without image** — text-only input (image removed)  

Performance is analyzed by question type:

- **Lookup** — direct value retrieval  
- **Compositional** — multi-step reasoning (difference, ratio, average, etc.)  
- **Yes/No** — binary comparisons  

---

## Current Findings (in progress)

Preliminary results across multiple models show a consistent pattern:

- Removing the image causes a large drop in **lookup** performance  
- **Compositional** performance drops less, but still consistently  
- **Yes/no** questions show weaker and more variable dependence  

This pattern appears:
- across model scale (Qwen2-VL 2B → 7B)  
- across model families (Qwen, InternVL, Phi)  

---

## Key Insight

Visual reliance is **task-dependent**, not uniform.

Benchmark accuracy reflects a mixture of:
- visually grounded retrieval  
- partially visual reasoning  
- weakly visual or non-visual judgment  

This suggests that standard multimodal benchmarks may conflate distinct capabilities rather than measuring a single notion of visual reasoning.

---

## Current Experiments

Completed:
- Qwen2-VL-2B (full ChartQA)
- Qwen2-VL-7B (subset + full)
- InternVL2-4B (subset)
- Phi-3.5-vision (subset)
- Subtype analysis (Qwen2-VL-2B)
- Statistical validation (confidence intervals)

In progress:
- InternVL2 full evaluation  
- Phi-3.5 full evaluation  
- Categorization standardization across all models  
- Robustness checks (prompt variation / alternative settings)  
- Final figure construction  

---

## Repository Structure

scripts/   # evaluation pipelines and analysis scripts
results/   # experiment outputs and intermediate results
notes/     # experiment logs and research framing

---

## Status

This is an **active research project**.

The core empirical pattern is stable, but:
- additional models are being run at full scale  
- results are being standardized across pipelines  
- final figures and analysis are being consolidated  

---

## Notes

See:
- `notes/experiment_log.md` for detailed experiment history  
- `notes/idea_filter.md` for research framing  
