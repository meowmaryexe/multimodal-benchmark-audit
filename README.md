# When Do Models Look? Understanding Visual Reliance in Chart Reasoning

## Overview

This project investigates whether vision-language models (VLMs) actually rely on visual input when answering chart-based questions.

While multimodal benchmarks such as ChartQA are often used as evidence of visual reasoning ability, it remains unclear whether models depend on the image itself or exploit learned textual patterns.

---

## Core Idea

We evaluate models under two conditions:

- **With image** (standard multimodal input)
- **Without image** (text-only input)

We then analyze performance across different types of questions:

- **Lookup** — direct value retrieval from the chart  
- **Compositional** — multi-step reasoning (differences, ratios, averages)  
- **Yes/No** — binary comparisons  

This allows us to measure **when visual input actually matters**.

---

## Key Question

Do models truly use visual information, or can they partially answer questions without it?

---

## Current Findings (Qwen2-VL-2B, ChartQA)

- Strong performance on **lookup** questions when images are available  
- Substantially lower performance on **compositional reasoning**  
- Large gap between retrieval and reasoning suggests different underlying mechanisms  

Full counterfactual (no-image) results are being finalized to measure differential reliance on visual input.

---

## Structure

- `scripts/` — evaluation pipelines (image / no-image)
- `results/` — experiment outputs and summaries
- `notes/` — research logs and direction tracking

---

## Status

Active research project. Experiments are being scaled and refined.