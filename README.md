# What Do Chart Question Answering Benchmarks Measure?
## Task-Specific Visual Evidence Dependence in Vision-Language Models

## Overview

This project studies how vision-language models (VLMs) use visual evidence when answering chart-based questions.

Multimodal benchmarks such as ChartQA are often interpreted as evidence of visual reasoning ability. However, aggregate benchmark accuracy may obscure whether models depend on visual evidence in the same way across different task types.

This work investigates task-specific visual evidence dependence using counterfactual modality interventions across multiple chart QA settings.

---

## Core Question

To what extent do vision-language models rely on matched visual evidence when answering chart questions, and how does this dependence vary across task types?

---

## Approach

Models are evaluated under multiple counterfactual conditions:

- **Matched image** — standard multimodal inference
- **No image** — image removed entirely
- **Distractor image** — unrelated chart paired with the question

Questions are partitioned into:

- **Lookup** — direct value extraction
- **Compositional** — multi-step numerical question answering
- **Yes/No** — binary comparisons and judgments

The project analyzes:

- task-specific accuracy degradation
- behavioral transitions after image removal
- dependence on matched versus generic chart-like visual input
- efficiency implications of selective multimodal inference

---

## Current Findings

Across multiple VLM families and scales, a consistent qualitative pattern emerges:

- Removing the image produces the largest and most consistent degradation for lookup questions
- Compositional questions also depend substantially on visual evidence, but remain difficult even with image access
- Yes/no questions show weaker and more variable dependence on visual input

Additional findings include:

- Lookup examples overwhelmingly transition from correct → incorrect after image removal
- Distractor-image performance for lookup and compositional questions falls near no-image levels
- Similar qualitative asymmetries appear on a PlotQA-derived evaluation subset
- Image-free inference substantially reduces latency relative to multimodal inference

The pattern currently holds across:

- Qwen2-VL-2B
- Qwen2-VL-7B
- Phi-3.5-vision-instruct
- InternVL2-4B

---

## Key Insight

Visual reliance is strongly task-dependent rather than uniform.

Aggregate chart QA benchmark accuracy appears to combine multiple qualitatively distinct behaviors, including:

- direct visually grounded extraction
- partially visually grounded compositional question answering
- weaker or less stable multimodal dependence

This suggests that standard multimodal benchmark scores may conflate distinct capabilities rather than measuring a single unified notion of multimodal reasoning.

The results also motivate adaptive multimodal systems that selectively invoke visual computation only when it is likely to improve answer quality.

---

## Experiments

### Completed

- Qwen2-VL-2B (full ChartQA)
- Qwen2-VL-7B (full ChartQA)
- Phi-3.5-vision-instruct (full ChartQA)
- InternVL2-4B (500-example evaluation)
- Prompt robustness evaluation
- Behavioral transition analysis
- Distractor-image evaluation
- Compositional subtype analysis
- Wilson confidence interval analysis
- PlotQA-derived replication subset
- Latency benchmarking for multimodal vs text-only inference

### Ongoing

- Full-scale InternVL2-4B evaluation
- PlotQA replication across additional model families
- Stronger perturbation-based counterfactuals
- Additional multimodal benchmark validation
- Extended qualitative failure analysis
- Exploratory routing-oriented efficiency analysis

---

## Repository Structure

```text
scripts/      evaluation pipelines and analysis scripts
results/      experiment outputs and processed metrics
figures/      consolidated figures and visualizations
notes/        experiment logs, framing notes, and research directions
```

---

## Main Experimental Directions

### Counterfactual Modality Interventions

- image removal
- distractor-image replacement
- prompt variation

### Behavioral Analysis

- correct → incorrect transition tracking
- task-specific modality dependence
- qualitative failure inspection

### Cross-Model Validation

- model scale comparisons
- model family comparisons
- benchmark replication

### Efficiency Diagnostics

- multimodal latency benchmarking
- image vs no-image inference cost analysis
- exploratory selective multimodal inference analysis

---

## Current Limitations

- Analysis is currently concentrated in chart QA
- Some evaluations remain subset-based
- Task categorization is heuristic
- Findings are behavioral rather than mechanistic

---

## Future Directions

Potential next steps include:

- broader PlotQA replication
- OCR perturbation experiments
- partial chart masking
- structured distractor generation
- adaptive routing policies
- extension to document QA and table QA benchmarks
- mechanistic analysis of cross-modal behavior

---

## Status

This is an active research project focused on multimodal evaluation and efficiency-sensitive QA analysis.

The main empirical pattern has remained qualitatively consistent across the evaluated models, prompts, and benchmark settings tested so far, while ongoing work focuses on stronger controls, broader replication, and efficiency-oriented multimodal evaluation.

---

## Notes

See:

- `notes/experiment_log.md` for chronological experiment history
- `notes/idea_filter.md` for framing notes, failed directions, risks, and future plans