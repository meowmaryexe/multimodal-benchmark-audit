# What Do Chart Question Answering Benchmarks Measure?
## Task-Specific Visual Evidence Dependence in Vision-Language Models

## Overview

This project studies how vision-language models (VLMs) use visual evidence when answering chart-based questions.

Multimodal benchmarks such as ChartQA are often interpreted as evidence of visual reasoning ability. However, aggregate benchmark accuracy may obscure whether models depend on visual information in the same way across different task types.

This work investigates task-specific visual evidence dependence using counterfactual modality interventions.

---

## Core Question

To what extent do vision-language models rely on matched visual evidence when answering chart questions, and how does this dependence vary across task types?

---

## Approach

Models are evaluated under multiple counterfactual conditions:

- **Matched image** — standard multimodal input  
- **No image** — image removed entirely  
- **Distractor image** — unrelated chart paired with the question  

Questions are partitioned into:

- **Lookup** — direct value extraction  
- **Compositional** — multi-step numerical reasoning (difference, ratio, average, total, etc.)  
- **Yes/No** — binary comparisons and judgments  

The project analyzes:
- task-specific accuracy changes
- behavioral transitions after image removal
- dependence on matched versus generic chart-like visual input

---

## Current Findings

Across multiple VLM families and scales, a consistent qualitative pattern emerges:

- Removing the image produces the largest and most consistent degradation for **lookup** questions
- **Compositional** questions also depend substantially on visual input, but remain difficult even with images
- **Yes/no** questions show weaker and more variable dependence on visual evidence

Additional findings:
- Lookup examples overwhelmingly transition from correct → incorrect after image removal
- Distractor-image performance for lookup and compositional questions falls close to no-image levels
- The same qualitative asymmetry appears on a PlotQA-derived evaluation subset

The pattern currently holds across:
- Qwen2-VL-2B
- Qwen2-VL-7B
- InternVL2-4B
- Phi-3.5-vision-instruct

---

## Key Insight

Visual reliance is strongly task-dependent rather than uniform.

Aggregate chart QA benchmark accuracy appears to combine multiple qualitatively distinct behaviors, including:
- direct visually grounded extraction
- partially grounded compositional reasoning
- weaker or less stable multimodal dependence

This suggests that standard multimodal benchmark scores may conflate distinct capabilities rather than measuring a single notion of visual reasoning.

---

## Experiments

### Completed
- Qwen2-VL-2B (full ChartQA)
- Qwen2-VL-7B (full ChartQA)
- Phi-3.5-vision-instruct (full ChartQA)
- InternVL2-4B (500-sample evaluation)
- Prompt robustness evaluation
- Behavioral transition analysis
- Distractor-image evaluation
- Compositional subtype analysis
- Wilson confidence interval analysis
- PlotQA-derived replication subset
- Final NeurIPS submission experiments

### Ongoing
- Full-scale InternVL2-4B evaluation
- PlotQA replication for additional model families
- Stronger perturbation-based counterfactuals
- Additional multimodal benchmark validation
- Extended qualitative failure analysis
- Go ask nicely for more compute... 

---

## Repository Structure

```text
scripts/      evaluation pipelines and analysis scripts
results/      experiment outputs and processed results
figures/      consolidated paper figures and appendix figures
notes/        experiment logs, framing notes, and research directions

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

---

## Current Limitations

- Analysis is currently concentrated in chart QA
- Some evaluations remain subset-based
- Task categorization is heuristic
- Findings are behavioral/descriptive rather than mechanistic

---

## Future Directions

Potential next steps include:

- full InternVL evaluation
- broader PlotQA replication
- OCR perturbation experiments
- partial chart masking
- structured distractor generation
- mechanistic analysis of cross-modal behavior
- extension to document QA and table QA benchmarks

---

## Status

This is an active research project.

The main empirical pattern is stable across models, scales, prompts, and benchmarks tested so far, but the project is still evolving through:

- expanded evaluations
- stronger counterfactual controls
- additional benchmark validation
- deeper behavioral analysis

---

## Notes

See:

- `notes/experiment_log.md` for chronological experiment history
- `notes/idea_filter.md` for framing, failed directions, risks, and future plans