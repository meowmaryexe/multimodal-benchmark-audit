# Direction filter

## Selected Direction
Counterfactual modality reliance audits for multimodal benchmarks

## Core Question
To what extent do vision-language models rely on visual input when answering chart-based questions?

## Motivation
Multimodal benchmarks such as ChartQA are often used as evidence of visual reasoning ability. However, aggregate accuracy does not distinguish between:
- retrieving values from the chart
- performing multi-step reasoning over those values

This makes it unclear whether strong performance reflects genuine visual grounding or partial reliance on non-visual patterns.

---

## Approach
Evaluate models under two conditions:
- image + question (standard)
- question only (image removed)

Decompose questions into:
- lookup (direct value extraction)
- compositional (difference, ratio, average, etc.)
- yes/no (binary comparisons)

Compare accuracy across conditions and categories.

---

## Current Signal
Observed across Qwen2-VL 2B (full set) and 7B (subset):

- Removing the image causes a large drop in lookup performance (~−0.70)
- Compositional performance drops less (~−0.29)
- This gap is consistent across model scale
- Full subtype analysis shows:
  - all compositional subtypes benefit significantly from visual input
  - performance remains uniformly limited across operations (~0.32–0.44 with image)
  - large drops without image across all compositional types
- Yes/no questions show relatively small performance drop, suggesting partial reliance on language patterns

---

## Interpretation (updated)
- Visual dependence is strongly task-dependent (lookup vs compositional vs yes/no)
- Retrieval behavior is tightly coupled to visual input
- Compositional reasoning is not highly subtype-dependent, but is uniformly constrained across operations
- Visual input helps compositional reasoning, but does not close the performance gap with lookup tasks
- Aggregate benchmark scores mask these structural differences

---

## Open Questions
- Why does visual input improve compositional reasoning without enabling high accuracy?
- What limits performance across all compositional operations?
- To what extent are yes/no questions answerable via language priors?
- Does this pattern generalize beyond the Qwen model family?

---

## Risks
- May be specific to ChartQA
- Only one model family tested
- Results are descriptive without a mechanistic explanation
- Could be interpreted as known multimodal limitations without stronger generalization evidence

---

## Next Steps
- Add second model (different architecture) to test generalization of observed pattern
- Compare image vs no-image behavior across model families
- Build final table/figure highlighting task-dependent visual reliance

---

## Status
Core experiments complete:
- Image vs no-image (2B full, 7B subset)
- Full subtype analysis (2B)

Next step: cross-model validation.