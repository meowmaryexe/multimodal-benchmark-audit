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
- Yes/no results are less reliable due to small sample size

---

## Interpretation (tentative)
- Visual dependence appears strongly task-dependent
- Retrieval behavior is tightly coupled to the image
- Compositional performance may rely more on language patterns than expected
- Aggregate benchmark scores may hide this structure

---

## Open Questions
- Why is compositional reasoning less sensitive to removing the image?
- Is the model actually reasoning, or relying on dataset-specific patterns?
- Would incorrect images (image swap) actively hurt performance?
- Does this pattern hold beyond ChartQA?

---

## Relation to Prior Work
- Language priors in VQA (models answering without images)
- Multimodal ablation / robustness studies
- ChartQA and structured reasoning benchmarks

---

## Differentiation
- Focus on structured chart reasoning (not general VQA)
- Explicit decomposition of task types
- Direct comparison of image vs no-image conditions
- Cross-scale validation (2B → 7B)

---

## Risks
- May be specific to ChartQA
- Only one model family tested
- Could be interpreted as a known limitation without stronger framing

---

## Next Steps
- Decide whether to run image-swap experiment
- Improve result presentation (figure + table)
- Start writing while results are fresh

---

## Status
Core experiments complete; pattern appears stable but not yet stress-tested