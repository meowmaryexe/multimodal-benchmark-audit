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
- Initial subtype analysis suggests compositional reasoning is not uniform:
  - difference questions show stronger visual dependence
  - average and ratio questions remain weak even with image input
  - yes/no results are less reliable due to sample size and possible language priors

---

## Interpretation (tentative)
- Visual dependence appears strongly task-dependent
- Retrieval behavior is highly sensitive to image removal
- Compositional reasoning contains multiple failure modes rather than one broad category
- Aggregate benchmark scores may hide these differences

---

## Open Questions
- Why do average and ratio questions remain weak even with visual input?
- Which compositional operations benefit most from image access?
- Are yes/no questions partially answerable from language priors?
- Does this subtype structure hold on the full dataset?

---

## Risks
- May be specific to ChartQA
- Only one model family tested
- Subtype labels are heuristic and need careful framing
- Could be interpreted as known benchmark shortcut behavior without clear contribution

---

## Next Steps
- Run full subtype analysis with checkpointing
- Use subtype breakdown to build a stronger table/figure
- Decide whether image-swap belongs as a small robustness note or should be left out

---

## Status
Core image/no-image experiments are complete; 500-sample subtype pilot shows promising structure. Full subtype run pending.