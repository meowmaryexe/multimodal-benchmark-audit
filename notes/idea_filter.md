# Direction filter

## Selected Direction
Counterfactual modality reliance audits for multimodal benchmarks

## Core Question
To what extent do vision-language models rely on visual input when answering chart-based questions?

## Motivation
Multimodal benchmarks such as ChartQA are often used as evidence of visual reasoning ability. However, aggregate accuracy does not distinguish between:
- retrieving values from the chart
- performing multi-step reasoning over those values
- answering questions using partial or non-visual cues

This makes it unclear whether strong performance reflects genuine visual grounding or mixed reliance on visual and non-visual signals.

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

## Current Signal (validated)

Observed across:
- Qwen2-VL 2B (full test set)
- Qwen2-VL 7B (500 samples)
- InternVL2-4B (500 samples)

Key patterns:

- Removing the image causes a large drop in lookup performance (~−0.60 to −0.70)
- Compositional performance drops less (~−0.27 to −0.29)
- Yes/no performance shows weaker and less consistent dependence on visual input

Additional validation:
- Patterns are consistent across model scale (2B → 7B)
- Patterns generalize across model families (Qwen → InternVL)
- Confidence intervals show:
  - strong separation for lookup
  - consistent but smaller separation for compositional
  - overlapping intervals for yes/no in some cases

---

## Interpretation (updated)

- Visual reliance is strongly task-dependent
- Retrieval behavior is tightly coupled to visual input
- Compositional reasoning benefits from visual input but remains limited
- Yes/no questions are partially answerable without strong visual grounding
- Aggregate benchmark accuracy conflates distinct capabilities:
  - visually grounded retrieval
  - partially visual reasoning
  - weakly visual or non-visual judgment

---

## Open Questions

- Why does visual input improve compositional reasoning without enabling high accuracy?
- What limits performance across compositional operations?
- To what extent are yes/no questions driven by language priors?
- Does this pattern generalize beyond ChartQA?

---

## Risks

- May be specific to ChartQA
- Limited number of model families tested
- Results are descriptive rather than mechanistic
- Could be interpreted as known multimodal limitations without clear domain-specific insight

---

## Next Steps

- Finalize figure and paper framing
- Position work relative to VQA language priors and multimodal ablation literature
- Emphasize benchmark interpretation contribution rather than performance improvement

---

## Status

Core experiments complete:
- Image vs no-image (Qwen2B full, Qwen7B subset, InternVL subset)
- Subtype analysis (Qwen2B full)
- Cross-model validation (InternVL)
- Statistical validation (confidence intervals)

Current focus:
- Final framing, writing, and positioning for submission