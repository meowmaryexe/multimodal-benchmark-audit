# Direction filter

## Selected Direction
Counterfactual modality reliance audits for multimodal benchmarks

## Core Question
To what extent do vision-language models rely on visual input when answering chart-based questions?

## Working Hypothesis
Visual input is critical for direct value retrieval tasks, but compositional reasoning performance is substantially weaker and less dependent on the presence of visual input.

## Motivation
Multimodal benchmarks such as ChartQA are often used as evidence of visual reasoning ability. However, aggregate accuracy does not distinguish between:
- retrieving values from the chart
- performing multi-step reasoning over those values

This raises the question of whether strong performance reflects genuine visual grounding or partial reliance on non-visual heuristics.

## Approach
We evaluate a vision-language model under two conditions:
- standard setting (image + question)
- counterfactual setting (question only, no image)

We further decompose questions into:
- lookup (direct value extraction)
- compositional (difference, ratio, average, etc.)
- yes/no (binary comparisons)

This allows us to measure how visual dependence varies across task types.

## Empirical Signal (in progress)
- Significant performance gap between lookup and compositional questions in the image condition
- Early evidence that:
  - lookup performance drops sharply without image input
  - compositional performance degrades less

## Key Insight (target)
Visual grounding is not uniform across tasks: models rely heavily on images for retrieval, but compositional reasoning remains limited and partially independent of visual input.

## Relation to Prior Work
- Language priors in VQA (models answering without images)
- Multimodal robustness and ablation studies
- ChartQA and structured reasoning benchmarks

## Differentiation
- Focus on structured chart reasoning rather than general VQA
- Explicit task decomposition (lookup vs compositional)
- Direct comparison of visual vs non-visual conditions

## Risks
- Findings may be dataset-specific (ChartQA only)
- Single-model evidence may limit generality
- Results could be interpreted as a known limitation without deeper analysis

## Strengthening Plan
- Full dataset evaluation (image and no-image)
- Add a second strong model for validation
- Report absolute and relative performance drops by category
- Present results with a single clear comparison figure

## Significance
If validated, these results suggest that benchmark performance may overestimate true visual reasoning ability, particularly for compositional tasks.

## Verdict
PURSUE — initial results indicate a clear and interpretable failure mode worth formalizing