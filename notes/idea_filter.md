# Direction filter

## Selected Direction
Counterfactual modality reliance audits for multimodal benchmarks

## Core Question
Do multimodal benchmarks actually require images for correct answers?

## Why this might be novel
- Many benchmarks assume multimodal reasoning
- Few standardized counterfactual audits across datasets
- Potential mismatch between reported performance and actual visual reliance

## Closest prior work
- VQA language priors
- Benchmark ablations (no-image tests)
- Multimodal robustness work

## Why reviewers might reject it
- “This is already known”
- Weak or obvious results
- Too small-scale
- No generalizable insight

## Smallest convincing experiment (Day 1–3)
- Run model on 30 ChartQA samples
- Compare:
  - normal vs no-image
- Measure accuracy drop

## Data needed
- 30 ChartQA samples
- images + questions

## Time risk
- Low for initial test
- Medium for scaling across benchmarks

## Acceptance upside
- Medium → High if strong signal found
- Especially if results generalize across benchmarks

## Verdict
PURSUE (pending dead-end test)