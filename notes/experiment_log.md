# Experiment log

---

## Date: April 13
## Direction:
Counterfactual modality reliance audits

## Goal of today:
Set up project + GitHub + notes

## What I implemented:
- Created repo
- Set up folder structure
- Initialized git

## What worked:
- Git setup successful

## What failed:
- Minor confusion with git commands

## Observations:
- Setup took longer than expected

## Next smallest step:
Load 30 samples from ChartQA and print one example (image + question)

## Date: May 1
## Direction: 
Counterfactual modality reliance audits

## Goal of today:
Run ChartQA experiments with and without images and see how much performance actually depends on vision

## What I implemented:
- Full evaluation pipeline for Qwen2-VL-2B on ChartQA (full test set)
- Image condition (normal input)
- No-image condition (text-only)
- Same pipeline reused for Qwen2-VL-7B (500 sample subset due to compute)
- Question categorization:
  - lookup
  - compositional
  - yes/no
- Accuracy tracking:
  - overall
  - per category

## What worked:
- Both 2B and 7B pipelines ran end-to-end without major bugs
- Category breakdown stayed consistent across runs
- No-image condition behaved as expected (large drops, not random outputs)
- Results were stable enough to compare across models

## What didn’t / friction:
- Colab GPU limits slowed iteration
- Model loading / setup took longer than expected
- Had to manually save outputs to avoid losing results
- Earlier bug where image input wasn’t actually being passed (fixed)

## Observations:
- Removing the image hurts lookup much more than compositional
- This pattern shows up in both 2B and 7B
- Compositional performance drops, but not nearly as much
- 7B improves compositional accuracy overall, but still shows the same pattern
- Yes/no feels noisy because there are very few samples

## Open questions:
- Why is compositional reasoning less affected by removing the image?
- Is the model actually reasoning, or using language shortcuts?
- Would giving the wrong image actively hurt performance (instead of just removing it)?
- Is this specific to ChartQA or more general?

## Next smallest step:
- Decide whether to run an image-swap (wrong image) experiment
- Start organizing results + writing while everything is fresh

## Update: Image Swap Experiment (May 1)

Ran 7B with wrong-image condition (200 samples).

Observations:
- Lookup performance remains relatively high (~0.60)
- Compositional performance similar to no-image (~0.17)
- Result is less clean than expected

Notes:
- Adjacent image swap may not be a strong counterfactual
- Needs better control or reinterpretation

## Update: Subtype Analysis (May 1)

Ran 500-sample pilot for compositional subtype breakdown.

Key observations:
- Compositional performance is highly non-uniform
- Difference questions show strong visual dependence
- Average questions show near-zero improvement from image input
- Ratio questions remain very low accuracy
- Yes/no questions largely insensitive to image

Conclusion:
Compositional reasoning is not a single failure mode; performance varies significantly by operation type.

## Update: Subtype Analysis (Full Dataset)

Ran full ChartQA test set for subtype breakdown.

Observations:
- All compositional subtypes show similar performance (~0.32–0.44 with image)
- Large drop without image across all compositional types
- Earlier pilot overestimated subtype differences
- Visual grounding consistently helps, but reasoning remains limited

Conclusion:
Compositional reasoning is uniformly constrained rather than highly subtype-dependent.