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
Run full-scale ChartQA evaluation to measure visual reliance (image vs no-image) and category-level behavior

## What I implemented:
- Full evaluation pipeline for Qwen2-VL-2B on ChartQA test set (~2500 samples)
- Image condition (standard VLM input)
- No-image condition (text-only ablation)
- Question categorization:
  - lookup (direct value retrieval)
  - compositional (difference, ratio, average, etc.)
  - yes/no
- Accuracy tracking:
  - overall accuracy
  - per-category accuracy
- Clean normalization + exact match evaluation

## What worked:
- Successfully ran full dataset with image condition
- Stable category distributions:
  - lookup: 1938
  - compositional: 534
  - yes/no: 28
- Strong and interpretable performance gap:
  - lookup: 0.740
  - compositional: 0.375
- Category split clearly separates retrieval vs reasoning behavior

## What failed:
- Earlier script versions accidentally dropped image inputs (fixed)
- LLaVA baseline produced weak/unreliable results → not suitable as primary comparison
- Minor debugging with indentation / processor inputs

## Observations:
- Large performance gap between lookup and compositional (~36.5 points)
- Model performs well at extracting values from charts
- Compositional reasoning remains significantly weaker even with image access
- Category decomposition is meaningful and not noisy
- Suggests different underlying mechanisms for retrieval vs reasoning

## Next smallest step:
- Complete NO-IMAGE full dataset run
- Measure:
  - lookup drop
  - compositional drop
- Compute differential dependence on visual input
