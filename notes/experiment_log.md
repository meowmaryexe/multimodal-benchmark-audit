
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

## Observations:
- Setup took longer than expected

## Next smallest step:
Load 30 samples from ChartQA and print one example (image + question)

---

## Date: May 1
## Direction: 
Counterfactual modality reliance audits

## Goal of today:
Run ChartQA experiments with and without images and measure dependence on visual input

## What I implemented:
- Full evaluation pipeline for Qwen2-VL-2B on ChartQA (full test set)
- Image vs no-image conditions
- Extended pipeline to Qwen2-VL-7B (500-sample subset)
- Question categorization:
  - lookup
  - compositional
  - yes/no
- Accuracy tracking (overall + per category)

## Observations:
- Removing the image causes a large drop in lookup accuracy
- Compositional performance drops less significantly
- Pattern is consistent across 2B and 7B models
- Yes/no results are noisy due to small sample size

## Interpretation:
- Visual reliance is not uniform across tasks
- Retrieval appears tightly coupled to visual input
- Compositional reasoning may partially rely on non-visual patterns

## Open questions:
- Why is compositional reasoning less affected by removing the image?
- Is the model reasoning or using language shortcuts?
- Would incorrect images actively hurt performance?
- Does this generalize beyond ChartQA?

## Next step:
Test alternative counterfactuals and refine analysis

---

## Update: Image Swap Experiment (May 1)

Tested wrong-image condition (Qwen7B, 200 samples).

## Observations:
- Lookup accuracy remains relatively high (~0.60)
- Compositional performance similar to no-image condition
- Results are difficult to interpret

## Interpretation:
- Adjacent-image swap is not a strong counterfactual
- Models may not be tightly grounded in specific visual content
- Setup introduces ambiguity

## Decision:
Exclude from main analysis. Image removal is a cleaner intervention.

---

## Update: Subtype Analysis (May 1)

Ran 500-sample pilot for compositional subtypes.

## Observations:
- Large variation across operations (difference, average, ratio)
- Some tasks show strong visual dependence, others do not

## Interpretation:
Compositional reasoning is not uniform.

---

## Update: Subtype Analysis (Full Dataset)

## Observations:
- Subtypes converge to similar performance (~0.32–0.44 with image)
- Large drop without image across all subtypes
- Pilot overestimated differences

## Conclusion:
Compositional reasoning is uniformly constrained rather than subtype-specific.

---

## Update: Cross-Model Validation (May 2)

Added InternVL2-4B (500 samples).

## Observations:
- Lookup drop (~−0.60) matches Qwen
- Compositional drop (~−0.28) matches prior results
- Yes/no remains weakly dependent

## Conclusion:
Pattern generalizes across model families, not just scale.

---

## Update: Statistical Validation (May 2)

Computed Wilson 95% confidence intervals.

## Observations:
- Clear separation for lookup
- Consistent separation for compositional
- Overlap for yes/no

## Conclusion:
Results are statistically robust for lookup and compositional tasks.

---

## Update: Final Consolidation (May 2, updated after May 5)

Constructed unified figure across models and task types.

## Observations:
- Strong separation for lookup
- Consistent but smaller gap for compositional
- Weak or overlapping separation for yes/no
- Pattern holds across:
  - Qwen2B (full)
  - Qwen7B (subset → later full)
  - InternVL2-4B
  - Phi-3.5-vision (added later)

## Conclusion:
ChartQA performance reflects a mixture of behaviors:
- visually grounded retrieval
- partially visual reasoning
- weakly visual or non-visual judgment

Aggregate accuracy conflates these capabilities.

---

## Date: May 4
## Direction:
Counterfactual modality reliance audits

## Goal of today:
Finalize framing and prepare submission

## Observations:
- Core contribution is about evaluation interpretation, not performance
- Simplicity of the method strengthens the argument
- Figure communicates the main result clearly

## Interpretation:
The work is best framed as:
- a diagnostic audit of multimodal evaluation
- not a model improvement

## Open questions:
- Will reviewers view this as sufficiently novel?
- Is the contribution framed strongly enough?

## Next step:
Strengthen empirical validation

---

---

## Date: May 5
## Direction:
Counterfactual modality reliance audits

## Goal of today:
Complete full-scale validation and finalize experimental pipeline

## What I implemented:
- Ran full ChartQA test set for Phi-3.5-vision
- Ran full ChartQA test set for Qwen2-VL-7B (completing full coverage for Qwen models)
- Verified and standardized categorization across all models (added "total")
- Validated category distributions match exactly across:
  - Qwen2B (full)
  - Qwen7B (full)
  - Phi-3.5-vision (full)
  - InternVL2-4B (subset)
- Consolidated final results into a single consistent evaluation format

## Observations:
- Task-dependent visual reliance holds across all models and dataset scales
- Lookup shows the largest drop without image (~−0.70 to −0.76 across full models)
- Compositional performance is moderate (~0.40–0.55 with image) and strongly affected by image removal
- Yes/no shows weaker and more variable dependence on visual input
- Phi full results align closely with Qwen7B full results, strengthening cross-model consistency
- Category counts are identical across all models after standardization

## Interpretation:
- The observed pattern is stable across:
  - model scale (2B → 7B)
  - model family (Qwen, InternVL, Phi)
  - dataset size (subset → full)
- Visual reliance is consistently ordered:
  lookup > compositional > yes/no
- Compositional reasoning requires visual input but remains limited in accuracy
- Yes/no questions are partially solvable without strong visual grounding and vary more across models

## Conclusion:
The experimental pipeline is complete and internally consistent.

The core result is strongly supported:

- Visual reliance is task-dependent
- Retrieval is tightly coupled to visual input
- Compositional reasoning is partially visual but constrained
- Binary judgments are weakly grounded and less stable

## Takeaway:
Adding Phi full-scale evaluation did not change the qualitative pattern, but significantly strengthens confidence in its generality across models and scales.

---

## Update: Prompt Robustness Check (May 5)

Evaluated prompt variation on Qwen2-VL-2B (500 samples) using an alternate instruction:
"Provide only the answer."

## Observations:
- Absolute accuracies change under prompt variation
- Compositional no-image performance increases relative to the original prompt
- Lookup still shows the largest drop without image
- Compositional shows a moderate drop
- Yes/no remains more variable

## Interpretation:
- The qualitative ordering of visual reliance is preserved:
  lookup > compositional > yes/no
- The pattern is not dependent on the exact answer-only prompt formulation
- Prompt changes affect absolute performance, but not the structure of visual dependence

## Conclusion:
The core result is robust to prompt variation. Visual reliance remains task-dependent even under changes to prompting.