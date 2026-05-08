
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

---

## Update: Behavioral Transition Analysis (May 5)

## Goal:
Understand how individual examples change behavior after image removal.

## What I implemented:
- Computed transition buckets:
  - both correct
  - image-only correct
  - no-image-only correct
  - both wrong
- Added visual dependence rate (VDR):
  fraction of image-correct examples that become incorrect after image removal
- Ran analysis across:
  - Qwen2B
  - Qwen7B
  - Phi-3.5-vision
  - InternVL2-4B

## Observations:
- Lookup examples are overwhelmingly image-only correct
- Lookup VDR is extremely high across models
- Compositional questions also show strong image dependence
- Many compositional examples remain incorrect even with image access
- Yes/no behavior is more variable

## Interpretation:
- Lookup behavior is tightly coupled to visual extraction
- Compositional reasoning requires visual input but is also limited by reasoning failures
- Yes/no questions are less consistently grounded in visual evidence

## Next step:
Add representative qualitative examples for transition categories

---

## Update: Qualitative Example Collection (May 5)

## Goal:
Collect representative examples illustrating different transition behaviors.

## What I implemented:
- Extracted examples for:
  - lookup image-only correct
  - lookup both wrong
  - compositional image-only correct
  - compositional both wrong
  - yes/no both correct
  - yes/no no-image-only correct
- Generated candidate qualitative panels for appendix figures

## Observations:
- Lookup failures without images are often severe
- Compositional failures persist even with visual access
- Some yes/no questions remain answerable without image grounding

## Interpretation:
Different task types exhibit qualitatively different dependence on visual input.

## Next step:
Revisit distractor-image experiments with stronger controls

---

## Update: Distractor-Image Evaluation (May 6)

## Goal:
Test whether models rely on matched chart evidence versus generic chart-like input.

## What I implemented:
- Built distractor-image evaluation on 500-example ChartQA subset
- Replaced charts with unrelated random charts
- Fixed random seed for reproducibility
- Evaluated Qwen2-VL-7B under:
  - matched image
  - no image
  - distractor image

## Observations:
- Lookup distractor accuracy falls close to no-image performance
- Compositional distractor accuracy also approaches no-image condition
- Yes/no remains more variable

## Interpretation:
- Strong distractor controls are much cleaner than the earlier adjacent-image swap
- Models depend on matched chart evidence rather than generic image presence

## Decision:
Include distractor-image analysis in final experiments

---

## Update: PlotQA Replication (May 6)

## Goal:
Test whether the observed asymmetry generalizes beyond ChartQA.

## What I implemented:
- Built PlotQA-derived evaluation subset:
  - 1000 charts
  - up to 5 QA pairs per chart
- Implemented evaluation pipeline for:
  - Qwen2-VL-2B
  - Qwen2-VL-7B
- Applied same:
  - categorization
  - prompting
  - normalization
  - image-removal intervention

## Observations:
- Same qualitative ordering appears on PlotQA-derived evaluation
- Lookup nearly collapses without image input
- Compositional drops substantially
- Yes/no remains comparatively stable

## Interpretation:
Task-dependent visual reliance generalizes beyond a single benchmark.

## Next step:
Finalize consolidated figures and tables

---

## Update: Final Experimental Consolidation (May 6–7)

## Goal:
Standardize and finalize all experimental outputs.

## What I implemented:
- Finalized:
  - transition analysis tables
  - distractor analysis tables
  - PlotQA replication tables
  - appendix figures
- Verified consistency across:
  - category counts
  - normalization
  - prompts
  - evaluation scripts
- Consolidated final figure set across all models

## Observations:
- The qualitative ordering remains stable across:
  - models
  - scales
  - datasets
  - prompt variants
- Lookup consistently shows strongest visual dependence
- Compositional dependence is substantial but incomplete
- Yes/no remains weaker and less stable

## Interpretation:
Aggregate chart QA accuracy combines behaviors with substantially different dependence on visual evidence. The results consistently suggest that benchmark-level performance can obscure important differences between direct visual extraction, partially grounded compositional reasoning, and weaker forms of multimodal dependence.

## Remaining questions:
- To what extent are compositional failures driven by reasoning limitations versus visual extraction failures?
- How sensitive are the observed patterns to alternative prompting strategies or evaluation formats?
- Would stronger chart-domain generalization tests change the relative dependence patterns?
- How much of yes/no performance is driven by dataset priors versus genuine visual grounding?
- Do similar asymmetries appear in other multimodal QA domains beyond charts?

## Ongoing direction:
Continue refining the evaluation framing, extending cross-benchmark validation, and improving analysis of modality dependence beyond aggregate benchmark accuracy.