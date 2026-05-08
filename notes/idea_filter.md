# Direction Filter

## Selected Direction  
Counterfactual modality reliance audits for multimodal benchmarks

---

## Core Question  
To what extent do vision-language models rely on visual input when answering chart-based questions, and how does this dependence vary by task type?

---

## Motivation  
Multimodal benchmarks such as ChartQA are commonly used as evidence of visual reasoning ability. However, aggregate accuracy does not distinguish between qualitatively different behaviors:

- retrieving values from the chart  
- performing multi-step reasoning over those values  
- answering using partial or non-visual cues  

As a result, it remains unclear whether strong performance reflects genuine visual grounding or a mixture of visual and non-visual strategies.

---

## Approach  
Evaluate models under multiple counterfactual conditions:

- image + question (standard multimodal input)  
- question only (image removed)  
- mismatched distractor image + question  

Decompose questions into:

- lookup (direct value extraction)  
- compositional (difference, ratio, average, total, etc.)  
- yes/no (binary comparisons)  

Compare accuracy across conditions and task types using exact match, transition analysis, and confidence intervals.

---

## Current Signal (validated)

Observed across:

- Qwen2-VL-2B (full test set)  
- Qwen2-VL-7B (full test set)  
- InternVL2-4B (500 samples)  
- Phi-3.5-vision-instruct (full test set)  
- PlotQA-derived evaluation subset (Qwen2B + Qwen7B)

Key patterns:

- Removing the image causes a large drop in lookup performance (~−0.65 to −0.75)  
- Compositional performance drops less (~−0.30 to −0.45)  
- Yes/no performance shows weaker and more variable dependence on visual input  
- Distractor-image performance for lookup/compositional falls close to no-image levels  
- Lookup examples overwhelmingly transition from correct → incorrect after image removal  

Additional validation:

- Pattern is consistent across model scale (2B → 7B)  
- Pattern generalizes across model families (Qwen, InternVL, Phi)  
- Pattern replicates on a second chart QA benchmark  
- Wilson 95% confidence intervals show:
  - clear separation for lookup  
  - consistent but smaller separation for compositional  
  - overlapping or unstable intervals for yes/no  

---

## Interpretation (refined)

- Visual reliance is strongly task-dependent  
- Retrieval behavior is tightly coupled to visual input  
- Compositional reasoning benefits from visual input but remains limited  
- Yes/no questions are often partially solvable without strong visual grounding  

Crucially:

- Benchmark accuracy does not reflect a single capability  
- Instead, it conflates:
  - visually grounded retrieval  
  - partially visual reasoning  
  - weakly visual or non-visual judgment  

The results suggest that aggregate multimodal benchmark accuracy can obscure important differences in how models use visual evidence.

---

## Contribution (clarified)

- Introduces a simple counterfactual framework for auditing visual reliance  
- Demonstrates that ChartQA accuracy conflates distinct multimodal behaviors  
- Shows that task-dependent visual reliance is consistent across models and scales  
- Provides evidence that standard multimodal evaluation does not isolate visual grounding  
- Adds behavioral transition analysis to characterize modality dependence at the example level  
- Introduces distractor-image evaluation to test dependence on matched chart evidence  

---

## Experiments / Ideas That Did Not Work Well

### Adjacent-image swap experiment
Initial distractor setup paired examples with nearby charts rather than fully random charts.

Observations:
- Lookup accuracy stayed surprisingly high (~0.60)
- Results were difficult to interpret
- Counterfactual was too weak because swapped charts sometimes shared structure or semantics

Decision:
- Excluded from main analysis
- Replaced with stronger random distractor-image condition

---

### Early compositional subtype hypothesis
Initial 500-sample pilot suggested large differences between:
- ratio
- average
- difference
- total

However, full-dataset evaluation showed much smaller separation.

Conclusion:
- Pilot overestimated subtype effects
- Compositional reasoning appears broadly constrained rather than strongly subtype-specific

---

### Concerns about prompt sensitivity
There was concern that:
- answer formatting
- instruction wording
- prompt strictness

might fully explain the observed effects.

Result:
- Prompt changes altered absolute accuracies
- But the qualitative ordering remained stable:
  lookup > compositional > yes/no

Conclusion:
- Core pattern appears robust to prompting variation

---

### Concern about benchmark-specific artifacts
A major risk was that the pattern might be unique to ChartQA.

Response:
- Constructed PlotQA-derived replication subset
- Observed same qualitative asymmetry

Remaining issue:
- Still limited to chart-domain multimodal QA

---

## Major Issues Encountered During the Project

- Full evaluation runs were computationally slow and difficult to monitor consistently
- Category standardization initially differed across models
- Early subset experiments created instability in yes/no estimates
- Distractor-image design required multiple iterations before becoming interpretable
- Some compositional labels were ambiguous or noisy under heuristic categorization
- PlotQA preprocessing required additional filtering and normalization work
- Needed repeated consistency checks across:
  - prompts
  - normalization
  - evaluation scripts
  - category counts

---

## Open Questions

- Why does visual input improve compositional reasoning without enabling high accuracy?  
- What fraction of compositional failures come from reasoning versus extraction failures?  
- To what extent are yes/no questions driven by language priors?  
- How broadly does this pattern generalize beyond chart QA?  
- Would stronger perturbations reveal different forms of visual dependence?  
- Can visual dependence be measured continuously rather than categorically?  

---

## Risks

- Analysis remains concentrated in chart QA  
- Subset evaluation for InternVL  
- Task categorization is heuristic  
- Findings are descriptive rather than mechanistic  
- No training or intervention component  
- Visual dependence does not necessarily imply correct reasoning

---

## Future Directions / Continued Work

### Additional evaluations
- Run full-dataset InternVL2-4B evaluation
- Extend PlotQA replication to:
  - InternVL2-4B
  - Phi-3.5-vision
- Evaluate additional VLM families

### Stronger counterfactuals
- Structured perturbations
- OCR corruption
- Partial chart masking
- Axis/value scrambling
- Saliency-targeted interventions

### Better task decomposition
- Finer compositional subtypes
- Numeric reasoning depth
- OCR-heavy vs relation-heavy questions
- Confidence-calibrated modality dependence

### Behavioral analysis
- Analyze error transitions in greater detail
- Cluster recurring failure modes
- Study cases where wrong images still produce correct answers

### Mechanistic direction
- Connect behavioral modality dependence to:
  - attention patterns
  - cross-modal token usage
  - internal representations
  - modality routing behavior

### Broader evaluation framing
- Extend beyond chart QA into:
  - document QA
  - table QA
  - diagram reasoning
  - multimodal scientific benchmarks

---

## Status

Core experiments complete:

- Image vs no-image:
  - Qwen2B full
  - Qwen7B full
  - Phi full
  - InternVL subset
- Distractor-image evaluation
- Transition analysis
- PlotQA replication
- Prompt robustness check
- Statistical validation (confidence intervals)
- Final figure set complete

Ongoing:
- Experiment logging cleanup
- GitHub/codebase organization
- Additional cross-benchmark validation
- Extended model coverage