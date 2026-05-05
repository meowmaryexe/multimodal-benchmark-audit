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
Evaluate models under two conditions:

- image + question (standard multimodal input)  
- question only (image removed)  

Decompose questions into:

- lookup (direct value extraction)  
- compositional (difference, ratio, average, total, etc.)  
- yes/no (binary comparisons)  

Compare accuracy across conditions and task types using exact match and confidence intervals.

---

## Current Signal (validated)

Observed across:

- Qwen2-VL-2B (full test set)  
- Qwen2-VL-7B (full test set)  
- InternVL2-4B (500 samples)  
- Phi-3.5-vision-instruct (500 samples)

Key patterns:

- Removing the image causes a large drop in lookup performance (~−0.65 to −0.75)  
- Compositional performance drops less (~−0.30 to −0.45)  
- Yes/no performance shows weaker and more variable dependence on visual input  

Additional validation:

- Pattern is consistent across model scale (2B → 7B)  
- Pattern generalizes across model families (Qwen, InternVL, Phi)  
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

---

## Contribution (clarified)

- Introduces a simple counterfactual framework for auditing visual reliance  
- Demonstrates that ChartQA accuracy conflates distinct capabilities  
- Shows that task-dependent visual reliance is consistent across models and scales  
- Provides evidence that standard multimodal evaluation does not isolate visual grounding  

---

## Open Questions

- Why does visual input improve compositional reasoning without enabling high accuracy?  
- What limits performance across compositional operations?  
- To what extent are yes/no questions driven by language priors?  
- How broadly does this pattern generalize beyond ChartQA?  

---

## Risks

- Analysis is limited to a single benchmark  
- Subset evaluation for some models (InternVL, Phi)  
- Task categorization is heuristic  
- Findings are descriptive rather than mechanistic  

---

## Next Steps (post-submission direction)

- Extend to additional structured multimodal benchmarks  
- Explore alternative counterfactuals (e.g., image perturbations, distractors)  
- Develop finer-grained task decompositions  
- Connect to mechanistic analysis of multimodal reasoning  

---

## Status

Core experiments complete:

- Image vs no-image (Qwen2B full, Qwen7B full, InternVL subset, Phi subset)  
- Subtype analysis (Qwen2B full)  
- Cross-model validation (InternVL, Phi)  
- Statistical validation (confidence intervals)  
- Final figure and paper draft complete  

Current focus:

- Final submission polish  
- Repo cleanup and commit structuring  
- Positioning for review and future work  