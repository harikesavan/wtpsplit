# Master's Thesis Plan: Length-Constrained Text Segmentation

## 1. Executive Summary

This thesis proposes implementing and evaluating length-constrained segmentation methods for neural sentence boundary detection models. While existing approaches (WtP, SaT) use threshold-based segmentation, this work will enable explicit control over segment lengths by combining model probabilities with prior distributions over chunk lengths, solving the optimization problem via greedy and Viterbi algorithms.

## 2. Research Motivation

### 2.1 Problem Statement
Current neural segmenters (SaT, WtP) only provide implicit length control through threshold adjustment. However, many downstream applications require explicit length constraints:
- Chunking for RAG (Retrieval-Augmented Generation) systems with token limits
- Document preprocessing for transformer models with fixed context windows
- Subtitle generation with display constraints
- Mobile text display optimization

### 2.2 Research Questions
1. How do different prior distributions over chunk lengths affect segmentation quality?
2. What is the trade-off between segmentation accuracy and length constraint satisfaction?
3. How does the Viterbi algorithm compare to greedy search in terms of quality and computational efficiency?
4. Can length-constrained segmentation improve downstream task performance in specific use cases?

## 3. Technical Approach

### 3.1 Theoretical Foundation

#### Optimization Problem
```
argmax_C ∏ Prior(C_i - C_{i-1}) · P(C_i)
```

Where:
- `C_i` = position of i-th selected boundary
- `P(C_i)` = segmenter probability at position i
- `Prior(C_i - C_{i-1})` = prior probability over chunk length

#### Log-space Formulation (for numerical stability)
```
argmax_C ∑ [log Prior(C_i - C_{i-1}) + log P(C_i)]
```

### 3.2 Implementation Approaches

#### **Approach 1: Greedy Selection**
**Algorithm:**
```
1. Start at position 0
2. For each chunk:
   a. Calculate posterior = prior(length) · prob(position) for all valid next positions
   b. Select position with maximum posterior
   c. Set as new starting position
3. Repeat until end of text
```

**Pros:**
- Simple to implement
- Fast (O(n·k) where k = max_length)
- Easy to debug and understand

**Cons:**
- Not globally optimal
- May make suboptimal early choices

#### **Approach 2: Viterbi Algorithm**
**Algorithm:**
```
1. Initialize: δ[0] = 1, backpointer[0] = None
2. For each position t:
   For each possible previous position s (within length constraints):
     score = δ[s] · prior(t-s) · prob(t)
     if score > δ[t]:
       δ[t] = score
       backpointer[t] = s
3. Backtrack from final position to recover optimal path
```

**Pros:**
- Globally optimal solution
- Theoretically principled
- Efficient (O(n·k²) or O(n·k) with optimization)

**Cons:**
- More complex implementation
- Higher memory requirements
- Slightly slower than greedy

### 3.3 Prior Distribution Designs

#### Prior 1: Uniform Hard Constraint
```python
def uniform_prior(length, max_length):
    return 1.0 if length <= max_length else 0.0
```
**Use case:** Strict maximum length requirement (e.g., token limits)

#### Prior 2: Gaussian (Normal) Distribution
```python
def gaussian_prior(length, target_length, sigma):
    return exp(-0.5 * ((length - target_length) / sigma) ** 2)
```
**Use case:** Prefer specific length with soft boundaries

#### Prior 3: Clipped Polynomial
```python
def clipped_polynomial_prior(length, target_length, coefficient):
    return max(1 - coefficient * (length - target_length) ** 2, 0.0)
```
**Use case:** Sharp preference with hard cutoff

#### Prior 4: Exponential Decay
```python
def exponential_prior(length, max_length, decay_rate):
    return exp(-decay_rate * (length - max_length)) if length > max_length else 1.0
```
**Use case:** Strong preference for shorter segments

#### Prior 5: Piecewise Linear
```python
def piecewise_linear_prior(length, min_length, target_length, max_length):
    if length < min_length: return 0.0
    elif length < target_length: return (length - min_length) / (target_length - min_length)
    elif length < max_length: return 1.0 - (length - target_length) / (max_length - target_length)
    else: return 0.0
```
**Use case:** Explicit min/target/max bounds with smooth transitions

## 4. Implementation Plan

### 4.1 Code Architecture

#### **Module Structure**
```
wtpsplit/
├── __init__.py                    # [MODIFY] Add split_length_constrained method to SaT/WtP
├── extract.py                     # [NO CHANGE] Core inference remains unchanged
├── utils/
│   ├── __init__.py               # [MODIFY] Add constraint solvers
│   └── constraints.py            # [NEW] Prior distributions and optimization algorithms
└── evaluation/
    └── length_constraints.py     # [NEW] Evaluation metrics
```

#### **New API Methods**

**In `wtpsplit/__init__.py` (SaT class):**
```python
def split_length_constrained(
    self,
    text_or_texts: Union[str, List[str]],
    prior: Union[str, Callable] = "uniform",
    max_length: Optional[int] = None,
    target_length: Optional[int] = None,
    min_length: int = 1,
    algorithm: str = "viterbi",  # or "greedy"
    prior_params: Optional[Dict] = None,
    lang_code: Optional[str] = None,
    verbose: bool = False,
    **kwargs
) -> Union[List[str], List[List[str]]]:
    """
    Segment text with explicit length constraints.

    Args:
        text_or_texts: Input text(s) to segment
        prior: Prior distribution ("uniform", "gaussian", "polynomial", or callable)
        max_length: Maximum segment length (characters)
        target_length: Preferred segment length
        min_length: Minimum segment length
        algorithm: "viterbi" or "greedy"
        prior_params: Additional parameters for prior function
        lang_code: Language code (auto-detect if None)
        verbose: Print debug information

    Returns:
        List of segmented sentences (or list of lists for batch)
    """
```

### 4.2 Implementation Phases

#### **Phase 1: Core Algorithm Implementation** (Week 1-2)
**Files to create/modify:**
1. `wtpsplit/utils/constraints.py` - Core algorithms
   - [ ] Implement greedy search algorithm
   - [ ] Implement Viterbi algorithm
   - [ ] Add numerical stability (log-space computation)
   - [ ] Unit tests for both algorithms

2. `wtpsplit/utils/priors.py` - Prior distributions
   - [ ] Implement 5 prior distribution functions
   - [ ] Create prior factory/registry
   - [ ] Support custom callable priors
   - [ ] Unit tests for each prior

**Deliverables:**
- Working constraint solver functions
- Comprehensive unit tests (>90% coverage)
- Documentation with mathematical formulation

#### **Phase 2: Integration with WtPSplit** (Week 2-3)
**Files to modify:**
1. `wtpsplit/__init__.py`
   - [ ] Add `split_length_constrained()` method to `SaT` class
   - [ ] Add `split_length_constrained()` method to `WtP` class
   - [ ] Handle batch processing
   - [ ] Preserve compatibility with existing API

2. Integration tests
   - [ ] Test with real SaT models
   - [ ] Test with different languages
   - [ ] Test edge cases (very short/long texts)
   - [ ] Benchmark performance

**Deliverables:**
- Fully integrated API
- Integration tests
- Performance benchmarks

#### **Phase 3: Evaluation Framework** (Week 3-4)
**Files to create:**
1. `wtpsplit/evaluation/length_constraints.py`
   - [ ] Implement evaluation metrics (see Section 5)
   - [ ] Create benchmark datasets
   - [ ] Statistical significance testing

2. `experiments/length_constrained_eval.py`
   - [ ] Experimental scripts
   - [ ] Result visualization
   - [ ] Comparison tables

**Deliverables:**
- Evaluation metrics implementation
- Benchmark results on standard datasets
- Visualization scripts

#### **Phase 4: Experiments & Analysis** (Week 4-6)
**Experiments to run:**
1. **Prior Comparison Study**
   - Test all 5 prior types on same dataset
   - Vary parameters systematically
   - Measure quality vs. constraint satisfaction trade-off

2. **Algorithm Comparison**
   - Greedy vs. Viterbi quality difference
   - Runtime analysis (time complexity empirically)
   - Memory profiling

3. **Language Generalization**
   - Test on 10+ languages
   - Analyze if optimal priors generalize

4. **Downstream Task Evaluation**
   - RAG chunking quality (retrieval accuracy)
   - Summarization with length-constrained input
   - Machine translation with segment length matching

**Deliverables:**
- Experimental results with statistical analysis
- Comparison charts and tables
- Error analysis

#### **Phase 5: Documentation & Thesis Writing** (Week 6-8)
1. **Code Documentation**
   - [ ] Docstrings for all public APIs
   - [ ] Tutorial notebook with examples
   - [ ] README section on length constraints
   - [ ] API reference documentation

2. **Thesis Chapters**
   - [ ] Introduction & Related Work
   - [ ] Methodology (algorithm description)
   - [ ] Implementation Details
   - [ ] Experiments & Results
   - [ ] Discussion & Future Work
   - [ ] Conclusion

**Deliverables:**
- Complete API documentation
- Jupyter notebook tutorial
- Thesis draft

## 5. Evaluation Methodology

### 5.1 Metrics

#### **Primary Metrics**

1. **Constraint Satisfaction Rate (CSR)**
   ```
   CSR = (# segments within [min_length, max_length]) / (total # segments)
   ```
   Target: >95%

2. **Segmentation F1 Score**
   - Compare against ground truth sentence boundaries
   - Standard precision/recall/F1
   - Measures if we're still finding "natural" boundaries

3. **Boundary Quality Score**
   ```
   BQS = mean(probability[selected_boundaries])
   ```
   - Higher = we're selecting high-confidence boundaries
   - Compare with unconstrained baseline

#### **Secondary Metrics**

4. **Length Distribution Analysis**
   - Mean, median, std dev of segment lengths
   - Compare distribution to target prior

5. **Computational Efficiency**
   - Runtime (seconds per 1000 characters)
   - Memory usage
   - Compare greedy vs. Viterbi

6. **Downstream Task Performance**
   - RAG retrieval: MRR (Mean Reciprocal Rank)
   - Summarization: ROUGE scores
   - Translation: BLEU scores

### 5.2 Datasets

#### **Intrinsic Evaluation**
1. **UD Treebanks** (Multilingual, gold sentence boundaries)
   - English EWT, German GSD, French GSD, Chinese GSD
   - 10+ languages for generalization testing

2. **OPUS-100** (Parallel sentences)
   - Test length matching for translation pairs

3. **Custom Dataset** (Generate from long-form text)
   - Wikipedia articles
   - arXiv papers
   - News articles

#### **Extrinsic Evaluation**
1. **RAG Chunking**: MS MARCO / Natural Questions
   - Measure retrieval quality with different chunk sizes

2. **Summarization**: CNN/DailyMail
   - Fixed-length input segments

3. **Translation**: WMT test sets
   - Match source/target segment lengths

### 5.3 Baselines

1. **Naive Threshold** (current default)
2. **Sliding Window** (fixed-length chunks, ignore boundaries)
3. **Recursive Splitting** (split until length satisfied)
4. **spaCy Sentencizer** (rule-based)
5. **NLTK Punkt** (unsupervised ML)

### 5.4 Experimental Design

#### **Experiment 1: Prior Sensitivity Analysis**
- **Variables:** Prior type, max_length ∈ {50, 100, 200, 500}
- **Fixed:** Algorithm (Viterbi), dataset (UD-English)
- **Metrics:** CSR, F1, BQS
- **Hypothesis:** Gaussian prior will balance quality and constraint satisfaction

#### **Experiment 2: Algorithm Comparison**
- **Variables:** Algorithm (greedy vs. Viterbi)
- **Fixed:** Prior (uniform), max_length = 100
- **Metrics:** F1, runtime, memory
- **Hypothesis:** Viterbi gives +2-5% F1 with <2x runtime

#### **Experiment 3: Cross-Lingual Generalization**
- **Variables:** Language (10+ from UD)
- **Fixed:** Prior (best from Exp 1), algorithm (Viterbi)
- **Metrics:** CSR, F1 per language
- **Hypothesis:** Method generalizes across languages

#### **Experiment 4: Downstream Task Impact**
- **Variables:** Chunking method (constrained vs. baselines)
- **Fixed:** Downstream model (fixed RAG/summarization system)
- **Metrics:** Task-specific (MRR, ROUGE)
- **Hypothesis:** Length constraints improve consistency and task performance

## 6. Timeline

### 8-Week Timeline

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| 1 | Setup + Algorithm Implementation | Implement greedy & Viterbi, unit tests | Working algorithms |
| 2 | Prior Distributions + Integration | Implement priors, integrate with SaT | API ready |
| 3 | Testing + Evaluation Framework | Integration tests, metrics implementation | Test suite, metrics |
| 4 | Experiment 1 & 2 | Prior comparison, algorithm comparison | Results for Exp 1-2 |
| 5 | Experiment 3 & 4 | Cross-lingual, downstream tasks | Results for Exp 3-4 |
| 6 | Analysis + Visualization | Statistical analysis, create figures | Analysis complete |
| 7 | Documentation + Writing | Code docs, tutorial, thesis draft | Documentation |
| 8 | Revision + Final Submission | Incorporate feedback, final polish | Final thesis |

### Milestone Checkpoints
- **Week 2:** Working demo to show supervisor
- **Week 4:** Midpoint review with initial results
- **Week 6:** Complete experimental results
- **Week 8:** Thesis submission

## 7. Expected Contributions

### 7.1 Technical Contributions
1. **Novel segmentation method** combining neural probabilities with length priors
2. **Open-source implementation** in widely-used WtPSplit library
3. **Comprehensive evaluation** across languages and use cases
4. **Performance optimization** for production deployment

### 7.2 Scientific Contributions
1. **Empirical analysis** of prior distributions for segmentation
2. **Comparison** of greedy vs. Viterbi for this specific problem
3. **Cross-lingual study** of length-constrained segmentation
4. **Downstream impact** analysis on RAG and other applications

### 7.3 Practical Impact
- Enable better chunking for RAG systems (direct industry application)
- Improve preprocessing for LLMs with context limits
- Provide flexible API for researchers and practitioners

## 8. Risk Mitigation

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Viterbi too slow for long texts | Medium | High | Implement beam search variant, optimize with Numba/Cython |
| Constraints too restrictive (low F1) | Medium | Medium | Provide soft constraints via prior smoothing |
| Memory issues with long documents | Low | Medium | Process in chunks with boundary resolution |
| Prior parameters hard to tune | Medium | Low | Auto-tuning via grid search on validation set |

### 8.2 Schedule Risks

| Risk | Mitigation |
|------|------------|
| Implementation takes longer than expected | Start with greedy only, add Viterbi if time permits |
| Experiments take too long to run | Use smaller datasets, parallel processing |
| Thesis writing falls behind | Write methods section during implementation |

## 9. Success Criteria

### 9.1 Minimum Viable Thesis
- [x] Working implementation (greedy or Viterbi)
- [x] At least 2 prior distributions
- [x] Evaluation on 3+ languages
- [x] Comparison with baseline threshold method
- [x] Complete thesis document

### 9.2 Target Success
- [x] Both greedy and Viterbi implemented
- [x] 5 prior distributions with tuning
- [x] Evaluation on 10+ languages
- [x] Downstream task evaluation (RAG)
- [x] Statistical significance testing
- [x] Code merged into WtPSplit main branch

### 9.3 Stretch Goals
- [ ] Real-time streaming variant for online segmentation
- [ ] Learned priors (train small model to predict optimal prior)
- [ ] Multi-objective optimization (length + coherence + topic)
- [ ] Published paper at NLP conference

## 10. References & Related Work

### 10.1 Key Papers
1. **WtPSplit Original Paper**: "Where's the Point? Self-Supervised Multilingual Punctuation-Agnostic Sentence Segmentation" (Minixhofer et al., ACL 2023)
2. **Viterbi Algorithm**: Viterbi (1967) - Error bounds for convolutional codes
3. **Text Segmentation**: Hearst (1997) - TextTiling: Segmenting text into multi-paragraph subtopic passages
4. **RAG Chunking**: Lewis et al. (2020) - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

### 10.2 Related Methods
- **Hidden Markov Models** for segmentation (similar Viterbi application)
- **Constrained decoding** in neural MT (inspiration for constraints)
- **Dynamic programming** for sentence compression (similar optimization)

## 11. Appendix: Code Examples

### 11.1 Example Usage API

```python
from wtpsplit import SaT

# Load model
model = SaT("sat-3l-sm")

# Example 1: Hard maximum length
sentences = model.split_length_constrained(
    text="Very long document...",
    prior="uniform",
    max_length=200,  # characters
    algorithm="viterbi"
)

# Example 2: Preferred length (Gaussian prior)
sentences = model.split_length_constrained(
    text="Very long document...",
    prior="gaussian",
    target_length=100,
    prior_params={"sigma": 20},
    algorithm="greedy"
)

# Example 3: Custom prior function
def my_prior(length):
    if length < 50: return 0.5
    elif length < 150: return 1.0
    else: return 0.1

sentences = model.split_length_constrained(
    text="Very long document...",
    prior=my_prior,
    algorithm="viterbi"
)

# Example 4: Token-based constraints (for RAG)
sentences = model.split_length_constrained(
    text="Very long document...",
    prior="uniform",
    max_length=512,  # Will count tokens, not chars
    length_unit="tokens",
    tokenizer="gpt-4"  # Specify tokenizer
)
```

### 11.2 Expected Output Format

```python
# Input
text = "This is sentence one. This is sentence two. This is sentence three."

# Output (with max_length=30)
[
    "This is sentence one.",  # 21 chars
    "This is sentence two.",  # 21 chars
    "This is sentence three." # 24 chars
]

# Each segment respects the constraint while preferring natural boundaries
```

## 12. Questions for Supervisor Review

1. **Scope:** Is the proposed scope appropriate for a master's thesis (8-week timeline)?
2. **Priorities:** Should I focus on Viterbi (optimal quality) or greedy (faster, simpler)?
3. **Evaluation:** Are the proposed metrics sufficient, or should I add specific ones?
4. **Downstream Tasks:** Which application (RAG, summarization, MT) is most valuable to evaluate?
5. **Publication:** Should we target a workshop paper (e.g., EMNLP Findings)?
6. **Baselines:** Are there other methods I should compare against?
7. **Collaboration:** Should this work coordinate with the main WtPSplit authors?

---

**Document Version:** 1.0
**Date:** 2025-10-26
**Author:** [Your Name]
**Supervisor:** [Supervisor Name]
**Institution:** [Your University]
