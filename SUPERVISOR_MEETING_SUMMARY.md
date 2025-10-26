# Supervisor Meeting: Length-Constrained Segmentation - Quick Summary

**Date:** 2025-10-26
**Student:** [Your Name]
**Thesis Topic:** Length-Constrained Text Segmentation for Neural Sentence Boundary Detection

---

## 1. One-Sentence Summary

Implement explicit length control for neural text segmentation by combining model probabilities with prior distributions over chunk lengths, optimized via greedy or Viterbi algorithms.

---

## 2. The Problem (30 seconds)

**Current situation:**
- WtPSplit/SaT models give probability scores for each character being a boundary
- Current approach: threshold-based segmentation (`prob > 0.025 → split here`)
- **Problem:** Only **implicit** length control (adjust threshold, hope for desired length)

**What we need:**
- **Explicit** length control: "Give me segments of ~100 characters" or "No segment > 200 chars"
- Real applications: RAG systems (token limits), mobile displays, subtitles

---

## 3. The Solution (1 minute)

### Mathematical Framework

Instead of just thresholding, we optimize:

```
argmax_C ∏ Prior(length) × Probability(boundary)
```

**Components:**
1. **Prior(length)**: Our preference for segment length
   - Uniform: "Any length up to 200 is fine"
   - Gaussian: "Prefer length ~100, but allow variation"
   - Custom: User-defined function

2. **Probability(boundary)**: Model's confidence (unchanged)

3. **Optimization**: Find best boundaries satisfying both

### Two Algorithms

| Algorithm | Description | Pros | Cons |
|-----------|-------------|------|------|
| **Greedy** | Pick best boundary one at a time | Fast, simple | Not globally optimal |
| **Viterbi** | Dynamic programming for optimal solution | Best quality | Slightly slower |

---

## 4. Example (Visual)

```
Text: "This is a test. Another sentence here. And one more."
Probabilities: [0.1, 0.3, 0.9, 0.2, 0.8, 0.1, 0.95]
               (model's boundary scores at each position)

Current (threshold=0.5):
  → "This is a test." | "Another sentence here." | "And one more."
  Lengths: [15, 23, 13] ✓ Uncontrolled

Length-Constrained (max_length=20):
  Prior penalizes length > 20
  → "This is a test." | "Another sentence" | "here. And one more."
  Lengths: [15, 17, 19] ✓ All under 20!
```

---

## 5. Implementation Plan (High-Level)

### Phase 1: Core (Week 1-2)
- Implement greedy + Viterbi algorithms
- 5 prior distributions (uniform, gaussian, polynomial, exponential, piecewise)
- Unit tests

### Phase 2: Integration (Week 2-3)
- Add `split_length_constrained()` method to SaT/WtP classes
- API design: clean, easy-to-use
- Integration tests

### Phase 3: Evaluation (Week 3-4)
- Metrics: constraint satisfaction rate, segmentation F1, boundary quality
- Datasets: UD Treebanks (10+ languages), OPUS, Wikipedia

### Phase 4: Experiments (Week 4-6)
1. Prior comparison: which prior works best?
2. Algorithm comparison: greedy vs Viterbi quality/speed
3. Cross-lingual: does it generalize?
4. Downstream: does it help RAG systems?

### Phase 5: Writing (Week 6-8)
- Documentation + tutorial
- Thesis chapters
- Final revision

---

## 6. Expected Contributions

### Technical
- ✅ Novel segmentation method (neural + length constraints)
- ✅ Open-source implementation in WtPSplit
- ✅ Production-ready code

### Scientific
- ✅ Empirical study of prior distributions
- ✅ Greedy vs Viterbi comparison for this problem
- ✅ Cross-lingual evaluation (10+ languages)
- ✅ Downstream impact analysis (RAG, summarization)

### Practical
- ✅ Better chunking for RAG systems (industry impact)
- ✅ Flexible API for researchers

---

## 7. Deliverables

### Code
- `wtpsplit/utils/constraints.py` - Core algorithms
- `wtpsplit/utils/priors.py` - Prior distributions
- Modified `wtpsplit/__init__.py` - API integration
- `wtpsplit/evaluation/length_constraints.py` - Metrics
- Unit tests + integration tests (>90% coverage)
- Jupyter notebook tutorial

### Thesis
- Introduction & Related Work
- Methodology (algorithms, priors, optimization)
- Implementation Details
- Experiments & Results (4 main experiments)
- Discussion & Future Work
- Conclusion

### Optional
- Workshop paper (EMNLP Findings, *ACL workshops)
- Pull request to WtPSplit main repository

---

## 8. Key Decisions Needed

| Decision | Options | Recommendation | Rationale |
|----------|---------|----------------|-----------|
| **Primary Algorithm** | Greedy vs Viterbi | **Viterbi** | Better quality, reasonable speed |
| **Number of Priors** | 3 vs 5 vs more | **5** | Covers main use cases |
| **Main Downstream Task** | RAG, Summarization, MT | **RAG** | Most practical impact |
| **Languages** | 5 vs 10 vs 20 | **10** | Balance coverage & time |
| **Publication Target** | Thesis only vs Workshop | **Both** | Maximize impact |

---

## 9. Timeline & Milestones

```
Week 1-2: [████████░░░░░░░░░░░░] Implementation
Week 3-4: [░░░░░░░░████████░░░░] Evaluation
Week 5-6: [░░░░░░░░░░░░░░██████] Experiments
Week 7-8: [░░░░░░░░░░░░░░░░░░██] Writing

Checkpoints:
  ✓ Week 2: Demo to supervisor
  ✓ Week 4: Midpoint review (initial results)
  ✓ Week 6: Complete experiments
  ✓ Week 8: Final submission
```

---

## 10. Success Metrics

### Minimum (Must Have)
- [ ] Working implementation (at least greedy)
- [ ] 2+ prior distributions
- [ ] Evaluation on 3+ languages
- [ ] Comparison with baseline
- [ ] Complete thesis

### Target (Should Have)
- [ ] Both greedy and Viterbi
- [ ] 5 prior distributions
- [ ] 10+ languages
- [ ] Downstream evaluation (RAG)
- [ ] Statistical tests
- [ ] Code merged to main branch

### Stretch (Nice to Have)
- [ ] Real-time streaming variant
- [ ] Learned priors (ML-based)
- [ ] Published workshop paper

---

## 11. Risks & Mitigation

### Technical Risks

**Risk:** Viterbi too slow for long documents
**Mitigation:** Optimize with Numba, implement beam search variant

**Risk:** Constraints hurt segmentation quality too much
**Mitigation:** Soft constraints via prior smoothing, parameter tuning

**Risk:** Hard to tune prior parameters
**Mitigation:** Auto-tuning via validation set grid search

### Schedule Risks

**Risk:** Implementation takes longer than expected
**Mitigation:** Start with greedy only, Viterbi if time permits

**Risk:** Experiments take too long
**Mitigation:** Smaller datasets, parallel processing

---

## 12. Questions for Supervisor

### Scope
1. Is 8 weeks realistic for this scope?
2. Should I prioritize quality (Viterbi) or speed (greedy)?
3. Are 4 experiments enough, or should I add more?

### Evaluation
4. Which downstream task is most valuable: RAG, summarization, or MT?
5. Should I include user studies (qualitative evaluation)?
6. Are the proposed metrics sufficient?

### Publication
7. Should we target a workshop paper?
8. Timeline for paper submission if yes?

### Collaboration
9. Should I coordinate with WtPSplit authors (Minixhofer et al.)?
10. Will you provide feedback on code/writing weekly or bi-weekly?

---

## 13. Backup Plans

### If Behind Schedule
- **Week 3:** Drop Viterbi, keep only greedy
- **Week 5:** Reduce languages from 10 to 5
- **Week 6:** Drop downstream evaluation, focus on intrinsic metrics
- **Week 7:** Reduce thesis depth, focus on core contributions

### If Ahead of Schedule
- **Week 4:** Add learned priors (ML model to predict optimal prior)
- **Week 5:** Implement streaming/online variant
- **Week 6:** Add multi-objective optimization (length + coherence + topic)
- **Week 7:** Write workshop paper draft

---

## 14. Example API (Show-and-Tell)

```python
from wtpsplit import SaT

# Load model
model = SaT("sat-3l-sm")

# Simple usage: hard limit
sentences = model.split_length_constrained(
    text="Your long document...",
    prior="uniform",
    max_length=200
)

# Advanced: prefer specific length
sentences = model.split_length_constrained(
    text="Your long document...",
    prior="gaussian",
    target_length=100,
    sigma=20
)

# Custom prior
def my_prior(length):
    if length < 50: return 0.0
    elif length < 150: return 1.0
    else: return 0.1

sentences = model.split_length_constrained(
    text="Your long document...",
    prior=my_prior,
    algorithm="viterbi"
)
```

**Key features:**
- Simple for common cases (just set `max_length`)
- Flexible for advanced users (custom priors)
- Backward compatible (doesn't break existing code)

---

## 15. Next Steps (After This Meeting)

### Immediate (This Week)
1. Get supervisor feedback on plan
2. Adjust scope based on feedback
3. Set up development environment
4. Create GitHub branch for thesis work

### Week 1
1. Implement greedy algorithm
2. Implement 2 basic priors (uniform, gaussian)
3. Write unit tests
4. Demo working prototype to supervisor

### Week 2
1. Implement Viterbi algorithm
2. Add remaining priors
3. Integrate with SaT class
4. Schedule midpoint review

---

## 16. References

### Key Papers to Read
1. **WtPSplit Paper** (Minixhofer et al., ACL 2023)
2. **Viterbi Algorithm** (Viterbi, 1967)
3. **RAG** (Lewis et al., NeurIPS 2020)
4. **Text Segmentation Survey** (Hearst, 1997; Beeferman et al., 1999)

### Related Work
- TextTiling (Hearst, 1997)
- Constrained decoding in NMT
- Hidden Markov Models for segmentation
- Sentence compression with DP

---

## 17. Contact & Resources

**Weekly meetings:** [Day/Time TBD]
**Communication:** Email / Slack / Teams
**Code repository:** `github.com/[your-username]/wtpsplit` (fork)
**Main WtPSplit repo:** `github.com/segment-any-text/wtpsplit`
**Documentation:** `docs.claude.com` (for thesis writing tips)

---

## Appendix: Visual Diagrams

### Algorithm Comparison

```
Greedy Algorithm:
  Start → [Check all next positions] → Pick best → Repeat
  Time: O(n·k)
  Quality: ~95% of optimal

Viterbi Algorithm:
  Forward pass → [DP table] → Backward pass → Optimal path
  Time: O(n·k²) or O(n·k)
  Quality: 100% optimal

Choice: Viterbi for thesis (quality matters most)
```

### Data Flow

```
Input Text
    ↓
SaT Model
    ↓
Character Probabilities [0.1, 0.3, 0.9, ...]
    ↓
Prior Function + Probabilities
    ↓
Constrained Optimization (Greedy/Viterbi)
    ↓
Selected Boundaries [10, 25, 41, ...]
    ↓
Segmented Sentences ["Sent 1", "Sent 2", ...]
```

### Prior Distributions Visualization

```
Uniform (max=100):
    │██████████████████████
    │                      └─── 0
    0                      100

Gaussian (target=100, σ=20):
         ╱──╲
    │   ╱    ╲
    │  ╱      ╲___
    │ ╱           ╲___
    0  50   100  150  200

Clipped Polynomial (target=100):
         ╱──╲
    │   ╱    ╲
    │  ╱      ╲
    │ ╱        └────
    0  50  100 150  200
```

---

**End of Meeting Summary**

Remember to bring:
- [ ] This summary
- [ ] Full plan document
- [ ] Questions list
- [ ] Laptop (for demo if requested)
