# Algorithm Pseudocode and Complexity Analysis

## 1. Greedy Algorithm

### Pseudocode
```python
def greedy_constrained_segmentation(probs, prior_fn, min_length=1):
    """
    Select boundaries greedily based on posterior = prior × probability

    Args:
        probs: array of shape (n,) with boundary probabilities
        prior_fn: function(length) -> prior_probability
        min_length: minimum segment length

    Returns:
        boundaries: list of selected boundary indices
    """
    n = len(probs)
    boundaries = []
    current_start = 0

    while current_start < n:
        best_score = -inf
        best_pos = None

        # Try all valid next positions
        for pos in range(current_start + min_length, n + 1):
            length = pos - current_start
            posterior = log(prior_fn(length)) + log(probs[pos - 1])

            if posterior > best_score:
                best_score = posterior
                best_pos = pos

        # If no valid position found (prior returns 0 everywhere),
        # fall back to maximum probability
        if best_pos is None:
            best_pos = argmax(probs[current_start + min_length:]) + current_start + min_length

        boundaries.append(best_pos)
        current_start = best_pos

    return boundaries[:-1]  # Exclude final position
```

### Example Walkthrough

**Input:**
```
text = "This is a test"
positions:  T  h  i  s     i  s     a     t  e  s  t
indices:    0  1  2  3  4  5  6  7  8  9  10 11 12 13
probs:     .1 .05 .1 .3 .05 .1 .7 .05 .1 .05 .1 .9 .05
```

**Prior:** `f(l) = max(1 - 0.5 * (l - 3)**2, 0)`
- Length 1: 0.0
- Length 2: 0.5
- Length 3: 1.0
- Length 4: 0.5
- Length 5+: 0.0

**Step-by-step:**
```
Start: pos=0

Iteration 1 (find first boundary):
  Try pos=1 (length=1): posterior = log(0.0) + log(0.05) = -inf ❌
  Try pos=2 (length=2): posterior = log(0.5) + log(0.1) = -0.69 + (-2.30) = -2.99
  Try pos=3 (length=3): posterior = log(1.0) + log(0.1) = 0 + (-2.30) = -2.30
  Try pos=4 (length=4): posterior = log(0.5) + log(0.3) = -0.69 + (-1.20) = -1.89 ✓ BEST

  → Select boundary at pos=4 ("This ")

Start: pos=4

Iteration 2 (find second boundary):
  Try pos=5 (length=1): posterior = -inf ❌
  Try pos=6 (length=2): posterior = log(0.5) + log(0.1) = -2.99
  Try pos=7 (length=3): posterior = log(1.0) + log(0.7) = 0 + (-0.36) = -0.36 ✓ BEST
  Try pos=8 (length=4): posterior = log(0.5) + log(0.05) = -0.69 + (-3.00) = -3.69

  → Select boundary at pos=7 ("is a ")

Start: pos=7

Iteration 3 (find third boundary):
  Try pos=8 (length=1): posterior = -inf ❌
  Try pos=9 (length=2): posterior = log(0.5) + log(0.05) = -3.69
  Try pos=10 (length=3): posterior = log(1.0) + log(0.1) = -2.30
  Try pos=11 (length=4): posterior = log(0.5) + log(0.9) = -0.69 + (-0.11) = -0.80 ✓ BEST

  → Select boundary at pos=11 ("test")

Final segmentation:
  - [0:4] "This "
  - [4:7] "is a "
  - [7:11] "test"
```

### Complexity Analysis

**Time Complexity:** O(n · k)
- n = text length
- k = maximum segment length (search window)
- For each of n positions, we try up to k candidates

**Space Complexity:** O(n)
- Store probabilities array
- Store boundaries list (at most n elements)

**Optimizations:**
1. Early stopping if prior returns 0 beyond certain length
2. Vectorized posterior computation for all candidates at once
3. Use numba JIT compilation for the inner loop

---

## 2. Viterbi Algorithm

### Pseudocode
```python
def viterbi_constrained_segmentation(probs, prior_fn, min_length=1, max_length=None):
    """
    Find globally optimal boundary selection using dynamic programming

    Args:
        probs: array of shape (n,) with boundary probabilities
        prior_fn: function(length) -> prior_probability
        min_length: minimum segment length
        max_length: maximum segment length (None = no limit)

    Returns:
        boundaries: list of selected boundary indices
    """
    n = len(probs)

    # Dynamic programming tables (log-space for numerical stability)
    delta = np.full(n + 1, -inf)  # Best score to reach each position
    backpointer = np.zeros(n + 1, dtype=int)  # Previous boundary for best path

    # Base case: start at position 0 with score 1 (log = 0)
    delta[0] = 0.0

    # Forward pass: compute best score for each position
    for t in range(1, n + 1):
        # Try all valid previous boundary positions
        min_prev = max(0, t - max_length) if max_length else 0
        max_prev = t - min_length

        for s in range(min_prev, max_prev + 1):
            # Calculate score for placing boundary at position t,
            # with previous boundary at position s
            segment_length = t - s

            # Posterior = prior(length) × prob(boundary) in log-space
            prior_score = log(prior_fn(segment_length))
            prob_score = log(probs[t - 1])  # Probability at position t
            score = delta[s] + prior_score + prob_score

            # Update if this is the best path to position t
            if score > delta[t]:
                delta[t] = score
                backpointer[t] = s

    # Backward pass: reconstruct optimal path
    boundaries = []
    current = n

    while current > 0:
        prev = backpointer[current]
        if prev > 0:  # Don't include starting position
            boundaries.append(prev)
        current = prev

    boundaries.reverse()
    return boundaries
```

### Example Walkthrough

**Same input as greedy example:**
```
positions:  0  1  2  3  4  5  6  7  8  9 10 11 12 13
probs:     .1 .05 .1 .3 .05 .1 .7 .05 .1 .05 .1 .9 .05
prior:     length 3 preferred (gaussian-like)
```

**Forward Pass (compute delta):**

```
Initialize:
  delta[0] = 0.0 (log of probability 1)
  delta[1..n] = -inf

Position t=1:
  From s=0 (length=1): delta[1] = 0.0 + log(prior(1)) + log(0.1) = -inf (prior=0)

Position t=2:
  From s=0 (length=2): delta[2] = 0.0 + log(0.5) + log(0.05) = -3.69
  From s=1 (length=1): delta[2] = -inf (can't reach s=1)
  → delta[2] = -3.69, backpointer[2] = 0

Position t=3:
  From s=0 (length=3): delta[3] = 0.0 + log(1.0) + log(0.1) = -2.30
  From s=2 (length=1): -inf
  → delta[3] = -2.30, backpointer[3] = 0

Position t=4:
  From s=0 (length=4): delta[4] = 0.0 + log(0.5) + log(0.3) = -1.89 ✓
  From s=2 (length=2): delta[4] = -3.69 + log(0.5) + log(0.3) = -4.58
  From s=3 (length=1): -inf
  → delta[4] = -1.89, backpointer[4] = 0

Position t=7:
  From s=4 (length=3): delta[7] = -1.89 + log(1.0) + log(0.7) = -2.25 ✓
  From s=5 (length=2): delta[7] = ... (compute all candidates)
  From s=6 (length=1): -inf
  → delta[7] = -2.25, backpointer[7] = 4

Position t=11:
  From s=7 (length=4): delta[11] = -2.25 + log(0.5) + log(0.9) = -3.05 ✓
  From s=8 (length=3): delta[11] = ...
  ...
  → delta[11] = -3.05, backpointer[11] = 7

... (continue for all positions)
```

**Backward Pass (reconstruct path):**
```
Start at position 13 (end)
  backpointer[13] = 11 → boundary at 11
  backpointer[11] = 7  → boundary at 7
  backpointer[7] = 4   → boundary at 4
  backpointer[4] = 0   → reached start

Boundaries: [4, 7, 11]
Same as greedy in this case!
```

### Complexity Analysis

**Time Complexity:** O(n · k²) naive, O(n · k) optimized
- n = text length
- k = max_length (search window)
- For each position t (n iterations):
  - Try k possible previous positions
  - Each requires O(1) computation
- **Optimization:** If max_length is fixed, it becomes O(n · k)

**Space Complexity:** O(n)
- delta array: O(n)
- backpointer array: O(n)
- boundaries list: O(n) worst case

**When Viterbi differs from Greedy:**
```
Example where greedy fails:

positions:  0  1  2  3  4  5  6
probs:     .9 .8 .2 .3 .9 .9 .9
prior:     Prefer length 3

Greedy:
  - Selects position 1 (high prob .9, but length=1 not preferred)
  - Then forced to make suboptimal choices

Viterbi:
  - Considers full path
  - Might skip position 1 to get better global score
  - Optimal: boundaries at [3, 6] instead of [1, 4, 6]
```

---

## 3. Numerical Stability Considerations

### Log-Space Computation

**Why:** Multiplying many probabilities (0 < p < 1) causes underflow

```python
# Bad: Direct multiplication
score = prior(length) * prob[pos]  # Can underflow!

# Good: Log-space
log_score = log(prior(length)) + log(prob[pos])
```

**Handling zero probabilities:**
```python
def safe_log(x, epsilon=1e-10):
    """Avoid log(0) = -inf"""
    return np.log(np.maximum(x, epsilon))
```

### Numerical Precision

**Problem:** Deep recursion in Viterbi can accumulate errors

**Solution:** Use `float64` for delta array, even if input is `float16`

```python
delta = np.zeros(n + 1, dtype=np.float64)  # Higher precision
```

---

## 4. Edge Cases & Handling

### Case 1: No Valid Segmentation
**Problem:** Prior is too restrictive (e.g., max_length=5 but no good boundaries in first 10 chars)

**Solution:** Fall back to best available boundary
```python
if best_score == -inf:
    # No valid boundary found, take highest probability
    best_pos = argmax(probs[current:current+max_length]) + current
```

### Case 2: Very Long Documents
**Problem:** O(n · k) becomes slow for n > 100,000

**Solution 1:** Process in chunks
```python
def chunk_and_process(text, chunk_size=10000):
    # Split text into chunks with overlap
    # Process each chunk independently
    # Merge boundaries
```

**Solution 2:** Beam search (approximate Viterbi)
```python
# Keep only top-B candidates at each position
# Reduces complexity to O(n · B) where B << k
```

### Case 3: Empty or Very Short Text
```python
if len(probs) < min_length:
    return []  # No segmentation possible

if len(probs) < min_length * 2:
    return [len(probs)]  # Single segment
```

---

## 5. Implementation Checklist

### Core Functions
- [ ] `greedy_segment(probs, prior_fn, min_length, max_length)`
- [ ] `viterbi_segment(probs, prior_fn, min_length, max_length)`
- [ ] `safe_log(x, epsilon)` - numerical stability
- [ ] `select_algorithm(algorithm_name)` - factory pattern

### Prior Functions
- [ ] `uniform_prior(length, max_length)`
- [ ] `gaussian_prior(length, target, sigma)`
- [ ] `clipped_polynomial_prior(length, target, coef)`
- [ ] `exponential_prior(length, max_length, decay)`
- [ ] `piecewise_linear_prior(length, min, target, max)`

### Integration
- [ ] `SaT.split_length_constrained()` - main API
- [ ] Handle character vs. token-based lengths
- [ ] Batch processing support
- [ ] Progress bar for long documents

### Testing
- [ ] Unit tests for each algorithm
- [ ] Edge case tests (empty, very short, very long)
- [ ] Numerical stability tests
- [ ] Comparison tests (greedy vs viterbi)
- [ ] Performance benchmarks

---

## 6. Performance Optimization Strategies

### Strategy 1: Numba JIT Compilation
```python
from numba import jit

@jit(nopython=True)
def viterbi_core(probs, prior_values, min_length, max_length):
    # Pure numerical computation
    # 10-50x speedup possible
    ...
```

### Strategy 2: Vectorization
```python
# Instead of loop:
for s in range(min_prev, max_prev):
    score = delta[s] + prior[t-s] + log_prob[t]

# Vectorize:
candidates = np.arange(min_prev, max_prev)
scores = delta[candidates] + prior[t - candidates] + log_prob[t]
best_idx = np.argmax(scores)
```

### Strategy 3: Early Stopping
```python
# If prior is zero beyond max_length, don't search further
if max_length and segment_length > max_length:
    break
```

### Strategy 4: Caching Prior Values
```python
# Precompute prior for all possible lengths
prior_cache = np.array([prior_fn(i) for i in range(max_length + 1)])

# Then just index:
prior_score = log(prior_cache[segment_length])
```

---

## 7. Testing Strategy

### Unit Tests

```python
def test_greedy_simple():
    probs = np.array([0.1, 0.9, 0.1, 0.9])
    prior = lambda l: 1.0 if l <= 2 else 0.0

    boundaries = greedy_segment(probs, prior, min_length=1)
    assert boundaries == [1, 3] or boundaries == [2, 4]

def test_viterbi_optimal():
    # Case where greedy fails but viterbi succeeds
    probs = np.array([0.9, 0.5, 0.5, 0.9])
    prior = lambda l: 1.0 if l == 2 else 0.1

    boundaries = viterbi_segment(probs, prior, min_length=1)
    # Should prefer [2, 4] over [1, 3] due to prior
    assert boundaries == [2, 4]

def test_numerical_stability():
    # Many small probabilities
    probs = np.full(1000, 0.001)
    prior = lambda l: 1.0 if l <= 100 else 0.0

    boundaries = viterbi_segment(probs, prior, min_length=10)
    # Should not overflow/underflow
    assert len(boundaries) > 0
    assert all(b < 1000 for b in boundaries)
```

### Integration Tests

```python
def test_sat_integration():
    model = SaT("sat-3l-sm")
    text = "This is a test. " * 100  # Long text

    sentences = model.split_length_constrained(
        text,
        prior="uniform",
        max_length=50,
        algorithm="viterbi"
    )

    # All sentences should respect constraint
    assert all(len(s) <= 50 for s in sentences)

    # Should return reasonable number of segments
    assert 50 < len(sentences) < 200
```

---

## 8. Comparison: Greedy vs Viterbi

| Aspect | Greedy | Viterbi |
|--------|--------|---------|
| **Optimality** | Local optimum | Global optimum |
| **Time Complexity** | O(n · k) | O(n · k²) or O(n · k) |
| **Space Complexity** | O(n) | O(n) |
| **Implementation** | Simple (~30 lines) | Moderate (~60 lines) |
| **Debugging** | Easy | Moderate |
| **Typical Quality** | 90-95% of optimal | 100% optimal |
| **Use Case** | Real-time, long documents | High-quality, offline |

**Recommendation:**
- Default: Viterbi (quality matters most for research)
- Option: Greedy for production/real-time systems
