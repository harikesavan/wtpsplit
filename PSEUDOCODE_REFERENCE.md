# Pseudocode Reference for Length-Constrained Segmentation

This document provides simplified pseudocode for understanding the core algorithms.

## Core Idea

Given:
- `probs[]`: Array of boundary probabilities from the model
- `prior(length)`: Function that returns preference for a given segment length

Find: Best positions to place boundaries that maximize:
```
posterior = prior(segment_length) × prob(boundary_position)
```

---

## Algorithm 1: Greedy Search (Fast, Locally Optimal)

```
FUNCTION greedy_segment(probs, prior, min_length, max_length):
    boundaries = []
    current_start = 0

    WHILE current_start < length(probs):
        best_score = -infinity
        best_position = null

        // Try all valid next positions
        FOR position FROM (current_start + min_length) TO (current_start + max_length):
            segment_length = position - current_start

            // Calculate posterior (in log space)
            score = log(prior(segment_length)) + log(probs[position])

            IF score > best_score:
                best_score = score
                best_position = position
            END IF
        END FOR

        // Move to next segment
        boundaries.append(best_position)
        current_start = best_position
    END WHILE

    RETURN boundaries
END FUNCTION
```

**Time Complexity:** O(n × k) where n = text length, k = max_length

**Example:**
```
Input:
  probs = [0.1, 0.3, 0.9, 0.2, 0.8, 0.1]
  prior = uniform(max_length=3)
  min_length = 1

Step 1: Start at position 0
  Try position 1: length=1, score = log(1.0) + log(0.1) = -2.3
  Try position 2: length=2, score = log(1.0) + log(0.3) = -1.2
  Try position 3: length=3, score = log(1.0) + log(0.9) = -0.1  ← BEST
  → Select boundary at position 3

Step 2: Start at position 3
  Try position 4: length=1, score = log(1.0) + log(0.2) = -1.6
  Try position 5: length=2, score = log(1.0) + log(0.8) = -0.2  ← BEST
  Try position 6: length=3, score = log(1.0) + log(0.1) = -2.3
  → Select boundary at position 5

Result: boundaries = [3, 5]
```

---

## Algorithm 2: Viterbi (Slower, Globally Optimal)

```
FUNCTION viterbi_segment(probs, prior, min_length, max_length):
    n = length(probs)

    // DP tables
    delta[0..n] = -infinity    // Best score to reach each position
    backpointer[0..n] = 0      // Previous boundary for best path
    delta[0] = 0               // Start with score 1 (log = 0)

    // FORWARD PASS: Compute best score for each position
    FOR t FROM 1 TO n:
        // Try all valid previous positions
        FOR s FROM max(0, t-max_length) TO (t-min_length):
            IF delta[s] == -infinity:
                CONTINUE  // Can't reach position s
            END IF

            segment_length = t - s

            // Calculate score
            score = delta[s] + log(prior(segment_length)) + log(probs[t])

            // Update if better
            IF score > delta[t]:
                delta[t] = score
                backpointer[t] = s
            END IF
        END FOR
    END FOR

    // BACKWARD PASS: Reconstruct optimal path
    boundaries = []
    current = n

    WHILE current > 0:
        previous = backpointer[current]
        IF previous > 0:
            boundaries.prepend(previous)
        END IF
        current = previous
    END WHILE

    RETURN boundaries
END FUNCTION
```

**Time Complexity:** O(n × k²) naive, O(n × k) optimized

**Example:**
```
Input:
  probs = [0.1, 0.3, 0.9, 0.2, 0.8, 0.1]
  prior = uniform(max_length=3)
  min_length = 1

Forward Pass:
  delta[0] = 0.0

  Position 1:
    From 0: score = 0 + log(1.0) + log(0.1) = -2.3
    → delta[1] = -2.3, backpointer[1] = 0

  Position 2:
    From 0: score = 0 + log(1.0) + log(0.3) = -1.2  ← BEST
    From 1: score = -2.3 + log(1.0) + log(0.3) = -3.5
    → delta[2] = -1.2, backpointer[2] = 0

  Position 3:
    From 0: score = 0 + log(1.0) + log(0.9) = -0.1  ← BEST
    From 1: score = -2.3 + log(1.0) + log(0.9) = -2.4
    From 2: score = -1.2 + log(1.0) + log(0.9) = -1.3
    → delta[3] = -0.1, backpointer[3] = 0

  ... continue for all positions ...

  Position 6:
    Best path ends with delta[6] = -0.4
    backpointer[6] = 3

Backward Pass:
  current = 6 → backpointer[6] = 3 → add 3 to boundaries
  current = 3 → backpointer[3] = 0 → stop

Result: boundaries = [3]
```

---

## Prior Functions

### 1. Uniform Prior
```
FUNCTION uniform_prior(length, max_length):
    IF length <= max_length:
        RETURN 1.0
    ELSE:
        RETURN 0.0
    END IF
END FUNCTION
```

**Use case:** Hard constraint, no segments > max_length

### 2. Gaussian Prior
```
FUNCTION gaussian_prior(length, target, sigma):
    RETURN exp(-0.5 × ((length - target) / sigma)²)
END FUNCTION
```

**Use case:** Prefer specific length, allow variation

### 3. Clipped Polynomial Prior
```
FUNCTION clipped_polynomial_prior(length, target, coefficient):
    value = 1.0 - coefficient × (length - target)²
    RETURN max(value, 0.0)
END FUNCTION
```

**Use case:** Sharp preference with hard cutoff

### 4. Exponential Prior
```
FUNCTION exponential_prior(length, max_length, decay_rate):
    IF length <= max_length:
        RETURN 1.0
    ELSE:
        RETURN exp(-decay_rate × (length - max_length))
    END IF
END FUNCTION
```

**Use case:** Soft limit with exponential penalty

### 5. Piecewise Linear Prior
```
FUNCTION piecewise_linear_prior(length, min_len, target, max_len):
    IF length < min_len:
        RETURN 0.0
    ELSE IF length < target:
        RETURN (length - min_len) / (target - min_len)  // Ramp up
    ELSE IF length <= max_len:
        RETURN 1.0                                       // Plateau
    ELSE:
        decay_end = 2 × max_len - target
        IF length >= decay_end:
            RETURN 0.0
        ELSE:
            RETURN 1.0 - (length - max_len) / (max_len - target)  // Ramp down
        END IF
    END IF
END FUNCTION
```

**Use case:** Explicit min/optimal/max with smooth transitions

---

## Utility Functions

### Safe Log (Numerical Stability)
```
FUNCTION safe_log(x, epsilon=1e-10):
    RETURN log(max(x, epsilon))
END FUNCTION
```

Prevents log(0) = -infinity from causing numerical issues.

### Algorithm Selection (Factory Pattern)
```
FUNCTION select_algorithm(name):
    IF name == "greedy":
        RETURN greedy_segment
    ELSE IF name == "viterbi":
        RETURN viterbi_segment
    ELSE:
        ERROR "Unknown algorithm"
    END IF
END FUNCTION
```

### Convert Boundaries to Text Segments
```
FUNCTION indices_to_segments(text, boundaries):
    segments = []
    start = 0

    FOR boundary IN boundaries:
        segment = text[start:boundary]
        segments.append(segment)
        start = boundary
    END FOR

    // Add final segment
    segments.append(text[start:end])

    RETURN segments
END FUNCTION
```

---

## Complete Usage Example

```
// 1. Load model and get probabilities
model = load_model("sat-3l-sm")
text = "Your long document here..."
probs = model.predict_proba(text)  // Get boundary probabilities

// 2. Create prior function
prior = create_prior_function("uniform", {max_length: 100})

// 3. Find optimal boundaries
boundaries = viterbi_segment(
    probs=probs,
    prior=prior,
    min_length=10,
    max_length=100
)

// 4. Convert to text chunks
chunks = indices_to_segments(text, boundaries)

// 5. Print results
FOR i, chunk IN enumerate(chunks):
    PRINT "Chunk", i+1, "(", length(chunk), "chars):", chunk
END FOR
```

---

## Complexity Comparison

| Algorithm | Time | Space | Quality | Use Case |
|-----------|------|-------|---------|----------|
| **Greedy** | O(n·k) | O(n) | 90-95% | Real-time, long docs |
| **Viterbi** | O(n·k) | O(n) | 100% | Best quality, offline |

Where:
- n = text length (number of characters)
- k = max_length (search window size)

**Note:** With optimization, Viterbi can achieve O(n·k) time complexity (same as greedy).

---

## Key Differences: Greedy vs Viterbi

### Greedy
- Makes local decisions
- Picks best immediate choice at each step
- Cannot "undo" previous decisions
- Faster to implement and debug
- May miss global optimum

### Viterbi
- Considers all possible paths
- Uses dynamic programming
- Guaranteed global optimum
- Slightly more complex
- Slightly higher memory usage

### When They Differ

```
Example where greedy fails:

probs = [0.9, 0.5, 0.5, 0.9]
prior = "prefer length 2"

Greedy:
  Step 1: Sees 0.9 at position 1, picks it (length=1, not preferred but high prob)
  Step 2: Forced to make suboptimal choice from position 1
  Result: [1, 3] → lengths [1, 2]

Viterbi:
  Considers all paths:
    Path [1, 3]: score = prior(1)×0.9 + prior(2)×0.9 = 0.1×0.9 + 1.0×0.9 = 0.99
    Path [2]:    score = prior(2)×0.5 + prior(2)×0.9 = 1.0×0.5 + 1.0×0.9 = 1.40 ← BEST
  Result: [2] → lengths [2, 2]
```

---

## Implementation Checklist

When implementing, make sure to:

- [ ] Use log-space computation to avoid underflow
- [ ] Handle edge cases (empty input, very short text)
- [ ] Validate prior parameters
- [ ] Add fallback for impossible constraints
- [ ] Test with various prior types
- [ ] Benchmark performance on long documents
- [ ] Add verbose mode for debugging
- [ ] Document all functions with examples
- [ ] Write unit tests for each algorithm
- [ ] Compare greedy vs Viterbi on test cases
