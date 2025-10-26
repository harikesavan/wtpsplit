# Implementation Roadmap: Step-by-Step Guide

This document provides a detailed, file-by-file implementation roadmap for the length-constrained segmentation feature.

---

## Phase 1: Core Algorithms (Week 1-2)

### Step 1.1: Create Prior Distributions Module

**File:** `wtpsplit/utils/priors.py` (NEW)

**Purpose:** Define all prior distribution functions

**Implementation:**
```python
"""
Prior distribution functions for length-constrained segmentation.

Each prior function takes a segment length and returns a probability.
"""

import numpy as np
from typing import Callable, Dict, Any


class PriorRegistry:
    """Registry for prior distribution functions."""

    _priors: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a prior function."""
        def decorator(func):
            cls._priors[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a prior function by name."""
        if name not in cls._priors:
            raise ValueError(f"Unknown prior: {name}. Available: {list(cls._priors.keys())}")
        return cls._priors[name]

    @classmethod
    def list_priors(cls):
        """List all available priors."""
        return list(cls._priors.keys())


@PriorRegistry.register("uniform")
def uniform_prior(length: int, max_length: int, **kwargs) -> float:
    """
    Uniform prior: equal probability up to max_length, then zero.

    Args:
        length: Segment length
        max_length: Maximum allowed length

    Returns:
        1.0 if length <= max_length, else 0.0
    """
    return 1.0 if length <= max_length else 0.0


@PriorRegistry.register("gaussian")
def gaussian_prior(length: int, target_length: int, sigma: float = 20.0, **kwargs) -> float:
    """
    Gaussian prior: prefer target_length with normal distribution.

    Args:
        length: Segment length
        target_length: Preferred length
        sigma: Standard deviation (controls spread)

    Returns:
        Gaussian probability
    """
    return np.exp(-0.5 * ((length - target_length) / sigma) ** 2)


@PriorRegistry.register("clipped_polynomial")
def clipped_polynomial_prior(
    length: int,
    target_length: int,
    coefficient: float = 0.01,
    **kwargs
) -> float:
    """
    Clipped polynomial: quadratic decay with hard cutoff.

    Args:
        length: Segment length
        target_length: Preferred length
        coefficient: Decay rate

    Returns:
        max(1 - coefficient * (length - target_length)^2, 0)
    """
    return max(1.0 - coefficient * (length - target_length) ** 2, 0.0)


@PriorRegistry.register("exponential")
def exponential_prior(
    length: int,
    max_length: int,
    decay_rate: float = 0.1,
    **kwargs
) -> float:
    """
    Exponential prior: constant up to max_length, then exponential decay.

    Args:
        length: Segment length
        max_length: Start of decay
        decay_rate: Decay rate

    Returns:
        Exponential probability
    """
    if length <= max_length:
        return 1.0
    else:
        return np.exp(-decay_rate * (length - max_length))


@PriorRegistry.register("piecewise_linear")
def piecewise_linear_prior(
    length: int,
    min_length: int,
    target_length: int,
    max_length: int,
    **kwargs
) -> float:
    """
    Piecewise linear prior: ramp up, plateau, ramp down.

    Args:
        length: Segment length
        min_length: Minimum length (0 probability below)
        target_length: Start of plateau
        max_length: End of plateau (0 probability above)

    Returns:
        Piecewise linear probability
    """
    if length < min_length:
        return 0.0
    elif length < target_length:
        return (length - min_length) / (target_length - min_length)
    elif length <= max_length:
        return 1.0
    else:
        return max(1.0 - (length - max_length) / (max_length - target_length), 0.0)


def create_prior_function(
    prior: str | Callable,
    prior_params: Dict[str, Any] = None
) -> Callable[[int], float]:
    """
    Create a prior function from name or callable.

    Args:
        prior: Prior name (string) or custom callable
        prior_params: Parameters for the prior function

    Returns:
        Prior function that takes length and returns probability
    """
    prior_params = prior_params or {}

    if callable(prior):
        # Custom prior function
        return prior
    elif isinstance(prior, str):
        # Named prior from registry
        prior_func = PriorRegistry.get(prior)

        # Create closure with parameters
        def parameterized_prior(length: int) -> float:
            return prior_func(length, **prior_params)

        return parameterized_prior
    else:
        raise TypeError(f"Prior must be string or callable, got {type(prior)}")
```

**Tests:** `tests/test_priors.py` (NEW)
```python
import pytest
import numpy as np
from wtpsplit.utils.priors import (
    uniform_prior,
    gaussian_prior,
    clipped_polynomial_prior,
    exponential_prior,
    piecewise_linear_prior,
    create_prior_function,
    PriorRegistry
)


def test_uniform_prior():
    # Within limit
    assert uniform_prior(50, max_length=100) == 1.0
    assert uniform_prior(100, max_length=100) == 1.0

    # Beyond limit
    assert uniform_prior(101, max_length=100) == 0.0
    assert uniform_prior(200, max_length=100) == 0.0


def test_gaussian_prior():
    # Peak at target
    target = 100
    sigma = 20
    assert gaussian_prior(target, target, sigma) == 1.0

    # Decreases away from target
    assert 0.5 < gaussian_prior(80, target, sigma) < 1.0
    assert 0.0 < gaussian_prior(150, target, sigma) < 0.5


def test_clipped_polynomial_prior():
    target = 100
    # At target
    assert clipped_polynomial_prior(target, target, coefficient=0.01) == 1.0

    # Decays
    assert 0.0 < clipped_polynomial_prior(90, target, coefficient=0.01) < 1.0

    # Clips to zero
    far_value = clipped_polynomial_prior(10, target, coefficient=0.01)
    assert far_value == 0.0


def test_create_prior_function():
    # From string
    prior_fn = create_prior_function("uniform", {"max_length": 100})
    assert prior_fn(50) == 1.0
    assert prior_fn(150) == 0.0

    # From callable
    custom_prior = lambda length: 0.5
    prior_fn = create_prior_function(custom_prior)
    assert prior_fn(100) == 0.5


def test_prior_registry():
    # List priors
    priors = PriorRegistry.list_priors()
    assert "uniform" in priors
    assert "gaussian" in priors

    # Get prior
    prior_func = PriorRegistry.get("uniform")
    assert prior_func(50, max_length=100) == 1.0

    # Unknown prior
    with pytest.raises(ValueError):
        PriorRegistry.get("nonexistent")
```

---

### Step 1.2: Create Constraint Algorithms Module

**File:** `wtpsplit/utils/constraints.py` (NEW)

**Purpose:** Implement greedy and Viterbi algorithms

**Implementation:**
```python
"""
Constraint optimization algorithms for length-constrained segmentation.
"""

import numpy as np
from typing import Callable, List, Optional
import warnings


def safe_log(x: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Numerically stable logarithm.

    Args:
        x: Input array
        epsilon: Small value to avoid log(0)

    Returns:
        log(max(x, epsilon))
    """
    return np.log(np.maximum(x, epsilon))


def greedy_constrained_segmentation(
    probs: np.ndarray,
    prior_fn: Callable[[int], float],
    min_length: int = 1,
    max_length: Optional[int] = None,
    verbose: bool = False
) -> List[int]:
    """
    Greedy algorithm for length-constrained segmentation.

    Iteratively selects the next boundary that maximizes:
        posterior = prior(length) × prob(boundary)

    Args:
        probs: Array of boundary probabilities (length n)
        prior_fn: Function mapping length -> prior probability
        min_length: Minimum segment length
        max_length: Maximum segment length (None = no limit)
        verbose: Print debug information

    Returns:
        List of boundary indices (0-indexed, excluding start/end)
    """
    n = len(probs)
    if n == 0:
        return []

    boundaries = []
    current_start = 0

    while current_start < n:
        best_score = -np.inf
        best_pos = None

        # Determine search range
        search_start = current_start + min_length
        search_end = min(current_start + max_length, n) if max_length else n

        if search_start >= n:
            # No valid boundary (remaining text too short)
            break

        # Try all valid positions
        for pos in range(search_start, search_end + 1):
            length = pos - current_start

            # Calculate posterior (in log-space for stability)
            prior_prob = prior_fn(length)
            if prior_prob <= 0:
                continue  # Skip if prior is zero

            # Log posterior = log(prior) + log(prob)
            log_posterior = np.log(prior_prob) + safe_log(np.array([probs[pos - 1]]))[0]

            if log_posterior > best_score:
                best_score = log_posterior
                best_pos = pos

        # Fallback if no valid position found
        if best_pos is None:
            if verbose:
                warnings.warn(
                    f"No valid boundary found from position {current_start}. "
                    "Using highest probability boundary as fallback."
                )
            # Take highest probability in range
            search_probs = probs[search_start:search_end]
            if len(search_probs) > 0:
                best_pos = search_start + np.argmax(search_probs)
            else:
                break  # Can't proceed

        if verbose:
            length = best_pos - current_start
            print(f"Selected boundary at {best_pos} (length={length}, score={best_score:.3f})")

        boundaries.append(best_pos)
        current_start = best_pos

    # Remove final boundary if it's exactly at the end
    if boundaries and boundaries[-1] >= n:
        boundaries.pop()

    return boundaries


def viterbi_constrained_segmentation(
    probs: np.ndarray,
    prior_fn: Callable[[int], float],
    min_length: int = 1,
    max_length: Optional[int] = None,
    verbose: bool = False
) -> List[int]:
    """
    Viterbi algorithm for length-constrained segmentation.

    Finds the globally optimal boundary selection using dynamic programming.

    Args:
        probs: Array of boundary probabilities (length n)
        prior_fn: Function mapping length -> prior probability
        min_length: Minimum segment length
        max_length: Maximum segment length (None = no limit)
        verbose: Print debug information

    Returns:
        List of boundary indices (0-indexed, excluding start/end)
    """
    n = len(probs)
    if n == 0:
        return []

    # DP tables (use float64 for numerical precision)
    delta = np.full(n + 1, -np.inf, dtype=np.float64)
    backpointer = np.zeros(n + 1, dtype=np.int32)

    # Base case: start at position 0 with probability 1 (log = 0)
    delta[0] = 0.0

    # Forward pass: compute best score for each position
    for t in range(1, n + 1):
        # Determine valid previous positions
        if max_length:
            min_prev = max(0, t - max_length)
        else:
            min_prev = 0

        max_prev = t - min_length

        if max_prev < 0:
            continue  # No valid previous position

        # Try all valid previous positions
        for s in range(min_prev, max_prev + 1):
            if delta[s] == -np.inf:
                continue  # Can't reach position s

            length = t - s

            # Calculate score
            prior_prob = prior_fn(length)
            if prior_prob <= 0:
                continue

            log_prior = np.log(prior_prob)
            log_prob = safe_log(np.array([probs[t - 1]]))[0]
            score = delta[s] + log_prior + log_prob

            # Update if better
            if score > delta[t]:
                delta[t] = score
                backpointer[t] = s

        if verbose and t % 100 == 0:
            print(f"Processed position {t}/{n}, best score: {delta[t]:.3f}")

    # Check if we reached the end
    if delta[n] == -np.inf:
        warnings.warn(
            "Could not find valid segmentation with given constraints. "
            "Falling back to greedy algorithm."
        )
        return greedy_constrained_segmentation(probs, prior_fn, min_length, max_length, verbose)

    # Backward pass: reconstruct optimal path
    boundaries = []
    current = n

    while current > 0:
        prev = backpointer[current]
        if prev > 0:  # Don't include starting position
            boundaries.append(prev)
        current = prev

    boundaries.reverse()

    # Remove final boundary if it's at the end
    if boundaries and boundaries[-1] >= n:
        boundaries.pop()

    return boundaries


def constrained_segmentation(
    probs: np.ndarray,
    prior_fn: Callable[[int], float],
    algorithm: str = "viterbi",
    min_length: int = 1,
    max_length: Optional[int] = None,
    verbose: bool = False
) -> List[int]:
    """
    Main entry point for constrained segmentation.

    Args:
        probs: Array of boundary probabilities
        prior_fn: Prior distribution function
        algorithm: "greedy" or "viterbi"
        min_length: Minimum segment length
        max_length: Maximum segment length
        verbose: Print debug information

    Returns:
        List of boundary indices
    """
    if algorithm == "greedy":
        return greedy_constrained_segmentation(
            probs, prior_fn, min_length, max_length, verbose
        )
    elif algorithm == "viterbi":
        return viterbi_constrained_segmentation(
            probs, prior_fn, min_length, max_length, verbose
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'greedy' or 'viterbi'.")
```

**Tests:** `tests/test_constraints.py` (NEW)
```python
import pytest
import numpy as np
from wtpsplit.utils.constraints import (
    greedy_constrained_segmentation,
    viterbi_constrained_segmentation,
    constrained_segmentation,
    safe_log
)


def test_safe_log():
    x = np.array([0.0, 1e-20, 1.0])
    result = safe_log(x)
    assert np.all(np.isfinite(result))
    assert result[2] == 0.0  # log(1) = 0


def test_greedy_simple():
    probs = np.array([0.1, 0.9, 0.1, 0.9])
    prior = lambda l: 1.0 if l <= 2 else 0.0

    boundaries = greedy_constrained_segmentation(probs, prior, min_length=1)

    # Should select high-probability positions within length constraint
    assert len(boundaries) > 0
    assert all(0 <= b < len(probs) for b in boundaries)


def test_viterbi_optimal():
    # Case where greedy might not find optimal solution
    probs = np.array([0.5, 0.9, 0.5, 0.9])
    prior = lambda l: 1.0 if l == 2 else 0.1  # Strong preference for length 2

    boundaries = viterbi_constrained_segmentation(probs, prior, min_length=1)

    # Should find global optimum
    assert len(boundaries) > 0


def test_algorithm_selector():
    probs = np.array([0.1, 0.5, 0.9, 0.1])
    prior = lambda l: 1.0

    # Greedy
    b1 = constrained_segmentation(probs, prior, algorithm="greedy")
    assert isinstance(b1, list)

    # Viterbi
    b2 = constrained_segmentation(probs, prior, algorithm="viterbi")
    assert isinstance(b2, list)

    # Invalid
    with pytest.raises(ValueError):
        constrained_segmentation(probs, prior, algorithm="invalid")


def test_max_length_constraint():
    probs = np.array([0.1] * 100)
    prior = lambda l: 1.0 if l <= 10 else 0.0

    boundaries = viterbi_constrained_segmentation(
        probs, prior, min_length=1, max_length=10
    )

    # Check all segments respect max_length
    boundaries_with_ends = [0] + boundaries + [len(probs)]
    for i in range(len(boundaries_with_ends) - 1):
        segment_length = boundaries_with_ends[i + 1] - boundaries_with_ends[i]
        assert segment_length <= 10


def test_empty_input():
    probs = np.array([])
    prior = lambda l: 1.0

    boundaries = greedy_constrained_segmentation(probs, prior)
    assert boundaries == []

    boundaries = viterbi_constrained_segmentation(probs, prior)
    assert boundaries == []
```

---

## Phase 2: Integration (Week 2-3)

### Step 2.1: Modify Main WtPSplit Module

**File:** `wtpsplit/__init__.py` (MODIFY)

**Add to SaT class:**
```python
def split_length_constrained(
    self,
    text_or_texts: Union[str, List[str]],
    prior: Union[str, Callable] = "uniform",
    max_length: Optional[int] = None,
    target_length: Optional[int] = None,
    min_length: int = 1,
    algorithm: str = "viterbi",
    prior_params: Optional[Dict] = None,
    lang_code: Optional[str] = None,
    verbose: bool = False,
    **kwargs
) -> Union[List[str], List[List[str]]]:
    """
    Segment text with explicit length constraints.

    This method combines the neural model's boundary probabilities with
    a prior distribution over segment lengths to explicitly control
    segmentation length.

    Args:
        text_or_texts: Input text(s) to segment
        prior: Prior distribution name ("uniform", "gaussian", etc.) or custom callable
        max_length: Maximum segment length (characters). Required for "uniform" prior.
        target_length: Preferred segment length. Required for "gaussian" prior.
        min_length: Minimum segment length (default: 1)
        algorithm: Optimization algorithm: "viterbi" (optimal) or "greedy" (faster)
        prior_params: Additional parameters for the prior function
        lang_code: Language code (auto-detect if None)
        verbose: Print debug information
        **kwargs: Additional arguments passed to predict_proba()

    Returns:
        List of segmented sentences (or list of lists for batch input)

    Examples:
        >>> model = SaT("sat-3l-sm")

        >>> # Hard maximum length
        >>> sentences = model.split_length_constrained(
        ...     "Long text...",
        ...     prior="uniform",
        ...     max_length=200
        ... )

        >>> # Preferred length (soft constraint)
        >>> sentences = model.split_length_constrained(
        ...     "Long text...",
        ...     prior="gaussian",
        ...     target_length=100,
        ...     prior_params={"sigma": 20}
        ... )

        >>> # Custom prior
        >>> def my_prior(length):
        ...     return 1.0 if 50 <= length <= 150 else 0.1
        >>> sentences = model.split_length_constrained(
        ...     "Long text...",
        ...     prior=my_prior
        ... )
    """
    from wtpsplit.utils.priors import create_prior_function
    from wtpsplit.utils.constraints import constrained_segmentation
    from wtpsplit.utils import indices_to_sentences

    # Handle batch vs single input
    is_batch = isinstance(text_or_texts, list)
    texts = text_or_texts if is_batch else [text_or_texts]

    # Set up prior parameters
    if prior_params is None:
        prior_params = {}

    # Add max_length and target_length to prior_params if provided
    if max_length is not None:
        prior_params.setdefault("max_length", max_length)
    if target_length is not None:
        prior_params.setdefault("target_length", target_length)

    # Validate parameters
    if isinstance(prior, str):
        if prior == "uniform" and "max_length" not in prior_params:
            raise ValueError("max_length is required for uniform prior")
        if prior == "gaussian" and "target_length" not in prior_params:
            raise ValueError("target_length is required for gaussian prior")

    # Create prior function
    prior_fn = create_prior_function(prior, prior_params)

    # Get probabilities from model
    probs_generator = self.predict_proba(texts, lang_code=lang_code, **kwargs)

    # Process each text
    all_sentences = []
    for text, probs in zip(texts, probs_generator):
        if len(probs) == 0:
            all_sentences.append([])
            continue

        # Apply constrained segmentation
        boundaries = constrained_segmentation(
            probs,
            prior_fn,
            algorithm=algorithm,
            min_length=min_length,
            max_length=max_length,
            verbose=verbose
        )

        # Convert boundaries to sentences
        sentences = indices_to_sentences(
            text,
            np.array(boundaries),
            strip_whitespace=kwargs.get("strip_whitespace", False)
        )

        all_sentences.append(sentences)

    # Return single list if single input
    return all_sentences if is_batch else all_sentences[0]
```

---

## Phase 3: Evaluation (Week 3-4)

### Step 3.1: Create Evaluation Module

**File:** `wtpsplit/evaluation/length_constraints.py` (NEW)

**Implementation:**
```python
"""
Evaluation metrics for length-constrained segmentation.
"""

import numpy as np
from typing import List, Tuple
from sklearn.metrics import precision_recall_fscore_support


def constraint_satisfaction_rate(
    segments: List[str],
    min_length: int = 0,
    max_length: int = float('inf')
) -> float:
    """
    Calculate the proportion of segments that satisfy length constraints.

    Args:
        segments: List of segmented strings
        min_length: Minimum length
        max_length: Maximum length

    Returns:
        CSR: fraction of segments within [min_length, max_length]
    """
    if len(segments) == 0:
        return 1.0

    satisfied = sum(1 for s in segments if min_length <= len(s) <= max_length)
    return satisfied / len(segments)


def boundary_quality_score(
    selected_boundaries: List[int],
    all_probs: np.ndarray
) -> float:
    """
    Calculate mean probability of selected boundaries.

    Args:
        selected_boundaries: Indices of selected boundaries
        all_probs: Probability array for all positions

    Returns:
        BQS: mean probability of selected boundaries
    """
    if len(selected_boundaries) == 0:
        return 0.0

    selected_probs = [all_probs[b] for b in selected_boundaries if b < len(all_probs)]
    return np.mean(selected_probs) if selected_probs else 0.0


def segmentation_f1(
    predicted_boundaries: List[int],
    gold_boundaries: List[int],
    tolerance: int = 0
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, F1 for boundary detection.

    Args:
        predicted_boundaries: Predicted boundary indices
        gold_boundaries: Ground truth boundary indices
        tolerance: Allow matches within ±tolerance positions

    Returns:
        (precision, recall, f1)
    """
    if len(predicted_boundaries) == 0:
        return 0.0, 0.0, 0.0

    if len(gold_boundaries) == 0:
        return 0.0, 1.0 if len(predicted_boundaries) == 0 else 0.0, 0.0

    # Calculate matches
    matched_pred = set()
    matched_gold = set()

    for pred_idx, pred_b in enumerate(predicted_boundaries):
        for gold_idx, gold_b in enumerate(gold_boundaries):
            if abs(pred_b - gold_b) <= tolerance:
                matched_pred.add(pred_idx)
                matched_gold.add(gold_idx)

    precision = len(matched_pred) / len(predicted_boundaries)
    recall = len(matched_gold) / len(gold_boundaries)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def length_distribution_stats(segments: List[str]) -> dict:
    """
    Calculate statistics about segment length distribution.

    Args:
        segments: List of segmented strings

    Returns:
        Dictionary with mean, median, std, min, max
    """
    lengths = [len(s) for s in segments]

    if len(lengths) == 0:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "count": 0}

    return {
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "std": np.std(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
        "count": len(lengths)
    }
```

---

## Implementation Summary

### Files to Create (NEW)
1. `wtpsplit/utils/priors.py` - Prior distributions
2. `wtpsplit/utils/constraints.py` - Optimization algorithms
3. `wtpsplit/evaluation/length_constraints.py` - Evaluation metrics
4. `tests/test_priors.py` - Prior tests
5. `tests/test_constraints.py` - Algorithm tests
6. `tests/test_length_constrained_integration.py` - Integration tests

### Files to Modify (MODIFY)
1. `wtpsplit/__init__.py` - Add `split_length_constrained()` to `SaT` and `WtP` classes

### Dependencies to Add
```python
# requirements.txt
# (most are already there)
numpy>=1.20.0
scikit-learn>=0.24.0  # for metrics
```

---

## Testing Strategy

### Unit Tests (Week 1-2)
```bash
pytest tests/test_priors.py -v
pytest tests/test_constraints.py -v
```

### Integration Tests (Week 2-3)
```bash
pytest tests/test_length_constrained_integration.py -v
```

### Manual Testing
```python
# test_manual.py
from wtpsplit import SaT

model = SaT("sat-3l-sm")

text = "This is a test. " * 100

# Test uniform prior
sentences = model.split_length_constrained(
    text,
    prior="uniform",
    max_length=50
)

print(f"Generated {len(sentences)} sentences")
print(f"Max length: {max(len(s) for s in sentences)}")
print(f"All under 50: {all(len(s) <= 50 for s in sentences)}")
```

---

## Development Workflow

### Week 1: Day-by-Day
- **Monday:** Create `priors.py`, implement uniform + gaussian
- **Tuesday:** Add remaining priors, write tests
- **Wednesday:** Create `constraints.py`, implement greedy
- **Thursday:** Implement Viterbi, write tests
- **Friday:** Code review, fix bugs

### Week 2: Day-by-Day
- **Monday:** Integrate with SaT class
- **Tuesday:** Add batch processing support
- **Wednesday:** Write integration tests
- **Thursday:** Test with real models
- **Friday:** Week 2 demo to supervisor

---

## Git Workflow

```bash
# Create feature branch
git checkout -b length-constrained-segmentation

# Commit frequently
git commit -m "Add prior distribution functions"
git commit -m "Implement greedy algorithm"
git commit -m "Implement Viterbi algorithm"
git commit -m "Integrate with SaT class"

# Push to remote
git push origin length-constrained-segmentation
```

---

## Debugging Tips

### If Viterbi is too slow:
1. Profile with `cProfile`
2. Add numba JIT: `@numba.jit(nopython=True)`
3. Reduce max_length search window

### If constraints not satisfied:
1. Check prior function returns non-zero values
2. Verify min_length < max_length
3. Test with verbose=True

### If F1 score too low:
1. Relax constraints (increase max_length)
2. Adjust prior smoothing
3. Check if model probabilities are reasonable

---

This roadmap should provide all the details needed to implement the feature step by step!
