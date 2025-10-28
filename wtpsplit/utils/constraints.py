"""
Constraint optimization algorithms for length-constrained segmentation.

Implements greedy and Viterbi algorithms for finding optimal segment boundaries
given boundary probabilities and length constraints.
"""

import numpy as np
from typing import Callable, List, Optional
import warnings


def safe_log(x: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Numerically stable logarithm that avoids log(0) = -inf.

    Args:
        x: Input array (can be scalar or array)
        epsilon: Small value to avoid log(0)

    Returns:
        log(max(x, epsilon))

    Example:
        >>> safe_log(np.array([0.0, 1e-20, 1.0]))
        array([-23.02585093, -23.02585093,   0.        ])
    """
    x = np.asarray(x)
    return np.log(np.maximum(x, epsilon))


def greedy_segment(
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

    This is a local optimization approach that may not find the global optimum
    but is fast and simple.

    Args:
        probs: Array of boundary probabilities (length n)
        prior_fn: Function mapping length -> prior probability
        min_length: Minimum segment length (default: 1)
        max_length: Maximum segment length (None = no limit)
        verbose: Print debug information

    Returns:
        List of boundary indices (0-indexed, excluding final position)

    Example:
        >>> probs = np.array([0.1, 0.9, 0.1, 0.9, 0.1])
        >>> prior = lambda l: 1.0 if l <= 2 else 0.0
        >>> greedy_segment(probs, prior, min_length=1)
        [1, 3]
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

        if search_start > n:
            # No valid boundary (remaining text too short)
            break

        # Try all valid positions
        for pos in range(search_start, search_end + 1):
            if pos > n:
                break

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
            if search_end <= n:
                search_probs = probs[search_start - 1:search_end]
                if len(search_probs) > 0:
                    best_pos = search_start + np.argmax(search_probs)
                else:
                    break  # Can't proceed
            else:
                break

        if verbose:
            length = best_pos - current_start
            print(f"Selected boundary at {best_pos} (length={length}, score={best_score:.3f})")

        boundaries.append(best_pos)
        current_start = best_pos

    # Remove final boundary if it's exactly at the end
    if boundaries and boundaries[-1] >= n:
        boundaries.pop()

    return boundaries


def viterbi_segment(
    probs: np.ndarray,
    prior_fn: Callable[[int], float],
    min_length: int = 1,
    max_length: Optional[int] = None,
    verbose: bool = False
) -> List[int]:
    """
    Viterbi algorithm for length-constrained segmentation.

    Finds the globally optimal boundary selection using dynamic programming.
    This guarantees the best solution but is slightly slower than greedy.

    The algorithm uses the Viterbi dynamic programming approach:
    1. Forward pass: compute best score to reach each position
    2. Backward pass: reconstruct optimal path

    Args:
        probs: Array of boundary probabilities (length n)
        prior_fn: Function mapping length -> prior probability
        min_length: Minimum segment length (default: 1)
        max_length: Maximum segment length (None = no limit)
        verbose: Print debug information

    Returns:
        List of boundary indices (0-indexed, excluding final position)

    Example:
        >>> probs = np.array([0.9, 0.5, 0.5, 0.9])
        >>> prior = lambda l: 1.0 if l == 2 else 0.1
        >>> viterbi_segment(probs, prior, min_length=1)
        [2]
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
        if verbose:
            warnings.warn(
                "Could not find valid segmentation with given constraints. "
                "Falling back to greedy algorithm."
            )
        return greedy_segment(probs, prior_fn, min_length, max_length, verbose)

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


def select_algorithm(algorithm_name: str) -> Callable:
    """
    Factory function to select segmentation algorithm by name.

    Args:
        algorithm_name: Algorithm name ("greedy" or "viterbi")

    Returns:
        Algorithm function

    Raises:
        ValueError: If algorithm name is unknown

    Example:
        >>> algo = select_algorithm("greedy")
        >>> probs = np.array([0.1, 0.5, 0.9])
        >>> prior = lambda l: 1.0
        >>> boundaries = algo(probs, prior, min_length=1)
    """
    algorithms = {
        "greedy": greedy_segment,
        "viterbi": viterbi_segment
    }

    algorithm_name = algorithm_name.lower()

    if algorithm_name not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available: {list(algorithms.keys())}"
        )

    return algorithms[algorithm_name]


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

    Convenience function that selects and runs the specified algorithm.

    Args:
        probs: Array of boundary probabilities
        prior_fn: Prior distribution function
        algorithm: "greedy" or "viterbi" (default: "viterbi")
        min_length: Minimum segment length
        max_length: Maximum segment length
        verbose: Print debug information

    Returns:
        List of boundary indices

    Example:
        >>> probs = np.array([0.1, 0.5, 0.9, 0.2])
        >>> prior = lambda l: 1.0 if l <= 3 else 0.0
        >>> boundaries = constrained_segmentation(probs, prior, algorithm="viterbi")
    """
    algo_fn = select_algorithm(algorithm)
    return algo_fn(probs, prior_fn, min_length, max_length, verbose)


def indices_to_segments(
    text: str,
    boundaries: List[int],
    strip_whitespace: bool = False
) -> List[str]:
    """
    Convert boundary indices to text segments.

    Args:
        text: Original text
        boundaries: List of boundary indices
        strip_whitespace: Whether to strip leading/trailing whitespace

    Returns:
        List of text segments

    Example:
        >>> text = "Hello world! How are you?"
        >>> boundaries = [13, 17]  # After "!" and "How "
        >>> indices_to_segments(text, boundaries)
        ['Hello world! ', 'How ', 'are you?']
    """
    if not boundaries:
        return [text]

    segments = []
    start = 0

    for boundary in boundaries:
        segment = text[start:boundary]
        if strip_whitespace:
            segment = segment.strip()
        if segment:  # Only add non-empty segments
            segments.append(segment)
        start = boundary

    # Add final segment
    final_segment = text[start:]
    if strip_whitespace:
        final_segment = final_segment.strip()
    if final_segment:
        segments.append(final_segment)

    return segments
