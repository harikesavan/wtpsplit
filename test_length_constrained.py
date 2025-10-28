#!/usr/bin/env python3
"""
Test script for length-constrained segmentation.

This script demonstrates how to use the prior functions and
constraint algorithms to chunk text with explicit length control.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/wtpsplit')

from wtpsplit.utils.priors import (
    uniform_prior,
    gaussian_prior,
    clipped_polynomial_prior,
    exponential_prior,
    piecewise_linear_prior,
    create_prior_function,
    PriorRegistry
)

from wtpsplit.utils.constraints import (
    safe_log,
    greedy_segment,
    viterbi_segment,
    select_algorithm,
    constrained_segmentation,
    indices_to_segments
)


def test_safe_log():
    """Test safe_log function."""
    print("=" * 60)
    print("TEST: safe_log")
    print("=" * 60)

    x = np.array([0.0, 1e-20, 0.5, 1.0])
    result = safe_log(x)

    print(f"Input:  {x}")
    print(f"Output: {result}")
    print(f"All finite: {np.all(np.isfinite(result))}")
    print()


def test_prior_functions():
    """Test all prior distribution functions."""
    print("=" * 60)
    print("TEST: Prior Distribution Functions")
    print("=" * 60)

    lengths = [10, 50, 100, 150, 200]

    # Test uniform prior
    print("\n1. Uniform Prior (max_length=100)")
    print("-" * 40)
    for length in lengths:
        prob = uniform_prior(length, max_length=100)
        print(f"  length={length:3d}: {prob:.3f}")

    # Test gaussian prior
    print("\n2. Gaussian Prior (target=100, sigma=20)")
    print("-" * 40)
    for length in lengths:
        prob = gaussian_prior(length, target_length=100, sigma=20)
        print(f"  length={length:3d}: {prob:.3f}")

    # Test clipped polynomial prior
    print("\n3. Clipped Polynomial Prior (target=100, coef=0.001)")
    print("-" * 40)
    for length in lengths:
        prob = clipped_polynomial_prior(length, target_length=100, coefficient=0.001)
        print(f"  length={length:3d}: {prob:.3f}")

    # Test exponential prior
    print("\n4. Exponential Prior (max=100, decay=0.1)")
    print("-" * 40)
    for length in lengths:
        prob = exponential_prior(length, max_length=100, decay_rate=0.1)
        print(f"  length={length:3d}: {prob:.3f}")

    # Test piecewise linear prior
    print("\n5. Piecewise Linear Prior (min=20, target=80, max=120)")
    print("-" * 40)
    for length in lengths:
        prob = piecewise_linear_prior(
            length, min_length=20, target_length=80, max_length=120
        )
        print(f"  length={length:3d}: {prob:.3f}")

    print()


def test_prior_registry():
    """Test prior registry."""
    print("=" * 60)
    print("TEST: Prior Registry")
    print("=" * 60)

    # List all available priors
    print("Available priors:", PriorRegistry.list_priors())

    # Create prior function from string
    prior_fn = create_prior_function("uniform", {"max_length": 100})
    print(f"\nCreated uniform prior: {prior_fn(50)=}, {prior_fn(150)=}")

    # Create prior function from callable
    custom_prior = lambda length: 0.5 if length < 100 else 0.1
    prior_fn = create_prior_function(custom_prior)
    print(f"Created custom prior: {prior_fn(50)=}, {prior_fn(150)=}")

    print()


def test_greedy_algorithm():
    """Test greedy segmentation algorithm."""
    print("=" * 60)
    print("TEST: Greedy Algorithm")
    print("=" * 60)

    # Create sample probabilities (simulating model output)
    probs = np.array([0.1, 0.3, 0.9, 0.2, 0.1, 0.8, 0.1, 0.4, 0.95, 0.1])
    print(f"Probabilities: {probs}")

    # Test with uniform prior (max_length=5)
    print("\nTest 1: Uniform prior (max_length=5)")
    print("-" * 40)
    prior = lambda l: 1.0 if l <= 5 else 0.0
    boundaries = greedy_segment(probs, prior, min_length=2, max_length=5, verbose=True)
    print(f"Boundaries: {boundaries}")

    # Test with gaussian prior
    print("\nTest 2: Gaussian prior (target=4, sigma=1)")
    print("-" * 40)
    prior = lambda l: gaussian_prior(l, target_length=4, sigma=1)
    boundaries = greedy_segment(probs, prior, min_length=2, verbose=True)
    print(f"Boundaries: {boundaries}")

    print()


def test_viterbi_algorithm():
    """Test Viterbi segmentation algorithm."""
    print("=" * 60)
    print("TEST: Viterbi Algorithm")
    print("=" * 60)

    # Create sample probabilities
    probs = np.array([0.1, 0.3, 0.9, 0.2, 0.1, 0.8, 0.1, 0.4, 0.95, 0.1])
    print(f"Probabilities: {probs}")

    # Test with uniform prior
    print("\nTest 1: Uniform prior (max_length=5)")
    print("-" * 40)
    prior = lambda l: 1.0 if l <= 5 else 0.0
    boundaries = viterbi_segment(probs, prior, min_length=2, max_length=5, verbose=True)
    print(f"Boundaries: {boundaries}")

    # Test with gaussian prior
    print("\nTest 2: Gaussian prior (target=4, sigma=1)")
    print("-" * 40)
    prior = lambda l: gaussian_prior(l, target_length=4, sigma=1)
    boundaries = viterbi_segment(probs, prior, min_length=2, verbose=True)
    print(f"Boundaries: {boundaries}")

    print()


def test_algorithm_comparison():
    """Compare greedy and Viterbi algorithms."""
    print("=" * 60)
    print("TEST: Algorithm Comparison (Greedy vs Viterbi)")
    print("=" * 60)

    # Create a case where greedy might not be optimal
    probs = np.array([0.9, 0.5, 0.5, 0.9, 0.5, 0.5, 0.9])
    print(f"Probabilities: {probs}")

    # Prior strongly prefers length=3
    prior = lambda l: 1.0 if l == 3 else 0.1

    print("\nGreedy algorithm:")
    print("-" * 40)
    boundaries_greedy = greedy_segment(probs, prior, min_length=1)
    print(f"Boundaries: {boundaries_greedy}")

    print("\nViterbi algorithm:")
    print("-" * 40)
    boundaries_viterbi = viterbi_segment(probs, prior, min_length=1)
    print(f"Boundaries: {boundaries_viterbi}")

    print("\nComparison:")
    print(f"  Greedy:  {boundaries_greedy}")
    print(f"  Viterbi: {boundaries_viterbi}")
    print(f"  Same result: {boundaries_greedy == boundaries_viterbi}")

    print()


def test_algorithm_factory():
    """Test algorithm selection factory."""
    print("=" * 60)
    print("TEST: Algorithm Factory (select_algorithm)")
    print("=" * 60)

    probs = np.array([0.1, 0.5, 0.9, 0.2])
    prior = lambda l: 1.0

    # Test greedy
    algo = select_algorithm("greedy")
    boundaries = algo(probs, prior, min_length=1)
    print(f"greedy: {boundaries}")

    # Test viterbi
    algo = select_algorithm("viterbi")
    boundaries = algo(probs, prior, min_length=1)
    print(f"viterbi: {boundaries}")

    # Test constrained_segmentation convenience function
    boundaries = constrained_segmentation(probs, prior, algorithm="viterbi")
    print(f"constrained_segmentation (viterbi): {boundaries}")

    print()


def test_text_chunking():
    """Test chunking real text with length constraints."""
    print("=" * 60)
    print("TEST: Text Chunking with Length Constraints")
    print("=" * 60)

    # Sample text
    text = "This is the first sentence. This is the second sentence. This is the third sentence. And a fourth one. And a fifth. Sixth sentence here. Seventh sentence now. Eighth one too."

    print(f"Original text ({len(text)} chars):")
    print(f'"{text}"')
    print()

    # Simulate model probabilities (higher at sentence boundaries)
    # In real usage, these would come from the SaT model
    # For demo, we manually set high probabilities at sentence ends
    sentence_ends = [26, 57, 88, 104, 117, 136, 155]  # positions after periods
    probs = np.array([0.1] * len(text))
    for pos in sentence_ends:
        if pos < len(text):
            probs[pos] = 0.9

    print(f"Simulated probabilities (shape={probs.shape}):")
    print(f"  High prob at positions: {sentence_ends}")
    print()

    # Test 1: Hard limit of 50 characters
    print("Test 1: Uniform prior (max_length=50)")
    print("-" * 40)
    prior = lambda l: 1.0 if l <= 50 else 0.0
    boundaries = viterbi_segment(probs, prior, min_length=10, max_length=50)
    segments = indices_to_segments(text, boundaries, strip_whitespace=True)

    print(f"Boundaries: {boundaries}")
    print(f"Number of chunks: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"  Chunk {i+1} ({len(seg):2d} chars): {seg[:60]}...")

    print()

    # Test 2: Prefer length ~40 characters
    print("Test 2: Gaussian prior (target=40, sigma=10)")
    print("-" * 40)
    prior = lambda l: gaussian_prior(l, target_length=40, sigma=10)
    boundaries = viterbi_segment(probs, prior, min_length=10)
    segments = indices_to_segments(text, boundaries, strip_whitespace=True)

    print(f"Boundaries: {boundaries}")
    print(f"Number of chunks: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"  Chunk {i+1} ({len(seg):2d} chars): {seg[:60]}...")

    print()

    # Test 3: Greedy vs Viterbi comparison
    print("Test 3: Greedy vs Viterbi (max_length=60)")
    print("-" * 40)
    prior = lambda l: 1.0 if l <= 60 else 0.0

    boundaries_greedy = greedy_segment(probs, prior, min_length=10, max_length=60)
    segments_greedy = indices_to_segments(text, boundaries_greedy, strip_whitespace=True)

    boundaries_viterbi = viterbi_segment(probs, prior, min_length=10, max_length=60)
    segments_viterbi = indices_to_segments(text, boundaries_viterbi, strip_whitespace=True)

    print(f"Greedy boundaries:  {boundaries_greedy}")
    print(f"Viterbi boundaries: {boundaries_viterbi}")
    print(f"Same result: {boundaries_greedy == boundaries_viterbi}")
    print()
    print("Greedy chunks:")
    for i, seg in enumerate(segments_greedy):
        print(f"  {i+1}. ({len(seg):2d} chars) {seg[:60]}...")
    print()
    print("Viterbi chunks:")
    for i, seg in enumerate(segments_viterbi):
        print(f"  {i+1}. ({len(seg):2d} chars) {seg[:60]}...")

    print()


def test_edge_cases():
    """Test edge cases."""
    print("=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)

    # Empty input
    print("Test 1: Empty input")
    print("-" * 40)
    probs = np.array([])
    prior = lambda l: 1.0
    boundaries = viterbi_segment(probs, prior)
    print(f"Empty probs: {boundaries}")

    # Very short input
    print("\nTest 2: Very short input (length=2, min_length=1)")
    print("-" * 40)
    probs = np.array([0.5, 0.9])
    boundaries = viterbi_segment(probs, prior, min_length=1)
    print(f"Short probs: {boundaries}")

    # All same probabilities
    print("\nTest 3: Uniform probabilities")
    print("-" * 40)
    probs = np.array([0.5] * 10)
    prior = lambda l: 1.0 if l <= 5 else 0.0
    boundaries = viterbi_segment(probs, prior, min_length=2, max_length=5)
    print(f"Uniform probs: {boundaries}")

    # Very restrictive prior
    print("\nTest 4: Very restrictive prior")
    print("-" * 40)
    probs = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    prior = lambda l: 1.0 if l == 2 else 0.0  # Only length 2 allowed
    boundaries = viterbi_segment(probs, prior, min_length=2, max_length=2)
    print(f"Restrictive prior: {boundaries}")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LENGTH-CONSTRAINED SEGMENTATION - TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_safe_log()
        test_prior_functions()
        test_prior_registry()
        test_greedy_algorithm()
        test_viterbi_algorithm()
        test_algorithm_comparison()
        test_algorithm_factory()
        test_text_chunking()
        test_edge_cases()

        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
