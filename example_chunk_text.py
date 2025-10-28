#!/usr/bin/env python3
"""
Simple example: Chunk text with explicit length constraints.

This script demonstrates the basic usage of length-constrained segmentation
to chunk paragraphs with a specified maximum size.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/wtpsplit')

from wtpsplit.utils.priors import create_prior_function
from wtpsplit.utils.constraints import constrained_segmentation, indices_to_segments


def chunk_text_with_max_length(text, max_chunk_size=100, algorithm="viterbi"):
    """
    Chunk text with a maximum chunk size.

    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        algorithm: "greedy" or "viterbi"

    Returns:
        List of text chunks
    """
    # In a real scenario, you would get probabilities from the SaT model
    # For this example, we simulate probabilities
    # High probability = likely sentence boundary

    # Simple heuristic: put high probability after punctuation
    probs = np.array([0.1] * len(text))
    for i, char in enumerate(text):
        if char in '.!?':
            if i + 1 < len(text):
                probs[i + 1] = 0.9  # High prob after punctuation

    # Create uniform prior: allow any length up to max_chunk_size
    prior_fn = create_prior_function(
        "uniform",
        {"max_length": max_chunk_size}
    )

    # Find optimal boundaries
    boundaries = constrained_segmentation(
        probs=probs,
        prior_fn=prior_fn,
        algorithm=algorithm,
        min_length=10,  # Don't make chunks too small
        max_length=max_chunk_size
    )

    # Convert boundaries to text chunks
    chunks = indices_to_segments(text, boundaries, strip_whitespace=True)

    return chunks


def chunk_text_with_target_length(text, target_length=80, sigma=20, algorithm="viterbi"):
    """
    Chunk text with a preferred target length (soft constraint).

    Args:
        text: Input text to chunk
        target_length: Preferred chunk length
        sigma: How much variance to allow
        algorithm: "greedy" or "viterbi"

    Returns:
        List of text chunks
    """
    # Simulate probabilities (same as above)
    probs = np.array([0.1] * len(text))
    for i, char in enumerate(text):
        if char in '.!?':
            if i + 1 < len(text):
                probs[i + 1] = 0.9

    # Create gaussian prior: prefer target_length, but allow variation
    prior_fn = create_prior_function(
        "gaussian",
        {"target_length": target_length, "sigma": sigma}
    )

    # Find optimal boundaries
    boundaries = constrained_segmentation(
        probs=probs,
        prior_fn=prior_fn,
        algorithm=algorithm,
        min_length=10
    )

    # Convert boundaries to text chunks
    chunks = indices_to_segments(text, boundaries, strip_whitespace=True)

    return chunks


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_1_hard_limit():
    """Example 1: Hard maximum length constraint."""
    print("=" * 70)
    print("EXAMPLE 1: Hard Maximum Length (100 characters)")
    print("=" * 70)

    text = """
    Machine learning is a subset of artificial intelligence. It focuses on
    developing systems that can learn from data. These systems improve their
    performance over time. Neural networks are a popular approach. They mimic
    the human brain's structure. Deep learning uses multiple layers. This
    enables complex pattern recognition. Applications include image recognition
    and natural language processing.
    """.strip()

    print(f"\nOriginal text ({len(text)} characters):\n")
    print(text)

    print("\n" + "-" * 70)
    print("Chunking with max_chunk_size=100...")
    print("-" * 70 + "\n")

    chunks = chunk_text_with_max_length(text, max_chunk_size=100, algorithm="viterbi")

    print(f"Generated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(f"  {chunk}")
        print()

    # Verify all chunks respect the constraint
    max_len = max(len(c) for c in chunks)
    print(f"Maximum chunk length: {max_len}")
    print(f"Constraint satisfied: {max_len <= 100}")
    print()


def example_2_preferred_length():
    """Example 2: Preferred target length (soft constraint)."""
    print("=" * 70)
    print("EXAMPLE 2: Preferred Target Length (80 characters)")
    print("=" * 70)

    text = """
    Python is a high-level programming language. It emphasizes code readability.
    The language supports multiple programming paradigms. These include procedural
    and object-oriented programming. Python has a comprehensive standard library.
    This makes it suitable for many applications. Data science is one popular use.
    Web development is another common application.
    """.strip()

    print(f"\nOriginal text ({len(text)} characters):\n")
    print(text)

    print("\n" + "-" * 70)
    print("Chunking with target_length=80, sigma=15...")
    print("-" * 70 + "\n")

    chunks = chunk_text_with_target_length(
        text, target_length=80, sigma=15, algorithm="viterbi"
    )

    print(f"Generated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        deviation = len(chunk) - 80
        print(f"Chunk {i} ({len(chunk)} chars, deviation: {deviation:+d}):")
        print(f"  {chunk}")
        print()

    # Statistics
    lengths = [len(c) for c in chunks]
    print(f"Length statistics:")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Std:  {np.std(lengths):.1f}")
    print(f"  Min:  {min(lengths)}")
    print(f"  Max:  {max(lengths)}")
    print()


def example_3_algorithm_comparison():
    """Example 3: Compare greedy vs Viterbi."""
    print("=" * 70)
    print("EXAMPLE 3: Greedy vs Viterbi Comparison")
    print("=" * 70)

    text = """
    Natural language processing enables computers to understand human language.
    It combines computational linguistics with machine learning. Common tasks
    include sentiment analysis and machine translation. Named entity recognition
    is also important. Text summarization reduces document length. Question
    answering systems provide direct responses.
    """.strip()

    print(f"\nOriginal text ({len(text)} characters):\n")
    print(text)

    max_size = 100

    print("\n" + "-" * 70)
    print(f"Chunking with max_chunk_size={max_size}...")
    print("-" * 70 + "\n")

    # Greedy
    print("GREEDY Algorithm:")
    chunks_greedy = chunk_text_with_max_length(text, max_chunk_size=max_size, algorithm="greedy")
    for i, chunk in enumerate(chunks_greedy, 1):
        print(f"  {i}. ({len(chunk):2d} chars) {chunk[:60]}...")

    print()

    # Viterbi
    print("VITERBI Algorithm:")
    chunks_viterbi = chunk_text_with_max_length(text, max_chunk_size=max_size, algorithm="viterbi")
    for i, chunk in enumerate(chunks_viterbi, 1):
        print(f"  {i}. ({len(chunk):2d} chars) {chunk[:60]}...")

    print()
    print(f"Number of chunks - Greedy: {len(chunks_greedy)}, Viterbi: {len(chunks_viterbi)}")
    print(f"Algorithms produced same result: {chunks_greedy == chunks_viterbi}")
    print()


def example_4_custom_prior():
    """Example 4: Using a custom prior function."""
    print("=" * 70)
    print("EXAMPLE 4: Custom Prior Function")
    print("=" * 70)

    text = """
    Version control systems track changes to code. Git is the most popular system.
    It enables collaboration among developers. Branches allow parallel development.
    Merging combines different code versions. Pull requests facilitate code review.
    """.strip()

    print(f"\nOriginal text ({len(text)} characters):\n")
    print(text)

    print("\n" + "-" * 70)
    print("Using custom prior: prefer 40-80 chars, allow 20-120...")
    print("-" * 70 + "\n")

    # Simulate probabilities
    probs = np.array([0.1] * len(text))
    for i, char in enumerate(text):
        if char in '.!?':
            if i + 1 < len(text):
                probs[i + 1] = 0.9

    # Custom prior function
    def custom_prior(length):
        """
        Custom prior:
        - Zero probability if length < 20 or length > 120
        - Probability 1.0 if 40 <= length <= 80
        - Linear ramp between these regions
        """
        if length < 20 or length > 120:
            return 0.0
        elif 20 <= length < 40:
            return (length - 20) / 20.0  # Ramp up
        elif 40 <= length <= 80:
            return 1.0  # Optimal range
        elif 80 < length <= 120:
            return 1.0 - (length - 80) / 40.0  # Ramp down
        else:
            return 0.0

    # Use the custom prior
    boundaries = constrained_segmentation(
        probs=probs,
        prior_fn=custom_prior,
        algorithm="viterbi",
        min_length=20,
        max_length=120
    )

    chunks = indices_to_segments(text, boundaries, strip_whitespace=True)

    print(f"Generated {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        in_optimal = 40 <= len(chunk) <= 80
        marker = "✓" if in_optimal else "○"
        print(f"  {marker} Chunk {i} ({len(chunk):2d} chars): {chunk[:60]}...")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("LENGTH-CONSTRAINED TEXT CHUNKING - EXAMPLES")
    print("=" * 70 + "\n")

    example_1_hard_limit()
    example_2_preferred_length()
    example_3_algorithm_comparison()
    example_4_custom_prior()

    print("=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
