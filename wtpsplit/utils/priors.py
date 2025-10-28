"""
Prior distribution functions for length-constrained segmentation.

Each prior function takes a segment length and returns a probability.
Priors control the preference for different segment lengths.
"""

import numpy as np
from typing import Callable, Dict, Any, Union


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
            raise ValueError(
                f"Unknown prior: {name}. Available: {list(cls._priors.keys())}"
            )
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

    Example:
        >>> uniform_prior(50, max_length=100)
        1.0
        >>> uniform_prior(150, max_length=100)
        0.0
    """
    return 1.0 if length <= max_length else 0.0


@PriorRegistry.register("gaussian")
def gaussian_prior(length: int, target_length: int, sigma: float = 20.0, **kwargs) -> float:
    """
    Gaussian prior: prefer target_length with normal distribution.

    Args:
        length: Segment length
        target_length: Preferred length (mean of distribution)
        sigma: Standard deviation (controls spread)

    Returns:
        Gaussian probability density

    Example:
        >>> gaussian_prior(100, target_length=100, sigma=20)
        1.0
        >>> gaussian_prior(80, target_length=100, sigma=20)
        0.6065...
    """
    return float(np.exp(-0.5 * ((length - target_length) / sigma) ** 2))


@PriorRegistry.register("clipped_polynomial")
def clipped_polynomial_prior(
    length: int,
    target_length: int,
    coefficient: float = 0.01,
    **kwargs
) -> float:
    """
    Clipped polynomial: quadratic decay with hard cutoff at zero.

    Args:
        length: Segment length
        target_length: Preferred length (peak of distribution)
        coefficient: Decay rate (larger = faster decay)

    Returns:
        max(1 - coefficient * (length - target_length)^2, 0)

    Example:
        >>> clipped_polynomial_prior(100, target_length=100, coefficient=0.01)
        1.0
        >>> clipped_polynomial_prior(90, target_length=100, coefficient=0.01)
        0.0
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
        max_length: Start of decay region
        decay_rate: Decay rate (larger = faster decay)

    Returns:
        1.0 if length <= max_length, else exp(-decay_rate * (length - max_length))

    Example:
        >>> exponential_prior(50, max_length=100, decay_rate=0.1)
        1.0
        >>> exponential_prior(110, max_length=100, decay_rate=0.1)
        0.367...
    """
    if length <= max_length:
        return 1.0
    else:
        return float(np.exp(-decay_rate * (length - max_length)))


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

    Creates a trapezoidal distribution:
    - 0 below min_length
    - Linear ramp from min_length to target_length
    - Constant 1.0 from target_length to max_length
    - Linear ramp down after max_length

    Args:
        length: Segment length
        min_length: Minimum length (0 probability below)
        target_length: Start of plateau
        max_length: End of plateau

    Returns:
        Piecewise linear probability

    Example:
        >>> piecewise_linear_prior(50, min_length=20, target_length=40, max_length=100)
        1.0
        >>> piecewise_linear_prior(10, min_length=20, target_length=40, max_length=100)
        0.0
    """
    if length < min_length:
        return 0.0
    elif length < target_length:
        # Ramp up
        return (length - min_length) / (target_length - min_length)
    elif length <= max_length:
        # Plateau
        return 1.0
    else:
        # Ramp down - decay to 0 at 2*max_length - target_length
        decay_end = 2 * max_length - target_length
        if length >= decay_end:
            return 0.0
        return 1.0 - (length - max_length) / (max_length - target_length)


def create_prior_function(
    prior: Union[str, Callable],
    prior_params: Dict[str, Any] = None
) -> Callable[[int], float]:
    """
    Create a prior function from name or callable.

    Args:
        prior: Prior name (string) or custom callable
        prior_params: Parameters for the prior function

    Returns:
        Prior function that takes length and returns probability

    Example:
        >>> # Using named prior
        >>> prior_fn = create_prior_function("uniform", {"max_length": 100})
        >>> prior_fn(50)
        1.0

        >>> # Using custom function
        >>> custom = lambda length: 0.5 if length < 100 else 0.1
        >>> prior_fn = create_prior_function(custom)
        >>> prior_fn(50)
        0.5
    """
    prior_params = prior_params or {}

    if callable(prior):
        # Custom prior function provided
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


def validate_prior_params(prior: str, prior_params: Dict[str, Any]) -> None:
    """
    Validate that required parameters are provided for a named prior.

    Args:
        prior: Prior name
        prior_params: Parameters dictionary

    Raises:
        ValueError: If required parameters are missing
    """
    required_params = {
        "uniform": ["max_length"],
        "gaussian": ["target_length"],
        "clipped_polynomial": ["target_length"],
        "exponential": ["max_length"],
        "piecewise_linear": ["min_length", "target_length", "max_length"]
    }

    if prior not in required_params:
        return  # Unknown prior, will fail later

    missing = [p for p in required_params[prior] if p not in prior_params]
    if missing:
        raise ValueError(
            f"Prior '{prior}' requires parameters: {missing}. "
            f"Got: {list(prior_params.keys())}"
        )
