# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
import numpy as np

from .eval import Array


def split_rng_key(base_rng_key, num_splits=2):
    # Poorly faking jax's random module here
    rng = np.random.default_rng(base_rng_key)
    return [rng.integers(0x7FFFFFFF) for _ in range(num_splits)]


def rand_uniform(shape, low=0.0, high=1.0, *, rng_key):
    rng = np.random.default_rng(rng_key)
    return Array(rng.uniform(low, high, shape))
