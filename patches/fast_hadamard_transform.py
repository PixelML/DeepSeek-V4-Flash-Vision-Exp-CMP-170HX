"""Pure-torch shim for the fast_hadamard_transform package (SM80 fallback).

model.py imports this by name; patches/ is on sys.path ahead of the reference
code, so this module satisfies 'from fast_hadamard_transform import
hadamard_transform' without the CUDA extension build.
"""

from sm80_fallbacks import hadamard_transform  # noqa: F401
