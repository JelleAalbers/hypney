import math

import numpy as np

import hypney

export, __all__ = hypney.exporter()

try:
    import numba

    have_numba = True
except ImportError:
    have_numba = False
    print("Get numba man, it's great")


@export
def maybe_jit(f):
    if have_numba:
        return numba.jit(f)
    else:
        return f


# See https://stackoverflow.com/questions/44346188
# and https://stackoverflow.com/questions/62056035
# (21! and above are no longer integers)
FACTORIALS = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ],
    dtype="int64",
)


@maybe_jit
def factorial(n):
    if np.max(n) > 20:
        raise OverflowError("Factorials of n>20 are not int64s")
    return FACTORIALS[n]
