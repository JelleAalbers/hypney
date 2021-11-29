
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
