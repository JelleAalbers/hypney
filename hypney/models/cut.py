import itertools
import math

import hypney

import eagerpy as ep
import numpy as np

export, __all__ = hypney.exporter()


@export
class NoCut:
    """Instruction to not cut data"""

    pass


@export
class CutModel(hypney.WrappedModel):
    """A model limiting observables to a rectangular region

    Args (beyond those of Model):
     - orig_model: original model taking transformed parameters
     - cut: NoCut, or a tuple of (low, right) bounds for each observables
        None can be put in place for +-inf
    """

    cut = NoCut
    _passes_cut: ep.Tensor

    def __init__(
        self, orig_model: hypney.Model = hypney.NotChanged, cut=NoCut, *args, **kwargs
    ):
        self.cut = self.validate_cut(cut)
        if cut != NoCut:
            kwargs.setdefault(
                "observables",
                tuple(
                    [
                        obs._replace(min=_min, max=_max)
                        for (_min, _max), obs in zip(self.cut, orig_model.observables)
                    ]
                ),
            )
        super().__init__(orig_model=orig_model, *args, **kwargs)

    def _init_data(self):
        # Compute which events pass cut
        passed = 1 + 0 * self.data[:, 0]
        if self.cut == NoCut:
            return passed
        for dim_i, (l, r) in enumerate(self.cut):
            passed *= (l <= self.data[:, dim_i]) * (self.data[:, dim_i] < r)
        self._passes_cut = passed

        return super()._init_data()

    # New methods specific to CutModel

    def validate_cut(self, cut):
        """Return a valid cut, i.e. NoCut or tuple of (l, r) tuples for each observable.
        """
        if cut == NoCut:
            return cut
        if cut is None:
            raise ValueError("None is not a valid cut, use NoCut")
        if isinstance(cut, (list, np.ndarray)):
            cut = tuple(cut)
        if isinstance(cut, dict):
            cut = tuple([cut.get(pname, (None, None)) for pname in self.param_names])
        if not isinstance(cut, tuple):
            raise ValueError("Cut should be a tuple")
        if len(cut) == 2 and len(self.observables) == 1:
            cut = (cut,)
        if len(cut) != len(self.observables):
            raise ValueError("Cut should have same length as observables")
        if any([len(c) != 2 for c in cut]):
            raise ValueError("Cut should be a tuple of 2-tuples")
        # Replace any None's by +-inf
        cut = tuple(
            [
                (-float("inf") if l is None else l, float("inf") if r is None else r,)
                for l, r in cut
            ]
        )
        return cut

    def cut_efficiency(self, params: dict = None, **kwargs) -> float:
        params = self.validate_params(params, **kwargs)
        return self._cut_efficiency(params).raw

    def _cut_efficiency(self, params: dict):
        if self.cut is NoCut:
            return 1.0
        if not hasattr(self, "cdf"):
            raise NotImplementedError("Nontrivial cuts require a cdf")
        # Evaluate CDF at rectangular endpoints, add up with alternating signs,
        # always + in upper right.
        # TODO: Not sure this is correct for n > 2!
        # (for n=3 looks OK, for higher n I can't draw/visualize)
        signs = [math.prod(x) for x in self._signs()]
        points = [
            [c[int(0.5 * j + 0.5)] for (c, j) in zip(self.cut, _signs)]
            for _signs in self._signs()
        ]
        # Careful here, super().cdf_ would call our (not implemented) _cdf!
        return ((signs) * self._orig_model.cdf_(data=points, params=params)).sum()

    def _signs(self):
        ndim = len(self.observables)  # TODO: make attribute
        base = (-1) ** len(self.observables)
        yield from itertools.product(*[[base, -base]] * ndim)

    def apply_cut(self, data=hypney.NotChanged):
        return self.apply_cut_(data=data).raw

    def apply_cut_(self, data=hypney.NotChanged):
        return self(data=data)._apply_cut()

    def _apply_cut(self):
        return self.data[self._passes_cut]

    # Simulation

    def _simulate(self, params) -> np.ndarray:
        return self._apply_cut(self._simulate(params))

    def _rvs(self, size) -> np.ndarray:
        # Simulate significant excess, cut, hope it is enough?
        raise NotImplementedError

    # Methods not using data

    def _rate(self, params: dict):
        return self._orig_model._rate(params) * self._cut_efficiency(params)

    # Methods using data / quantiles

    def _diff_rate(self, params: dict):
        return ep.where(
            self._passes_cut,
            # rate fell but pdf rose
            super()._diff_rate(params=params),
            0,
        )

    def _pdf(self, params: dict):
        return ep.where(
            self._passes_cut, super()._pdf(params) / self._cut_efficiency(params), 0
        )

    def _cdf(self, params: dict):
        raise NotImplementedError

    def _ppf(self, params: dict):
        raise NotImplementedError
