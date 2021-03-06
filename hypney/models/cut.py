import itertools
import math
import typing as ty

import hypney

import eagerpy as ep
import numpy as np
from scipy import stats

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

    # TODO: add docs warning about not autocutting data

    _cut: ty.Any = NoCut  # .cut is a model Method
    _passes_cut: ep.Tensor
    _fixed_eff = None

    def __init__(
        self,
        orig_model: hypney.Model = hypney.NotChanged,
        cut=NoCut,
        cut_type=hypney.DEFAULT_CUT_TYPE,
        cut_data=False,
        fixed_cut_efficiency=None,
        *args,
        **kwargs,
    ):
        self._cut = self.validate_cut(cut)
        self.cut_type = cut_type
        if cut != NoCut:
            kwargs.setdefault(
                "observables",
                tuple(
                    [
                        obs._replace(min=_min, max=_max)
                        for (_min, _max), obs in zip(self._cut, orig_model.observables)
                    ]
                ),
            )

        # TODO: sneaking an arg to _init_data...
        # TODO: this is bad. cut_data only works on first init..
        # I can see this is a nice optimization, but not what people expect
        self._cut_data = cut_data
        super().__init__(orig_model=orig_model, *args, **kwargs)
        self._cut_data = False

        if fixed_cut_efficiency:
            # Don't try to vectorize ... :-(
            self._fixed_eff = fixed_cut_efficiency

    def _init_data(self):
        super()._init_data()

        # Compute which events pass cut
        # Start with all passed
        passed = self.data.ones(len(self.data)) > 0
        if self._cut == NoCut:
            return passed
        for dim_i, (l, r) in enumerate(self._cut):
            x = self.data[:, dim_i]
            if self.cut_type == "open":
                passed *= (l < x) * (x < r)
            elif self.cut_type == "halfopen":
                passed *= (l <= x) * (x < r)
            elif self.cut_type == "closed":
                passed *= (l <= x) * (x <= r)
            else:
                raise ValueError(f"Unkown cut type {self.cut_type}")

        self._passes_cut = passed

        if self._cut_data:
            self.data = self.data[self._passes_cut]
            # Could create new bool instead, maybe easier/clearer..
            self._passes_cut = self._passes_cut[self._passes_cut]

    def _init_quantiles(self):
        # quantiles for the original model depend on parameters;
        # can't compute them until params known
        # (I suppose we could compute them for the defaults, but why bother)
        pass

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
        if not all([isinstance(a, (int, float, type(None))) for c in cut for a in c]):
            raise ValueError("Cut should be a tuple of 2-tuples of scalars/None")
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

    def _signs(self):
        yield from itertools.product(*[[-1, 1]] * self.n_dim)

    def _corner_points(self):
        """Return (n_corners, n_dim) nested list of corner points"""
        return [
            [c[int(0.5 * _sign + 0.5)] for (c, _sign) in zip(self._cut, _signs)]
            for _signs in self._signs()
        ]

    def _corners_cdf(self, params: dict):
        """Return ({batch_shape}, n_corners)) of CDF at the self._corner_points()"""
        # Must call high-level CDF here; low-level one would broadcast over
        # final 1 in the expanded batch shape
        return self._orig_model.cdf(data=self._corner_points(), params=params)

    def _cut_efficiency(self, params: dict, corners_cdf=None):
        if self._fixed_eff is not None:
            return self._fixed_eff
        if self._cut is NoCut:
            return 1.0
        if not hasattr(self, "cdf"):
            raise NotImplementedError("Nontrivial cuts require a cdf")
        if any([o.integer for o in self.observables]):
            raise NotImplementedError(
                "Cut efficiency not yet implemented for discrete models"
                " (would depend on cut_type"
            )

        # Evaluate CDF at rectangular endpoints, add up with alternating signs,
        # + in upper right.
        # TODO: Not sure this is correct for n > 2!
        # (for n=3 looks OK, for higher n I can't draw/visualize)

        if corners_cdf is None:
            corners_cdf = self._corners_cdf(params)
        # Get the sign for each corner; reshape to ({batch_shape}, n_corners)
        corner_signs = self._to_tensor([math.prod(signs) for signs in self._signs()])
        # Compute efficiency:
        result = (corner_signs * corners_cdf).sum(axis=-1)
        # assert result.max() <= 1
        return result

    def apply_cut(self, data=hypney.NotChanged):
        return self(data=data)._apply_cut().raw

    def _apply_cut(self):
        return self.data[self._passes_cut]

    # Simulation

    def _simulate(self, params) -> np.ndarray:
        return self.apply_cut(data=self._orig_model._simulate(params))

    def _rvs(self, size, params: dict) -> np.ndarray:
        if size == 0:
            return np.zeros((0, len(self.observables)))
        # Simulate an excess, enough that we almost always complete in one go
        n_needed = int(
            size
            + stats.nbinom(p=self._cut_efficiency(params).numpy(), n=size).ppf(1 - 1e-6)
        )
        while True:
            d = self._orig_model.rvs(size=n_needed, params=params)
            d = self.apply_cut(d)
            if len(d) >= size:
                break
        return d[:size]

    # Methods not using data

    def _rate(self, params: dict):
        return self._orig_model._rate(params) * self._cut_efficiency(params)

    # Methods using data / quantiles

    def _diff_rate(self, params: dict):
        return ep.where(
            self._passes_cut,
            # rate fell but pdf rose
            self._orig_model._diff_rate(params=params),
            0,
        )

    def _log_pdf(self, params: dict):
        return ep.where(
            self._passes_cut,
            self._orig_model._logpdf(params)
            - self._tensorlib.log(self._cut_efficiency(params)),
            -float("inf"),
        )

    def _pdf(self, params: dict):
        return ep.where(
            self._passes_cut,
            self._orig_model._pdf(params) / self._cut_efficiency(params),
            0,
        )

    def _cdf(self, params: dict):
        if self.n_dim > 1:
            raise NotImplementedError("nD cut CDF still todo...")
        corners_cdf = self._corners_cdf(params)
        c_low, c_high = corners_cdf[..., 0], corners_cdf[..., 1]
        return ((self._orig_model._cdf(params) - c_low) / (c_high - c_low)).clip(0, 1)

    def _ppf(self, params: dict):
        if self.n_dim > 1:
            raise NotImplementedError("nD cut PPF still todo...")
        corners_cdf = self._corners_cdf(params)
        c_low, c_high = corners_cdf[..., 0], corners_cdf[..., 1]

        # self.quantiles == (orig_quantiles - c_low).clip(0, None) / (c_high - c_low)
        # Cut always shrinks region and 0 <= quantiles <= 1, so clip never binds
        # ppf(0) = c_low.
        orig_quantiles = self.quantiles * (c_high - c_low) + c_low

        # Don't call the external .ppf, it would expand the parameters once more.
        result = ep.astensor(
            self._orig_model.set(quantiles=orig_quantiles)._ppf(params=params)
        )

        return result

    def _min(self, params: dict):
        assert self.n_dim == 1, "Support methods assume univariate dists"
        return hypney.utils.misc.flexible_clip(
            self._orig_model.min(params=params), self._cut[0][0]
        )

    def _max(self, params: dict):
        assert self.n_dim == 1, "Support methods assume univariate dists"
        return hypney.utils.misc.flexible_clip(
            self._orig_model.max(params=params), None, self._cut[0][1]
        )
