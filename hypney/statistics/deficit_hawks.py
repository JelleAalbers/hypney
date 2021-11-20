import hypney
import numpy as np

export, __all__ = hypney.exporter()


@export
class DeficitHawk(hypney.Statistic):

    def _compute(self, params):
        return min(self.score_cuts(params))

    def score_cuts(self, params):
        raise NotImplementedError

    # For extra info / debugging

    def best_cut(self, params):
        best_i = self.model.backend.argmin(self.score_cuts(params))
        return self._cut_info(params, best_i)

    def _cut_info(self, params, cut_i):
        raise NotImplementedError



@export
class FixedRegionHawk(DeficitHawk):

    cuts: tuple
    cut_lrs: list

    def __init__(self, *args, cuts, **kwargs):
        self.cuts = cuts
        super().__init__(*args, **kwargs)

    def _init_data(self):
        super()._init_data()
        self.cut_lrs = [
            hypney.statistics.SignedPLR(self.model.cut(cut, cut_data=True))
            for cut in self.cuts]

    def score_cuts(self, params):
        return [lr._compute(params) for lr in self.cut_lrs]

    def _cut_info(self, params, cut_i):
        return dict(
            i=cut_i,
            cut=self.cuts[cut_i],
            lr=self.cut_lrs[cut_i],
            model=self.cut_lrs[cut_i].model)

    # def extra_hash_dict(self):
    #     return dict(regions=self.regions)