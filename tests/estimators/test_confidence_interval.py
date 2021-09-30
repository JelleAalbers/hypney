import numpy as np
import pytest
from scipy import stats

import hypney


def poisson_ul(n, mu_bg=0, cl=0.9):
    """Upper limit on mu_signal, from observing n events
    where mu_bg background events were expected

    NB: can be negative if mu_bg large enough.
    It's your responsibility to clip to 0...
    """
    return stats.chi2.ppf(cl, 2 * n + 2) / 2 - mu_bg


def test_poisson_upper_limit():
    m = hypney.models.uniform(data=np.array([]))
    stat = hypney.statistics.Count(m)

    ul = hypney.estimators.UpperLimit(stat, anchors=[0, 5])()
    np.testing.assert_allclose(ul, poisson_ul(0))

    with pytest.raises(ValueError):
        hypney.estimators.UpperLimit(stat, anchors=[-5, 5])()
