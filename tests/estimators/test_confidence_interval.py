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

    ul = hypney.estimators.UpperLimit(stat, poi="rate", anchors=[0, 5], cl=0.9)
    np.testing.assert_allclose(ul, poisson_ul(0))

    # Invalid anchor
    with pytest.raises(ValueError):
        hypney.estimators.UpperLimit(stat, poi="rate", anchors=[-5, 5], cl=0.9)

    # Limit is above last anchor
    with pytest.raises(ValueError):
        hypney.estimators.UpperLimit(stat, poi="rate", anchors=[0, 2], cl=0.9)

    # Limit is below first anchor
    with pytest.raises(ValueError):
        hypney.estimators.UpperLimit(
            stat.set(data=np.ones(10)), poi="rate", anchors=[0, 2], cl=0.9
        )

    # Test anchors from toy MC process are preserved
    stat2 = stat.set(dist=stat.interpolate_dist_from_toys(anchors=dict(rate=(0, 5))))
    ul = hypney.estimators.UpperLimit(stat2, poi="rate", cl=0.9)
    assert 0 < ul < 5
