import numpy as np
import pytest
from scipy import stats

import hypney
from hypney.basics import ParameterSpec


def poisson_ul(n, mu_bg=0, cl=0.9):
    """Upper limit on mu_signal, from observing n events
    where mu_bg background events were expected

    NB: can be negative if mu_bg large enough.
    It's your responsibility to clip to 0...
    """
    # Adapted from https://stackoverflow.com/a/14832525
    return stats.chi2.ppf(cl, 2 * n + 2) / 2 - mu_bg


def poisson_ll(n, cl=0.9):
    # Adapted from https://stackoverflow.com/a/14832525
    # Checked through
    #     n = np.arange(0, 100)
    #     stats.poisson(poisson_ul(n)).cdf(n)
    # (and similarly for upper limit)
    return stats.chi2.ppf(1 - cl, 2 * n + 2) / 2


def test_poisson_upper_limit():
    m = hypney.models.uniform(data=np.array([]))
    stat = hypney.statistics.Count(m)

    ul = hypney.estimators.UpperLimit(
        stat, poi="rate", anchors=[0, 5], cl=0.9, use_cdf=False
    )
    np.testing.assert_allclose(ul, poisson_ul(0))

    ul = hypney.estimators.UpperLimit(
        stat, poi="rate", anchors=[0, 5], cl=0.9, use_cdf=True
    )
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

    # Test case where statistic decreases as parameter increases
    # Data is still empty, so UL should be all the way at 0
    mneg = m.reparametrize(
        lambda params: dict(rate=-params["neg_rate"]),
        param_specs=(
            ParameterSpec(name="neg_rate", default=0, min=-float("inf"), max=0),
        ),
    )
    stat3 = hypney.statistics.Count(mneg)
    ul = hypney.estimators.UpperLimit(
        stat3, poi="neg_rate", anchors=[-20, 0], cl=0.9, sign=-1
    )
    assert ul == 0.0

    # Now data is not empty, UL finite (and negative)
    ul = hypney.estimators.UpperLimit(
        stat3(data=np.ones(50)), poi="neg_rate", anchors=[-50, -20], cl=0.9, sign=-1
    )
    np.testing.assert_allclose(ul, -poisson_ll(50))
