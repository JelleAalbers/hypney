import numpy as np
import pytest
from scipy import stats

import hypney.all as hp


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
    n = np.asarray(n)
    return np.where(n == 0, np.zeros_like(n), stats.chi2.ppf(1 - cl, 2 * n + 2) / 2)


def test_poisson_upper_limit():
    m = hp.uniform(data=np.array([]))
    stat = hp.Count(m)

    ul = hp.UpperLimit(stat, poi="rate", anchors=[0, 5], cl=0.9, use_cdf=False)()
    np.testing.assert_allclose(ul, poisson_ul(0))

    ul = hp.UpperLimit(stat, poi="rate", anchors=[0, 5], cl=0.9, use_cdf=True)()
    np.testing.assert_allclose(ul, poisson_ul(0))

    # Invalid anchor
    with pytest.raises(ValueError):
        hp.UpperLimit(stat, poi="rate", anchors=[-5, 5], cl=0.9)()

    # Limit is above last anchor
    with pytest.raises(hp.FullIntervalError):
        hp.UpperLimit(stat, poi="rate", anchors=[0, 2], cl=0.9)()

    # Limit is below first anchor
    with pytest.raises(hp.EmptyIntervalError):
        hp.UpperLimit(stat.set(data=np.ones(0)), poi="rate", anchors=[20, 40], cl=0.9)()

    # Test anchors from toy MC process are preserved.
    # -- Actually, I don't see how to do this with reparametrized distributions
    # stat2 = stat.set(dist=stat.interpolate_dist_from_toys(anchors=dict(rate=(0, 5))))
    # ul = hypney.estimators.UpperLimit(stat2, poi="rate", cl=0.9)()
    # assert 0 < ul < 5

    # Test case where statistic decreases as parameter increases
    # Data is still empty, so UL = bestfit = highest possible value = 0
    mneg = m.reparametrize(
        lambda params: dict(rate=-params["neg_rate"]),
        param_specs=(
            hp.Parameter(name="neg_rate", default=0, min=-float("inf"), max=0),
        ),
    )
    stat3 = hp.Count(mneg)
    ul = hp.UpperLimit(stat3, poi="neg_rate", anchors=[-20, 0], cl=0.9, sign=-1)()
    assert ul == -poisson_ll(0) == 0

    # Now data is not empty, UL finite (and negative)
    ul = hp.UpperLimit(
        stat3(data=np.ones(50)), poi="neg_rate", anchors=[-50, -20], cl=0.9, sign=-1
    )()
    np.testing.assert_allclose(ul, -poisson_ll(50))


def test_poisson_lower_limit():
    m = hp.uniform(data=np.array([]))
    stat = hp.Count(m)

    ll = hp.LowerLimit(stat, poi="rate", anchors=[0, 5], cl=0.9, use_cdf=False)()
    np.testing.assert_allclose(ll, poisson_ll(0))

    # Test case where statistic decreases as parameter increases
    mneg = m.reparametrize(
        lambda params: dict(rate=-params["neg_rate"]),
        param_specs=(
            hp.Parameter(name="neg_rate", default=0, min=-float("inf"), max=0),
        ),
    )
    stat3 = hp.Count(mneg)
    ll = hp.LowerLimit(stat3, poi="neg_rate", anchors=[-20, 0], cl=0.9, sign=-1)()
    np.testing.assert_allclose(ll, -poisson_ul(0))

    ll = hp.LowerLimit(
        stat3(data=np.ones(50)), poi="neg_rate", anchors=[-100, -50], cl=0.9, sign=-1
    )()
    np.testing.assert_allclose(ll, -poisson_ul(50))


def test_poisson_central_interval():
    m = hp.uniform(data=np.array([]))
    stat = hp.Count(m)
    ll, ul = hp.CentralInterval(stat, poi="rate", anchors=[0, 5], cl=0.8)()
    assert ll == 0
    np.testing.assert_allclose(ul, poisson_ul(0))

    ll, ul = hp.CentralInterval(
        stat(data=np.zeros(50)), poi="rate", anchors=[10, 50, 90], cl=0.8
    )()
    np.testing.assert_allclose(ll, poisson_ll(50))
    np.testing.assert_allclose(ul, poisson_ul(50))
