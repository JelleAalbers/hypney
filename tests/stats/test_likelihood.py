import numpy as np
from scipy import stats

import numpy as np

import hypney


def test_likelihood():
    # Five events at x=0
    m = hypney.models.norm(rate=1)
    data = np.array([0, 0, 0, 0, 0])

    lf = hypney.statistics.LogLikelihood(m, data=data)
    assert len(lf.data) == 5

    np.testing.assert_almost_equal(lf.compute(), -1 + 5 * stats.norm.logpdf(0))

    np.testing.assert_almost_equal(
        lf.compute(params=dict(rate=2)), -2 + 5 * np.log(2 * stats.norm.pdf(0))
    )

    # Test vectorization
    rates = np.linspace(0.1, 10, 10)
    np.testing.assert_equal(
        lf.compute(rate=rates), np.array([lf.compute(rate=x) for x in rates])
    )
    locs = np.linspace(-2, 2, 10)
    np.testing.assert_equal(
        lf.compute(loc=locs), np.array([lf.compute(loc=x) for x in locs])
    )


def test_lr():
    # One event at x=0
    m = hypney.models.norm(rate=1)
    data = np.array([0,])

    # Likelihood with all parameters free
    lr = hypney.statistics.LikelihoodRatio(m, data=data)

    # Best fit should be a very sharp Gaussian with rate = 1
    min_scale = m.param_spec_for("scale").min
    assert np.isclose(lr.bestfit["rate"], 1)
    assert np.isclose(lr.bestfit["scale"], min_scale)
    assert np.isclose(lr.bestfit["loc"], 0)

    double_rate = lr.compute(params={**lr.bestfit, **dict(rate=2)})
    assert np.isclose(
        double_rate,
        -2
        * (
            # LL of hypothesis ...
            (-2 + np.log(2 * stats.norm(0, min_scale).pdf(0)))
            # ... minus LL of bestfit
            - (-1 + stats.norm(0, min_scale).logpdf(0))
        ),
    )

    # Likelihood with only the rate free
    lr = hypney.statistics.LikelihoodRatio(m.fix_except("rate"), data=data)
    assert len(lr.bestfit) == 1
    assert np.isclose(lr.bestfit["rate"], 1)
