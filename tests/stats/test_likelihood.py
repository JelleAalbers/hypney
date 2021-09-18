import numpy as np
from scipy import stats

import hypney


def test_likelihood():
    # Single event at x=0
    m = hypney.models.Norm(rate=1)
    data = np.array([0,])

    lf = hypney.stats.LogLikelihood(m, data=data)
    assert len(lf.data) == 1

    assert lf() == -1 + stats.norm.logpdf(0)
    assert lf(params=dict(rate=2)) == -2 + np.log(2 * stats.norm.pdf(0))


def test_lr():
    # Single event at x=0
    m = hypney.models.Norm(rate=1)
    data = np.array([0,])

    # Likelihood with all parameters free
    lr = hypney.stats.LikelihoodRatio(m, data=data)

    # Best fit should be a very sharp Gaussian with rate = 1
    min_scale = m.param_spec_for("scale").min
    assert np.isclose(lr.bestfit["rate"], 1)
    assert np.isclose(lr.bestfit["scale"], min_scale)
    assert np.isclose(lr.bestfit["loc"], 0)

    double_rate = lr(params={**lr.bestfit, **dict(rate=2)}).numpy()
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

    # # Likelihood with only the rate free
    lr = hypney.stats.LikelihoodRatio(m.filter_params(keep="rate"), data=data)
    assert np.isclose(lr.bestfit["rate"], 1)
    assert len(lr.bestfit) == 1
