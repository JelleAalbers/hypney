import pickle
import tempfile

import numpy as np
import pytest
from scipy import stats

import hypney


def test_uniform():
    m = hypney.Uniform()
    assert m.rate() == hypney.DEFAULT_RATE_PARAM.default
    assert m.rate(params=dict(rate=100.0)) == 100.0

    # Test setting params on init
    m = hypney.Uniform(rate=100)
    assert m.rate() == 100.0
    assert m.simulate().shape[0] > 0

    # Test simulate
    m = hypney.Uniform(rate=0)
    data = m.simulate()
    assert data.shape == (0, 1)
    data = m.rvs(size=5)
    assert data.shape == (5, 1)

    # Test different data formats and pdf
    assert (
        m.pdf(data=0)
        == m.pdf(data=[0])
        == m.pdf(data=np.array([0]))
        == m.pdf(data=np.array([[0]]))
    )
    assert m.pdf(data=0) == 1.0

    # Test cdf
    assert np.all(m.cdf(data=[0.0, 0.5, 1.0]) == np.array([0.0, 0.5, 1.0]))

    # Test diff rate
    assert m.diff_rate(data=0.0) == 0.0

    # Test making models with new defaults
    m2 = m(rate=50)
    assert m2 != m
    assert m2.rate() == 50.0

    # Test cut efficiency
    m = hypney.Uniform(rate=100)
    assert m.cut_efficiency(cut=(0, 0.5)) == 0.5
    assert m.rate(cut=(0, 0.5)) == 50.0

    # Test freezing data
    m = hypney.Uniform(rate=100)
    with pytest.raises(Exception):
        m.pdf()
    m2 = m(data=0)
    assert m2 is not m
    assert m2.pdf() == 1.0
    assert m2(data=1) not in (m, m2)

    # Models can be pickled and unpickled
    m = hypney.Uniform(loc=0.5)
    with tempfile.NamedTemporaryFile() as tempf:
        fn = tempf.name
        with open(fn, mode="wb") as f:
            pickle.dump(m, f)
        with open(fn, mode="rb") as f:
            m = pickle.load(f)
    assert m.defaults["loc"] == 0.5


def test_beta():
    m = hypney.Beta(a=0.5, b=0.5, rate=100)

    data = m.simulate()
    assert len(data)
    assert data.min() > 0
    assert data.max() < 1

    np.testing.assert_equal(m.pdf(data=data), stats.beta(a=0.5, b=0.5).pdf(data[:, 0]))
    assert m.rate() == 100.0

    m2 = m(rate=20, loc=-100, scale=10)
    assert m2.defaults["a"] == 0.5
    assert m2.rate() == 20.0
    assert m2.dist == m.dist
    data = m2.simulate()
    assert len(data)
    assert data.min() < 0
    assert (data.max() - data.min()) > 1


def test_poisson():
    m = hypney.Poisson(mu=3, rate=100)
    data = m.simulate()
    np.testing.assert_equal(m.pdf(data=data), stats.poisson(mu=3).pmf(data[:, 0]))
    assert m.rate() == 100.0


def test_from_histogram():
    hist, edges = np.array([1, 2, 1]), np.array([0, 1, 2, 3])
    m = hypney.From1DHistogram(hist, edges)
    data = m.simulate()
    np.testing.assert_equal(
        m.pdf(data=data), stats.rv_histogram((hist, edges),).pdf(data[:, 0])
    )


def test_mixture():
    m1 = hypney.Uniform(rate=40)
    m2 = hypney.Uniform(rate=20)
    m3 = hypney.Uniform(rate=30)

    mix = hypney.Mixture(m1, m2)
    # assert mix.param_names == (
    #     "m0_expected_events", "m0_loc", "m1_expected_events")

    assert mix.rate() == 60.0
    assert mix.rate(params=dict(m0_rate=0)) == 20.0

    assert mix.pdf(data=0) == 1.0

    assert mix.diff_rate(data=0) == 60.0

    assert np.all(mix.cdf(data=[0.0, 0.5, 1.0]) == np.array([0.0, 0.5, 1.0]))

    assert mix.simulate().shape[0] > 0
    assert mix.rvs(size=50).shape[0] > 0

    # Test forming mixtures by +
    mix2 = m1 + m2
    assert mix2.diff_rate(data=0) == 60.0

    mix3 = m3 + mix2
    assert mix3.diff_rate(data=0) == 90.0
    assert len(mix3.models) == 3, "Should unpack mixtures"
    mix4 = m1 + m2 + m3
    assert mix4.diff_rate(data=0) == 90.0

    # TODO: Test parameter sharing
