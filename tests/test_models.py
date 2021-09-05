import pickle
import tempfile

import numpy as np
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
    data = m.simulate_n(5)
    assert data.shape == (5, 1)

    # Test different data formats and pdf
    assert m.pdf(0) == m.pdf([0]) == m.pdf(np.array([0])) == m.pdf(np.array([[0]]))
    assert m.pdf(0) == 1.0

    # Test cdf
    assert np.all(m.cdf([0.0, 0.5, 1.0]) == np.array([0.0, 0.5, 1.0]))

    # Test diff rate
    assert m.diff_rate(0.0) == 0.0

    # Test making models with new defaults
    m2 = m(rate=50)
    assert m2 != m
    assert m2.rate() == 50.0

    # Test cut efficiency
    m = hypney.Uniform(rate=100)
    assert m.cut_efficiency(cut=(0, 0.5)) == 0.5
    assert m.rate(cut=(0, 0.5)) == 50.0

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
    np.testing.assert_equal(m.pdf(data), stats.beta(a=0.5, b=0.5).pdf(data[:, 0]))
    assert m.rate() == 100.0
    assert len(data)
    assert data.min() > 0
    assert data.max() < 1

    m2 = m(rate=20, loc=-100, scale=10)
    assert m2.defaults["a"] == 0.5
    assert m2.rate() == 20.0
    assert m2.dist == m.dist
    data = m2.simulate()
    assert len(data)
    assert data.min() < 0
    assert (data.max() - data.min()) > 1


def test_mixture():
    m1 = hypney.Uniform(rate=40)
    m2 = hypney.Uniform(rate=20)
    m3 = hypney.Uniform(rate=30)

    mix = hypney.Mixture(m1, m2)
    # assert mix.param_names == (
    #     "m0_expected_events", "m0_loc", "m1_expected_events")

    assert mix.rate() == 60.0
    assert mix.rate(params=dict(m0_rate=0)) == 20.0

    assert mix.pdf(0) == 1.0

    assert mix.diff_rate(0) == 60.0

    assert np.all(mix.cdf([0.0, 0.5, 1.0]) == np.array([0.0, 0.5, 1.0]))

    assert mix.simulate().shape[0] > 0
    assert mix.simulate_n(50).shape[0] > 0

    # Test forming mixtures by +
    mix2 = m1 + m2
    assert mix2.diff_rate(0) == 60.0

    mix3 = m3 + mix2
    assert mix3.diff_rate(0) == 90.0
    assert len(mix3.models) == 3, "Should unpack mixtures"
    mix4 = m1 + m2 + m3
    assert mix4.diff_rate(0) == 90.0

    # TODO: Test parameter sharing
