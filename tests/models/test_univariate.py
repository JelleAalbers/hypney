import pickle
import tempfile

import numpy as np
import pytest
from scipy import stats

import hypney


def test_uniform():
    m = hypney.models.uniform()
    assert m.rate() == hypney.DEFAULT_RATE_PARAM.default
    assert m.rate(params=dict(rate=100.0)) == 100.0

    # Test setting params on init
    m = hypney.models.uniform(rate=100)
    assert m.rate() == 100.0
    assert m.simulate().shape[0] > 0

    # Test simulate
    m = hypney.models.uniform(rate=0)
    data = m.simulate()
    assert data.shape == (0, 1)
    data = m.rvs(size=5)
    assert data.shape == (5, 1)

    # Test different data formats and pdf
    assert m.pdf(0) == m.pdf([0]) == m.pdf(np.array([0])) == m.pdf(np.array([[0]]))
    assert m.pdf(0) == 1.0

    # Ensure we don't get back whacky types (0-element arrays, ep-wrapped scalars)
    assert isinstance(m.pdf(0), (float, np.float64))

    assert m.logpdf(0) == stats.uniform().logpdf(0)

    # Test cdf and ppf
    np.testing.assert_array_equal(m.cdf([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))
    np.testing.assert_array_equal(m.ppf([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))

    # Test diff rate
    m = hypney.models.uniform(rate=2)
    assert m.diff_rate(0.0) == 2.0
    assert m.log_diff_rate(0.0) == np.log(2.0)

    # Test mean and std
    assert m.mean() == 0.5
    assert m.mean(loc=1, scale=2) == 2
    assert m.std() == stats.uniform().std()

    # Test making models with new defaults
    m2 = m(rate=50)
    assert m2 != m
    assert m2.rate() == 50.0

    # Test freezing data
    m = hypney.models.uniform(rate=100)
    with pytest.raises(Exception):
        m.pdf()
    m2 = m(data=0)
    assert m2 is not m
    assert m2.pdf() == 1.0
    assert m2(data=1) not in (m, m2)

    # Models can be pickled and unpickled
    m = hypney.models.uniform(loc=0.5)
    with tempfile.NamedTemporaryFile() as tempf:
        fn = tempf.name
        with open(fn, mode="wb") as f:
            pickle.dump(m, f)
        with open(fn, mode="rb") as f:
            m = pickle.load(f)
    assert m.defaults["loc"] == 0.5


def test_beta():
    m = hypney.models.beta(a=0.5, b=0.5, rate=100)

    data = m.simulate()
    assert len(data)
    assert data.min() > 0
    assert data.max() < 1

    np.testing.assert_equal(m.pdf(data), stats.beta(a=0.5, b=0.5).pdf(data[:, 0]))
    assert m.rate() == 100.0

    m2 = m(rate=20, loc=-100, scale=10)
    assert m2.defaults["a"] == 0.5
    assert m2.rate() == 20.0
    assert m2._dists["scipy"] == m._dists["scipy"]
    data = m2.simulate()
    assert len(data)
    assert data.min() < 0
    assert (data.max() - data.min()) > 1

    params = dict(a=0.5, b=0.5, loc=-100, scale=10)
    data = [-100, -93, -98, -34]
    np.testing.assert_equal(m2.cdf(data), stats.beta(**params).cdf(data))
    quantiles = [0.1, 0.8, 0.3, 0.2, 1, 0]
    np.testing.assert_equal(m2.ppf(quantiles), stats.beta(**params).ppf(quantiles))


def test_poisson():
    m = hypney.models.poisson(mu=3, rate=100)
    data = m.simulate()
    np.testing.assert_equal(m.pdf(data), stats.poisson(mu=3).pmf(data[:, 0]))
    assert m.rate() == 100.0


def test_vectorization():
    m = hypney.models.norm()

    ##
    # Multiple parameters (nontrivial batch_shape)
    ##

    # Param vector
    data = np.array([1, 2, 3])
    rates = [0, 1, 2]
    m = m(data=data)
    np.testing.assert_equal(m.rate(rate=rates), [m.rate(rate=x) for x in rates])
    np.testing.assert_equal(
        m.diff_rate(rate=rates), np.stack([m.diff_rate(rate=x) for x in rates])
    )

    # Param matrix
    rates = [0, 1, 2]
    locs = [0, 1, 3, 2]
    _r, _l = np.meshgrid(rates, locs, indexing="ij")
    np.testing.assert_equal(
        m.rate(rate=_r, loc=_l), [[m.rate(rate=x, loc=y) for y in locs] for x in rates]
    )

    # 'lazy matrix', two vectors with [:,None] and [None,:], auto-broadcasted up
    _r, _l = np.array(rates)[:, None], np.array(locs)[None, :]
    np.testing.assert_equal(
        m.rate(rate=_r, loc=_l), [[m.rate(rate=x, loc=y) for y in locs] for x in rates]
    )

    ##
    # Multiple datasets (nontrivial sample_shape)
    # This is experimental...
    ##
    data = np.array([[1, 2, 3], [3, 4, 5]])[..., None]
    assert data.shape == (2, 3, 1)
    pdf = m.pdf(data)
    assert isinstance(pdf, np.ndarray)
    assert pdf.shape == (2, 3)
    np.testing.assert_equal(pdf, [m.pdf(data[0]), m.pdf(data[1])])
