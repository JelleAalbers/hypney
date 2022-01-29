import hypney

import numpy as np
from scipy import stats


def test_mixture():
    m1 = hypney.models.uniform(rate=40)
    m2_free = hypney.models.uniform(rate=20)
    m2_frozen = m2_free.freeze()
    m3 = hypney.models.uniform(rate=30)

    for m2 in m2_free, m2_frozen:
        mix = hypney.models.mixture(m1, m2)
        assert mix.rate() == 60.0

        assert mix.pdf(data=0) == 1.0
        assert mix.diff_rate(data=0) == 60.0

        np.testing.assert_array_equal(
            mix.cdf(data=[0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])
        )

        # Test vectorization
        rates = np.linspace(1, 2, 10)
        pname = mix.param_specs[0].name  # Changes w free/frozen
        np.testing.assert_almost_equal(
            mix.pdf(0, params={pname: rates}),
            [mix.pdf(0, params={pname: rate}) for rate in rates],
        )

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

    # Test mean and std
    mix = hypney.models.norm() + hypney.models.uniform(loc=5, scale=2)
    data = mix.rvs(100_000)
    np.testing.assert_allclose(mix.mean(), data.mean(), rtol=0.05)
    np.testing.assert_allclose(mix.std(), data.std(), rtol=0.05)

    # Test parameter after renaming
    mix = m1 + m2_free
    assert mix.rate(params=dict(m0_rate=1)) == 21.0

    mix = m1 + m2_frozen
    assert mix.rate(params=dict(rate=1)) == 21.0

    m2 = m2_free

    # Test parameter sharing
    m_shared = hypney.models.mixture(m1, m2, m3, share="scale")
    assert "scale" in m_shared.param_names
    assert "scale_0" not in m_shared.param_names
    assert "scale_1" not in m_shared.param_names
    assert "scale_2" not in m_shared.param_names
    assert m_shared(scale=2).pdf(2) == 0.5 * (m1 + m2 + m3).pdf(1)


def test_tensor_product():
    m1 = hypney.models.uniform(rate=40)
    m2 = hypney.models.uniform(rate=20)
    m3 = hypney.models.uniform(rate=30)

    prod = m1 ** m2 ** m3

    data = np.array([[0, 0, 0], [1, 1, 1]])

    np.testing.assert_array_equal(prod.pdf(data=data), np.array([1, 1]))

    np.testing.assert_array_equal(
        prod.logpdf(data=data), stats.uniform().logpdf(0) ** 3
    )

    assert prod.rate() == 40.0
    data = prod.simulate()
    assert data.shape[0] > 0
    assert data.shape[1] == 3
