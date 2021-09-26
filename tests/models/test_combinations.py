import hypney

import numpy as np


def test_mixture():
    m1 = hypney.models.uniform(rate=40)
    m2 = hypney.models.uniform(rate=20)
    m3 = hypney.models.uniform(rate=30)

    mix = hypney.models.Mixture(m1, m2)
    # assert mix.param_names == (
    #     "m0_expected_events", "m0_loc", "m1_expected_events")

    assert mix.rate() == 60.0
    assert mix.rate(params=dict(m0_rate=0)) == 20.0

    assert mix.pdf(data=0) == 1.0

    assert mix.diff_rate(data=0) == 60.0

    np.testing.assert_array_equal(
        mix.cdf(data=[0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])
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

    # TODO: Test parameter sharing

    # Test mean and std
    mix = hypney.models.norm() + hypney.models.uniform(loc=5, scale=2)
    data = mix.rvs(100_000)
    np.testing.assert_allclose(mix.mean(), data.mean(), rtol=0.05)
    np.testing.assert_allclose(mix.std(), data.std(), rtol=0.05)


def test_tensor_product():
    m1 = hypney.models.uniform(rate=40)
    m2 = hypney.models.uniform(rate=20)
    m3 = hypney.models.uniform(rate=30)

    prod = m1 ** m2 ** m3

    data = np.array([[0, 0, 0], [1, 1, 1]])

    np.testing.assert_array_equal(prod.pdf(data=data), np.array([1, 1]))
    np.testing.assert_array_equal(prod.cdf(data=data), np.array([0, 1]))

    assert prod.rate() == 40.0
    data = prod.simulate()
    assert data.shape[0] > 0
    assert data.shape[1] == 3
