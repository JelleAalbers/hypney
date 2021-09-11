import hypney
import numpy as np


def test_interpolated():
    def builder(params):
        return hypney.Uniform(**params, rate=1000)

    m = hypney.Interpolation(builder, param_specs=dict(loc=(-0.5, 0, 0.5)))

    data = m.simulate()
    assert len(data)
    assert 0.4 < data.mean() < 0.6

    assert m.rate() == 1000.0

    # TODO: Derive these analytically. But they look plausible.
    x = [-0.01, 0.01, 0.49, 0.51, 0.99, 1.01, 1.49, 1.51]
    y = np.array([0, 0.6, 0.6, 1.0, 1.0, 0.4, 0.4, 0.0])

    m2 = m(data=x)
    np.testing.assert_array_almost_equal(m2.pdf(params=dict(loc=0.2)), y)
    np.testing.assert_array_almost_equal(m2.diff_rate(params=dict(loc=0.2)), y * 1000.0)

    # Test two dimensions of anchors
    m2 = hypney.Interpolation(
        builder, param_specs=dict(loc=(-0.5, 0, 0.5), scale=(0.5, 1, 1.5)), data=x
    )

    # TODO: same here
    y = np.array([0.0, 1.6, 1.6, 0.4, 0.4, 0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(m2.pdf(params=dict(scale=0.7)), y)
