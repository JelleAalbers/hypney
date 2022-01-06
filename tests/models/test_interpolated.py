import hypney
import numpy as np


def test_interpolated():
    def builder(params):
        return hypney.models.uniform(**params, rate=1000)

    m = hypney.models.Interpolation(
        builder, param_specs=dict(loc=(-0.5, 0, 0.5)), methods=("mean", "pdf")
    )

    data = m.simulate()
    assert len(data)
    assert 0.4 < data.mean() < 0.6

    assert m.rate() == 1000.0

    # TODO: Derive these analytically. But they look plausible.
    x = np.array([-0.01, 0.01, 0.49, 0.51, 0.99, 1.01, 1.49, 1.51])
    y = np.array([0, 0.6, 0.6, 1.0, 1.0, 0.4, 0.4, 0.0])

    m2 = m(data=x)
    np.testing.assert_array_almost_equal(m2.pdf(params=dict(loc=0.2)), y)
    np.testing.assert_array_almost_equal(m2.diff_rate(params=dict(loc=0.2)), y * 1000.0)

    # Test vectorization
    locs = np.array([0.2, 0, -0.2])
    rate_list = np.array([m2.rate(loc=x) for x in locs])
    rate_arr = m2.rate(loc=locs)
    np.testing.assert_array_equal(rate_list, rate_arr)
    pdf_list = np.stack([m2.pdf(loc=x) for x in locs])
    pdf_arr = m2.pdf(loc=locs)
    np.testing.assert_array_equal(pdf_list, pdf_arr)

    # No, linearly interpolated CDF is not the inverse of the linearly interpolated PPF
    # (nor is it the integral of the linearly interpolated PDF.. pretty tricky)

    # Test two dimensions of anchors
    m2 = hypney.models.Interpolation(
        builder,
        param_specs=dict(loc=(-0.5, 0, 0.5), scale=(0.5, 1, 1.5)),
        data=x,
        methods="pdf",
    )

    # TODO: same here
    y = np.array([0.0, 1.6, 1.6, 0.4, 0.4, 0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal(m2.pdf(params=dict(scale=0.7)), y)
