import numpy as np

import hypney


def test_negative_data():

    m = hypney.models.Uniform()
    m_flip = hypney.models.NegativeData(m)

    assert m_flip.pdf(data=-0.3) == m.pdf(data=0.3)
    assert m_flip.diff_rate(data=-0.3) == m.pdf(data=0.3)
    assert m_flip.rate() == m.rate()
    assert m_flip.cdf(data=-0.3) == 1 - m.cdf(data=0.3)


def test_filter_params():
    m = hypney.models.Uniform()
    data = m.rvs(size=100)

    m2 = m.filter_params(keep="rate")
    assert len(m2.param_specs) == 1
    np.testing.assert_array_equal(m2.pdf(data=data).numpy(), m.pdf(data=data).numpy())

    fix = dict(loc=3, scale=2)
    m2 = m.filter_params(fix=fix)
    np.testing.assert_array_equal(
        m2.pdf(data=data).numpy(), m.pdf(params=fix, data=data).numpy()
    )
