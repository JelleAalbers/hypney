import hypney


def test_negative_data():

    m = hypney.models.Uniform()
    m_flip = hypney.models.NegativeData(m)

    assert m_flip.pdf(data=-0.3) == m.pdf(data=0.3)
    assert m_flip.diff_rate(data=-0.3) == m.pdf(data=0.3)
    assert m_flip.rate() == m.rate()
    assert m_flip.cdf(data=-0.3) == 1 - m.cdf(data=0.3)
