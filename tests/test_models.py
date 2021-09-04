import hypney

import numpy as np


def test_uniform():
    m = hypney.Uniform()
    assert m.expected_count() == 0.0
    assert m.expected_count(params=dict(expected_events=100.0)) == 100.0

    data = m.simulate()
    assert data.shape == (0, 1)
    data = m.simulate_n(5)
    assert data.shape == (5, 1)

    assert m.pdf(0) == m.pdf([0]) == m.pdf(np.array([0])) == m.pdf(np.array([[0]]))
    assert m.pdf(0) == 1.0

    assert np.all(m.cdf([0.0, 0.5, 1.0]) == np.array([0.0, 0.5, 1.0]))

    assert m.diff_rate(0.0) == 0.0

    m = hypney.Uniform(expected_events=100)
    assert m.expected_count() == 100.0
    assert m.simulate().shape[0] > 0

    assert m.expected_count(cut=(0, 0.5)) == 50.0


def test_mixture():
    m1 = hypney.Uniform(expected_events=40)
    m2 = hypney.Uniform(expected_events=20)
    m3 = hypney.Uniform(expected_events=30)

    mix = hypney.Mixture(m1, m2)
    assert mix.param_names == ("m0_expected_events", "m1_expected_events")

    assert mix.expected_count() == 60.0
    assert mix.expected_count(params=dict(m0_expected_events=0)) == 20.0

    assert mix.pdf(0) == 1.0

    assert mix.diff_rate(0) == 60.0

    assert np.all(mix.cdf([0.0, 0.5, 1.0]) == np.array([0.0, 0.5, 1.0]))

    assert mix.simulate().shape[0] > 0

    # Test forming mixtures by +
    mix2 = m1 + m2
    assert mix2.diff_rate(0) == 60.0

    mix3 = m3 + mix2
    assert mix3.diff_rate(0) == 90.0
    assert len(mix3.models) == 3, "Should unpack mixtures"
    mix4 = m1 + m2 + m3
    assert mix4.diff_rate(0) == 90.0

    # TODO: Test parameter sharing
