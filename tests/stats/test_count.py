import pytest
import numpy as np

import hypney


def test_count():
    # Five events at x=0
    m = hypney.models.norm(rate=1)
    data = np.array([0, 0, 0, 0, 0])

    stat = hypney.statistics.Count(m, data=data)
    assert len(stat.data) == 5

    np.testing.assert_equal(stat.compute(), 5)
    np.testing.assert_equal(stat.compute(rate=42), 5)

    # Test vectorization
    vec_result = stat.compute(rate=np.array([-42, 0]))
    assert len(vec_result) == 2
    np.testing.assert_equal(vec_result, np.array([5, 5]))

    # Test pararms are still filtered
    with pytest.raises(ValueError):
        stat(kwarg=0)

    # Test building new stats with .set
    stat2 = stat.set(rate=5)
    assert stat2.model != stat.model
    np.testing.assert_equal(
        stat2.model.defaults["rate"].numpy(), 5 * stat.model.defaults["rate"].numpy()
    )

    stat3 = stat.set(params=dict(rate=7))
    assert stat3.model != stat.model
    np.testing.assert_equal(
        stat3.model.defaults["rate"].numpy(), 7 * stat.model.defaults["rate"].numpy()
    )
