import pytest
import numpy as np

import hypney


def test_count():
    # Five events at x=0
    m = hypney.models.norm(rate=1)
    data = np.array([0, 0, 0, 0, 0])

    stat = hypney.statistics.Count(m, data=data)
    assert len(stat.data) == 5

    np.testing.assert_equal(stat(), 5)
    np.testing.assert_equal(stat(rate=42), 5)

    # Test vectorization
    vec_result = stat(rate=np.array([-42, 0]))
    assert len(vec_result) == 2
    np.testing.assert_equal(vec_result, np.array([5, 5]))

    # Test pararms are still filtered
    with pytest.raises(ValueError):
        stat(kwarg=0)
