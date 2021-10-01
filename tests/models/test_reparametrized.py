import numpy as np

import hypney


def test_filter_params():
    m = hypney.models.uniform()
    data = m.rvs(size=100)

    m2 = m.fix_except("rate")
    assert len(m2.param_specs) == 1
    np.testing.assert_array_equal(m2.pdf(data=data), m.pdf(data=data))

    fix = dict(loc=3, scale=2)
    m2 = m.fix(fix)
    np.testing.assert_array_equal(m2.pdf(data=data), m.pdf(params=fix, data=data))

    # Test fixing model with data already set
    md = m(data=data)
    md2 = md.freeze()
    assert md.data is md2.data
