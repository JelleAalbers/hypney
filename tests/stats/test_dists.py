import hypney


def test_dist():
    """Test statistics' distributions behave as expected"""
    m = hypney.models.norm(rate=10)
    count = hypney.statistics.Count(m)
    dist = count.dist
    assert dist.param_names == m.param_names
    assert dist.defaults["rate"] == 10
    assert dist.mean() == 10


def test_dist_from_toys():
    """Test building distributions by toy MC"""
    # cannot test this with count, its dist params collapses to mu
    # which makes no sense for a histogram
    # (we could build a toy MC distribution for Count using
    # interpolate_dist_from_toys)
    class Count2(hypney.statistics.Count):
        def _dist_params(self, params):
            return params

    m = hypney.models.norm(rate=10)
    count = Count2(m)

    dist = count.dist_from_toys(n_toys=2000)
    assert (dist.ppf(0.5) - 10) / 10 < 0.1

    dist = count.dist_from_toys(rate=20, n_toys=2000)
    assert (dist.ppf(0.5) - 20) / 20 < 0.1

    # Setting this distribution just adds some dummy params
    c2 = count.set(dist=dist)
    c2_dist = c2.dist
    assert not isinstance(c2_dist._orig_model, hypney.models.poisson)
    assert c2.dist.param_names == m.param_names
    assert c2.dist.defaults["rate"] == 10
    assert c2.dist.ppf(0.5) == dist.ppf(0.5)
