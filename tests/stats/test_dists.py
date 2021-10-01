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
    m = hypney.models.norm(rate=10)
    count = hypney.statistics.Count(m)

    dist = count.dist_from_toys(n_toys=5000)
    assert (dist.mean() - 10) / 10 < 0.1

    dist = count.interpolate_dist_from_toys(
        anchors=dict(rate=[1, 2, 5, 10, 20, 50, 100]), methods="mean"
    )
    assert 14 < dist(rate=15).mean() < 16

    # Setting this distribution just adds some dummy params
    c2 = count.set(dist=dist)
    c2_dist = c2.dist
    assert not isinstance(c2_dist._orig_model, hypney.models.poisson)
    assert c2.dist.param_names == m.param_names
    assert c2.dist.defaults["rate"] == 10
    assert c2.dist.mean() == dist.mean()
