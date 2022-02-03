import numpy as np
from scipy import stats

import hypney.all as hp


def test_cut_indices():
    def cut_indices_numpy(n):
        indices = np.stack(np.indices((n, n)), axis=-1).reshape(-1, 2)
        return indices[indices[:, 1] > indices[:, 0]]

    for n in [0, 1, 10]:
        np.testing.assert_almost_equal(cut_indices_numpy(n), hp.all_cut_indices(n))


def test_simple_and_full():
    """Test simplified hawks against full hawks"""
    model = hp.uniform(rate=7).fix_except("rate")
    cuts = [(0.0, 1.0), (0.5, 1), (0.1, 0.3)]

    for toy_data in [], [0.05, 0.7, 0.4], [0.2, 0.7]:
        toy_data = np.array(toy_data)[:, None]

        for statname, (simple_all, full_all, simple_fixed, full_fixed) in [
            (
                "pn",
                (
                    hp.PNAllRegionHawk,
                    hp.PNAllRegionHawkSlow,
                    hp.PNFixedRegionHawk,
                    hp.PNFixedRegionHawkSlow,
                ),
            ),
            (
                "lr",
                (
                    hp.AllRegionSimpleHawk,
                    hp.AllRegionFullHawk,
                    hp.FixedRegionSimpleHawk,
                    hp.FixedRegionFullHawk,
                ),
            ),
            # Yellin's CN statistic would have to be generalized
            # to allow fixed region hawks.
            ("yellin_cn", (hp.YellinCNHawk, hp.YellinCNFullHawk, None, None),),
        ]:
            print(f"\nNow testing {statname}\n")
            simple_result = simple_all(model, data=toy_data).compute()
            full_result = full_all(model, data=toy_data).compute()
            np.testing.assert_almost_equal(simple_result, full_result)

            if simple_fixed is None:
                continue
            fast_stat = simple_fixed(model, cuts=cuts, data=toy_data)
            slow_stat = full_fixed(model, cuts=cuts, data=toy_data)
            simple_result = fast_stat.compute()
            full_result = slow_stat.compute()
            np.testing.assert_almost_equal(simple_result, full_result)


def test_pn():
    """Test the PN all region hawk (equivalent to Yellin's pmax/pmin method)"""
    for mu_true in [0, 2, 4]:

        model_simple = hp.uniform(rate=mu_true).fix_except("rate")
        # Normal on uniform background, clip to [0, 1]
        model_withbg = (
            hp.norm(loc=0.5, scale=0.1).fix_except("rate")
            # + hp.uniform(rate=10).freeze()
        ).cut(0, 1)

        for m, signal_only in [(model_simple, True), (model_withbg, False)]:

            d = m.simulate()
            stat = hp.PNAllRegionHawk(model=m, signal_only=signal_only, data=d)

            for mu_test in [0, 2, 25]:

                pmin_obs = stat.compute(rate=mu_test)

                # Sanity check on the best cut
                bestcut_dict = stat.best_cut(rate=mu_test)
                best_cut = bestcut_dict["cut"]
                # bestcut_stat = bestcut_dict["stat"]
                # TODO: setting data on stats is weird...
                bestcut_stat = hp.PNOneCut(m.cut(*best_cut))
                bestcut_stat = bestcut_stat(
                    data=d[(best_cut[0] < d[:, 0]) & (d[:, 0] < best_cut[1])]
                )
                bestcut_stat.compute(rate=mu_test) == pmin_obs
                assert np.all(
                    np.asarray(bestcut_stat.model._cut[0]) == np.asarray(best_cut)
                )
                assert bestcut_stat.compute(rate=mu_test) == pmin_obs

                # Exhaustively compute the ps for each every interval..
                xs = np.concatenate([[-float("inf")], np.sort(d[:, 0]), [float("inf")]])
                min_seen = 1
                for left_i, left_x in enumerate(xs):
                    for j, right_x in enumerate(xs[left_i + 1 :]):
                        n_obs = j
                        mu = m.cut(left_x, right_x).rate(rate=mu_test)
                        # mu_full * (right_x - left_x)
                        poisson_p = stats.poisson(mu=mu).cdf(n_obs)
                        assert poisson_p >= pmin_obs, "pmin missed smaller p"
                        min_seen = min(poisson_p, min_seen)
                assert min_seen == pmin_obs, "pmin found too small p'"


def test_pn_vectorization():
    model = hp.uniform(rate=40).fix_except("rate")
    toy_data = model.simulate()

    rates = np.array([0, 2, 5, 7.2])

    for stat_class in [hp.PNAllRegionHawkSlow, hp.PNAllRegionHawk]:
        print(stat_class)
        stat = stat_class(model, data=toy_data)
        y = stat.compute(rate=rates)
        assert isinstance(y, np.ndarray)
        assert isinstance(y[0], float)
        np.testing.assert_array_equal(
            y, np.array([stat.compute(rate=r) for r in rates])
        )

    for stat_class in [hp.PNFixedRegionHawkSlow, hp.PNFixedRegionHawk]:
        print(stat_class)
        cuts = [(0.0, 1.0), (0.5, 1), (0.1, 0.3)]
        stat = stat_class(model, cuts=cuts, data=toy_data)
        y = stat.compute(rate=rates)
        assert isinstance(y, np.ndarray)
        assert isinstance(y[0], float)
        np.testing.assert_array_equal(
            y, np.array([stat.compute(rate=r) for r in rates])
        )
