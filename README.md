Hypney
======

Hypney (probably not its final name) is an embryonic statistical inference package.

Synopsis
--------

```python
import hypney.all as hp

model = hp.Norm() + hp.Norm(loc=5).fix(rate=7)
data = model.simulate()
statistic = hp.LogLikelihood(model, data)
bestfit = hp.Maximum(statistic)()
```

This defines a model that mixes two Gaussians (one with a fixed amplitude/rate), simulates a dataset, then computes a maximum likelihood fit.


Features
--------
  * Build models in multiple dimensions, with shared or independent parameters, using analytic distributions (from scipy.stats) or histograms as building blocks.
  * Supports [numpy](https://numpy.org/), [pytorch](https://pytorch.org/), [tensorflow](https://www.tensorflow.org/), and [JAX](https://github.com/google/jax). Provide your data as a tensor or array, and hypney uses the corresponding library under the hood, including autodifferentiation to speed up optimizations. (Thanks to [eagerpy](https://github.com/jonasrauber/eagerpy)!)
  * Light dependencies: numpy, scipy, and pure-python only.


Features to come
-----------------
 * Convenient support for exact / non-asymptotic frequentist inference, such as confidence intervals from [Neyman constructions](https://en.wikipedia.org/wiki/Neyman_construction).

 * Robust limit-setting statistics, such as [optimum interval](https://arxiv.org/abs/physics/0203002).
 * Adapters for models developed in other packages such as [zfit](https://github.com/zfit/zfit), [pyhf](https://github.com/scikit-hep/pyhf), [pymc3](https://github.com/pymc-devs/pymc3)?


Basic structure
----------------
Hypney builds analyses from three elements: models, statistics, and estimators.

  * **Models** take data and parameters, and assign probabilities to events. They can also simulate new datasets.

  * **Statistics** take a model, data and parameters, and output some value -- perhaps the count of all events or a profile likelihood.

  * **Estimators** take a statistic, model, and data, and output some statement about parameters. For example, a point estimate, confidence interval, or chain of samples.

These elements are the 'plugins' of hypney. If you already have a model (e.g. from zfit), statistic (e.g. external code to compute an optimum interval), or an estimator (some custom minimizer), you can wrap it in hypney's API and use it with other components.

You can safely share hypney's built-in elements in threads, or copy them out to forked processes. They are serializable through pickle, and are immutable/stateless post-initialization (if you stick with the public API).



Alternatives
-------------
  * Hypney's use case is most similar to  the [zfit](https://github.com/zfit/zfit)/[hepstats](https://github.com/scikit-hep/hepstats) combination, though these have a rather different API similar to [RooFit](https://root.cern/manual/roofit/) and [GooFit](https://github.com/GooFit/GooFit). You may like any of these packages better, especially if you work for an LHC experiment.

  * [pyhf](https://github.com/scikit-hep/pyhf) specializes in models that interpolate between histograms, just like [histfactory](https://root.cern/doc/master/group__HistFactory.html). Hypney can also interpolate histograms, though only linearly, and works with unbinned data.

  * [pymc3](https://github.com/pymc-devs/pymc3) specializes in Bayesian analysis. Bayesians could certainly adapt their samplers to work on hypney's models, but might wonder why we treat observables, parameters and statistics so differently -- isn't everything just a random variable?

  * [statsmodels](https://github.com/statsmodels/statsmodels) specializes in linear/formula-style models like Y ~ X + np.log(Q).

  * [pomegranate](https://github.com/jmschrei/pomegranate) contains especially interesting functionality for models with discrete variables.
