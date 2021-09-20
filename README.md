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

This defines a model made from two Gaussians (one with a fixed amplitude / event rate), simulates a dataset, then computes a maximum likelihood fit.


Features
--------
  * Build models in multiple dimensions, with shared or independent parameters, using analytic distributions or histograms as building blocks.
  * Support for [numpy](https://numpy.org/), [pytorch](https://pytorch.org/), [tensorflow](https://www.tensorflow.org/), and [JAX](https://github.com/google/jax). Just provide your data as a tensor or array, and hypney will use the corresponding library under the hood, including autodifferentiation to speed up optimizations. (Thanks to [eagerpy](https://github.com/jonasrauber/eagerpy)!)
  * Light dependencies: numpy, scipy, and pure-python packages only.
  * Safely share hypney's built-in elements in threads, or copy them out to forked processes. They are serializable through pickle, and are stateless / immutable after they are built.

Features on the way
-------------------
 * Convenient support for exact / non-asymptotic frequentist inference, such as confidence intervals from [Neyman constructions](https://en.wikipedia.org/wiki/Neyman_construction).

 * Robust limit-setting statistics, such as [optimum interval](https://arxiv.org/abs/physics/0203002).

 * Adapters for models developed in other packages such as [zfit](https://github.com/zfit/zfit), [pyhf](https://github.com/scikit-hep/pyhf), [pymc3](https://github.com/pymc-devs/pymc3)?


Basic structure
----------------
Hypney builds analyses from three **elements**: models, statistics, and estimators.

  * **Models** take data and parameters, and assign probabilities to events. They can also simulate new datasets.

  * **Statistics** take a model, data and parameters, and output some value -- perhaps the count of all events or a profile likelihood.

  * **Estimators** take a statistic, model, and data, and output some statement about parameters. For example, a point estimate, confidence interval, or chain of samples.

Think of these elements are the 'plugins' of hypney. If you already have a model, statistic, or estimator, you can wrap them in hypney's API and use it seamlessly with other components.

Hypney has several built-in elements, but other packages may have elements more suited to your application. There are many statistical python packages, and hypney certainly does not replace them!

I made hypney so I didn't have to remember all their different APIs, and to provide some often-missing aspects like Neyman constructions and some specialized statistics.


Other packages
--------------

  * Hypney's is probably most similar to  the [zfit](https://github.com/zfit/zfit)/[hepstats](https://github.com/scikit-hep/hepstats) combination, though these have a rather different API similar to [RooFit](https://root.cern/manual/roofit/) and [GooFit](https://github.com/GooFit/GooFit). You may like these packages better, especially if you work for an LHC experiment.

  * [pyhf](https://github.com/scikit-hep/pyhf) specializes in models that interpolate between histograms, as does [histfactory](https://root.cern/doc/master/group__HistFactory.html). Hypney can also interpolate histograms, though only linearly, and works with unbinned data.

  * [pymc3](https://github.com/pymc-devs/pymc3) specializes in Bayesian analysis. Bayesians could certainly adapt their samplers to work on hypney's models, but might wonder why we treat observables, parameters and statistics so differently -- isn't everything just a random variable?

  * [statsmodels](https://github.com/statsmodels/statsmodels) specializes in linear/formula-style models like Y ~ X + np.log(Q).

  * [pomegranate](https://github.com/jmschrei/pomegranate) contains especially interesting functionality for models with discrete variables.
