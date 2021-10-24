Hypney
======

Hypney (probably not its final name) is an embryonic statistical inference package.

![Build Status](https://github.com/JelleAalbers/hypney/actions/workflows/pytest.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/hypney/badge/?version=latest)](https://hypney.readthedocs.io/en/latest/?badge=latest)

Synopsis
--------

```python
import hypney.all as hp

model = hp.norm() + hp.norm(loc=5).fix(rate=7)
data = model.simulate()
statistic = hp.LogLikelihood(model, data)
bestfit = hp.Maximum(statistic)
```

This defines a model made from two Gaussians (the second with a fixed event rate / known amplitude), simulates a dataset, then computes a maximum likelihood fit.


Features
--------
  * Build models in multiple dimensions, with shared or independent parameters, using analytic distributions or histograms as building blocks.
  * Support for [numpy](https://numpy.org/), [pytorch](https://pytorch.org/) and [tensorflow](https://www.tensorflow.org/). Hypney will use the library corresponding to your data, including autodifferentiation to speed up optimizations. (Thanks to [eagerpy](https://github.com/jonasrauber/eagerpy)!)
  * Depends only on numpy, scipy, and pure-python packages.
  * Models/statistics are immutable and pickle-able, enabling serialization, multithreading and multiprocessing.
  * Exact / non-asymptotic frequentist inference, such as confidence intervals from [Neyman constructions](https://en.wikipedia.org/wiki/Neyman_construction).

Features on the way
-------------------

 * Robust limit-setting statistics, such as [optimum interval](https://arxiv.org/abs/physics/0203002).

 * Adapters for models developed in other packages such as [zfit](https://github.com/zfit/zfit), [pyhf](https://github.com/scikit-hep/pyhf), [pymc3](https://github.com/pymc-devs/pymc3)?


Basic structure
----------------
Hypney builds analyses from three **elements**: models, statistics, and estimators.

  * **Models** take data and parameters, and assign probabilities to events. They also simulate new data.

  * **Statistics** take a model, data and parameters, and output some value -- perhaps the count of all events or a profile likelihood.

  * **Estimators** take a statistic, model and data, and output some statement about parameters -- for example, a point estimate, confidence interval, or chain of samples drawn from a posterior.

Think of these elements are the 'plugins' of hypney. If you already have a model, statistic, or estimator, you can wrap it in a small hypney class, then use it seamlessly with existing elements.

Indeed, other packages may have the most appropriate elements for your application. There are many statistical python packages, and hypney does not aim to replace them! It just aims to provide a useful API, with some basic tools you might like.

Other packages
--------------

  * Hypney's has several similarities to [zfit](https://github.com/zfit/zfit) with the hypotest package in [hepstats](https://github.com/scikit-hep/hepstats). These have a somewhat different API, similar to [RooFit](https://root.cern/manual/roofit/) and [GooFit](https://github.com/GooFit/GooFit).

  * [pyhf](https://github.com/scikit-hep/pyhf) specializes in models that interpolate between histograms, as does [histfactory](https://root.cern/doc/master/group__HistFactory.html). Hypney can also interpolate histograms, but works with unbinned data. If you have many more events than bins, pyhf will be much faster.

  * [pymc3](https://github.com/pymc-devs/pymc3) specializes in Bayesian analysis. Bayesians could certainly adapt their samplers to work on hypney's models, but might wonder why we treat observables, parameters and statistics so differently -- isn't everything a random variable?

  * [statsmodels](https://github.com/statsmodels/statsmodels) specializes in linear/formula-style models like Y ~ X + np.log(Q).

  * [pomegranate](https://github.com/jmschrei/pomegranate) contains interesting functionality for models with discrete variables, among other things.
