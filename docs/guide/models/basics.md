Basic models
=============

You can use analytic probability distributions and histograms as basic models, described below.

Distributions
--------------

Hypney supports all univariate probability from [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html), accessed by the same name:

```python
hp.norm().plot_pdf()
hp.cosine().plot_pdf()
hp.cauchy().plot_pdf(x=np.linspace(-5, 5, 500))
```

Hypney uses scipy.stats for computations on numpy data. For tensorflow data, it will use corresponding distributions from [tensorflow-probability](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions) (if available), and similarly [torch.distributions](https://pytorch.org/docs/stable/distributions.html) for PyTorch data and [jax.scipy.stats](https://jax.readthedocs.io/en/latest/jax.scipy.html#jax-scipy-stats) for JAX data.

Hypney assumes all shape parameters have a default of 0, and that the distributions are implemented consistently between scipy, tensorflow-probability, torch, and JAX. We should probably test that explicitly soon...


Histograms
----------

You can also build a model from a histogram:

```
hp.from_histogram...
```
