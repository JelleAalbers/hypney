import hypney
from hypney import RATE_LOC_PARAMS, RATE_LOC_SCALE_PARAMS
from .univariate import UnivariateDistribution, UnivariateDiscreteDistribution

export, __all__ = hypney.exporter()


def shape_params(*names):
    return tuple(
        [
            hypney.Parameter(name=name, default=0, min=0, max=float("inf"))
            for name in names
        ]
    )


@export
class alpha(UnivariateDistribution):
    r"""An alpha continuous random variable.

    Notes
    -----
    The probability density function for `alpha` ([1]_, [2]_) is:

    .. math::

        f(x, a) = \frac{1}{x^2 \Phi(a) \sqrt{2\pi}} *
                  \exp(-\frac{1}{2} (a-1/x)^2)

    where :math:`\Phi` is the normal CDF, :math:`x > 0`, and :math:`a > 0`.

    `alpha` takes ``a`` as a shape parameter.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a")

    scipy_name = "alpha"


@export
class anglit(UnivariateDistribution):
    r"""An anglit continuous random variable.

    Notes
    -----
    The probability density function for `anglit` is:

    .. math::

        f(x) = \sin(2x + \pi/2) = \cos(2x)

    for :math:`-\pi/4 \le x \le \pi/4`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "anglit"


@export
class arcsine(UnivariateDistribution):
    r"""An arcsine continuous random variable.

    Notes
    -----
    The probability density function for `arcsine` is:

    .. math::

        f(x) = \frac{1}{\pi \sqrt{x (1-x)}}

    for :math:`0 < x < 1`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "arcsine"


@export
class argus(UnivariateDistribution):
    r"""
    Argus distribution

    Notes
    -----
    The probability density function for `argus` is:

    .. math::

        f(x, \chi) = \frac{\chi^3}{\sqrt{2\pi} \Psi(\chi)} x \sqrt{1-x^2}
                     \exp(-\chi^2 (1 - x^2)/2)

    for :math:`0 < x < 1` and :math:`\chi > 0`, where

    .. math::

        \Psi(\chi) = \Phi(\chi) - \chi \phi(\chi) - 1/2

    with :math:`\Phi` and :math:`\phi` being the CDF and PDF of a standard
    normal distribution, respectively.

    `argus` takes :math:`\chi` as shape a parameter.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("chi")

    scipy_name = "argus"


@export
class bernoulli(UnivariateDiscreteDistribution):
    r"""A Bernoulli discrete random variable.

    Notes
    -----
    The probability mass function for `bernoulli` is:

    .. math::

       f(k) = \begin{cases}1-p  &\text{if } k = 0\\
                           p    &\text{if } k = 1\end{cases}

    for :math:`k` in :math:`\{0, 1\}`, :math:`0 \leq p \leq 1`

    `bernoulli` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("p")

    scipy_name = "bernoulli"
    torch_name = "Bernoulli"
    tfp_name = "Bernoulli"


@export
class beta(UnivariateDistribution):
    r"""A beta continuous random variable.

    Notes
    -----
    The probability density function for `beta` is:

    .. math::

        f(x, a, b) = \frac{\Gamma(a+b) x^{a-1} (1-x)^{b-1}}
                          {\Gamma(a) \Gamma(b)}

    for :math:`0 <= x <= 1`, :math:`a > 0`, :math:`b > 0`, where
    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).

    `beta` takes :math:`a` and :math:`b` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "b")

    scipy_name = "beta"
    torch_name = "Beta"
    tfp_name = "Beta"


@export
class betabinom(UnivariateDiscreteDistribution):
    r"""A beta-binomial discrete random variable.

    Notes
    -----
    The beta-binomial distribution is a binomial distribution with a
    probability of success `p` that follows a beta distribution.

    The probability mass function for `betabinom` is:

    .. math::

       f(k) = \binom{n}{k} \frac{B(k + a, n - k + b)}{B(a, b)}

    for ``k`` in ``{0, 1,..., n}``, :math:`n \geq 0`, :math:`a > 0`,
    :math:`b > 0`, where :math:`B(a, b)` is the beta function.

    `betabinom` takes :math:`n`, :math:`a`, and :math:`b` as shape parameters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution
    """

    param_specs = RATE_LOC_PARAMS + shape_params("n", "a", "b")

    scipy_name = "betabinom"
    tfp_name = "BetaBinomial"


@export
class betaprime(UnivariateDistribution):
    r"""A beta prime continuous random variable.

    Notes
    -----
    The probability density function for `betaprime` is:

    .. math::

        f(x, a, b) = \frac{x^{a-1} (1+x)^{-a-b}}{\beta(a, b)}

    for :math:`x >= 0`, :math:`a > 0`, :math:`b > 0`, where
    :math:`\beta(a, b)` is the beta function (see `scipy.special.beta`).

    `betaprime` takes ``a`` and ``b`` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "b")

    scipy_name = "betaprime"


@export
class binom(UnivariateDiscreteDistribution):
    r"""A binomial discrete random variable.

    Notes
    -----
    The probability mass function for `binom` is:

    .. math::

       f(k) = \binom{n}{k} p^k (1-p)^{n-k}

    for ``k`` in ``{0, 1,..., n}``, :math:`0 \leq p \leq 1`

    `binom` takes ``n`` and ``p`` as shape parameters,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("n", "p")

    scipy_name = "binom"
    torch_name = "Binomial"
    tfp_name = "Binomial"


@export
class boltzmann(UnivariateDiscreteDistribution):
    r"""A Boltzmann (Truncated Discrete Exponential) random variable.

    Notes
    -----
    The probability mass function for `boltzmann` is:

    .. math::

        f(k) = (1-\exp(-\lambda)) \exp(-\lambda k) / (1-\exp(-\lambda N))

    for :math:`k = 0,..., N-1`.

    `boltzmann` takes :math:`\lambda > 0` and :math:`N > 0` as shape parameters.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("lambda_", "N")

    scipy_name = "boltzmann"


@export
class bradford(UnivariateDistribution):
    r"""A Bradford continuous random variable.

    Notes
    -----
    The probability density function for `bradford` is:

    .. math::

        f(x, c) = \frac{c}{\log(1+c) (1+cx)}

    for :math:`0 <= x <= 1` and :math:`c > 0`.

    `bradford` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "bradford"


@export
class burr(UnivariateDistribution):
    r"""A Burr (Type III) continuous random variable.

    Notes
    -----
    The probability density function for `burr` is:

    .. math::

        f(x, c, d) = c d x^{-c - 1} / (1 + x^{-c})^{d + 1}

    for :math:`x >= 0` and :math:`c, d > 0`.

    `burr` takes :math:`c` and :math:`d` as shape parameters.

    This is the PDF corresponding to the third CDF given in Burr's list;
    specifically, it is equation (11) in Burr's paper [1]_. The distribution
    is also commonly referred to as the Dagum distribution [2]_. If the
    parameter :math:`c < 1` then the mean of the distribution does not
    exist and if :math:`c < 2` the variance does not exist [2]_.
    The PDF is finite at the left endpoint :math:`x = 0` if :math:`c * d >= 1`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c", "d")

    scipy_name = "burr"


@export
class burr12(UnivariateDistribution):
    r"""A Burr (Type XII) continuous random variable.

    Notes
    -----
    The probability density function for `burr` is:

    .. math::

        f(x, c, d) = c d x^{c-1} / (1 + x^c)^{d + 1}

    for :math:`x >= 0` and :math:`c, d > 0`.

    `burr12` takes ``c`` and ``d`` as shape parameters for :math:`c`
    and :math:`d`.

    This is the PDF corresponding to the twelfth CDF given in Burr's list;
    specifically, it is equation (20) in Burr's paper [1]_.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c", "d")

    scipy_name = "burr12"


@export
class cauchy(UnivariateDistribution):
    r"""A Cauchy continuous random variable.

    Notes
    -----
    The probability density function for `cauchy` is

    .. math::

        f(x) = \frac{1}{\pi (1 + x^2)}

    for a real number :math:`x`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "cauchy"
    torch_name = "Cauchy"
    tfp_name = "Cauchy"


@export
class chi(UnivariateDistribution):
    r"""A chi continuous random variable.

    Notes
    -----
    The probability density function for `chi` is:

    .. math::

        f(x, k) = \frac{1}{2^{k/2-1} \Gamma \left( k/2 \right)}
                   x^{k-1} \exp \left( -x^2/2 \right)

    for :math:`x >= 0` and :math:`k > 0` (degrees of freedom, denoted ``df``
    in the implementation). :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).

    Special cases of `chi` are:

        - ``chi(1, loc, scale)`` is equivalent to `halfnorm`
        - ``chi(2, 0, scale)`` is equivalent to `rayleigh`
        - ``chi(3, 0, scale)`` is equivalent to `maxwell`

    `chi` takes ``df`` as a shape parameter.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("df")

    scipy_name = "chi"
    tfp_name = "Chi"


@export
class chi2(UnivariateDistribution):
    r"""A chi-squared continuous random variable.

    For the noncentral chi-square distribution, see `ncx2`.

    Notes
    -----
    The probability density function for `chi2` is:

    .. math::

        f(x, k) = \frac{1}{2^{k/2} \Gamma \left( k/2 \right)}
                   x^{k/2-1} \exp \left( -x/2 \right)

    for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
    in the implementation).

    `chi2` takes ``df`` as a shape parameter.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("df")

    scipy_name = "chi2"
    torch_name = "Chi2"
    tfp_name = "Chi2"


@export
class cosine(UnivariateDistribution):
    r"""A cosine continuous random variable.

    Notes
    -----
    The cosine distribution is an approximation to the normal distribution.
    The probability density function for `cosine` is:

    .. math::

        f(x) = \frac{1}{2\pi} (1+\cos(x))

    for :math:`-\pi \le x \le \pi`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "cosine"


@export
class crystalball(UnivariateDistribution):
    r"""
    Crystalball distribution

    Notes
    -----
    The probability density function for `crystalball` is:

    .. math::

        f(x, \beta, m) =  \begin{cases}
                            N \exp(-x^2 / 2),  &\text{for } x > -\beta\\
                            N A (B - x)^{-m}  &\text{for } x \le -\beta
                          \end{cases}

    where :math:`A = (m / |\beta|)^n  \exp(-\beta^2 / 2)`,
    :math:`B = m/|\beta| - |\beta|` and :math:`N` is a normalisation constant.

    `crystalball` takes :math:`\beta > 0` and :math:`m > 1` as shape
    parameters.  :math:`\beta` defines the point where the pdf changes
    from a power-law to a Gaussian distribution.  :math:`m` is the power
    of the power-law tail.

    References
    ----------
    .. [1] "Crystal Ball Function",
           https://en.wikipedia.org/wiki/Crystal_Ball_function
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("beta", "m")

    scipy_name = "crystalball"


@export
class dgamma(UnivariateDistribution):
    r"""A double gamma continuous random variable.

    Notes
    -----
    The probability density function for `dgamma` is:

    .. math::

        f(x, a) = \frac{1}{2\Gamma(a)} |x|^{a-1} \exp(-|x|)

    for a real number :math:`x` and :math:`a > 0`. :math:`\Gamma` is the
    gamma function (`scipy.special.gamma`).

    `dgamma` takes ``a`` as a shape parameter for :math:`a`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a")

    scipy_name = "dgamma"


@export
class dlaplace(UnivariateDiscreteDistribution):
    r"""A  Laplacian discrete random variable.

    Notes
    -----
    The probability mass function for `dlaplace` is:

    .. math::

        f(k) = \tanh(a/2) \exp(-a |k|)

    for integers :math:`k` and :math:`a > 0`.

    `dlaplace` takes :math:`a` as shape parameter.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("a")

    scipy_name = "dlaplace"


@export
class dweibull(UnivariateDistribution):
    r"""A double Weibull continuous random variable.

    Notes
    -----
    The probability density function for `dweibull` is given by

    .. math::

        f(x, c) = c / 2 |x|^{c-1} \exp(-|x|^c)

    for a real number :math:`x` and :math:`c > 0`.

    `dweibull` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "dweibull"


@export
class erlang(UnivariateDistribution):
    r"""An Erlang continuous random variable.

    Notes
    -----
    The Erlang distribution is a special case of the Gamma distribution, with
    the shape parameter `a` an integer.  Note that this restriction is not
    enforced by `erlang`. It will, however, generate a warning the first time
    a non-integer value is used for the shape parameter.

    Refer to `gamma` for examples.


    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a")

    scipy_name = "erlang"


@export
class expon(UnivariateDistribution):
    r"""An exponential continuous random variable.

    Notes
    -----
    The probability density function for `expon` is:

    .. math::

        f(x) = \exp(-x)

    for :math:`x \ge 0`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "expon"


@export
class exponnorm(UnivariateDistribution):
    r"""An exponentially modified Normal continuous random variable.

    Notes
    -----
    The probability density function for `exponnorm` is:

    .. math::

        f(x, K) = \frac{1}{2K} \exp\left(\frac{1}{2 K^2} - x / K \right)
                  \text{erfc}\left(-\frac{x - 1/K}{\sqrt{2}}\right)

    where :math:`x` is a real number and :math:`K > 0`.

    It can be thought of as the sum of a standard normal random variable
    and an independent exponentially distributed random variable with rate
    ``1/K``.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("K")

    scipy_name = "exponnorm"


@export
class exponpow(UnivariateDistribution):
    r"""An exponential power continuous random variable.

    Notes
    -----
    The probability density function for `exponpow` is:

    .. math::

        f(x, b) = b x^{b-1} \exp(1 + x^b - \exp(x^b))

    for :math:`x \ge 0`, :math:`b > 0`.  Note that this is a different
    distribution from the exponential power distribution that is also known
    under the names "generalized normal" or "generalized Gaussian".

    `exponpow` takes ``b`` as a shape parameter for :math:`b`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("b")

    scipy_name = "exponpow"


@export
class exponweib(UnivariateDistribution):
    r"""An exponentiated Weibull continuous random variable.

    Notes
    -----
    The probability density function for `exponweib` is:

    .. math::

        f(x, a, c) = a c [1-\exp(-x^c)]^{a-1} \exp(-x^c) x^{c-1}

    and its cumulative distribution function is:

    .. math::

        F(x, a, c) = [1-\exp(-x^c)]^a

    for :math:`x > 0`, :math:`a > 0`, :math:`c > 0`.

    `exponweib` takes :math:`a` and :math:`c` as shape parameters:

    * :math:`a` is the exponentiation parameter,
      with the special case :math:`a=1` corresponding to the
      (non-exponentiated) Weibull distribution `weibull_min`.
    * :math:`c` is the shape parameter of the non-exponentiated Weibull law.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "c")

    scipy_name = "exponweib"


@export
class f(UnivariateDistribution):
    r"""An F continuous random variable.

    For the noncentral F distribution, see `ncf`.

    Notes
    -----
    The probability density function for `f` is:

    .. math::

        f(x, df_1, df_2) = \frac{df_2^{df_2/2} df_1^{df_1/2} x^{df_1 / 2-1}}
                                {(df_2+df_1 x)^{(df_1+df_2)/2}
                                 B(df_1/2, df_2/2)}

    for :math:`x > 0`.

    `f` takes ``dfn`` and ``dfd`` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("dfn", "dfd")

    scipy_name = "f"
    torch_name = "FisherSnedecor"


@export
class fatiguelife(UnivariateDistribution):
    r"""A fatigue-life (Birnbaum-Saunders) continuous random variable.

    Notes
    -----
    The probability density function for `fatiguelife` is:

    .. math::

        f(x, c) = \frac{x+1}{2c\sqrt{2\pi x^3}} \exp(-\frac{(x-1)^2}{2x c^2})

    for :math:`x >= 0` and :math:`c > 0`.

    `fatiguelife` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "fatiguelife"


@export
class fisk(UnivariateDistribution):
    r"""A Fisk continuous random variable.

    The Fisk distribution is also known as the log-logistic distribution.

    Notes
    -----
    The probability density function for `fisk` is:

    .. math::

        f(x, c) = c x^{-c-1} (1 + x^{-c})^{-2}

    for :math:`x >= 0` and :math:`c > 0`.

    `fisk` takes ``c`` as a shape parameter for :math:`c`.

    `fisk` is a special case of `burr` or `burr12` with ``d=1``.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "fisk"


@export
class foldcauchy(UnivariateDistribution):
    r"""A folded Cauchy continuous random variable.

    Notes
    -----
    The probability density function for `foldcauchy` is:

    .. math::

        f(x, c) = \frac{1}{\pi (1+(x-c)^2)} + \frac{1}{\pi (1+(x+c)^2)}

    for :math:`x \ge 0`.

    `foldcauchy` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "foldcauchy"


@export
class foldnorm(UnivariateDistribution):
    r"""A folded normal continuous random variable.

    Notes
    -----
    The probability density function for `foldnorm` is:

    .. math::

        f(x, c) = \sqrt{2/\pi} cosh(c x) \exp(-\frac{x^2+c^2}{2})

    for :math:`c \ge 0`.

    `foldnorm` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "foldnorm"


@export
class gamma(UnivariateDistribution):
    r"""A gamma continuous random variable.

    Notes
    -----
    The probability density function for `gamma` is:

    .. math::

        f(x, a) = \frac{x^{a-1} e^{-x}}{\Gamma(a)}

    for :math:`x \ge 0`, :math:`a > 0`. Here :math:`\Gamma(a)` refers to the
    gamma function.

    `gamma` takes ``a`` as a shape parameter for :math:`a`.

    When :math:`a` is an integer, `gamma` reduces to the Erlang
    distribution, and when :math:`a=1` to the exponential distribution.

    Gamma distributions are sometimes parameterized with two variables,
    with a probability density function of:

    .. math::

        f(x, \alpha, \beta) = \frac{\beta^\alpha x^{\alpha - 1} e^{-\beta x }}{\Gamma(\alpha)}

    Note that this parameterization is equivalent to the above, with
    ``scale = 1 / beta``.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a")

    scipy_name = "gamma"
    torch_name = "Gamma"
    tfp_name = "Gamma"


@export
class gausshyper(UnivariateDistribution):
    r"""A Gauss hypergeometric continuous random variable.

    Notes
    -----
    The probability density function for `gausshyper` is:

    .. math::

        f(x, a, b, c, z) = C x^{a-1} (1-x)^{b-1} (1+zx)^{-c}

    for :math:`0 \le x \le 1`, :math:`a > 0`, :math:`b > 0`, :math:`z > -1`,
    and :math:`C = \frac{1}{B(a, b) F[2, 1](c, a; a+b; -z)}`.
    :math:`F[2, 1]` is the Gauss hypergeometric function
    `scipy.special.hyp2f1`.

    `gausshyper` takes :math:`a`, :math:`b`, :math:`c` and :math:`z` as shape
    parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "b", "c", "z")

    scipy_name = "gausshyper"


@export
class genexpon(UnivariateDistribution):
    r"""A generalized exponential continuous random variable.

    Notes
    -----
    The probability density function for `genexpon` is:

    .. math::

        f(x, a, b, c) = (a + b (1 - \exp(-c x)))
                        \exp(-a x - b x + \frac{b}{c}  (1-\exp(-c x)))

    for :math:`x \ge 0`, :math:`a, b, c > 0`.

    `genexpon` takes :math:`a`, :math:`b` and :math:`c` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "b", "c")

    scipy_name = "genexpon"


@export
class genextreme(UnivariateDistribution):
    r"""A generalized extreme value continuous random variable.

    Notes
    -----
    For :math:`c=0`, `genextreme` is equal to `gumbel_r`.
    The probability density function for `genextreme` is:

    .. math::

        f(x, c) = \begin{cases}
                    \exp(-\exp(-x)) \exp(-x)              &\text{for } c = 0\\
                    \exp(-(1-c x)^{1/c}) (1-c x)^{1/c-1}  &\text{for }
                                                            x \le 1/c, c > 0
                  \end{cases}


    Note that several sources and software packages use the opposite
    convention for the sign of the shape parameter :math:`c`.

    `genextreme` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "genextreme"


@export
class gengamma(UnivariateDistribution):
    r"""A generalized gamma continuous random variable.

    Notes
    -----
    The probability density function for `gengamma` is ([1]_):

    .. math::

        f(x, a, c) = \frac{|c| x^{c a-1} \exp(-x^c)}{\Gamma(a)}

    for :math:`x \ge 0`, :math:`a > 0`, and :math:`c \ne 0`.
    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).

    `gengamma` takes :math:`a` and :math:`c` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "c")

    scipy_name = "gengamma"


@export
class genhalflogistic(UnivariateDistribution):
    r"""A generalized half-logistic continuous random variable.

    Notes
    -----
    The probability density function for `genhalflogistic` is:

    .. math::

        f(x, c) = \frac{2 (1 - c x)^{1/(c-1)}}{[1 + (1 - c x)^{1/c}]^2}

    for :math:`0 \le x \le 1/c`, and :math:`c > 0`.

    `genhalflogistic` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "genhalflogistic"


@export
class geninvgauss(UnivariateDistribution):
    r"""A Generalized Inverse Gaussian continuous random variable.

    Notes
    -----
    The probability density function for `geninvgauss` is:

    .. math::

        f(x, p, b) = x^{p-1} \exp(-b (x + 1/x) / 2) / (2 K_p(b))

    where `x > 0`, and the parameters `p, b` satisfy `b > 0` ([1]_).
    :math:`K_p` is the modified Bessel function of second kind of order `p`
    (`scipy.special.kv`).
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("p", "b")

    scipy_name = "geninvgauss"


@export
class genlogistic(UnivariateDistribution):
    r"""A generalized logistic continuous random variable.

    Notes
    -----
    The probability density function for `genlogistic` is:

    .. math::

        f(x, c) = c \frac{\exp(-x)}
                         {(1 + \exp(-x))^{c+1}}

    for :math:`x >= 0`, :math:`c > 0`.

    `genlogistic` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "genlogistic"


@export
class gennorm(UnivariateDistribution):
    r"""A generalized normal continuous random variable.

    Notes
    -----
    The probability density function for `gennorm` is [1]_:

    .. math::

        f(x, \beta) = \frac{\beta}{2 \Gamma(1/\beta)} \exp(-|x|^\beta)

    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).

    `gennorm` takes ``beta`` as a shape parameter for :math:`\beta`.
    For :math:`\beta = 1`, it is identical to a Laplace distribution.
    For :math:`\beta = 2`, it is identical to a normal distribution
    (with ``scale=1/sqrt(2)``).

    References
    ----------

    .. [1] "Generalized normal distribution, Version 1",
           https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("beta")

    scipy_name = "gennorm"


@export
class genpareto(UnivariateDistribution):
    r"""A generalized Pareto continuous random variable.

    Notes
    -----
    The probability density function for `genpareto` is:

    .. math::

        f(x, c) = (1 + c x)^{-1 - 1/c}

    defined for :math:`x \ge 0` if :math:`c \ge 0`, and for
    :math:`0 \le x \le -1/c` if :math:`c < 0`.

    `genpareto` takes ``c`` as a shape parameter for :math:`c`.

    For :math:`c=0`, `genpareto` reduces to the exponential
    distribution, `expon`:

    .. math::

        f(x, 0) = \exp(-x)

    For :math:`c=-1`, `genpareto` is uniform on ``[0, 1]``:

    .. math::

        f(x, -1) = 1
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "genpareto"


@export
class geom(UnivariateDiscreteDistribution):
    r"""A geometric discrete random variable.

    Notes
    -----
    The probability mass function for `geom` is:

    .. math::

        f(k) = (1-p)^{k-1} p

    for :math:`k \ge 1`, :math:`0 < p \leq 1`

    `geom` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("p")

    scipy_name = "geom"
    torch_name = "Geometric"
    tfp_name = "Geometric"


@export
class gilbrat(UnivariateDistribution):
    r"""A Gilbrat continuous random variable.

    Notes
    -----
    The probability density function for `gilbrat` is:

    .. math::

        f(x) = \frac{1}{x \sqrt{2\pi}} \exp(-\frac{1}{2} (\log(x))^2)

    `gilbrat` is a special case of `lognorm` with ``s=1``.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "gilbrat"


@export
class gompertz(UnivariateDistribution):
    r"""A Gompertz (or truncated Gumbel) continuous random variable.

    Notes
    -----
    The probability density function for `gompertz` is:

    .. math::

        f(x, c) = c \exp(x) \exp(-c (e^x-1))

    for :math:`x \ge 0`, :math:`c > 0`.

    `gompertz` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "gompertz"


@export
class gumbel_l(UnivariateDistribution):
    r"""A left-skewed Gumbel continuous random variable.

    Notes
    -----
    The probability density function for `gumbel_l` is:

    .. math::

        f(x) = \exp(x - e^x)

    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett
    distribution.  It is also related to the extreme value distribution,
    log-Weibull and Gompertz distributions.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "gumbel_l"


@export
class gumbel_r(UnivariateDistribution):
    r"""A right-skewed Gumbel continuous random variable.

    Notes
    -----
    The probability density function for `gumbel_r` is:

    .. math::

        f(x) = \exp(-(x + e^{-x}))

    The Gumbel distribution is sometimes referred to as a type I Fisher-Tippett
    distribution.  It is also related to the extreme value distribution,
    log-Weibull and Gompertz distributions.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "gumbel_r"
    torch_name = "Gumbel"
    tfp_name = "Gumbel"


@export
class halfcauchy(UnivariateDistribution):
    r"""A Half-Cauchy continuous random variable.

    Notes
    -----
    The probability density function for `halfcauchy` is:

    .. math::

        f(x) = \frac{2}{\pi (1 + x^2)}

    for :math:`x \ge 0`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "halfcauchy"
    torch_name = "HalfCauchy"
    tfp_name = "HalfCauchy"


@export
class halfgennorm(UnivariateDistribution):
    r"""The upper half of a generalized normal continuous random variable.

    Notes
    -----
    The probability density function for `halfgennorm` is:

    .. math::

        f(x, \beta) = \frac{\beta}{\Gamma(1/\beta)} \exp(-|x|^\beta)

    for :math:`x > 0`. :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).

    `gennorm` takes ``beta`` as a shape parameter for :math:`\beta`.
    For :math:`\beta = 1`, it is identical to an exponential distribution.
    For :math:`\beta = 2`, it is identical to a half normal distribution
    (with ``scale=1/sqrt(2)``).

    References
    ----------

    .. [1] "Generalized normal distribution, Version 1",
           https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("beta")

    scipy_name = "halfgennorm"


@export
class halflogistic(UnivariateDistribution):
    r"""A half-logistic continuous random variable.

    Notes
    -----
    The probability density function for `halflogistic` is:

    .. math::

        f(x) = \frac{ 2 e^{-x} }{ (1+e^{-x})^2 }
             = \frac{1}{2} \text{sech}(x/2)^2

    for :math:`x \ge 0`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "halflogistic"


@export
class halfnorm(UnivariateDistribution):
    r"""A half-normal continuous random variable.

    Notes
    -----
    The probability density function for `halfnorm` is:

    .. math::

        f(x) = \sqrt{2/\pi} \exp(-x^2 / 2)

    for :math:`x >= 0`.

    `halfnorm` is a special case of `chi` with ``df=1``.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "halfnorm"
    torch_name = "HalfNormal"
    tfp_name = "HalfNormal"


@export
class hypergeom(UnivariateDiscreteDistribution):
    r"""A hypergeometric discrete random variable.

    The hypergeometric distribution models drawing objects from a bin.
    `M` is the total number of objects, `n` is total number of Type I objects.
    The random variate represents the number of Type I objects in `N` drawn
    without replacement from the total population.

    Notes
    -----
    The symbols used to denote the shape parameters (`M`, `n`, and `N`) are not
    universally accepted.  See the Examples for a clarification of the
    definitions used here.

    The probability mass function is defined as,

    .. math:: p(k, M, n, N) = \frac{\binom{n}{k} \binom{M - n}{N - k}}
                                   {\binom{M}{N}}

    for :math:`k \in [\max(0, N - M + n), \min(n, N)]`, where the binomial
    coefficients are defined as,

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("M", "n", "N")

    scipy_name = "hypergeom"


@export
class hypsecant(UnivariateDistribution):
    r"""A hyperbolic secant continuous random variable.

    Notes
    -----
    The probability density function for `hypsecant` is:

    .. math::

        f(x) = \frac{1}{\pi} \text{sech}(x)

    for a real number :math:`x`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "hypsecant"


@export
class invgamma(UnivariateDistribution):
    r"""An inverted gamma continuous random variable.

    Notes
    -----
    The probability density function for `invgamma` is:

    .. math::

        f(x, a) = \frac{x^{-a-1}}{\Gamma(a)} \exp(-\frac{1}{x})

    for :math:`x >= 0`, :math:`a > 0`. :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).

    `invgamma` takes ``a`` as a shape parameter for :math:`a`.

    `invgamma` is a special case of `gengamma` with ``c=-1``.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a")

    scipy_name = "invgamma"
    tfp_name = "InverseGamma"


@export
class invgauss(UnivariateDistribution):
    r"""An inverse Gaussian continuous random variable.

    Notes
    -----
    The probability density function for `invgauss` is:

    .. math::

        f(x, \mu) = \frac{1}{\sqrt{2 \pi x^3}}
                    \exp(-\frac{(x-\mu)^2}{2 x \mu^2})

    for :math:`x >= 0` and :math:`\mu > 0`.

    `invgauss` takes ``mu`` as a shape parameter for :math:`\mu`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("mu")

    scipy_name = "invgauss"


@export
class invweibull(UnivariateDistribution):
    r"""An inverted Weibull continuous random variable.

    This distribution is also known as the Fréchet distribution or the
    type II extreme value distribution.

    Notes
    -----
    The probability density function for `invweibull` is:

    .. math::

        f(x, c) = c x^{-c-1} \exp(-x^{-c})

    for :math:`x > 0`, :math:`c > 0`.

    `invweibull` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "invweibull"


@export
class johnsonsb(UnivariateDistribution):
    r"""A Johnson SB continuous random variable.

    Notes
    -----
    The probability density function for `johnsonsb` is:

    .. math::

        f(x, a, b) = \frac{b}{x(1-x)}  \phi(a + b \log \frac{x}{1-x} )

    for :math:`0 <= x < =1` and :math:`a, b > 0`, and :math:`\phi` is the normal
    pdf.

    `johnsonsb` takes :math:`a` and :math:`b` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "b")

    scipy_name = "johnsonsb"


@export
class johnsonsu(UnivariateDistribution):
    r"""A Johnson SU continuous random variable.

    Notes
    -----
    The probability density function for `johnsonsu` is:

    .. math::

        f(x, a, b) = \frac{b}{\sqrt{x^2 + 1}}
                     \phi(a + b \log(x + \sqrt{x^2 + 1}))

    for all :math:`x, a, b > 0`, and :math:`\phi` is the normal pdf.

    `johnsonsu` takes :math:`a` and :math:`b` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "b")

    scipy_name = "johnsonsu"


@export
class kappa3(UnivariateDistribution):
    r"""Kappa 3 parameter distribution.

    Notes
    -----
    The probability density function for `kappa3` is:

    .. math::

        f(x, a) = a (a + x^a)^{-(a + 1)/a}

    for :math:`x > 0` and :math:`a > 0`.

    `kappa3` takes ``a`` as a shape parameter for :math:`a`.

    References
    ----------
    P.W. Mielke and E.S. Johnson, "Three-Parameter Kappa Distribution Maximum
    Likelihood and Likelihood Ratio Tests", Methods in Weather Research,
    701-707, (September, 1973),
    :doi:`10.1175/1520-0493(1973)101<0701:TKDMLE>2.3.CO;2`

    B. Kumphon, "Maximum Entropy and Maximum Likelihood Estimation for the
    Three-Parameter Kappa Distribution", Open Journal of Statistics, vol 2,
    415-419 (2012), :doi:`10.4236/ojs.2012.24050`
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a")

    scipy_name = "kappa3"


@export
class kappa4(UnivariateDistribution):
    r"""Kappa 4 parameter distribution.

    Notes
    -----
    The probability density function for kappa4 is:

    .. math::

        f(x, h, k) = (1 - k x)^{1/k - 1} (1 - h (1 - k x)^{1/k})^{1/h-1}

    if :math:`h` and :math:`k` are not equal to 0.

    If :math:`h` or :math:`k` are zero then the pdf can be simplified:

    h = 0 and k != 0::

        kappa4.pdf(x, h, k) = (1.0 - k*x)**(1.0/k - 1.0)*
                              exp(-(1.0 - k*x)**(1.0/k))

    h != 0 and k = 0::

        kappa4.pdf(x, h, k) = exp(-x)*(1.0 - h*exp(-x))**(1.0/h - 1.0)

    h = 0 and k = 0::

        kappa4.pdf(x, h, k) = exp(-x)*exp(-exp(-x))

    kappa4 takes :math:`h` and :math:`k` as shape parameters.

    The kappa4 distribution returns other distributions when certain
    :math:`h` and :math:`k` values are used.

    +------+-------------+----------------+------------------+
    | h    | k=0.0       | k=1.0          | -inf<=k<=inf     |
    +======+=============+================+==================+
    | -1.0 | Logistic    |                | Generalized      |
    |      |             |                | Logistic(1)      |
    |      |             |                |                  |
    |      | logistic(x) |                |                  |
    +------+-------------+----------------+------------------+
    |  0.0 | Gumbel      | Reverse        | Generalized      |
    |      |             | Exponential(2) | Extreme Value    |
    |      |             |                |                  |
    |      | gumbel_r(x) |                | genextreme(x, k) |
    +------+-------------+----------------+------------------+
    |  1.0 | Exponential | Uniform        | Generalized      |
    |      |             |                | Pareto           |
    |      |             |                |                  |
    |      | expon(x)    | uniform(x)     | genpareto(x, -k) |
    +------+-------------+----------------+------------------+

    (1) There are at least five generalized logistic distributions.
        Four are described here:
        https://en.wikipedia.org/wiki/Generalized_logistic_distribution
        The "fifth" one is the one kappa4 should match which currently
        isn't implemented in scipy:
        https://en.wikipedia.org/wiki/Talk:Generalized_logistic_distribution
        https://www.mathwave.com/help/easyfit/html/analyses/distributions/gen_logistic.html
    (2) This distribution is currently not in scipy.

    References
    ----------
    J.C. Finney, "Optimization of a Skewed Logistic Distribution With Respect
    to the Kolmogorov-Smirnov Test", A Dissertation Submitted to the Graduate
    Faculty of the Louisiana State University and Agricultural and Mechanical
    College, (August, 2004),
    https://digitalcommons.lsu.edu/gradschool_dissertations/3672

    J.R.M. Hosking, "The four-parameter kappa distribution". IBM J. Res.
    Develop. 38 (3), 25 1-258 (1994).

    B. Kumphon, A. Kaew-Man, P. Seenoi, "A Rainfall Distribution for the Lampao
    Site in the Chi River Basin, Thailand", Journal of Water Resource and
    Protection, vol. 4, 866-869, (2012).
    :doi:`10.4236/jwarp.2012.410101`

    C. Winchester, "On Estimation of the Four-Parameter Kappa Distribution", A
    Thesis Submitted to Dalhousie University, Halifax, Nova Scotia, (March
    2000).
    http://www.nlc-bnc.ca/obj/s4/f2/dsk2/ftp01/MQ57336.pdf
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("h", "k")

    scipy_name = "kappa4"


@export
class ksone(UnivariateDistribution):
    r"""Kolmogorov-Smirnov one-sided test statistic distribution.

    This is the distribution of the one-sided Kolmogorov-Smirnov (KS)
    statistics :math:`D_n^+` and :math:`D_n^-`
    for a finite sample size ``n`` (the shape parameter).

    Notes
    -----
    :math:`D_n^+` and :math:`D_n^-` are given by

    .. math::

        D_n^+ &= \text{sup}_x (F_n(x) - F(x)),\\
        D_n^- &= \text{sup}_x (F(x) - F_n(x)),\\

    where :math:`F` is a continuous CDF and :math:`F_n` is an empirical CDF.
    `ksone` describes the distribution under the null hypothesis of the KS test
    that the empirical CDF corresponds to :math:`n` i.i.d. random variates
    with CDF :math:`F`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("n")

    scipy_name = "ksone"


@export
class kstwo(UnivariateDistribution):
    r"""Kolmogorov-Smirnov two-sided test statistic distribution.

    This is the distribution of the two-sided Kolmogorov-Smirnov (KS)
    statistic :math:`D_n` for a finite sample size ``n``
    (the shape parameter).

    Notes
    -----
    :math:`D_n` is given by

    .. math::

        D_n &= \text{sup}_x |F_n(x) - F(x)|

    where :math:`F` is a (continuous) CDF and :math:`F_n` is an empirical CDF.
    `kstwo` describes the distribution under the null hypothesis of the KS test
    that the empirical CDF corresponds to :math:`n` i.i.d. random variates
    with CDF :math:`F`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("n")

    scipy_name = "kstwo"


@export
class kstwobign(UnivariateDistribution):
    r"""Limiting distribution of scaled Kolmogorov-Smirnov two-sided test statistic.

    This is the asymptotic distribution of the two-sided Kolmogorov-Smirnov
    statistic :math:`\sqrt{n} D_n` that measures the maximum absolute
    distance of the theoretical (continuous) CDF from the empirical CDF.
    (see `kstest`).

    Notes
    -----
    :math:`\sqrt{n} D_n` is given by

    .. math::

        D_n = \text{sup}_x |F_n(x) - F(x)|

    where :math:`F` is a continuous CDF and :math:`F_n` is an empirical CDF.
    `kstwobign`  describes the asymptotic distribution (i.e. the limit of
    :math:`\sqrt{n} D_n`) under the null hypothesis of the KS test that the
    empirical CDF corresponds to i.i.d. random variates with CDF :math:`F`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "kstwobign"


@export
class laplace(UnivariateDistribution):
    r"""A Laplace continuous random variable.

    Notes
    -----
    The probability density function for `laplace` is

    .. math::

        f(x) = \frac{1}{2} \exp(-|x|)

    for a real number :math:`x`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "laplace"
    torch_name = "Laplace"
    tfp_name = "Laplace"


@export
class laplace_asymmetric(UnivariateDistribution):
    r"""An asymmetric Laplace continuous random variable.

    Notes
    -----
    The probability density function for `laplace_asymmetric` is

    .. math::

       f(x, \kappa) &= \frac{1}{\kappa+\kappa^{-1}}\exp(-x\kappa),\quad x\ge0\\
                    &= \frac{1}{\kappa+\kappa^{-1}}\exp(x/\kappa),\quad x<0\\

    for :math:`-\infty < x < \infty`, :math:`\kappa > 0`.

    `laplace_asymmetric` takes ``kappa`` as a shape parameter for
    :math:`\kappa`. For :math:`\kappa = 1`, it is identical to a
    Laplace distribution.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("kappa")

    scipy_name = "laplace_asymmetric"


@export
class levy(UnivariateDistribution):
    r"""A Levy continuous random variable.

    Notes
    -----
    The probability density function for `levy` is:

    .. math::

        f(x) = \frac{1}{\sqrt{2\pi x^3}} \exp\left(-\frac{1}{2x}\right)

    for :math:`x >= 0`.

    This is the same as the Levy-stable distribution with :math:`a=1/2` and
    :math:`b=1`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "levy"


@export
class levy_l(UnivariateDistribution):
    r"""A left-skewed Levy continuous random variable.

    Notes
    -----
    The probability density function for `levy_l` is:

    .. math::
        f(x) = \frac{1}{|x| \sqrt{2\pi |x|}} \exp{ \left(-\frac{1}{2|x|} \right)}

    for :math:`x <= 0`.

    This is the same as the Levy-stable distribution with :math:`a=1/2` and
    :math:`b=-1`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "levy_l"


@export
class levy_stable(UnivariateDistribution):
    r"""A Levy-stable continuous random variable.

    Notes
    -----
    The distribution for `levy_stable` has characteristic function:

    .. math::

        \varphi(t, \alpha, \beta, c, \mu) =
        e^{it\mu -|ct|^{\alpha}(1-i\beta \operatorname{sign}(t)\Phi(\alpha, t))}

    where:

    .. math::

        \Phi = \begin{cases}
                \tan \left({\frac {\pi \alpha }{2}}\right)&\alpha \neq 1\\
                -{\frac {2}{\pi }}\log |t|&\alpha =1
                \end{cases}

    The probability density function for `levy_stable` is:

    .. math::

        f(x) = \frac{1}{2\pi}\int_{-\infty}^\infty \varphi(t)e^{-ixt}\,dt

    where :math:`-\infty < t < \infty`. This integral does not have a known closed form.

    For evaluation of pdf we use either Zolotarev :math:`S_0` parameterization with integration,
    direct integration of standard parameterization of characteristic function or FFT of
    characteristic function. If set to other than None and if number of points is greater than
    ``levy_stable.pdf_fft_min_points_threshold`` (defaults to None) we use FFT otherwise we use one
    of the other methods.

    The default method is 'best' which uses Zolotarev's method if alpha = 1 and integration of
    characteristic function otherwise. The default method can be changed by setting
    ``levy_stable.pdf_default_method`` to either 'zolotarev', 'quadrature' or 'best'.

    To increase accuracy of FFT calculation one can specify ``levy_stable.pdf_fft_grid_spacing``
    (defaults to 0.001) and ``pdf_fft_n_points_two_power`` (defaults to a value that covers the
    input range * 4). Setting ``pdf_fft_n_points_two_power`` to 16 should be sufficiently accurate
    in most cases at the expense of CPU time.

    For evaluation of cdf we use Zolatarev :math:`S_0` parameterization with integration or integral of
    the pdf FFT interpolated spline. The settings affecting FFT calculation are the same as
    for pdf calculation. Setting the threshold to ``None`` (default) will disable FFT. For cdf
    calculations the Zolatarev method is superior in accuracy, so FFT is disabled by default.

    Fitting estimate uses quantile estimation method in [MC]. MLE estimation of parameters in
    fit method uses this quantile estimate initially. Note that MLE doesn't always converge if
    using FFT for pdf calculations; so it's best that ``pdf_fft_min_points_threshold`` is left unset.

    .. warning::

        For pdf calculations implementation of Zolatarev is unstable for values where alpha = 1 and
        beta != 0. In this case the quadrature method is recommended. FFT calculation is also
        considered experimental.

        For cdf calculations FFT calculation is considered experimental. Use Zolatarev's method
        instead (default).
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("alpha", "beta")

    scipy_name = "levy_stable"


@export
class loggamma(UnivariateDistribution):
    r"""A log gamma continuous random variable.

    Notes
    -----
    The probability density function for `loggamma` is:

    .. math::

        f(x, c) = \frac{\exp(c x - \exp(x))}
                       {\Gamma(c)}

    for all :math:`x, c > 0`. Here, :math:`\Gamma` is the
    gamma function (`scipy.special.gamma`).

    `loggamma` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "loggamma"


@export
class logistic(UnivariateDistribution):
    r"""A logistic (or Sech-squared) continuous random variable.

    Notes
    -----
    The probability density function for `logistic` is:

    .. math::

        f(x) = \frac{\exp(-x)}
                    {(1+\exp(-x))^2}

    `logistic` is a special case of `genlogistic` with ``c=1``.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "logistic"
    tfp_name = "Logistic"


@export
class loglaplace(UnivariateDistribution):
    r"""A log-Laplace continuous random variable.

    Notes
    -----
    The probability density function for `loglaplace` is:

    .. math::

        f(x, c) = \begin{cases}\frac{c}{2} x^{ c-1}  &\text{for } 0 < x < 1\\
                               \frac{c}{2} x^{-c-1}  &\text{for } x \ge 1
                  \end{cases}

    for :math:`c > 0`.

    `loglaplace` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "loglaplace"


@export
class lognorm(UnivariateDistribution):
    r"""A lognormal continuous random variable.

    Notes
    -----
    The probability density function for `lognorm` is:

    .. math::

        f(x, s) = \frac{1}{s x \sqrt{2\pi}}
                  \exp\left(-\frac{\log^2(x)}{2s^2}\right)

    for :math:`x > 0`, :math:`s > 0`.

    `lognorm` takes ``s`` as a shape parameter for :math:`s`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("s")

    scipy_name = "lognorm"
    torch_name = "LogNormal"
    tfp_name = "LogNormal"


@export
class logser(UnivariateDiscreteDistribution):
    r"""A Logarithmic (Log-Series, Series) discrete random variable.

    Notes
    -----
    The probability mass function for `logser` is:

    .. math::

        f(k) = - \frac{p^k}{k \log(1-p)}

    for :math:`k \ge 1`, :math:`0 < p < 1`

    `logser` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("p")

    scipy_name = "logser"


@export
class loguniform(UnivariateDistribution):
    r"""A loguniform or reciprocal continuous random variable.

    Notes
    -----
    The probability density function for this class is:

    .. math::

        f(x, a, b) = \frac{1}{x \log(b/a)}

    for :math:`a \le x \le b`, :math:`b > a > 0`. This class takes
    :math:`a` and :math:`b` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "b")

    scipy_name = "loguniform"


@export
class lomax(UnivariateDistribution):
    r"""A Lomax (Pareto of the second kind) continuous random variable.

    Notes
    -----
    The probability density function for `lomax` is:

    .. math::

        f(x, c) = \frac{c}{(1+x)^{c+1}}

    for :math:`x \ge 0`, :math:`c > 0`.

    `lomax` takes ``c`` as a shape parameter for :math:`c`.

    `lomax` is a special case of `pareto` with ``loc=-1.0``.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "lomax"


@export
class maxwell(UnivariateDistribution):
    r"""A Maxwell continuous random variable.

    Notes
    -----
    A special case of a `chi` distribution,  with ``df=3``, ``loc=0.0``,
    and given ``scale = a``, where ``a`` is the parameter used in the
    Mathworld description [1]_.

    The probability density function for `maxwell` is:

    .. math::

        f(x) = \sqrt{2/\pi}x^2 \exp(-x^2/2)

    for :math:`x >= 0`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "maxwell"


@export
class mielke(UnivariateDistribution):
    r"""A Mielke Beta-Kappa / Dagum continuous random variable.

    Notes
    -----
    The probability density function for `mielke` is:

    .. math::

        f(x, k, s) = \frac{k x^{k-1}}{(1+x^s)^{1+k/s}}

    for :math:`x > 0` and :math:`k, s > 0`. The distribution is sometimes
    called Dagum distribution ([2]_). It was already defined in [3]_, called
    a Burr Type III distribution (`burr` with parameters ``c=s`` and
    ``d=k/s``).

    `mielke` takes ``k`` and ``s`` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("k", "s")

    scipy_name = "mielke"


@export
class moyal(UnivariateDistribution):
    r"""A Moyal continuous random variable.

    Notes
    -----
    The probability density function for `moyal` is:

    .. math::

        f(x) = \exp(-(x + \exp(-x))/2) / \sqrt{2\pi}

    for a real number :math:`x`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "moyal"
    tfp_name = "Moyal"


@export
class nakagami(UnivariateDistribution):
    r"""A Nakagami continuous random variable.

    Notes
    -----
    The probability density function for `nakagami` is:

    .. math::

        f(x, \nu) = \frac{2 \nu^\nu}{\Gamma(\nu)} x^{2\nu-1} \exp(-\nu x^2)

    for :math:`x >= 0`, :math:`\nu > 0`.

    `nakagami` takes ``nu`` as a shape parameter for :math:`\nu`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("nu")

    scipy_name = "nakagami"


@export
class nbinom(UnivariateDiscreteDistribution):
    r"""A negative binomial discrete random variable.

    Notes
    -----
    Negative binomial distribution describes a sequence of i.i.d. Bernoulli
    trials, repeated until a predefined, non-random number of successes occurs.

    The probability mass function of the number of failures for `nbinom` is:

    .. math::

       f(k) = \binom{k+n-1}{n-1} p^n (1-p)^k

    for :math:`k \ge 0`, :math:`0 < p \leq 1`

    `nbinom` takes :math:`n` and :math:`p` as shape parameters where n is the
    number of successes, :math:`p` is the probability of a single success,
    and :math:`1-p` is the probability of a single failure.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("n", "p")

    scipy_name = "nbinom"
    torch_name = "NegativeBinomial"
    tfp_name = "NegativeBinomial"


@export
class ncf(UnivariateDistribution):
    r"""A non-central F distribution continuous random variable.

    Notes
    -----
    The probability density function for `ncf` is:

    .. math::

        f(x, n_1, n_2, \lambda) =
            \exp\left(\frac{\lambda}{2} +
                      \lambda n_1 \frac{x}{2(n_1 x + n_2)}
                \right)
            n_1^{n_1/2} n_2^{n_2/2} x^{n_1/2 - 1} \\
            (n_2 + n_1 x)^{-(n_1 + n_2)/2}
            \gamma(n_1/2) \gamma(1 + n_2/2) \\
            \frac{L^{\frac{n_1}{2}-1}_{n_2/2}
                \left(-\lambda n_1 \frac{x}{2(n_1 x + n_2)}\right)}
            {B(n_1/2, n_2/2)
                \gamma\left(\frac{n_1 + n_2}{2}\right)}

    for :math:`n_1, n_2 > 0`, :math:`\lambda\geq 0`.  Here :math:`n_1` is the
    degrees of freedom in the numerator, :math:`n_2` the degrees of freedom in
    the denominator, :math:`\lambda` the non-centrality parameter,
    :math:`\gamma` is the logarithm of the Gamma function, :math:`L_n^k` is a
    generalized Laguerre polynomial and :math:`B` is the beta function.

    `ncf` takes ``df1``, ``df2`` and ``nc`` as shape parameters. If ``nc=0``,
    the distribution becomes equivalent to the Fisher distribution.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("dfn", "dfd", "nc")

    scipy_name = "ncf"


@export
class nct(UnivariateDistribution):
    r"""A non-central Student's t continuous random variable.

    Notes
    -----
    If :math:`Y` is a standard normal random variable and :math:`V` is
    an independent chi-square random variable (`chi2`) with :math:`k` degrees
    of freedom, then

    .. math::

        X = \frac{Y + c}{\sqrt{V/k}}

    has a non-central Student's t distribution on the real line.
    The degrees of freedom parameter :math:`k` (denoted ``df`` in the
    implementation) satisfies :math:`k > 0` and the noncentrality parameter
    :math:`c` (denoted ``nc`` in the implementation) is a real number.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("df", "nc")

    scipy_name = "nct"


@export
class ncx2(UnivariateDistribution):
    r"""A non-central chi-squared continuous random variable.

    Notes
    -----
    The probability density function for `ncx2` is:

    .. math::

        f(x, k, \lambda) = \frac{1}{2} \exp(-(\lambda+x)/2)
            (x/\lambda)^{(k-2)/4}  I_{(k-2)/2}(\sqrt{\lambda x})

    for :math:`x >= 0` and :math:`k, \lambda > 0`. :math:`k` specifies the
    degrees of freedom (denoted ``df`` in the implementation) and
    :math:`\lambda` is the non-centrality parameter (denoted ``nc`` in the
    implementation). :math:`I_\nu` denotes the modified Bessel function of
    first order of degree :math:`\nu` (`scipy.special.iv`).

    `ncx2` takes ``df`` and ``nc`` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("df", "nc")

    scipy_name = "ncx2"


@export
class nhypergeom(UnivariateDiscreteDistribution):
    r"""A negative hypergeometric discrete random variable.

    Consider a box containing :math:`M` balls:, :math:`n` red and
    :math:`M-n` blue. We randomly sample balls from the box, one
    at a time and *without* replacement, until we have picked :math:`r`
    blue balls. `nhypergeom` is the distribution of the number of
    red balls :math:`k` we have picked.

    Notes
    -----
    The symbols used to denote the shape parameters (`M`, `n`, and `r`) are not
    universally accepted. See the Examples for a clarification of the
    definitions used here.

    The probability mass function is defined as,

    .. math:: f(k; M, n, r) = \frac{{{k+r-1}\choose{k}}{{M-r-k}\choose{n-k}}}
                                   {{M \choose n}}

    for :math:`k \in [0, n]`, :math:`n \in [0, M]`, :math:`r \in [0, M-n]`,
    and the binomial coefficient is:

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    It is equivalent to observing :math:`k` successes in :math:`k+r-1`
    samples with :math:`k+r`'th sample being a failure. The former
    can be modelled as a hypergeometric distribution. The probability
    of the latter is simply the number of failures remaining
    :math:`M-n-(r-1)` divided by the size of the remaining population
    :math:`M-(k+r-1)`. This relationship can be shown as:

    .. math:: NHG(k;M,n,r) = HG(k;M,n,k+r-1)\frac{(M-n-(r-1))}{(M-(k+r-1))}

    where :math:`NHG` is probability mass function (PMF) of the
    negative hypergeometric distribution and :math:`HG` is the
    PMF of the hypergeometric distribution.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("M", "n", "r")

    scipy_name = "nhypergeom"


@export
class norm(UnivariateDistribution):
    r"""A normal continuous random variable.

    The location (``loc``) keyword specifies the mean.
    The scale (``scale``) keyword specifies the standard deviation.

    Notes
    -----
    The probability density function for `norm` is:

    .. math::

        f(x) = \frac{\exp(-x^2/2)}{\sqrt{2\pi}}

    for a real number :math:`x`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "norm"
    torch_name = "Normal"
    tfp_name = "Normal"


@export
class norminvgauss(UnivariateDistribution):
    r"""A Normal Inverse Gaussian continuous random variable.

    Notes
    -----
    The probability density function for `norminvgauss` is:

    .. math::

        f(x, a, b) = \frac{a \, K_1(a \sqrt{1 + x^2})}{\pi \sqrt{1 + x^2}} \,
                     \exp(\sqrt{a^2 - b^2} + b x)

    where :math:`x` is a real number, the parameter :math:`a` is the tail
    heaviness and :math:`b` is the asymmetry parameter satisfying
    :math:`a > 0` and :math:`|b| <= a`.
    :math:`K_1` is the modified Bessel function of second kind
    (`scipy.special.k1`).
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "b")

    scipy_name = "norminvgauss"


@export
class pareto(UnivariateDistribution):
    r"""A Pareto continuous random variable.

    Notes
    -----
    The probability density function for `pareto` is:

    .. math::

        f(x, b) = \frac{b}{x^{b+1}}

    for :math:`x \ge 1`, :math:`b > 0`.

    `pareto` takes ``b`` as a shape parameter for :math:`b`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("b")

    scipy_name = "pareto"
    torch_name = "Pareto"
    tfp_name = "Pareto"


@export
class pearson3(UnivariateDistribution):
    r"""A pearson type III continuous random variable.

    Notes
    -----
    The probability density function for `pearson3` is:

    .. math::

        f(x, \kappa) = \frac{|\beta|}{\Gamma(\alpha)}
                       (\beta (x - \zeta))^{\alpha - 1}
                       \exp(-\beta (x - \zeta))

    where:

    .. math::

            \beta = \frac{2}{\kappa}

            \alpha = \beta^2 = \frac{4}{\kappa^2}

            \zeta = -\frac{\alpha}{\beta} = -\beta

    :math:`\Gamma` is the gamma function (`scipy.special.gamma`).
    Pass the skew :math:`\kappa` into `pearson3` as the shape parameter
    ``skew``.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("skew")

    scipy_name = "pearson3"


@export
class planck(UnivariateDiscreteDistribution):
    r"""A Planck discrete exponential random variable.

    Notes
    -----
    The probability mass function for `planck` is:

    .. math::

        f(k) = (1-\exp(-\lambda)) \exp(-\lambda k)

    for :math:`k \ge 0` and :math:`\lambda > 0`.

    `planck` takes :math:`\lambda` as shape parameter. The Planck distribution
    can be written as a geometric distribution (`geom`) with
    :math:`p = 1 - \exp(-\lambda)` shifted by `loc = -1`.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("lambda_")

    scipy_name = "planck"


@export
class poisson(UnivariateDiscreteDistribution):
    r"""A Poisson discrete random variable.

    Notes
    -----
    The probability mass function for `poisson` is:

    .. math::

        f(k) = \exp(-\mu) \frac{\mu^k}{k!}

    for :math:`k \ge 0`.

    `poisson` takes :math:`\mu` as shape parameter.
    When mu = 0 then at quantile k = 0, ``pmf`` method
    returns `1.0`.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("mu")

    scipy_name = "poisson"
    torch_name = "Poisson"
    tfp_name = "Poisson"


@export
class powerlaw(UnivariateDistribution):
    r"""A power-function continuous random variable.

    Notes
    -----
    The probability density function for `powerlaw` is:

    .. math::

        f(x, a) = a x^{a-1}

    for :math:`0 \le x \le 1`, :math:`a > 0`.

    `powerlaw` takes ``a`` as a shape parameter for :math:`a`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a")

    scipy_name = "powerlaw"


@export
class powerlognorm(UnivariateDistribution):
    r"""A power log-normal continuous random variable.

    Notes
    -----
    The probability density function for `powerlognorm` is:

    .. math::

        f(x, c, s) = \frac{c}{x s} \phi(\log(x)/s)
                     (\Phi(-\log(x)/s))^{c-1}

    where :math:`\phi` is the normal pdf, and :math:`\Phi` is the normal cdf,
    and :math:`x > 0`, :math:`s, c > 0`.

    `powerlognorm` takes :math:`c` and :math:`s` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c", "s")

    scipy_name = "powerlognorm"


@export
class powernorm(UnivariateDistribution):
    r"""A power normal continuous random variable.

    Notes
    -----
    The probability density function for `powernorm` is:

    .. math::

        f(x, c) = c \phi(x) (\Phi(-x))^{c-1}

    where :math:`\phi` is the normal pdf, and :math:`\Phi` is the normal cdf,
    and :math:`x >= 0`, :math:`c > 0`.

    `powernorm` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "powernorm"


@export
class randint(UnivariateDiscreteDistribution):
    r"""A uniform discrete random variable.

    Notes
    -----
    The probability mass function for `randint` is:

    .. math::

        f(k) = \frac{1}{high - low}

    for ``k = low, ..., high - 1``.

    `randint` takes ``low`` and ``high`` as shape parameters.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("low", "high")

    scipy_name = "randint"


@export
class rayleigh(UnivariateDistribution):
    r"""A Rayleigh continuous random variable.

    Notes
    -----
    The probability density function for `rayleigh` is:

    .. math::

        f(x) = x \exp(-x^2/2)

    for :math:`x \ge 0`.

    `rayleigh` is a special case of `chi` with ``df=2``.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "rayleigh"


@export
class rdist(UnivariateDistribution):
    r"""An R-distributed (symmetric beta) continuous random variable.

    Notes
    -----
    The probability density function for `rdist` is:

    .. math::

        f(x, c) = \frac{(1-x^2)^{c/2-1}}{B(1/2, c/2)}

    for :math:`-1 \le x \le 1`, :math:`c > 0`. `rdist` is also called the
    symmetric beta distribution: if B has a `beta` distribution with
    parameters (c/2, c/2), then X = 2*B - 1 follows a R-distribution with
    parameter c.

    `rdist` takes ``c`` as a shape parameter for :math:`c`.

    This distribution includes the following distribution kernels as
    special cases::

        c = 2:  uniform
        c = 3:  `semicircular`
        c = 4:  Epanechnikov (parabolic)
        c = 6:  quartic (biweight)
        c = 8:  triweight
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "rdist"


@export
class recipinvgauss(UnivariateDistribution):
    r"""A reciprocal inverse Gaussian continuous random variable.

    Notes
    -----
    The probability density function for `recipinvgauss` is:

    .. math::

        f(x, \mu) = \frac{1}{\sqrt{2\pi x}}
                    \exp\left(\frac{-(1-\mu x)^2}{2\mu^2x}\right)

    for :math:`x \ge 0`.

    `recipinvgauss` takes ``mu`` as a shape parameter for :math:`\mu`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("mu")

    scipy_name = "recipinvgauss"


@export
class reciprocal(UnivariateDistribution):
    r"""A loguniform or reciprocal continuous random variable.

    Notes
    -----
    The probability density function for this class is:

    .. math::

        f(x, a, b) = \frac{1}{x \log(b/a)}

    for :math:`a \le x \le b`, :math:`b > a > 0`. This class takes
    :math:`a` and :math:`b` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "b")

    scipy_name = "reciprocal"


@export
class rice(UnivariateDistribution):
    r"""A Rice continuous random variable.

    Notes
    -----
    The probability density function for `rice` is:

    .. math::

        f(x, b) = x \exp(- \frac{x^2 + b^2}{2}) I_0(x b)

    for :math:`x >= 0`, :math:`b > 0`. :math:`I_0` is the modified Bessel
    function of order zero (`scipy.special.i0`).

    `rice` takes ``b`` as a shape parameter for :math:`b`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("b")

    scipy_name = "rice"


@export
class semicircular(UnivariateDistribution):
    r"""A semicircular continuous random variable.

    Notes
    -----
    The probability density function for `semicircular` is:

    .. math::

        f(x) = \frac{2}{\pi} \sqrt{1-x^2}

    for :math:`-1 \le x \le 1`.

    The distribution is a special case of `rdist` with `c = 3`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "semicircular"


@export
class skellam(UnivariateDiscreteDistribution):
    r"""A  Skellam discrete random variable.

    Notes
    -----
    Probability distribution of the difference of two correlated or
    uncorrelated Poisson random variables.

    Let :math:`k_1` and :math:`k_2` be two Poisson-distributed r.v. with
    expected values :math:`\lambda_1` and :math:`\lambda_2`. Then,
    :math:`k_1 - k_2` follows a Skellam distribution with parameters
    :math:`\mu_1 = \lambda_1 - \rho \sqrt{\lambda_1 \lambda_2}` and
    :math:`\mu_2 = \lambda_2 - \rho \sqrt{\lambda_1 \lambda_2}`, where
    :math:`\rho` is the correlation coefficient between :math:`k_1` and
    :math:`k_2`. If the two Poisson-distributed r.v. are independent then
    :math:`\rho = 0`.

    Parameters :math:`\mu_1` and :math:`\mu_2` must be strictly positive.

    For details see: https://en.wikipedia.org/wiki/Skellam_distribution

    `skellam` takes :math:`\mu_1` and :math:`\mu_2` as shape parameters.
    """

    param_specs = RATE_LOC_PARAMS + shape_params("mu1", "mu2")

    scipy_name = "skellam"
    tfp_name = "Skellam"


@export
class skewnorm(UnivariateDistribution):
    r"""A skew-normal random variable.

    Notes
    -----
    The pdf is::

        skewnorm.pdf(x, a) = 2 * norm.pdf(x) * norm.cdf(a*x)

    `skewnorm` takes a real number :math:`a` as a skewness parameter
    When ``a = 0`` the distribution is identical to a normal distribution
    (`norm`). `rvs` implements the method of [1]_.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a")

    scipy_name = "skewnorm"


@export
class t(UnivariateDistribution):
    r"""A Student's t continuous random variable.

    For the noncentral t distribution, see `nct`.

    Notes
    -----
    The probability density function for `t` is:

    .. math::

        f(x, \nu) = \frac{\Gamma((\nu+1)/2)}
                        {\sqrt{\pi \nu} \Gamma(\nu/2)}
                    (1+x^2/\nu)^{-(\nu+1)/2}

    where :math:`x` is a real number and the degrees of freedom parameter
    :math:`\nu` (denoted ``df`` in the implementation) satisfies
    :math:`\nu > 0`. :math:`\Gamma` is the gamma function
    (`scipy.special.gamma`).
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("df")

    scipy_name = "t"
    torch_name = "StudentT"
    tfp_name = "StudentT"


@export
class trapezoid(UnivariateDistribution):
    r"""A trapezoidal continuous random variable.

    Notes
    -----
    The trapezoidal distribution can be represented with an up-sloping line
    from ``loc`` to ``(loc + c*scale)``, then constant to ``(loc + d*scale)``
    and then downsloping from ``(loc + d*scale)`` to ``(loc+scale)``.  This
    defines the trapezoid base from ``loc`` to ``(loc+scale)`` and the flat
    top from ``c`` to ``d`` proportional to the position along the base
    with ``0 <= c <= d <= 1``.  When ``c=d``, this is equivalent to `triang`
    with the same values for `loc`, `scale` and `c`.
    The method of [1]_ is used for computing moments.

    `trapezoid` takes :math:`c` and :math:`d` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c", "d")

    scipy_name = "trapezoid"


@export
class trapz(UnivariateDistribution):
    r"""trapz is an alias for `trapezoid`
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c", "d")

    scipy_name = "trapz"


@export
class triang(UnivariateDistribution):
    r"""A triangular continuous random variable.

    Notes
    -----
    The triangular distribution can be represented with an up-sloping line from
    ``loc`` to ``(loc + c*scale)`` and then downsloping for ``(loc + c*scale)``
    to ``(loc + scale)``.

    `triang` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "triang"


@export
class truncexpon(UnivariateDistribution):
    r"""A truncated exponential continuous random variable.

    Notes
    -----
    The probability density function for `truncexpon` is:

    .. math::

        f(x, b) = \frac{\exp(-x)}{1 - \exp(-b)}

    for :math:`0 <= x <= b`.

    `truncexpon` takes ``b`` as a shape parameter for :math:`b`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("b")

    scipy_name = "truncexpon"


@export
class truncnorm(UnivariateDistribution):
    r"""A truncated normal continuous random variable.

    Notes
    -----
    The standard form of this distribution is a standard normal truncated to
    the range [a, b] --- notice that a and b are defined over the domain of the
    standard normal.  To convert clip values for a specific mean and standard
    deviation, use::

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    `truncnorm` takes :math:`a` and :math:`b` as shape parameters.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("a", "b")

    scipy_name = "truncnorm"


@export
class tukeylambda(UnivariateDistribution):
    r"""A Tukey-Lamdba continuous random variable.

    Notes
    -----
    A flexible distribution, able to represent and interpolate between the
    following distributions:

    - Cauchy                (:math:`lambda = -1`)
    - logistic              (:math:`lambda = 0`)
    - approx Normal         (:math:`lambda = 0.14`)
    - uniform from -1 to 1  (:math:`lambda = 1`)

    `tukeylambda` takes a real number :math:`lambda` (denoted ``lam``
    in the implementation) as a shape parameter.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("lam")

    scipy_name = "tukeylambda"


@export
class uniform(UnivariateDistribution):
    r"""A uniform continuous random variable.

    In the standard form, the distribution is uniform on ``[0, 1]``. Using
    the parameters ``loc`` and ``scale``, one obtains the uniform distribution
    on ``[loc, loc + scale]``.

    Notes
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "uniform"
    torch_name = "Uniform"
    tfp_name = "Uniform"

    ### CUSTOM!
    ### TODO: preserve on autogeneration
    def torch_param_transform(self, params):
        return dict(low=params["loc"], high=params["loc"] + params["scale"])


@export
class vonmises(UnivariateDistribution):
    r"""A Von Mises continuous random variable.

    Notes
    -----
    The probability density function for `vonmises` and `vonmises_line` is:

    .. math::

        f(x, \kappa) = \frac{ \exp(\kappa \cos(x)) }{ 2 \pi I_0(\kappa) }

    for :math:`-\pi \le x \le \pi`, :math:`\kappa > 0`. :math:`I_0` is the
    modified Bessel function of order zero (`scipy.special.i0`).

    `vonmises` is a circular distribution which does not restrict the
    distribution to a fixed interval. Currently, there is no circular
    distribution framework in scipy. The ``cdf`` is implemented such that
    ``cdf(x + 2*np.pi) == cdf(x) + 1``.

    `vonmises_line` is the same distribution, defined on :math:`[-\pi, \pi]`
    on the real line. This is a regular (i.e. non-circular) distribution.

    `vonmises` and `vonmises_line` take ``kappa`` as a shape parameter.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("kappa")

    scipy_name = "vonmises"
    torch_name = "VonMises"
    tfp_name = "VonMises"


@export
class vonmises_line(UnivariateDistribution):
    r"""A Von Mises continuous random variable.

    Notes
    -----
    The probability density function for `vonmises` and `vonmises_line` is:

    .. math::

        f(x, \kappa) = \frac{ \exp(\kappa \cos(x)) }{ 2 \pi I_0(\kappa) }

    for :math:`-\pi \le x \le \pi`, :math:`\kappa > 0`. :math:`I_0` is the
    modified Bessel function of order zero (`scipy.special.i0`).

    `vonmises` is a circular distribution which does not restrict the
    distribution to a fixed interval. Currently, there is no circular
    distribution framework in scipy. The ``cdf`` is implemented such that
    ``cdf(x + 2*np.pi) == cdf(x) + 1``.

    `vonmises_line` is the same distribution, defined on :math:`[-\pi, \pi]`
    on the real line. This is a regular (i.e. non-circular) distribution.

    `vonmises` and `vonmises_line` take ``kappa`` as a shape parameter.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("kappa")

    scipy_name = "vonmises_line"


@export
class wald(UnivariateDistribution):
    r"""A Wald continuous random variable.

    Notes
    -----
    The probability density function for `wald` is:

    .. math::

        f(x) = \frac{1}{\sqrt{2\pi x^3}} \exp(- \frac{ (x-1)^2 }{ 2x })

    for :math:`x >= 0`.

    `wald` is a special case of `invgauss` with ``mu=1``.
    """

    param_specs = RATE_LOC_SCALE_PARAMS

    scipy_name = "wald"


@export
class weibull_max(UnivariateDistribution):
    r"""Weibull maximum continuous random variable.

    The Weibull Maximum Extreme Value distribution, from extreme value theory
    (Fisher-Gnedenko theorem), is the limiting distribution of rescaled
    maximum of iid random variables. This is the distribution of -X
    if X is from the `weibull_min` function.

    Notes
    -----
    The probability density function for `weibull_max` is:

    .. math::

        f(x, c) = c (-x)^{c-1} \exp(-(-x)^c)

    for :math:`x < 0`, :math:`c > 0`.

    `weibull_max` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "weibull_max"


@export
class weibull_min(UnivariateDistribution):
    r"""Weibull minimum continuous random variable.

    The Weibull Minimum Extreme Value distribution, from extreme value theory
    (Fisher-Gnedenko theorem), is also often simply called the Weibull
    distribution. It arises as the limiting distribution of the rescaled
    minimum of iid random variables.

    Notes
    -----
    The probability density function for `weibull_min` is:

    .. math::

        f(x, c) = c x^{c-1} \exp(-x^c)

    for :math:`x > 0`, :math:`c > 0`.

    `weibull_min` takes ``c`` as a shape parameter for :math:`c`.
    (named :math:`k` in Wikipedia article and :math:`a` in
    ``numpy.random.weibull``).  Special shape values are :math:`c=1` and
    :math:`c=2` where Weibull distribution reduces to the `expon` and
    `rayleigh` distributions respectively.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "weibull_min"
    torch_name = "Weibull"
    tfp_name = "Weibull"


@export
class wrapcauchy(UnivariateDistribution):
    r"""A wrapped Cauchy continuous random variable.

    Notes
    -----
    The probability density function for `wrapcauchy` is:

    .. math::

        f(x, c) = \frac{1-c^2}{2\pi (1+c^2 - 2c \cos(x))}

    for :math:`0 \le x \le 2\pi`, :math:`0 < c < 1`.

    `wrapcauchy` takes ``c`` as a shape parameter for :math:`c`.
    """

    param_specs = RATE_LOC_SCALE_PARAMS + shape_params("c")

    scipy_name = "wrapcauchy"


@export
class yulesimon(UnivariateDiscreteDistribution):
    r"""A Yule-Simon discrete random variable.

    Notes
    -----

    The probability mass function for the `yulesimon` is:

    .. math::

        f(k) =  \alpha B(k, \alpha+1)

    for :math:`k=1,2,3,...`, where :math:`\alpha>0`.
    Here :math:`B` refers to the `scipy.special.beta` function.

    The sampling of random variates is based on pg 553, Section 6.3 of [1]_.
    Our notation maps to the referenced logic via :math:`\alpha=a-1`.

    For details see the wikipedia entry [2]_.

    References
    ----------
    .. [1] Devroye, Luc. "Non-uniform Random Variate Generation",
         (1986) Springer, New York.

    .. [2] https://en.wikipedia.org/wiki/Yule-Simon_distribution
    """

    param_specs = RATE_LOC_PARAMS + shape_params("alpha")

    scipy_name = "yulesimon"


@export
class zipf(UnivariateDiscreteDistribution):
    r"""A Zipf discrete random variable.

    Notes
    -----
    The probability mass function for `zipf` is:

    .. math::

        f(k, a) = \frac{1}{\zeta(a) k^a}

    for :math:`k \ge 1`.

    `zipf` takes :math:`a` as shape parameter. :math:`\zeta` is the
    Riemann zeta function (`scipy.special.zeta`)
    """

    param_specs = RATE_LOC_PARAMS + shape_params("a")

    scipy_name = "zipf"
    tfp_name = "Zipf"
