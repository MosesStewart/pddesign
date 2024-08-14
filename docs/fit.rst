rddesign.RDD.fit
****************
.. RDD.fit:

.. currentmodule:: RDD.fit

Method to fit the regression discontinuity design model with user specified
options.

Parameters
----------
model : str, optional
    Method used to fit the model. Should be one of ``"local linear"`` or
    ``"polynomial"``
design : str, optional
    Design for the RDD model. Should be one of ``"sharp"`` or ``"fuzzy"``.
bootstrap : boolean, optional
    Indicator for whether to use a bayesian bootstrap to compute confidence
    intervals and standard errors. Must be true to compute confidence intervals
    and standard errors when ``design = "fuzzy"``.
bandwidth : float, optional
    Choice of bandwidth to use with ``model = "local linear"``, or 1/2 the width of 
    the window to the left and right of the cutoff to used in local linear regression.
    When left unspecified, bandwidth is chosen using cross-validation procedure from
    :cite:t:`imbens_regression_2008`.
order : int, optional
    Choice of order of polynomial to use with ``model = "polynomial"``. When left
    unspecified, polynomial order is chosen by optimizing using the criterion chosen
    for ``optimization``.
optimization : str, optional
    Optimization method for polynomial order. Should be one of ``"aic"`` to use Akaike 
    Information Criterion or ``"significance bins"`` to use choose polynomial order
    based on hypothesis tests for bin dummies. Default is ``"aic"``.
nbins : int, optional
    Number of bins to use when choosing polynomial order by optimizing with 
    ``optimization = "significance bins"``. Default is no. observations/20.
pval : int, optional
    Significance cutoff for bin dummies when choosing polynomial order by optimizing 
    with ``optimization = "significance bins"``. The algorithm stops once the joint wald test 
    pvalue exceeds ``pval``. Default is ``pval = 0.05``.

Returns
-------
result : RegressionResult
    The optimization result represented as an object. Resembles `statsmodels` result
    object. Important attributes are:

- ``params``, which stores the estimated treatment effect as a `pd.Series`
- ``bse``, which stores the standard errors as a `pd.Series`
- ``summary()``, which summarizes the results
- ``predict(runv)``, which predicts new outcome variables :math:`Y_{\text{new}}` 
  given running variable observations :math:`D_{\text{new}}`

Notes
-----
*Local Linear Estimation* ``model.fit(model = "local linear")``

-----

Setting ``model = "local linear"`` estimates the pooled regression:

.. math::

    Y = \alpha_{l} + \theta_{0} A + \beta_{\ell} (D - c) 
    + (\beta_{r} − \beta_{\ell}) A (D − c) + \epsilon

For :math:`Y`, the outcome variable, :math:`c`, the cutoff, :math:`D`, 
the running variable, and :math:`A`, the treatment. Observations used in the 
regression are restricted to  :math:`c - h < d_{i} < c + h`, where `h` is 
the choice of bandwidth. :cite:t:`lee2010regression` shows that the treatment 
effect is given by  :math:`\theta_{0}`. When exogenous variables are 
specified, we residualize :math:`Y` with respect to the exogenous covariates 
and estimate:

.. math::

    Y - X\pi = \alpha_{l} + \theta_{0} A + \beta_{\ell} (D - c) 
    + (\beta_{r} − \beta_{\ell}) A (D − c) + \epsilon

For exogenous covariates :math:`X`.

-----

*Polynomial Estimation* ``model.fit(model = "polynomial")``

-----

Setting ``model = "polynomial"`` estimates the pooled regression:

.. math::

    Y = \alpha_{l} + \theta_{0} A + \beta_{\ell1} (D - c) 
    + (\beta_{r1} − \beta_{\ell1}) A (D − c) + \dots + 
    \beta_{\ell k} (D - c)^{k} +  (\beta_{rk} − \beta_{\ell k}) A (D − c)^{k}
    +  \epsilon

For :math:`Y`, the outcome variable, :math:`c`, the cutoff, and :math:`D`, 
the running variable, and the order of the polynomial :math:`k`. When 
exogenous variables are specified, we residualize :math:`Y` with respect 
to the exogenous covariates and estimate:

.. math::

    Y - X\pi = \alpha_{l} + \theta_{0} A + \beta_{\ell1} (D - c) 
    + (\beta_{r1} − \beta_{\ell1}) A (D − c) + \dots + 
    \beta_{\ell k} (D - c)^{k} +  (\beta_{rk} − \beta_{\ell k}) A (D − c)^{k}
    +  \epsilon

For exogenous covariates :math:`X`.

-----

*Bandwidth optimization*

-----

When choice for ``bandwidth`` is left unspecified, we use the cross-validation
procedure introduced in :cite:t:`imbens_regression_2008` to choose the optimal 
bandwidth. Specifically, for observations less than the cutoff (:math:`d_{i} < c`),
we predict the value of the outcome :math:`\hat{Y}(d_{i})` using only observations
within the bandwidth range and to the left of the point (:math:`d_{i} - h < d < d_{i}`).
Similarly, for points greater than the cutoff, (:math:`d_{i} > c`), we calculate 
:math:`\hat{Y}(d_{i})` using :math:`d` such that :math:`d_{i} < d < d_{i} + h`.
When then choose the bandwidth which minimizes the mean squared error of the
predictions:

.. math::

    CV(h) = \frac{1}{N} \sum_{i = 1}^{N} (Y_{i} - \hat{Y}(d_{i}))^{2}

-----

*Polynomial Order Optimization*

-----

When choice for ``order`` is left unspecified, we use one of two optimization
procedures introduced in :cite:t:`lee2010regression` to choose the order of the polynomial.
If ``optimization = 'aic'`` (default), then we minimize the Akaike Information Criterion
objective function,

.. math::

    AIC(p) = N \log(\hat{\sigma}^{2}) + 2p

For :math:`p`, the order of the polynomial and :math:`\hat{\sigma}^{2}`, the estimated mean
squared error of the regression. When ``optimization = 'significance bins'``, we follow
the procedure outlined in :cite:t:`lee2010regression`. Specifically, we add a set of 
bin dummies :math:`\phi_{j}` to the polynomial regression to estimate the model:

.. math::

    Y - X\pi = \alpha_{l} + \theta_{0} A + \beta_{\ell1} (D - c) 
    + (\beta_{r1} − \beta_{\ell1}) A (D − c) + \dots + 
    \beta_{\ell k} (D - c)^{k} +  (\beta_{rk} − \beta_{\ell k}) A (D − c)^{k}
    +  \sum_{j = 2}^{J - 1}\phi_{j} B_{j} +  \epsilon

And use a Wald Test to under the null hypothesis that :math:`\phi_{2}, \dots, \phi_{J - 1} = 0`.
The first polynomial order :math:`p` such that the pvalue of the wald test exceeds ``pval = 0.05`` 
is chosen (so we cannot reject the bin coefficients being zero). 

-----

*Fuzzy Design* ``model.fit(design = "fuzzy")``

-----

When specifying a fuzzy design, we estimate the 2SLS model,

.. math::

    Y = \alpha + \theta_{0} \hat{\tilde{A}} + f(D - c) + \epsilon\\
    \tilde{A} = \gamma + \delta A + f(D - c) + \epsilon

For :math:`Y`, the outcome variable, :math:`c`, the cutoff, :math:`D`, the running
variable, and :math:`A`, the treatment. :math:`f(\cdot)` represents the method
selected for the regression, which is polynomial for ``model = "polynomial"`` and
local linear when ``model = "local linear"``. :cite:t:`imbens_regression_2008`
shows that for 

.. math::

    \theta_{0} = \lim_{\epsilon \to 0^{+}}
    \frac{\mathbb{E}[Y \mid D = c + \epsilon] - \mathbb{E}[Y \mid D = c - \epsilon]}
    {\mathbb{E}[A \mid D = c + \epsilon] - \mathbb{E}[A \mid D = c - \epsilon]}

The estimated treatment effect :math:`\hat{\theta}_{0}` in 2SLS is equivalent to estimating 
:math:`\mathbb{E}[Y \mid D]` and :math:`\mathbb{E}[A \mid D]` using
regressions with the same function :math:`f(\cdot)`.

References
----------
.. bibliography:: refs.bib

Examples
--------

Polynomial regression under a sharp design with polynomial order selection based
on bin-coefficient hypothesis test.

>>> y, d, x, a = generate_sample_data()
>>> c = 0
>>> model = RDD(outcome = y, runv = d, cutoff = c,
>>>             exog = x)
>>> res = model.fit(model = "polynomial", 
>>>       optimization = 'significance bins')
>>> res.summary()
                                          Sharp Polynomial
====================================================================================================
Dep. Variable:                              None    Run. Variable:                              None
Model:                          Sharp Polynomial    No. Observations:                            400
Covariance Type:       Heteroskedasticity robust    Polynomial Order:                              1
====================================================================================================
                                              coef   std err         t     P>|t|    [0.025    0.975]
----------------------------------------------------------------------------------------------------
Treatment                                    1.291     0.270     4.788     0.000     0.761     1.821
====================================================================================================


Local linear regression under a fuzzy design with automatic bandwidth selection. 

>>> y, d, x, a = generate_sample_data()
>>> c = 0
>>> model = RDD(outcome = y, runv = d, cutoff = c, 
>>>             treatment = a, exog = x)
>>> res = model.fit(model = "local linear", 
>>>                 design = "fuzzy")
>>> res.summary()
                                         Fuzzy Local linear
====================================================================================================
Dep. Variable:                              None    Run. Variable:                              None
Model:                        Fuzzy Local linear    No. Observations:                            396
Variance Type:                   Bayes bootstrap    Bandwidth:                                 9.938
====================================================================================================
                                                         coef      std err       [0.025       0.975]
----------------------------------------------------------------------------------------------------
Treatment                                               1.733         None         None         None
====================================================================================================

Polynomial regression under a fuzzy design with AIC optimized polynomial order and 
bootstrap standard errors + confidence intervals.

>>> y, d, x, a = generate_sample_data()
>>> c = 0
>>> model = RDD(outcome = y, runv = d, cutoff = c, 
>>>             treatment = a, exog = x)
>>> res = model.fit(model = "polynomial", design = "fuzzy", 
>>>                 optimization = 'aic', bootstrap = True)
>>> res.summary()
                                          Fuzzy Polynomial
====================================================================================================
Dep. Variable:                              None    Run. Variable:                              None
Model:                          Fuzzy Polynomial    No. Observations:                            400
Variance Type:                   Bayes bootstrap    Polynomial Order:                              2
====================================================================================================
                                                         coef      std err       [0.025       0.975]
----------------------------------------------------------------------------------------------------
Treatment                                               2.011        0.540        1.110        3.076
====================================================================================================
