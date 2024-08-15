rddesign.RDD.continuity_test
****************************
.. RDD.continuity_test:

.. currentmodule:: RDD.continuity_test

Performs a joint Wald Test using exogenous covariates to test for discontinuities
in the distribution of the running variable or the no-treatment potential outcome
following :footcite:t:`lee2010regression`.

Parameters
----------
model : str, optional
    Method used to fit the model. Should be one of ``"local linear"`` or
    ``"polynomial"``
residualize : boolean, optional
    Indicator for whether to residualize each placebo outcome (exogenous covariate)
    using the other exogenous covariates before estimation. Default is ``True``.
kwargs : varies, optional
    All other keyword arguments are passed into
    
    .. toctree::
        :maxdepth: 1

        fit

Returns
-------
result : HTestResult
    The optimization result represented as an object. Resembles `statsmodels` result
    object. Important attributes are:

- ``params``, which stores the estimated treatment effect for each placebo outcome
  as a `pd.Series`
- ``bse``, which stores the standard errors for the treatment effect for each placebo
  outcome as a `pd.Series`
- ``waldstat``, which stores the standardized :math:`\chi^{2}` test statistic for the
  treatment effect for each placebo outcome
- ``pvalue``, which stores the joint pvalue under the alternative hypothesis 
  :math:`\theta_{0}^{(1)}, \dots, \theta_{0}^{(p)} \neq 0`
- ``summary()``, which summarizes the results

Notes
-----

For each unique exogenous variable (:math:`x_{p} \in X`), we estimate the equation,

.. math::

    x_{p} = \alpha_{l} + \theta_{0} A + f(D − c) + \epsilon

For treatment assignment :math:`A = 1(D > c)`, running variable :math:`D`, placebo
outcome :math:`x_{p}`, and model choice :math:`f(D - c)`. We then compute the wald 
test statistic for each faux treatment effect,

.. math::

    W_{p} = \frac{\hat{\theta_{0}}^{2}}{\hat{\sigma}^{2}}

Finally, the joint pvalue is calculated by using a :math:`\chi^{2}` test on the sum of
the wald statistics :math:`W_{1} + \dots + W_{p}`.

When ``model = 'local linear'`` 
is chosen, :math:`f(D - c) = \beta_{\ell} (D - c) + (\beta_{r} − \beta_{\ell}) A (D − c)`.
When ``model = 'polynomial'`` is chosen, 
:math:`f(D - c) = \beta_{\ell} (D - c) + (\beta_{r} − \beta_{\ell}) A (D − c) + \dots + \beta_{\ell} (D - c)^{k} + (\beta_{r} − \beta_{\ell}) A (D − c)^{k}`.

Bandwidth, polynomial order, and optimization algorithms, can be controlled by passing
keyword arguments.

References
----------
.. footbibliography::

Examples
--------

Perform continuity hypthesis test using local linear regression with automatic bandwidth
selection for each placebo outcome.

>>> y, d, x, a = generate_sample_data()
>>> c = 0
>>> model = RDD(outcome = y, runv = d, cutoff = c,
>>>             exog = x)
>>> res = model.continuity_test(model = 'local linear')
>>> res.summary()
                                 Local linear Continuity Wald Test
====================================================================================================
Model:                              local linear    Run. Variable:                              None
Covariance Type:       Heteroskedasticity-robust    No. Observations:                            400
Joint pvalue:                              0.307
====================================================================================================
Dep. Variable              Treatment ...        coef     std err    waldstat       order   bandwidth
----------------------------------------------------------------------------------------------------
0                                              0.173       0.117       2.177        None       9.938
1                                             -0.056       0.130       0.184        None       9.938
====================================================================================================

Perform the same test, but fix the bandwidth used in estimation:

>>> res = model.continuity_test(model = 'local linear',
>>>       bandwidth = 5)
>>> res.summary()
                                 Local linear Continuity Wald Test
====================================================================================================
Model:                              local linear    Run. Variable:                              None
Covariance Type:       Heteroskedasticity-robust    No. Observations:                            400
Joint pvalue:                              0.533
====================================================================================================
Dep. Variable              Treatment ...        coef     std err    waldstat       order   bandwidth
----------------------------------------------------------------------------------------------------
0                                              0.155       0.160       0.940        None       5.000
1                                             -0.099       0.174       0.319        None       5.000
====================================================================================================

Perform continuity hypthesis test using polynomial regression, not residualizing the 
placebo outcomes and fixing the polynomial order.

>>> res = model.continuity_test(model = 'polynomial',
>>>       order = 2, residulaize = False)
>>> res.summary()
                                  Polynomial Continuity Wald Test
====================================================================================================
Model:                                polynomial    Run. Variable:                              None
Covariance Type:       Heteroskedasticity-robust    No. Observations:                            400
Joint pvalue:                              0.777
====================================================================================================
Dep. Variable              Treatment ...        coef     std err    waldstat       order   bandwidth
----------------------------------------------------------------------------------------------------
0                                              0.115       0.171       0.449           2        None
1                                              0.044       0.188       0.054           2        None
====================================================================================================
