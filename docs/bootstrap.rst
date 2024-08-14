rddesign.RDD.bootstrap
**********************
.. RDD.bootstrap:

.. currentmodule:: RDD.bootstrap

Performs a Bayesian bootstrap on the treatment effect and returns the bootstrap
replicates.

Parameters
----------
model : str, optional
    Method used to fit the model. Should be one of ``"local linear"`` or
    ``"polynomial"``
design : str, optional
    Design for the RDD model. Should be one of ``"sharp"`` or ``"fuzzy"``.
nreps : int, optional
    Number of bootstrap replications to perform. Default is ``200``.
seed : int, optional
    Seed used when generating bootstrap weights. Default is ``1004``.
kwargs : varies, optional
    For each replication, all other keyword arguments are passed into
    
    .. toctree::
        :maxdepth: 1

        fit

Returns
-------
replicates : (N,) ndarray
    An array containing the estimated treatment effect for each bootstrap replication.

Notes
-----

Following :cite:t:`andrews_bootstrap_2024`, for each bootstrap replication :math:`j \in (1, \dots, J)`,
we generate a set of weights :math:`W_{j} \sim Dirichlet(\frac{1}{n}, \dots, \frac{1}{n})`.
For each set of weights :math:`W_{j}, J \in (1, \dots, J)`, we then estimate the treatment
effect :math:`\theta_{0}^{(j)}` by reweighting the observations. In the case of a fuzzy design, 
which uses the model,

.. math::

    Y = \alpha + \theta_{0} \hat{\tilde{A}} + f(D - c) + \epsilon\\
    \tilde{A} = \gamma + \delta A + f(D - c) + \epsilon

For each bootstrap replicate :math:`\theta_{0}^{(j)}` we minimize the equations,

.. math::

    (\hat{\theta_{0}}^{(j)}, \hat{\alpha}) = \text{arg min} \sum_{i = 1}^{N} w_{j, i} (Y - \alpha - \theta_{0}^{(j)} \hat{\tilde{A}} + f(D - c))\\
    (\hat{\delta}, \hat{\gamma}) = \text{arg min} \sum_{i = 1}^{N} w_{j, i} (A - \gamma - \delta A + f(D - c))

References
----------
.. bibliography:: refs.bib

Examples
--------

Get ``1000`` bootstrap replicates using local linear estimation with a fuzzy design and 
automatic bandwidth selection.

>>> y, d, x, a = generate_sample_data()
>>> c = 0
>>> model = RDD(outcome = y, runv = d, cutoff = c,
>>>             exog = x)
>>> replicates = model.bootstrap(model = 'local linear',
>>>              design = 'fuzzy', nreps = 1000)

Use ``bootstrap = True`` when fitting the model to see bootstrap standard errors.

>>> y, d, x, a = generate_sample_data()
>>> c = 0
>>> model = RDD(outcome = y, runv = d, cutoff = c, 
>>>             treatment = a, exog = x)
>>> res = model.fit(model = "polynomial", design = "fuzzy", 
>>>                 optimization = 'aic', bootstrap = True,
>>>                 nreps = 1000, seed = 2002)
>>> res.summary()
                                          Fuzzy Polynomial
====================================================================================================
Dep. Variable:                              None    Run. Variable:                              None
Model:                          Fuzzy Polynomial    No. Observations:                            400
Variance Type:                   Bayes bootstrap    Polynomial Order:                              2
====================================================================================================
                                                         coef      std err       [0.025       0.975]
----------------------------------------------------------------------------------------------------
Treatment                                               2.011        0.547        1.026        3.101
====================================================================================================
