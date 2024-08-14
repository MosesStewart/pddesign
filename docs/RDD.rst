rddesign.RDD
************
.. RDD:

.. currentmodule:: RDD

Regression discontinuity design model.

A regression discontinuity design model is a way of estimating a treatment
effect in an observational setting where the treatment is influenced by 
whether an observed running variable exceeds a known cutoff point.

This class implements the nonlinear estimators, cross-validation methods,
and hypothesis tests introduced in :cite:t:`lee2010regression` and :cite:t:`imbens_regression_2008`.

Parameters
----------
outcome : (N,) ndarray or (N,) dataframe
    The endogenous response variable, or dependent variable in the model.
runv : (N,) ndarray or (N,) dataframe
    The running variable. Also referred to as the forcing variable.
    Influences the treatment at the cutoff.
cutoff : float, optional
    The cutoff point where we are testing for a discontinuity. 
exog : (N,P) ndarray or (N,P) dataframe, optional
    A matrix of exogenous variables used to residualize the outcome variable
    before estimation. Necessary for continuity hypothesis testing.
treatment : (N,) ndarray or (N,) dataframe, optional
    An array of indicator random variables for whether or not each observation
    received the treatment. Necessary to implement a fuzzy design.
weights : (N,) ndarray or (N,) dataframe, optional
    An array of analytic weights given to each observation.

Methods
-------

.. toctree::
   :maxdepth: 1
   :caption: RDD Methods

   fit
   continuity_test

Notes
-----
Exogenous variables `X` passed into the ``RDD`` class are not included in the regression
when estimating the model. Instead, following :cite:t:`lee2010regression`, they are used to 
residualize the outcome variable `Y`, or subtract from `Y` a prediction of `Y` based on the
exogenous covariates ``exog`` (`X`). 

This procedure removes the portion of the variation in `Y`` we could have predicted using
the exogenous covariates `X`. It also weakens the continuity assumption imposed on 
:math:`\mathbb{E}[Y(0) \mid D]` to assume continuity only on :math:`\mathbb{E}[Y(0) \mid D, X]`.

References
----------
.. bibliography:: refs.bib

Examples
--------
Generating random data and building the model.

>>> y, d, x, a = generate_sample_data()
>>> model = RDD(outcome = y, runv = d, cutoff = c, 
>>>             treatment = a, exog = x)
>>> res = model.fit()
>>> hres = model.continuity_test()

