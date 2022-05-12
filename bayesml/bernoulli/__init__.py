# Document Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
r"""
The Bernoulli distribution with the beta prior distribution.

The stochastic data generative model is as follows:

* :math:`x \in \{ 0, 1\}`: a data point
* :math:`p \in [0, 1]`: a parameter 

.. math:: \text{Bern}(x|p) = p^x (1-p)^{1-x}.

The prior distribution is as follows:

* :math:`\alpha_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`B(\cdot,\cdot): \mathbb{R}_{>0} \times \mathbb{R}_{>0} \to \mathbb{R}_{>0}`: the Beta function

.. math:: \text{Beta}(p|\alpha_0,\beta_0) = \frac{1}{B(\alpha_0, \beta_0)} p^{\alpha_0} (1-p)^{\beta_0}.

The posterior distribution is as follows:

* :math:`x^n = (x_1, x_2, \dots , x_n) \in \{ 0, 1\}^n`: given data
* :math:`\alpha_n \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_n \in \mathbb{R}_{>0}`: a hyperparameter

.. math:: \text{Beta}(p|\alpha_n,\beta_n) = \frac{1}{B(\alpha_n, \beta_n)} p^{\alpha_n} (1-p)^{\beta_n},

where the updating rule of the hyperparameters is

.. math::
    \alpha_n = \alpha_0 + \sum_{i=1}^n I \{ x_i = 1 \},\\
    \beta_n = \beta_0 + \sum_{i=1}^n I \{ x_i = 0 \}.    

The predictive distribution is as follows:

* :math:`x \in \{ 0, 1\}`: a new data point
* :math:`\alpha_\mathrm{p} \in \mathbb{R}_{>0}`: a parameter
* :math:`\beta_\mathrm{p} \in \mathbb{R}_{>0}`: a parameter

.. math::
    p(x|\alpha_\mathrm{p}, \beta_\mathrm{p}) = \begin{cases}
    \frac{\alpha_\mathrm{p}}{\alpha_\mathrm{p} + \beta_\mathrm{p}} & x = 1,\\
    \frac{\beta_\mathrm{p}}{\alpha_\mathrm{p} + \beta_\mathrm{p}} & x = 0,
    \end{cases}

where the parameters are abtained from the hyperparameters of the posterior distribution as follows.

.. math::
    \alpha_\mathrm{p} &= \alpha_n,\\
    \beta_\mathrm{p} &= \beta_n
"""

from ._bernoulli import GenModel
from ._bernoulli import LearnModel

__all__ = ["GenModel","LearnModel"]
