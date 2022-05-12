# Document Author
# Noboru Namegaya <n.noboru20180403@toki.waseda.jp>
# Koshi Shimada <shimada.koshi.re@gmail.com>
r"""
The Normal distribution with the beta prior distribution.

The stochastic data generative model is as follows:

* :math:`x \in \mathbb{R}`: a data point
* :math:`mu \in \mathbb{R}`: a mean parameter
* :math:`sigma \in \mathbb{R}`: a variance parameter

.. math:: \text{Normal}(x|\mu,\sigma) = \frac{1}{\sqrt{2\pi \simga^2}} \exp\left( -\frac{(x-\mu)^2}{2\sigma^2}\right)

The prior distribution is as follows:

* :math:`\mu_0 \in \mathbb{R}`: a hyperparameter
* :math:`\sigma_0 \in \mathbb{R}_{>0}`: a hyperparameter


.. math:: \text{Normal}(\mu|\mu_0,\sigma_0) = \frac{1}{\sqrt{2\pi \simga_0^2}} \exp\left( -\frac{(\mu-\mu_0)^2}{2\sigma_0^2}\right)

The posterior distribution is as follows:

* :math:`x^n = (x_1, x_2, \dots , x_n) \in \mathbb{R}^n`: given data
* :math:`\mu_n \in \mathbb{R}`: a hyperparameter
* :math:`\sigma_n \in \mathbb{R}_{>0}`: a hyperparameter

.. math:: \text{Normal}(\mu_n,\sigma_n|x^n) = \mathcal{N}(\mu_n,\sigma_n)

where the updating rule of the hyperparameters is

.. math::
    \alpha_n = \frac{\sigma_0^2 \sum_{i}^n x_i +\mu_0\sigma^2}{n\sigma_0^2+\sigma^2}\\
    \beta_n =  \frac{\sigma_0^2 \sigma^2}{n\sigma_0^2+\sigma^2}  

The predictive distribution is as follows:

* :math:`x_n \in \mathbb{R}`: a new data point
* :math:`\mu_n \in \mathbb{R}`: the hyperparameter of the posterior
* :math:`\sigma_n \in \mathbb{R}_{>0}`: the hyperparameter of the posterior

.. math::
    p(x|\mu_n, \sigma_n) = \mathcal{N}(\mu_n,\sigma^2+\sigma_n^2)
"""

from ._normal import GenModel
from ._normal import LearnModel

__all__ = ["GenModel","LearnModel"]
