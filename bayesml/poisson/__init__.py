# Document Author
# Ryota Maniwa <r_maniwa115@fuji.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
r"""
The Poisson distribution with the gamma prior distribution.

The stochastic data generative model is as follows:

* :math:`x \in \mathbb{N}`: a data point
* :math:`\lambda \in \mathbb{N}`: a parameter 

.. math:: \text{Po}(x|\lambda) = \lambda^{x}e^{-\lambda}/{x!}

The prior distribution is as follows:

* :math:`\alpha_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\Gamma(\cdot): \mathbb{R}_{>0} \to \mathbb{R}_{>0}`: the Gamma function

.. math:: \text{Gamma}(x|\alpha_0,\beta_0) = \frac{\beta_0^{\alpha_0}}{\Gamma(\alpha_0)}x^{\alpha_0-1} e^{-\beta_0 x}

The posterior distribution is as follows:

* :math:`x^n = (x_1, x_2, \dots , x_n) \in \mathbb{N}^n`: given data
* :math:`\alpha_n \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_n \in \mathbb{R}_{>0}`: a hyperparameter

.. math:: \text{Gamma}(\lambda|x) = \frac{\beta_n^{\alpha_n}}{\Gamma(\alpha_n)}\lambda^{\alpha_n-1} e^{-\beta_n \lambda},

where the updating rule of the hyperparameters is

.. math::
    \alpha_n = \alpha_0 + \sum_{i=1}^n x_i\\
    \beta_n = \beta_0 + n    

The predictive distribution is as follows:

* :math:`x_n \in \mathbb{N}`: a new data point
* :math:`\alpha_n \in \mathbb{R}_{>0}`: the hyperparameter of the posterior
* :math:`\beta_n \in \mathbb{R}_{>0}`: the hyperparameter of the posterior

.. math::
    p(x|\alpha_n, \beta_n) = \frac{\Gamma(x_n+\alpha_n)}{\Gamma(x_n+1)\Gamma(\alpha_n)}p^{x_n}(1-p)^{\alpha_n},
    
where :math:`p = \frac{1}{\beta_n+1}`
"""

from ._poisson import GenModel
from ._poisson import LearnModel

__all__ = ["GenModel","LearnModel"]
