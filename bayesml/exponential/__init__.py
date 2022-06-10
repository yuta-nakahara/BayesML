# Document Author
# Koki Kazama <kokikazama@aoni.waseda.jp>
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
r"""
The exponential distribution with the gamma prior distribution.

The stochastic data generative model is as follows:

* :math:`x \in \mathbb{R}_{\geq 0}`: a data point
* :math:`\lambda \in \mathbb{R}_{>0}`: a parameter

.. math::
    p(x | \lambda) = \mathrm{Exp}(x|\lambda) = \lambda\exp(-\lambda x).

.. math::
    \mathbb{E}[x] &= \frac{1}{\lambda}, \\
    \mathbb{V}[x] &= \frac{1}{\lambda^2}.


The prior distribution is as follows:

* :math:`\alpha_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\Gamma(\cdot): \mathbb{R}_{>0} \to \mathbb{R}_{>0}`: the Gamma function

.. math::
    p(\lambda) = \mathrm{Gam}(\lambda|\alpha_0,\beta_0) = \frac{\beta_0^{\alpha_0}}{\Gamma (\alpha_0)} \lambda^{\alpha_0 - 1} \exp \{ -\beta_0 \lambda \},

.. math::
    \mathbb{E}[\lambda] &= \frac{\alpha_0}{\beta_0}, \\
    \mathbb{V}[\lambda] &= \frac{\alpha_0}{\beta_0^2}.


The posterior distribution is as follows:

* :math:`x^n = (x_1, x_2, \dots , x_n) \in \mathbb{R}_{\geq 0}^n`: given data
* :math:`\alpha_n \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_n \in \mathbb{R}_{>0}`: a hyperparameter

.. math::
    p(\lambda | x^n) = \mathrm{Gam}(\lambda|\alpha_n,\beta_n) = \frac{\beta_n^{\alpha_n}}{\Gamma (\alpha_n)} \lambda^{\alpha_n - 1} \exp \{ -\beta_n \lambda \},

.. math::
    \mathbb{E}[\lambda | x^n] &= \frac{\alpha_n}{\beta_n}, \\
    \mathbb{V}[\lambda | x^n] &= \frac{\alpha_n}{\beta_n^2},

where the updating rule of the hyperparameters is

.. math::
    \alpha_n &= \alpha_0 + n,\\
    \beta_n &= \beta_0 + \sum_{i=1}^n x_i.


The predictive distribution is as follows:

* :math:`x_{n+1} \in \mathbb{R}_{\geq 0}`: a new data point
* :math:`\alpha_\mathrm{p} \in \mathbb{R}_{>0}`: the hyperparameter of the posterior
* :math:`\eta_\mathrm{p} \in \mathbb{R}_{>0}`: the hyperparameter of the posterior

.. math::
    p(x_{n+1}|x^n)=\mathrm{Lomax}(x_{n+1}|\alpha_\mathrm{p},\eta_\mathrm{p}) = \frac{\alpha_\mathrm{p}}{\eta_\mathrm{p}}\left(1+\frac{x_{n+1}}{\eta_\mathrm{p}}\right)^{-(\alpha_\mathrm{p}+1)},

.. math::
    \mathbb{E}[x_{n+1} | x^n] &=
    \begin{cases}
    \frac{\eta_\mathrm{p}}{\alpha_\mathrm{p}-1}, & \alpha_\mathrm{p}>1,\\
    \mathrm{undefined}, & \text{otherwise},
    \end{cases}\\
    \mathbb{V}[x_{n+1} | x^n] &=
    \begin{cases}
    \frac{\eta_\mathrm{p}^2 \alpha_\mathrm{p}}{(\alpha_\mathrm{p}-1)^2(\alpha_\mathrm{p}-2)}, & \alpha_\mathrm{p}>2,\\
    \infty, & 1<\alpha_\mathrm{p}\leq 2,\\
    \mathrm{undefined}, & \text{otherwise},
    \end{cases}

where the parameters are obtained from the hyperparameters of the posterior distribution as follows:

.. math::
    &\alpha_\mathrm{p} = \alpha_n, \\
    &\eta_\mathrm{p} = \beta_n.
"""
from ._exponential import GenModel
from ._exponential import LearnModel

__all__ = ["GenModel", "LearnModel"]