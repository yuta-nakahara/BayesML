# Document Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
r"""
The linear autoregressive model with the normal-gamma prior distribution.

The stochastic data generative model is as follows:

* :math:`d \in \mathbb{N}`: the degree of the model
* :math:`n \in \mathbb{N}`: time index
* :math:`x_n \in \mathbb{R}`: a data point at :math:`n`
* :math:`\boldsymbol{x}'_n := [1, x_{n-d}, x_{n-d+1}, \dots , x_{n-1}]^\top \in \mathbb{R}^{d+1}`. Here we assume :math:`x_n` for :math:`n < 1` is given as a initial value.
* :math:`\boldsymbol{\theta} \in \mathbb{R}^{d+1}`: a regression coefficient parameter
* :math:`\tau \in \mathbb{R}_{>0}`: a precision parameter of noise

.. math:: 
    \mathcal{N}(x_n|\boldsymbol{\theta}^\top \boldsymbol{x}'_{n-1}, \tau^{-1})
    = \sqrt{\frac{\tau}{2 \pi}} \exp \left\{ -\frac{\tau}{2} (x_n - \boldsymbol{\theta}^\top \boldsymbol{x}'_{n-1})^2 \right\}.

The prior distribution is as follows:

* :math:`\boldsymbol{\mu}_0 \in \mathbb{R}^{d+1}`: a hyperparameter for :math:`\boldsymbol{\theta}`
* :math:`\boldsymbol{\Lambda}_0 \in \mathbb{R}^{(d+1) \times (d+1)}`: a hyperparameter for :math:`\boldsymbol{\theta}` (a positive definite matrix)
* :math:`| \boldsymbol{\Lambda}_0 | \in \mathbb{R}`: the determinant of :math:`\boldsymbol{\Lambda}_0`
* :math:`a_0 \in \mathbb{R}_{>0}`: a hyperparameter for :math:`\tau`
* :math:`b_0 \in \mathbb{R}_{>0}`: a hyperparameter for :math:`\tau`
* :math:`\Gamma(\cdot): \mathbb{R}_{>0} \to \mathbb{R}`: the Gamma function

.. math::
    &\mathcal{N}(\boldsymbol{\theta}|\boldsymbol{\mu}_0, (\tau \boldsymbol{\Lambda}_0)^{-1}) \text{Gam}(\tau|a_0,b_0)\\
    &= \frac{|\tau \boldsymbol{\Lambda}_0|^{1/2}}{(2 \pi)^{(d+1)/2}} 
    \exp \left\{ -\frac{\tau}{2} (\boldsymbol{\theta} - \boldsymbol{\mu}_0)^\top 
    \boldsymbol{\Lambda}_0 (\boldsymbol{\theta} - \boldsymbol{\mu}_0) \right\}
    \frac{b_0^{a_0}}{\Gamma (a_0)} \tau^{a_0 - 1} \exp \{ -b_0 \tau \} .

The posterior distribution is as follows:

* :math:`x^n := [x_1, x_2, \dots , x_n]^\top \in \mathbb{R}^n`: given data
* :math:`\boldsymbol{X}_n = [\boldsymbol{x}'_1, \boldsymbol{x}'_2, \dots , \boldsymbol{x}'_n]^\top \in \mathbb{R}^{n \times (d+1)}`
* :math:`\boldsymbol{\mu}_n \in \mathbb{R}^{d+1}`: a hyperparameter for :math:`\boldsymbol{\theta}`
* :math:`\boldsymbol{\Lambda}_n \in \mathbb{R}^{(d+1) \times (d+1)}`: a hyperparameter for :math:`\boldsymbol{\theta}` (a positive definite matrix)
* :math:`a_n \in \mathbb{R}_{>0}`: a hyperparameter for :math:`\tau`
* :math:`b_n \in \mathbb{R}_{>0}`: a hyperparameter for :math:`\tau`

.. math::
    &\mathcal{N}(\boldsymbol{\theta}|\boldsymbol{\mu}_n, (\tau \boldsymbol{\Lambda}_n)^{-1}) \text{Gam}(\tau|a_n,b_n)\\
    &= \frac{|\boldsymbol{\tau \Lambda}_n|^{1/2}}{(2 \pi)^{(d+1)/2}}
    \exp \left\{ -\frac{\tau}{2} (\boldsymbol{\theta} - \boldsymbol{\mu}_n)^\top 
    \boldsymbol{\Lambda}_n (\boldsymbol{\theta} - \boldsymbol{\mu}_n) \right\}
    \frac{b_n^{a_n}}{\Gamma (a_n)} \tau^{a_n - 1} \exp \{ -b_n \tau \} .

where the updating rules of the hyperparameters are

.. math::
    \boldsymbol{\Lambda}_n &= \boldsymbol{\Lambda}_0 + \boldsymbol{X}_n^\top \boldsymbol{X}_n,\\
    \boldsymbol{\mu}_n &= \boldsymbol{\Lambda}_n^{-1} (\boldsymbol{\Lambda}_0 \boldsymbol{\mu}_0 + \boldsymbol{X}_n^\top x^n),\\
    a_n &= a_0 + \frac{n}{2},\\
    b_n &= b_0 + \frac{1}{2} \left( -\boldsymbol{\mu}_n^\top \boldsymbol{\Lambda}_n \boldsymbol{\mu}_n 
    + (x^n)^\top x^n + \boldsymbol{\mu}_0^\top \boldsymbol{\Lambda}_0 \boldsymbol{\mu}_0 \right).

The predictive distribution is as follows:

* :math:`x_{n+1} \in \mathbb{R}`: a new data point
* :math:`m_\mathrm{p} \in \mathbb{R}`: a parameter
* :math:`\lambda_\mathrm{p} \in \mathbb{R}_{>0}`: a parameter
* :math:`\nu_\mathrm{p} \in \mathbb{R}_{>0}`: a parameter

.. math::
    \text{St}(x_{n+1}|m_\mathrm{p}, \lambda_\mathrm{p}, \nu_\mathrm{p})
    = \frac{\Gamma (\nu_\mathrm{p}/2 + 1/2)}{\Gamma (\nu_\mathrm{p}/2)}
    \left( \frac{m_\mathrm{p}}{\pi \nu_\mathrm{p}} \right)^{1/2}
    \left[ 1 + \frac{\lambda_\mathrm{p}(x_{n+1}-m_\mathrm{p})^2}{\nu_\mathrm{p}} \right]^{-\nu_\mathrm{p}/2 - 1/2}.

where the parameters are obtained from the hyperparameters of the posterior distribution as follows.

.. math::
    m_\mathrm{p} &= \mu_n^\top \boldsymbol{x}'_n,\\
    \lambda_\mathrm{p} &= \frac{a_n}{b_n} (1 + (\boldsymbol{x}'_n)^\top \boldsymbol{\Lambda}_n^{-1} \boldsymbol{x}'_n)^{-1},\\
    \nu_\mathrm{p} &= 2 a_n.
"""

from ._autoregressive import GenModel
from ._autoregressive import LearnModel

__all__ = ["GenModel","LearnModel"]
