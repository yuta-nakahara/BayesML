# Code Author
# Yasushi Esaki <esakiful@gmail.com>
r"""
The multinomial mixture model with the dirichlet prior distributions.

The stochastic data generative model is as follows:

* :math:`K \in \mathbb{N}`: number of latent classes
* :math:`\boldsymbol{z} \in \{ 0, 1 \}^K`: a one-hot vector representing the latent class (latent variable)
* :math:`\boldsymbol{\pi} \in [0, 1]^K`: a parameter for latent classes, :math:`(\sum_{k=1}^K \pi_k=1)`
* :math:`d \in \mathbb{Z}`: a dimension :math:`(d \geq 2)`
* :math:`m \in \mathbb{N}`: number of trials
* :math:`\boldsymbol{x} \in \mathbb{Z}_{\geq 0}^d`: a data point, :math:`(\sum_{l=0}^d x_l = m)`
* :math:`\boldsymbol{\theta}_k=(\theta_{k,1},\theta_{k,2},\dots,\theta_{k,d})^\top \in [0, 1]^d`: a parameter, :math:`(\sum_{l=1}^d \theta_{k,l}=1)`
* :math:`\boldsymbol{\theta} = \{ \boldsymbol{\theta}_k \}_{k=1}^K`
* :math:`\Gamma (\cdot)`: the gamma function

.. math::
    p(\boldsymbol{z} | \boldsymbol{\pi}) &= \mathrm{Cat}(\boldsymbol{z}|\boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{z_k},\\
    p(\boldsymbol{x} | \boldsymbol{\theta}, \boldsymbol{z}) &= \prod_{k=1}^K \mathrm{Multi}(\boldsymbol{x}|\boldsymbol{\theta}_k)^{z_k} \\
    &= \prod_{k=1}^K \left(\frac{\Gamma(\sum^d_{l=1}x_l+1)}{\prod^d_{l=1}\Gamma(x_l+1)}\prod^d_{l=1}\theta^{x_{l}}_{k,l}  \right)^{z_k}.


The prior distribution is as follows:


* :math:`\boldsymbol{\beta}_0=(\beta_{0,1},\beta_{0,2},\dots,\beta_{0,d})^\top \in \mathbb{R}^{d}_{>0}`: a hyperparameter
* :math:`\boldsymbol{\alpha}_0=(\alpha_{0,1},\alpha_{0,2},\dots,\alpha_{0,K})^\top \in \mathbb{R}_{> 0}^K`: a hyperparameter

.. math::
    p(\boldsymbol{\theta},\boldsymbol{\pi}) &= \left\{ \prod_{k=1}^K \mathrm{Dir}(\boldsymbol{\theta}_k|\boldsymbol{\beta}_0) \right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\alpha}_0) \\
    &=  \prod^K_{k=1}\left\{C(\boldsymbol{\beta}_0)\prod^d_{l=1}\theta_{k,l}^{\beta_{0,l}-1}\right\}\times C(\boldsymbol{\alpha}_0)\prod_{k=1}^K \pi_k^{\alpha_{0,k}-1},


where :math:`C(\cdot)` is defined as follows:

.. math::
    C(\boldsymbol{a}) &= \frac{\Gamma(\sum_{i=1}^n a_{i})}{\prod^n_{i=1}\Gamma(a_{i})}\ \ (n\in\mathbb{N},\ \boldsymbol{a}\in\mathbb{R}^n_{>0}).


The apporoximate posterior distribution in the :math:`t`-th iteration of a variational bayesian method is as follows:

* :math:`\boldsymbol{x}^n = (\boldsymbol{x}_1, \boldsymbol{x}_2, \dots , \boldsymbol{x}_n) \in \mathbb{Z}_{\geq 0}^{d \times n}`: given data
* :math:`\boldsymbol{z}^n = (\boldsymbol{z}_1, \boldsymbol{z}_2, \dots , \boldsymbol{z}_n) \in \{ 0, 1 \}^{K \times n}`: latent classes of given data
* :math:`\boldsymbol{r}_i^{(t)} = (r_{i,1}^{(t)}, r_{i,2}^{(t)}, \dots , r_{i,K}^{(t)}) \in [0, 1]^K`: a parameter for the :math:`i`-th latent class, :math:`(\sum_{k=1}^K r_{i, k}^{(t)} = 1)`
* :math:`\boldsymbol{\beta}_{n,k}^{(t)}=(\beta^{(t)}_{n,k,1},\beta^{(t)}_{n,k,2},\cdots,\beta^{(t)}_{n,k,d})^\top \in \mathbb{R}_{> 0}^d`: a hyperparameter
* :math:`\boldsymbol{\alpha}_n^{(t)}=(\alpha^{(t)}_{n,1},\alpha^{(t)}_{n,2},\cdots,\alpha^{(t)}_{n,K})^\top \in \mathbb{R}_{> 0}^K`: a hyperparameter
* :math:`\psi (\cdot)`: the digamma function

.. math::
    q(\boldsymbol{z}^n, \boldsymbol{\theta},\boldsymbol{\pi}) &= \left\{ \prod_{i=1}^n \mathrm{Cat} (\boldsymbol{z}_i | \boldsymbol{r}_i^{(t)}) \right\} \left\{  \prod^K_{k=1}\mathrm{Dir}(\boldsymbol{\theta}_k|\boldsymbol{\beta}^{(t)}_{n,k})\right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\alpha}_n^{(t)}) \\
    &= \left\{ \prod_{i=1}^n \prod_{k=1}^K \left(r_{i,k}^{(t)}\right)^{z_{i,k}} \right\}\times \prod^K_{k=1}\left\{C(\boldsymbol{\beta}_{n,k}^{(t)})\prod_{l=1}^d \theta^{\beta_{n,k,l}^{(t)}-1}_{k,l}\right\}\times C(\boldsymbol{\alpha}_n^{(t)})\prod_{k=1}^K \pi_k^{\alpha_{n,k}^{(t)}-1},


where the updating rule of the hyperparameters is as follows:

.. math::
    N_k^{(t)} &= \sum_{i=1}^n r_{i,k}^{(t)},\\
    s^{(t)}_{k,l}&=\sum_{i=1}^nr_{i,k}^{(t)}x_{l,i},\\
    \beta^{(t+1)}_{n,k,l}&=\beta_{0,l}+s^{(t)}_{k,l},\\
    \alpha^{(t+1)}_{n,k}&=\alpha_{0,k}+N^{(t)}_k,\\
    \ln \rho_{i,k}^{(t+1)}&=\psi\left(\alpha^{(t+1)}_{n,k}\right)-\psi\left(\sum^K_{k'=1}\alpha^{(t+1)}_{n,k'}\right)+\sum^d_{l=1}x_{i,l}\left\{\psi\left(\beta^{(t+1)}_{n,k,l}\right)-\psi\left(\sum^d_{l'=1}\beta^{(t+1)}_{n,k,l'}\right)\right\} \nonumber\\
    &\hspace{5cm}+\ln \Gamma\left(\sum^d_{l=1}x_{l,i}+1\right)-\sum^d_{l=1}\ln \Gamma(x_{l,i}+1),\\
    r^{(t+1)}_{i,k}&=\frac{\rho_{i,k}^{(t+1)}}{\sum_{k=1}^K \rho_{i,k}^{(t+1)}}.

"""

from ._multinomialmixture import GenModel

__all__ = ["GenModel"]
