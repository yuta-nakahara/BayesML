# Document Author
# Taisuke Ishiwatari <taisuke.ishiwatari@fuji.waseda.jp>
r"""
The  Baysian Linear Regression.

The stochastic data generative model is as follows:

* $d \in \mathbb N$: dimension
* $N \in \mathbb N$: sample size
* $\boldsymbol{x_n} \in \mathbb{R}^d$: n-th data point
* $\bold\Phi\in \mathbb{N}^{N\times d}$: design matrix of $x_n$
* $\bold y\in\mathbb{N}^N$: a objective variable
* $\bold w \in\mathbb{N}^d$: a parameter
* $\beta \in\mathbb{N}_{>0}$: a parameter

$$p(y|\bold\Phi,\bold w,\beta) = \mathcal N (\bold y| \bold \Phi \bold w,\beta\bold I_N)$$

The prior distribution is as follows:

* $\bold\mu_0\in \mathbb{R}^d$: a hyperparameter
* $\bold\Lambda_0\in \mathbb{R}^{d\times d}$: a hyperparameter
* $a_0\in \mathbb{R}_{>0}$: a hyperparameter
* $b_0\in \mathbb{R}_{>0}$: a hyperparameter

$$p(\bold w, \beta)= \mathcal N (\bold w|\bold \mu_0,(\beta\bold\Lambda)^{-1})\rm{Gam}(\beta|a_0,b_0)$$

The posterior distribution is as follows:

* $\bold\mu_N\in \mathbb{R}^d$: a hyperparameter
* $\bold\Lambda_N\in \mathbb{R}^{d\times d}$: a hyperparameter
* $a_N\in \mathbb{R}_{>0}$: a hyperparameter
* $b_N\in \mathbb{R}_{>0}$: a hyperparameter

$$
p(\boldsymbol{w}, \beta \mid \boldsymbol{\Phi}, \boldsymbol{y})=\mathcal{N}\left(\boldsymbol{w} \mid \boldsymbol{\mu}_{N},\left(\beta \boldsymbol{\Lambda}_{N}\right)^{-1}\right) \operatorname{Gam}\left(\beta \mid a_{N}, b_{N}\right)
$$
where
* $ \boldsymbol{\Lambda}_{N} =\boldsymbol{\Lambda}_{0}+\boldsymbol{\Phi}^{\top} \boldsymbol{\Phi}$
* $\boldsymbol{\mu}_{N} =\boldsymbol{\Lambda}_{N}^{-1}\left(\boldsymbol{\Phi}^{\top} \boldsymbol{y}+\boldsymbol{\Lambda}_{0} \boldsymbol{\mu}_{0}\right)$
* $a_{N} =a_{0}+\frac{N}{2}$
*  $b_{N} =b_{0}+\frac{1}{2}\left(-\boldsymbol{\mu}_{N}^{\top} \boldsymbol{\Lambda}_{N} \boldsymbol{\mu}_{N}+\boldsymbol{y}^{\top} \boldsymbol{y}+\boldsymbol{\mu}_{0}^{\top} \boldsymbol{\Lambda}_{0} \boldsymbol{\mu}_{0}\right)$

The predictive distribution is as follows:

* $x_{N+1}$\in \mathbb{R}^d:a new data point
* $y_{N+1}$\in \mathbb{R}:a objective variable
* $m\in \mathbb{R}$: a parameter
* $\lambda\in \mathbb{R}$: a parameter
* $\nu\in \mathbb{R}$: a parameter

$$
p\left(y_{N+1} \mid \boldsymbol{x}_{N+1}, \boldsymbol{\Phi}, \boldsymbol{y}\right)=\operatorname{St}\left(y_{N+1} \mid m, \lambda, \nu\right)
$$

where
* $m =\boldsymbol{x}_{N+1}^{\top} \boldsymbol{\mu}_{N} $
* $\lambda =\frac{a_{N}}{b_{N}}\left(1+\boldsymbol{x}_{N+1}^{\top} \boldsymbol{\Lambda}_{N} \boldsymbol{x}_{N+1}\right)^{-1}$
* $\nu =2 a_{N}$
"""

from ._linearregression import GenModel
from ._linearregression import LearnModel

__all__ = ["GenModel","LearnModel"]
