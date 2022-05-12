# Document Author
# Keito Tajima <wool812@akane.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
r"""
The multivariate normal distribution with normal-wishart prior distribution.

The stochastic data generative model is as follows:

* $x \in \mathbb{R}^D$: a data point
* $\bm{\mu} \in \mathbb{R}^D$: a parameter
* $\Sigma \in \mathbb{R}^{D\times D}$ : a parameter 

$$ \mathcal{N}(\bm{x}|\bm{\mu},\Sigma) = \frac{1}{(2\pi)^{\frac{D}{2}}|\Sigma|^{\frac{1}{2}}}e^{-\frac{1}{2}(\bm{x}-\bm{\mu})\Sigma^{-1}(\bm{x}-\bm{\mu})^\top}
$$

The prior distribution is as follows:

* $\bm{\mu}_0 \in \mathbb{R}^{D}$: a hyperparameter
* $\kappa_0 \in \mathbb{R}_{>0}$: a hyperparameter
* $\nu_0 \in \mathbb{R}_{>D-1}$: a hyperparameter
* $V_0 \in \mathbb{R}^{D\times D}$: a hyperparameter

$$ f(\bm{\mu},\Lambda|\bm{\mu}_0,\kappa_0,V_0,\nu_0) = \mathcal{N}(\bm{\mu}|\bm{\mu}_0,(\lambda_0\Lambda)^{-1})\mathcal{W}(\Lambda|V_0,\nu_0)
$$

The posterior distribution is as follows:

* $\bm{x}^n = (\bm{x}_1, \bm{x}_2, \dots , \bm{x}_n) \in \mathbb{R}^{D\times n}$: given data
* $\bm{\mu}_n \in \mathbb{R}^{D}$: a hyperparameter
* $\kappa_n \in \mathbb{R}_{>0}$: a hyperparameter
* $\nu_n \in \mathbb{R}_{>D-1}$: a hyperparameter
* $V_n \in \mathbb{R}^{D\times D}$: a hyperparameter

$$ f(\bm{\mu},\Lambda|\bm{\mu}_n,\kappa_n,V_n,\nu_n) = \mathcal{N}(\bm{\mu}|\bm{\mu}_n,(\lambda_n\Lambda)^{-1})\mathcal{W}(\Lambda|V_n,\nu_n)
$$
where the updating rule of the hyperparameters is

$$
    \bm{\mu}_n=\frac{\kappa_0\bm{\mu}+n\bar{\bm{x}}}{\kappa_0+n}\\
    \kappa_n=\kappa_0+n\\
    V_n=\left(V_0^{-1}+C+\frac{\kappa_0 n}{\kappa_0+n}(\bar{\bm{x}}-\bm{\mu}_0)(\bar{\bm{x}}-\bm{\mu}_0)^\top\right)^{-1}\\
    \nu_n=\nu_0+n\\
    C=\sum_{i=1}^{n}(\bm{x}_i-\bar{\bm{x}})(\bm{x}_i-\bar{\bm{x}})^\top
$$
The predictive distribution is as follows:

* $x \in \mathbb{R}$: a new data point
* $\bm{\mu}_p$: the hyperparameter of the posterior
* $\kappa_p$: the hyperparameter of the posterior
* $V_p$: the hyperparameter of the posterior
* $\nu_p$: the hyperparameter of the posterior

$$
    p(x|\bm{\mu}_p,\kappa_p,V_p,\nu_p) = t_{\nu_p-D+1}\left(\bm{\mu}_p,\frac{\kappa_p+1}{\kappa_p(\nu_p-D+1)}V_p^{-1}\right)
$$
"""

from ._multivariatenormal import GenModel
from ._multivariatenormal import LearnModel

__all__ = ["GenModel","LearnModel"]