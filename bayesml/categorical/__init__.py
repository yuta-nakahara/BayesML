# Document Author
# Kohei Horinouchi <horinochi_18@toki.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
r"""
The stochastic data generative model is as follows:

* $d\in \mathbb{Z}_{\ge 2}$: a dimension
* $\bm{x} \in \{ 0, 1\}^d$: a data point, ($\sum_{k=1}^dx_k=1$)
* $\bm{p} \in [0, 1]^d$: a parameter, ($\sum_{k=1}^dp_k=1$)

$$
\text{Categ}(\bm{x}|\bm{p}) = p_1^{x_1}p_2^{x_2}\cdots p_{d-1}^{x_{d-1}}p_d^{x_d}
$$

The prior distribution is as follows:

* $\bm{\alpha}_0 \in \mathbb{R}_{>0}$: a hyperparameter
* $C(\bm{\alpha})=\frac{\Gamma(\hat{\alpha})}{\Gamma(\alpha_1)\cdots\Gamma(\alpha_d)}$
* $\hat{\alpha}=\sum_{k=1}^d\alpha_k$

$$
\text{Dir}(\bm{p}|\bm{\alpha}_0) = C(\bm{\alpha}_0)\prod_{k=1}^dp_k^{\alpha_{0_k}-1}
$$

The posterior distribution is as follows:

* $\bm{x}^n = (\bm{x}_1, \bm{x}_2, \dots , \bm{x}_n) \in \{ 0, 1\}^{d\times n}$: given data
* $\bm{\alpha}_n \in \mathbb{R}_{>0}^d$: a hyperparameter

$$
\text{Dir}(\bm{p}|\bm{\alpha}_n) = C(\bm{\alpha}_n)\prod_{k=1}^dp_k^{\alpha_{n_k}-1}
$$

where the updating rule of the hyperparameters is

$$
\alpha_{n_1} = \alpha_{0_1} + \sum_{i=1}^n x_{i_1}\\
\vdots\\
\alpha_{n_d} = \alpha_{0_d} + \sum_{i=1}^n x_{i_d}
$$

The predictive distribution is as follows:

* $x_n \in \{ 0, 1\}^d$: a new data point
* $\bm{\alpha}_n \in \mathbb{R}_{>0}^d$: the hyperparameter of the posterior

$$
p(\bm{x}|\bm{\alpha}_n) = \begin{cases}
\frac{\alpha_{n_1}}{\sum_{k=1}^d\alpha_{n_k}} & x_{n_1} = 1\\
\quad\vdots\\
\frac{\alpha_{n_d}}{\sum_{k=1}^d\alpha_{n_k}} & x_{n_d} = 1
\end{cases}
$$
"""

from ._categorical import GenModel
from ._categorical import LearnModel

__all__ = ["GenModel", "LearnModel"]
