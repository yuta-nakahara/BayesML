<!-- Document Author
Haruka Murayama <h-murayama@ruri.waseda.jp>
-->

The mixture of linear regression model with the Gauss-Gamma prior distribution and the Dirichlet prior distribution.

The stochastic data generative model is as follows:

* $K \in \mathbb{N}$: number of latent classes
* $\boldsymbol{z} \in \{ 0, 1 \}^K$: a one-hot vector representing the latent class (latent variable)
* $\boldsymbol{\pi} \in [0, 1]^K$: a parameter for latent classes, ($\sum_{k=1}^K \pi_k=1$)
* $D \in \mathbb{N}$: a dimension of data
* $y\in\mathbb{R}$: an objective variable
* $\boldsymbol{x} \in \mathbb{R}^D$: a data point
* $\boldsymbol{\theta}_k\in\mathbb{R}^{D}$: a parameter
* $\boldsymbol{\theta} = \{ \boldsymbol{\theta}_k \}_{k=1}^K$
* $\tau_k \in \mathbb{R}_{>0}$ : a parameter 
* $\boldsymbol{\tau} = \{ \tau_k \}_{k=1}^K$


$$
\begin{align}
    p(\boldsymbol{z} | \boldsymbol{\pi}) &= \mathrm{Cat}(\boldsymbol{z}|\boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{z_k},\\
    p(y | \boldsymbol{x}, \boldsymbol{\theta}, \boldsymbol{\tau}, \boldsymbol{z}) &= \prod_{k=1}^K \mathcal{N}(y | \boldsymbol{\theta}^\top_k \boldsymbol{x},\tau_k^{-1})^{z_k} \\
    &= \prod_{k=1}^K \left( \sqrt{\frac{\tau_k}{2\pi}} \exp \left\{ -\frac{\tau_k}{2}(y - \boldsymbol{\theta}^\top_k\boldsymbol{x})^2 \right\} \right)^{z_k}.
\end{align}
$$

The prior distribution is as follows:

* $\boldsymbol{\mu}_0 \in \mathbb{R}^{D}$: a hyperparameter
* $\boldsymbol{\Lambda}_0 \in \mathbb{R}^{D\times D}$: a hyperparameter (a positive definite matrix)
* $\boldsymbol{\alpha}_0 \in \mathbb{R}_{> 0}^K$: a hyperparameter
* $\beta_0\in \mathbb{R}_{>0}$: a hyperparameter
* $\boldsymbol{\gamma}_0 \in \mathbb{R}_{>0}^K$: a hyper parameter 
* $\Gamma (\cdot)$: the gamma function


$$
\begin{align}
    p(\boldsymbol{\theta},\boldsymbol{\tau},\boldsymbol{\pi}) &= \left\{ \prod_{k=1}^K \mathcal{N}(\boldsymbol{\theta}_k|\boldsymbol{\mu}_0,(\tau_k \boldsymbol{\Lambda}_0)^{-1})\rm{Gam}(\tau_k|\alpha_0, \beta_0) \right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\gamma}_0) \\
    &= \Biggl[ \prod_{k=1}^K  \frac{|\tau_k \Lambda_0|^{1/2}}{(2\pi)^{d/2}} \exp \left\{ -\frac{\tau_k}{2}(\boldsymbol{\theta}_k -\boldsymbol{\mu}_0)^\top \boldsymbol{\Lambda}_0 (\boldsymbol{\theta}_k - \boldsymbol{\mu}_0) \right\} \\
    &\qquad \times \frac{\beta_0^{\alpha_0}}{\Gamma(\alpha_0)}\tau_k^{\alpha_0-1}\exp\{-\beta_0\tau_k\} \Biggl]\\
    &\qquad \times C(\boldsymbol{\gamma}_0)\prod_{k=1}^K \pi_k^{\gamma_{0,k}-1},\\
\end{align}
$$

where $C(\boldsymbol{\gamma}_0)$ are defined as follows:

$$
\begin{align}
    C(\boldsymbol{\gamma}_0) &= \frac{\Gamma(\sum_{k=1}^K \gamma_{0,k})}{\Gamma(\gamma_{0,1})\cdots\Gamma(\gamma_{0,K})}.
\end{align}
$$
