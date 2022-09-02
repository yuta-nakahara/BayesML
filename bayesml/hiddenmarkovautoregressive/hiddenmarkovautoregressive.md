<!-- Document Author
Koki Kazama <kokikazama@aoni.waseda.jp>
-->
The Hidden Markov model with the Gauss-Wishart prior distribution and the Dirichlet prior distribution.

The stochastic data generative model is as follows:

* $K \in \mathbb{N}$: number of latent classes
* $n \in \mathbb{N}$: time index
* $\boldsymbol{z}_{n} \in \{ 0, 1 \}^K$: a one-hot vector representing the latent class (latent variable)
* $\boldsymbol{\pi}=[\pi_{1},\dots,\pi_{K}]^{\top} \in [0, 1]^K$: a parameter for latent classes, ($\sum_{k=1}^K \pi_k=1$)
* $a_{jk}$ : transition probability to latent state k under latent state j
* $\boldsymbol{A}=(a_{jk})_{0\leq j,k\leq K} \in [0, 1]^{K\times K}$: a parameter for latent classes, ($\sum_{k=1}^K a_{jk}=1$)
* $d \in \mathbb{N}$: the degree of the model
* $\boldsymbol{x}'_n := [1, x_{n-d}, x_{n-d+1}, \dots , x_{n-1}]^\top \in \mathbb{R}^{d+1}$. Here we assume $x_n$ for $n < 1$ is given as a initial value.
* $\boldsymbol{\theta}_{k} \in \mathbb{R}^{d+1}$: a regression coefficient parameter
* $\boldsymbol{\theta}:=\{\boldsymbol{\theta}_{k}\}_{k=1}^{K}$
* $x_n \in \mathbb{R}$: a data point at $n$
* $\tau_{k} \in \mathbb{R}_{>0}$: a precision parameter of noise
* $\boldsymbol{\tau}:=[\tau_{1},\dots,\tau_{K}]^{\top}\in\mathbb{R}^{K}$.


$$
\begin{align}
    p(\boldsymbol{z}_{1} | \boldsymbol{\pi}) &= \mathrm{Cat}(\boldsymbol{z}_{1}|\boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{z_{1,k}},\\
    p(\boldsymbol{z}_{n} |\boldsymbol{z}_{n-1} ,\boldsymbol{A}) &= \prod_{k=1}^K \prod_{j=1}^K a_{jk}^{z_{n-1,j}z_{n,k}},\\
    p(x_n | \boldsymbol{x}'_{n-1}, \boldsymbol{\theta}, \boldsymbol{\tau}) &= \prod_{k=1}^{K}\mathcal{N}(x_n|\boldsymbol{\theta}_{k}^\top \boldsymbol{x}'_{n-1}, \tau_{k}^{-1})^{z_{k}} \\
    &= \prod_{k=1}^{K}\left(\sqrt{\frac{\tau_{k}}{2 \pi}} \exp \left\{ -\frac{\tau_{k}}{2} (x_n - \boldsymbol{\theta}_{k}^\top \boldsymbol{x}'_{n-1})^2 \right\}\right)^{z_{k}}
\end{align}
$$

The prior distribution is as follows:

* $\boldsymbol{\mu}_0 \in \mathbb{R}^{d+1}$: a hyperparameter for $\boldsymbol{\theta}$
* $\boldsymbol{\Lambda}_0 \in \mathbb{R}^{(d+1) \times (d+1)}$: a hyperparameter for $\boldsymbol{\theta}$ (a positive definite matrix)
* $| \boldsymbol{\Lambda}_0 | \in \mathbb{R}$: the determinant of $\boldsymbol{\Lambda}_0$
* $\alpha_0 \in \mathbb{R}_{>0}$: a hyperparameter for $\tau$
* $\beta_0 \in \mathbb{R}_{>0}$: a hyperparameter for $\tau$
* $\Gamma(\cdot): \mathbb{R}_{>0} \to \mathbb{R}$: the Gamma function
* $D \in \mathbb{N}$: a dimension of data
* $\boldsymbol{\mu}_k \in \mathbb{R}^D$: a parameter
* $\boldsymbol{\mu} := \{ \boldsymbol{\mu}_k \}_{k=1}^K$
* $\boldsymbol{\Lambda}_k \in \mathbb{R}^{D\times D}$ : a parameter (a positive definite matrix)
* $| \boldsymbol{\Lambda}_k | \in \mathbb{R}$: the determinant of $\boldsymbol{\Lambda}_k$
* $\boldsymbol{\Lambda} := \{ \boldsymbol{\Lambda}_k \}_{k=1}^K$
* $\boldsymbol{m}_0 \in \mathbb{R}^{D}$: a hyperparameter
* $\kappa_0 \in \mathbb{R}_{>0}$: a hyperparameter
* $\nu_0 \in \mathbb{R}$: a hyperparameter ($\nu_0 > D-1$)
* $\boldsymbol{W}_0 \in \mathbb{R}^{D\times D}$: a hyperparameter (a positive definite matrix)
* $\boldsymbol{\eta}_0=(\eta_{0,1},\dots,\eta_{0,K}) \in \mathbb{R}_{> 0}^K$: a hyperparameter
* $\boldsymbol{\zeta}_{0,j}=(\zeta_{0,j,1},\dots,\zeta_{0,j,K}) \in \mathbb{R}_{> 0}^K$: a hyperparameter for $j=1,\dots,K$
* $\boldsymbol{a}_{j}=\{a_{j,k}\}_{k=1}^{K}$ for $j=1,\dots,K$
* $\mathrm{Tr} \{ \cdot \}$: a trace of a matrix

$$
\begin{align}
    p(\boldsymbol{\theta}, \boldsymbol{\tau}) &=\prod_{k=1}^{K} \mathcal{N}(\boldsymbol{\theta}_{k}|\boldsymbol{\mu}_0, (\tau _{k}\boldsymbol{\Lambda}_0)^{-1}) \mathrm{Gam}(\tau_{k}|\alpha_0,\beta_0)\\
    &=\prod_{k=1}^{K}\frac{|\tau_{k}\boldsymbol{\Lambda}_0|^{1/2}}{(2 \pi)^{(d+1)/2}} 
    \exp \left\{ -\frac{\tau_{k}}{2} (\boldsymbol{\theta}_{k} - \boldsymbol{\mu}_0)^\top 
    \boldsymbol{\Lambda}_0 (\boldsymbol{\theta}_{k} - \boldsymbol{\mu}_0) \right\}
    \frac{\beta_0^{\alpha_0}}{\Gamma (\alpha_0)} \tau_{k}^{\alpha_0 - 1} \exp \{ -\beta_0 \tau_{k} \} .
\end{align}
$$

$$
\begin{align}
    p(\boldsymbol{\mu},\boldsymbol{\Lambda},\boldsymbol{\pi},\boldsymbol{A}) &= \left\{ \prod_{k=1}^K \mathcal{N}(\boldsymbol{\mu}_k|\boldsymbol{m}_0,(\kappa_0 \boldsymbol{\Lambda}_k)^{-1})\mathcal{W}(\boldsymbol{\Lambda}_k|\boldsymbol{W}_0, \nu_0) \right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\eta}_0) \prod_{j=1}^{K}\mathrm{Dir}(\boldsymbol{a}_{j}|\boldsymbol{\zeta}_{0,j}), \\
    &= \Biggl[ \prod_{k=1}^K \left( \frac{\kappa_0}{2\pi} \right)^{D/2} |\boldsymbol{\Lambda}_k|^{1/2} \exp \left\{ -\frac{\kappa_0}{2}(\boldsymbol{\mu}_k -\boldsymbol{m}_0)^\top \boldsymbol{\Lambda}_k (\boldsymbol{\mu}_k - \boldsymbol{m}_0) \right\} \\
    &\qquad \times B(\boldsymbol{W}_0, \nu_0) | \boldsymbol{\Lambda}_k |^{(\nu_0 - D - 1) / 2} \exp \left\{ -\frac{1}{2} \mathrm{Tr} \{ \boldsymbol{W}_0^{-1} \boldsymbol{\Lambda}_k \} \right\}\biggl] \\
    &\qquad \times \Biggl[ \prod_{k=1}^KC(\boldsymbol{\eta}_0)\pi_k^{\eta_{0,k}-1}\biggl]\times \biggl[\prod_{j=1}^KC(\boldsymbol{\zeta}_{0,j})\prod_{k=1}^K a_{jk}^{\zeta_{0,j,k}-1}\Biggr],\\
\end{align}
$$
​
where $B(\boldsymbol{W}_0, \nu_0)$ and $C(\boldsymbol{\eta}_0)$ are defined as follows:
​
$$
\begin{align}
    B(\boldsymbol{W}_0, \nu_0) &= | \boldsymbol{W}_0 |^{-\nu_0 / 2} \left( 2^{\nu_0 D / 2} \pi^{D(D-1)/4} \prod_{i=1}^D \Gamma \left( \frac{\nu_0 + 1 - i}{2} \right) \right)^{-1}, \\
    C(\boldsymbol{\eta}_0) &= \frac{\Gamma(\sum_{k=1}^K \eta_{0,k})}{\Gamma(\eta_{0,1})\cdots\Gamma(\eta_{0,K})},\\
    C(\boldsymbol{\zeta}_{0,j}) &= \frac{\Gamma(\sum_{k=1}^K \zeta_{0,j,k})}{\Gamma(\zeta_{0,j,1})\cdots\Gamma(\zeta_{0,j,K})}. 
\end{align}
$$