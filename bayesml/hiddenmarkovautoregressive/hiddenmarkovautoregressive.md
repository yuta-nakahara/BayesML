<!-- Document Author
Koki Kazama <kokikazama@aoni.waseda.jp>
-->
The Hidden Markov model with the Gauss-Wishart prior distribution and the Dirichlet prior distribution.

The stochastic data generative model is as follows:

* $K \in \mathbb{N}$: number of latent classes
* $\boldsymbol{z} \in \{ 0, 1 \}^K$: a one-hot vector representing the latent class (latent variable)
* $\boldsymbol{\pi} \in [0, 1]^K$: a parameter for latent classes, ($\sum_{k=1}^K \pi_k=1$)
* $a_{jk}$ : transition probability to latent state k under latent state j
* $\boldsymbol{A}=(a_{jk})_{0\leq j,k\leq K} \in [0, 1]^{K\times K}$: a parameter for latent classes, ($\sum_{k=1}^K a_{jk}=1$)
* $D \in \mathbb{N}$: a dimension of data
* $\boldsymbol{x} \in \mathbb{R}^D$: a data point
* $\boldsymbol{\mu}_k \in \mathbb{R}^D$: a parameter
* $\boldsymbol{\mu} = \{ \boldsymbol{\mu}_k \}_{k=1}^K$
* $\boldsymbol{\Lambda}_k \in \mathbb{R}^{D\times D}$ : a parameter (a positive definite matrix)
* $\boldsymbol{\Lambda} = \{ \boldsymbol{\Lambda}_k \}_{k=1}^K$
* $| \boldsymbol{\Lambda}_k | \in \mathbb{R}$: the determinant of $\boldsymbol{\Lambda}_k$
* $\boldsymbol{x}'_n := [1, x_{n-d}, x_{n-d+1}, \dots , x_{n-1}]^\top \in \mathbb{R}^{d+1}$. Here we assume $x_n$ for $n < 1$ is given as a initial value.
* $\boldsymbol{\theta}_{k} \in \mathbb{R}^{d+1}$: a regression coefficient parameter
* $d \in \mathbb{N}$: the degree of the model
* $n \in \mathbb{N}$: time index
* $x_n \in \mathbb{R}$: a data point at $n$
* $\tau_{k} \in \mathbb{R}_{>0}$: a precision parameter of noise
* $\boldsymbol{\tau}:=(\tau_{1},\dots,\tau_{K})$

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


$$
\begin{align}
    p(\boldsymbol{\theta}, \boldsymbol{\tau}) &=\prod_{k=1}^{K} \mathcal{N}(\boldsymbol{\theta}|\boldsymbol{\mu}_0, (\tau _{k}\boldsymbol{\Lambda}_0)^{-1}) \mathrm{Gam}(\tau_{k}|\alpha_0,\beta_0)\\
    &=\prod_{k=1}^{K}\frac{|\tau_{k}\boldsymbol{\Lambda}_0|^{1/2}}{(2 \pi)^{(d+1)/2}} 
    \exp \left\{ -\frac{\tau_{k}}{2} (\boldsymbol{\theta} - \boldsymbol{\mu}_0)^\top 
    \boldsymbol{\Lambda}_0 (\boldsymbol{\theta} - \boldsymbol{\mu}_0) \right\}
    \frac{\beta_0^{\alpha_0}}{\Gamma (\alpha_0)} \tau_{k}^{\alpha_0 - 1} \exp \{ -\beta_0 \tau_{k} \} .
\end{align}
$$