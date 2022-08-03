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

$$
\begin{align}
    p(\boldsymbol{z}_{1} | \boldsymbol{\pi}) &= \mathrm{Cat}(\boldsymbol{z}_{1}|\boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{z_{1,k}},\\
    p(\boldsymbol{z}_{n} |\boldsymbol{z}_{n-1} ,\boldsymbol{A}) &= \prod_{k=1}^K \prod_{j=1}^K a_{jk}^{z_{n-1,j}z_{n,k}},\\
    p(\boldsymbol{x}_{n} | \boldsymbol{\mu}, \boldsymbol{\Lambda}, \boldsymbol{z}_{n}) &= \prod_{k=1}^K \mathcal{N}(\boldsymbol{x}|\boldsymbol{\mu}_k,\boldsymbol{\Lambda}_k^{-1})^{z_{n,k}} \\
    &= \prod_{k=1}^K \left( \frac{| \boldsymbol{\Lambda}_{k} |^{1/2}}{(2\pi)^{D/2}} \exp \left\{ -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu}_{k})^\top \boldsymbol{\Lambda}_{k} (\boldsymbol{x}-\boldsymbol{\mu}_{k}) \right\} \right)^{z_{n,k}},
\end{align}
$$

The prior distribution is as follows:

* $\boldsymbol{m}_0 \in \mathbb{R}^{D}$: a hyperparameter
* $\kappa_0 \in \mathbb{R}_{>0}$: a hyperparameter
* $\nu_0 \in \mathbb{R}$: a hyperparameter ($\nu_0 > D-1$)
* $\boldsymbol{W}_0 \in \mathbb{R}^{D\times D}$: a hyperparameter (a positive definite matrix)
* $\boldsymbol{\eta}_0 \in \mathbb{R}_{> 0}^K$: a hyperparameter
* $\boldsymbol{\zeta}_{0,j} \in \mathbb{R}_{> 0}^K$: a hyperparameter
* $\mathrm{Tr} \{ \cdot \}$: a trace of a matrix
* $\Gamma (\cdot)$: the gamma function

$$
\begin{align}
    p(\boldsymbol{\mu},\boldsymbol{\Lambda},\boldsymbol{\pi},\boldsymbol{A}) &= \left\{ \prod_{k=1}^K \mathcal{N}(\boldsymbol{\mu}_k|\boldsymbol{m}_0,(\kappa_0 \boldsymbol{\Lambda}_k)^{-1})\mathcal{W}(\boldsymbol{\Lambda}_k|\boldsymbol{W}_0, \nu_0) \right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\eta}_0) \prod_{j=1}^{K}\mathrm{Dir}(\boldsymbol{a}_{j}|\boldsymbol{\zeta}_{0,j}), \\
    &= \Biggl[ \prod_{k=1}^K \left( \frac{\kappa_0}{2\pi} \right)^{D/2} |\boldsymbol{\Lambda}_k|^{1/2} \exp \left\{ -\frac{\kappa_0}{2}(\boldsymbol{\mu}_k -\boldsymbol{m}_0)^\top \boldsymbol{\Lambda}_k (\boldsymbol{\mu}_k - \boldsymbol{m}_0) \right\} \\
    &\qquad \times B(\boldsymbol{W}_0, \nu_0) | \boldsymbol{\Lambda}_k |^{(\nu_0 - D - 1) / 2} \exp \left\{ -\frac{1}{2} \mathrm{Tr} \{ \boldsymbol{W}_0^{-1} \boldsymbol{\Lambda}_k \} \right\}\biggl] \\
    &\qquad \times \Biggl[ \prod_{k=1}^KC(\boldsymbol{\eta}_0)\pi_k^{\eta_{0,k}-1}\biggl]\times \biggl[\prod_{j=1}^KC(\boldsymbol{\zeta}_{0,j})\prod_{k=1}^K a_{jk}^{\zeta_{0,j,k}-1}\Biggr],\\
\end{align}
$$

where $B(\boldsymbol{W}_0, \nu_0)$ and $C(\boldsymbol{\eta}_0)$ are defined as follows:

$$
\begin{align}
    B(\boldsymbol{W}_0, \nu_0) &= | \boldsymbol{W}_0 |^{-\nu_0 / 2} \left( 2^{\nu_0 D / 2} \pi^{D(D-1)/4} \prod_{i=1}^D \Gamma \left( \frac{\nu_0 + 1 - i}{2} \right) \right)^{-1}, \\
    C(\boldsymbol{\eta}_0) &= \frac{\Gamma(\sum_{k=1}^K \eta_{0,k})}{\Gamma(\eta_{0,1})\cdots\Gamma(\eta_{0,K})},\\
    C(\boldsymbol{\zeta}_{0,j}) &= \frac{\Gamma(\sum_{k=1}^K \zeta_{0,j,k})}{\Gamma(\zeta_{0,j,1})\cdots\Gamma(\zeta_{0,j,K})}. 
\end{align}
$$
