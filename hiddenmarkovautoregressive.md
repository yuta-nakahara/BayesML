<!-- Document Author
Koki Kazama <kokikazama@aoni.waseda.jp>
-->
The Hidden Markov model with the Gauss-Wishart prior distribution and the Dirichlet prior distribution.

The stochastic data generative model is as follows:

* $n \in \mathbb N$: a sample size
* $\boldsymbol{X} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \dots , \boldsymbol{x}_n]^\top \in \mathbb{R}^{n \times d}$
* $\boldsymbol{y} = [y_1, y_2, \dots , y_n]^\top \in \mathbb{R}^n$
* $\boldsymbol{\mu}_n\in \mathbb{R}^d$: a hyperparameter
* $\boldsymbol{\Lambda_n} \in \mathbb{R}^{d\times d}$: a hyperparameter (a positive definite matrix)
* $\alpha_n\in \mathbb{R}_{>0}$: a hyperparameter
* $\beta_n\in \mathbb{R}_{>0}$: a hyperparameter

$$
\begin{align}
   &p(\boldsymbol{\theta}, \tau | \boldsymbol{X}, \boldsymbol{y})
   \\
   &= \frac{|\tau \boldsymbol{\Lambda}_n|^{1/2}}{(2 \pi)^{d/2}} \exp \left\{ -\frac{\tau}{2} (\boldsymbol{\theta} - \boldsymbol{\mu}_n)^\top \boldsymbol{\Lambda}_n (\boldsymbol{\theta} - \boldsymbol{\mu}_n) \right\} \frac{\beta_n^{\alpha_n}}{\Gamma (\alpha_n)} \tau^{\alpha_n - 1} \exp \{ -\beta_n \tau \} 
\end{align}
$$

The prior distribution is as follows:

$$
\begin{align}
    p(\boldsymbol{\theta}, \tau | x^n) &= \mathcal{N}(\boldsymbol{\theta}|\boldsymbol{\mu}_n, (\tau \boldsymbol{\Lambda}_n)^{-1}) \mathrm{Gam}(\tau|\alpha_n,\beta_n)\\
    &= \frac{|\boldsymbol{\tau \Lambda}_n|^{1/2}}{(2 \pi)^{(d+1)/2}}
    \exp \left\{ -\frac{\tau}{2} (\boldsymbol{\theta} - \boldsymbol{\mu}_n)^\top 
    \boldsymbol{\Lambda}_n (\boldsymbol{\theta} - \boldsymbol{\mu}_n) \right\}
    \frac{\beta_n^{\alpha_n}}{\Gamma (\alpha_n)} \tau^{\alpha_n - 1} \exp \{ -\beta_n \tau \} .
\end{align}
$$

$$
\begin{align}
    \mathbb{E}[\boldsymbol{\theta} | x^n] &= \boldsymbol{\mu}_n & \left( \alpha_n > \frac{1}{2} \right), \\
    \mathrm{Cov}[\boldsymbol{\theta} | x^n] &= \frac{\beta_n}{\alpha_n - 1} \boldsymbol{\Lambda}_n^{-1} & (\alpha_n > 1), \\
    \mathbb{E}[\tau | x^n] &= \frac{\alpha_n}{\beta_n}, \\
    \mathbb{V}[\tau | x^n] &= \frac{\alpha_n}{\beta_n^2},
\end{align}
$$