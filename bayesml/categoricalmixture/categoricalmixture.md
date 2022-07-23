<!-- Document Author
Yasushi Esaki <esakiful@gmail.com>
-->

The Categorical mixture model with the Dirichlet prior distributions. The stochastic data generative model is as follows:
​

* $K \in \mathbb{N}$: number of latent classes
* $\boldsymbol{z} \in \{ 0, 1 \}^K$: a one-hot vector representing the latent class (latent variable)
* $\boldsymbol{\pi} \in [0, 1]^K$: a parameter for latent classes, ($\sum_{k=1}^K \pi_k=1$)
* $d \in \mathbb{Z}$: a dimension ($d \geq 2$)
* $\boldsymbol{x} \in \{ 0, 1\}^d$: a data point, (a one-hot vector, i.e., $\sum_{l=1}^d x_l=1$)
* $\boldsymbol{\theta}_k=(\theta_{k,1},\theta_{k,2},\cdots,\theta_{k,d})^\mathrm{T} \in [0, 1]^d$: a parameter, ($\sum_{l=1}^d \theta_{k,l}=1$)
* $\boldsymbol{\theta} = \{ \boldsymbol{\theta}_k \}_{k=1}^K$

​
$$
\begin{align}
    p(\boldsymbol{z} | \boldsymbol{\pi}) &= \mathrm{Cat}(\boldsymbol{z}|\boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{z_k},\\
    p(\boldsymbol{x} | \boldsymbol{\theta}, \boldsymbol{z}) &= \prod_{k=1}^K \mathrm{Cat}(\boldsymbol{x}|\boldsymbol{\theta}_k)^{z_k} \\
    &= \prod_{k=1}^K \left(\prod^d_{l=1}\theta^{x_{l}}_{k,l}  \right)^{z_k}.
\end{align}
$$
​
The prior distribution is as follows:
​

* $\boldsymbol{\beta}_0=(\beta_{0,1},\beta_{0,2},\cdots,\beta_{0,d})^{\mathrm{T}} \in \mathbb{R}^{d}_{>0}$: a hyperparameter
* $\boldsymbol{\alpha}_0=(\alpha_{0,1},\alpha_{0,2},\cdots,\alpha_{0,K})^{\mathrm{T}} \in \mathbb{R}_{> 0}^K$: a hyperparameter
* $\Gamma (\cdot)$: the gamma function
​
$$
\begin{align}
    p(\boldsymbol{\theta},\boldsymbol{\pi}) &= \left\{ \prod_{k=1}^K \mathrm{Dir}(\boldsymbol{\theta}_k|\boldsymbol{\beta}_0) \right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\alpha}_0) \\
    &=  \prod^K_{k=1}\left\{C(\boldsymbol{\beta_0})\prod^d_{l=1}\theta_{k,l}^{\beta_{0,l}-1}\right\}\times C(\boldsymbol{\alpha}_0)\prod_{k=1}^K \pi_k^{\alpha_{0,k}-1}\\
    &=C(\boldsymbol{\alpha}_0)C(\boldsymbol{\beta}_0)\prod^K_{k=1} \left(\pi_k^{\alpha_{0,k}-1}\prod^d_{l=1}\theta_{k,l}^{\beta_{0,l}-1}\right),
\end{align}
$$
​
where $C(\cdot)$ are defined as follows:
​
$$
\begin{align}
    C(\boldsymbol{a}) &= \frac{\Gamma(\sum_{i=1}^n a_{i})}{\Gamma(a_{1})\cdots\Gamma(a_{n})}\ \ (n\in\mathbb{N},\ \boldsymbol{a}\in\mathbb{R}^n_{>0}).
\end{align}
$$
