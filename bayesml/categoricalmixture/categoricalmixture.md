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
* $\boldsymbol{\theta}_k=(\theta_{k,1},\theta_{k,2},\dots,\theta_{k,d})^\top \in [0, 1]^d$: a parameter, ($\sum_{l=1}^d \theta_{k,l}=1$)
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

* $\boldsymbol{\beta}_0=(\beta_{0,1},\beta_{0,2},\dots,\beta_{0,d})^\top \in \mathbb{R}^{d}_{>0}$: a hyperparameter
* $\boldsymbol{\alpha}_0=(\alpha_{0,1},\alpha_{0,2},\dots,\alpha_{0,K})^\top \in \mathbb{R}_{> 0}^K$: a hyperparameter
* $\Gamma (\cdot)$: the gamma function
​
$$
\begin{align}
    p(\boldsymbol{\theta},\boldsymbol{\pi}) &= \left\{ \prod_{k=1}^K \mathrm{Dir}(\boldsymbol{\theta}_k|\boldsymbol{\beta}_0) \right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\alpha}_0) \\
    &=  \prod^K_{k=1}\left\{C(\boldsymbol{\beta}_0)\prod^d_{l=1}\theta_{k,l}^{\beta_{0,l}-1}\right\}\times C(\boldsymbol{\alpha}_0)\prod_{k=1}^K \pi_k^{\alpha_{0,k}-1},
\end{align}
$$
​
where $C(\cdot)$ is defined as follows:
​
$$
\begin{align}
    C(\boldsymbol{a}) &= \frac{\Gamma(\sum_{i=1}^n a_{i})}{\Gamma(a_{1})\cdots\Gamma(a_{n})}\ \ (n\in\mathbb{N},\ \boldsymbol{a}\in\mathbb{R}^n_{>0}).
\end{align}
$$

The apporoximate posterior distribution in the $t$-th iteration of a variational Bayesian method is as follows:

* $\boldsymbol{x}^n = (\boldsymbol{x}_1, \boldsymbol{x}_2, \dots , \boldsymbol{x}_n) \in \{ 0, 1\}^{d \times n}$: given data
* $\boldsymbol{z}^n = (\boldsymbol{z}_1, \boldsymbol{z}_2, \dots , \boldsymbol{z}_n) \in \{ 0, 1 \}^{K \times n}$: latent classes of given data
* $\boldsymbol{r}_i^{(t)} = (r_{i,1}^{(t)}, r_{i,2}^{(t)}, \dots , r_{i,K}^{(t)}) \in [0, 1]^K$: a parameter for the $i$-th latent class, ($\sum_{k=1}^K r_{i, k}^{(t)} = 1$)
* $\boldsymbol{\beta}_{n,k}^{(t)}=(\beta^{(t)}_{n,k,1},\beta^{(t)}_{n,k,2},\cdots,\beta^{(t)}_{n,k,d})^\top \in \mathbb{R}_{> 0}^d$: a hyperparameter
* $\boldsymbol{\alpha}_n^{(t)}=(\alpha^{(t)}_{n,1},\alpha^{(t)}_{n,2},\cdots,\alpha^{(t)}_{n,K})^\top \in \mathbb{R}_{> 0}^K$: a hyperparameter
* $\psi (\cdot)$: the digamma function

$$
\begin{align}
    q(\boldsymbol{z}^n, \boldsymbol{\theta},\boldsymbol{\pi}) &= \left\{ \prod_{i=1}^n \mathrm{Cat} (\boldsymbol{z}_i | \boldsymbol{r}_i^{(t)}) \right\} \left\{  \prod^K_{k=1}\mathrm{Dir}(\boldsymbol{\theta}_k|\boldsymbol{\beta}^{(t)}_{n,k})\right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\alpha}_n^{(t)}) \\
    &= \left\{ \prod_{i=1}^n \prod_{k=1}^K \left(r_{i,k}^{(t)}\right)^{z_{i,k}} \right\}\times \prod^K_{k=1}\left\{C(\boldsymbol{\beta}_{n,k}^{(t)})\prod_{l=1}^d \theta^{\beta_{n,k,l}^{(t)}-1}_{k,l}\right\}\times C(\boldsymbol{\alpha}_n^{(t)})\prod_{k=1}^K \pi_k^{\alpha_{n,k}^{(t)}-1},
\end{align}
$$

where the updating rule of the hyperparameters is as follows:

$$
\begin{align}
    N_k^{(t)} &= \sum_{i=1}^n r_{i,k}^{(t)},\\
    s^{(t)}_{k,l}&=\sum_{i=1}^nr_{i,k}^{(t)}x_{l,i},\\
    \beta^{(t+1)}_{n,k,l}&=\beta_{0,l}+s^{(t)}_{k,l},\\
    \alpha^{(t+1)}_{n,k}&=\alpha_{0,k}+N^{(t)}_k,\\
    \ln \rho_{i,k}^{(t+1)}&=\psi\left(\alpha^{(t+1)}_{n,k}\right)-\psi\left(\sum^K_{k'=1}\alpha^{(t+1)}_{n,k'}\right)+\sum^d_{l=1}x_{i,l}\psi\left(\beta^{(t+1)}_{n,k,l}\right)-\psi\left(\sum^d_{l=1}\beta^{(t+1)}_{n,k,l}\right),\\
    r^{(t+1)}_{i,k}&=\frac{\rho_{i,k}^{(t+1)}}{\sum_{k=1}^K \rho_{i,k}^{(t+1)}}.
\end{align}
$$

The approximate predictive distribution is as follows:

* $\boldsymbol{x}_{n+1} \in \{ 0, 1\}^d$: a new data point
* $\boldsymbol{\theta}_{\mathrm{p},k}=(\theta_{\mathrm{p},k,1},\theta_{\mathrm{p},k,2},\cdots,\theta_{\mathrm{p},k,d})^\top \in [0, 1]^d$: the parameter of the predictive distribution ($\sum_{l=1}^d \theta_{\mathrm{p}, k,l}=1$)

$$
\begin{align}
    p(\boldsymbol{x}_{n+1}|\boldsymbol{x}^n) &= \frac{1}{\sum_{k=1}^K \alpha_{n,k}^{(t)}} \sum_{k=1}^K \alpha_{n,k}^{(t)} \mathrm{Cat}(\boldsymbol{x}_{n+1}|\boldsymbol{\theta}_{\mathrm{p},k})\\
    &=\frac{1}{\sum_{k=1}^K \alpha_{n,k}^{(t)}} \sum_{k=1}^K \alpha_{n,k}^{(t)}\prod^d_{l=1}\theta^{x_{n+1,l}}_{\mathrm{p},k,l},
\end{align}
$$

where the parameters are obtained from the hyperparameters of the posterior distribution as follows:

$$
    \theta_{\mathrm{p},k,l}=\frac{\beta^{(t)}_{n,k,l}}{\sum^d_{l=1} \beta^{(t)}_{n,k,l}}.
$$