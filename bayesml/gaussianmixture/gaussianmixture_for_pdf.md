<!-- Document Author
Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
-->

The Gaussian mixture model with the Gauss-Wishart prior distribution and the Dirichlet prior distribution.

The stochastic data generative model is as follows:

* $K \in \mathbb{N}$: number of latent classes
* $\boldsymbol{z} \in \\{ 0, 1 \\}^K$: a one-hot vector representing the latent class (latent variable)
* $\boldsymbol{\pi} \in [0, 1]^K$: a parameter for latent classes, ($\sum\_{k=1}^K \pi\_k=1$)
* $D \in \mathbb{N}$: a dimension of data
* $\boldsymbol{x} \in \mathbb{R}^D$: a data point
* $\boldsymbol{\mu}\_k \in \mathbb{R}^D$: a parameter
* $\boldsymbol{\mu} = \\{ \boldsymbol{\mu}\_k \\}\_{k=1}^K$
* $\boldsymbol{\Lambda}\_k \in \mathbb{R}^{D\times D}$ : a parameter (a positive definite matrix)
* $\boldsymbol{\Lambda} = \\{ \boldsymbol{\Lambda}\_k \\}\_{k=1}^K$
* $| \boldsymbol{\Lambda}\_k | \in \mathbb{R}$: the determinant of $\boldsymbol{\Lambda}\_k$

$$
\begin{align}
    p(\boldsymbol{z} | \boldsymbol{\pi}) &= \mathrm{Cat}(\boldsymbol{z}|\boldsymbol{\pi}) = \prod\_{k=1}^K \pi\_k^{z\_k},\cr
    p(\boldsymbol{x} | \boldsymbol{\mu}, \boldsymbol{\Lambda}, \boldsymbol{z}) &= \prod\_{k=1}^K \mathcal{N}(\boldsymbol{x}|\boldsymbol{\mu}\_k,\boldsymbol{\Lambda}\_k^{-1})^{z\_k} \cr
    &= \prod\_{k=1}^K \left( \frac{| \boldsymbol{\Lambda} |^{1/2}}{(2\pi)^{D/2}} \exp \left\\{ -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^\top \boldsymbol{\Lambda} (\boldsymbol{x}-\boldsymbol{\mu}) \right\\} \right)^{z\_k},
\end{align}
$$

The prior distribution is as follows:

* $\boldsymbol{m}\_0 \in \mathbb{R}^{D}$: a hyperparameter
* $\kappa\_0 \in \mathbb{R}\_{>0}$: a hyperparameter
* $\nu\_0 \in \mathbb{R}$: a hyperparameter ($\nu\_0 > D-1$)
* $\boldsymbol{W}\_0 \in \mathbb{R}^{D\times D}$: a hyperparameter (a positive definite matrix)
* $\boldsymbol{\alpha}\_0 \in \mathbb{R}\_{> 0}^K$: a hyperparameter
* $\mathrm{Tr} \\{ \cdot \\}$: a trace of a matrix
* $\Gamma (\cdot)$: the gamma function

$$
\begin{align}
    p(\boldsymbol{\mu},\boldsymbol{\Lambda},\boldsymbol{\pi}) &= \left\\{ \prod\_{k=1}^K \mathcal{N}(\boldsymbol{\mu}\_k|\boldsymbol{m}\_0,(\kappa\_0 \boldsymbol{\Lambda}\_k)^{-1})\mathcal{W}(\boldsymbol{\Lambda}\_k|\boldsymbol{W}\_0, \nu\_0) \right\\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\alpha}\_0) \cr
    &= \Biggl[ \prod\_{k=1}^K \left( \frac{\kappa\_0}{2\pi} \right)^{D/2} |\boldsymbol{\Lambda}\_k|^{1/2} \exp \left\\{ -\frac{\kappa\_0}{2}(\boldsymbol{\mu}\_k -\boldsymbol{m}\_0)^\top \boldsymbol{\Lambda}\_k (\boldsymbol{\mu}\_k - \boldsymbol{m}\_0) \right\\} \cr
    &\qquad \times B(\boldsymbol{W}\_0, \nu\_0) | \boldsymbol{\Lambda}\_k |^{(\nu\_0 - D - 1) / 2} \exp \left\\{ -\frac{1}{2} \mathrm{Tr} \\{ \boldsymbol{W}\_0^{-1} \boldsymbol{\Lambda}\_k \\} \right\\} \Biggr] \cr
    &\qquad \times C(\boldsymbol{\alpha}\_0)\prod\_{k=1}^K \pi\_k^{\alpha\_{0,k}-1},\cr
\end{align}
$$

where $B(\boldsymbol{W}\_0, \nu\_0)$ and $C(\boldsymbol{\alpha}\_0)$ are defined as follows:

$$
\begin{align}
    B(\boldsymbol{W}\_0, \nu\_0) &= | \boldsymbol{W}\_0 |^{-\nu\_0 / 2} \left( 2^{\nu\_0 D / 2} \pi^{D(D-1)/4} \prod\_{i=1}^D \Gamma \left( \frac{\nu\_0 + 1 - i}{2} \right) \right)^{-1}, \cr
    C(\boldsymbol{\alpha}\_0) &= \frac{\Gamma(\sum\_{k=1}^K \alpha\_{0,k})}{\Gamma(\alpha\_{0,1})\cdots\Gamma(\alpha\_{0,K})}.
\end{align}
$$

The apporoximate posterior distribution in the $t$-th iteration of a variational Bayesian method is as follows:

* $\boldsymbol{x}^n = (\boldsymbol{x}\_1, \boldsymbol{x}\_2, \dots , \boldsymbol{x}\_n) \in \mathbb{R}^{D \times n}$: given data
* $\boldsymbol{z}^n = (\boldsymbol{z}\_1, \boldsymbol{z}\_2, \dots , \boldsymbol{z}\_n) \in \\{ 0, 1 \\}^{K \times n}$: latent classes of given data
* $\boldsymbol{r}\_i^{(t)} = (r\_{i,1}^{(t)}, r\_{i,2}^{(t)}, \dots , r\_{i,K}^{(t)}) \in [0, 1]^K$: a parameter for the $i$-th latent class. ($\sum\_{k=1}^K r\_{i, k}^{(t)} = 1$)
* $\boldsymbol{m}\_{n,k}^{(t)} \in \mathbb{R}^{D}$: a hyperparameter
* $\kappa\_{n,k}^{(t)} \in \mathbb{R}\_{>0}$: a hyperparameter
* $\nu\_{n,k}^{(t)} \in \mathbb{R}$: a hyperparameter $(\nu\_n > D-1)$
* $\boldsymbol{W}\_{n,k}^{(t)} \in \mathbb{R}^{D\times D}$: a hyperparameter (a positive definite matrix)
* $\boldsymbol{\alpha}\_n^{(t)} \in \mathbb{R}\_{> 0}^K$: a hyperparameter

$$
\begin{align}
    q(\boldsymbol{z}^n, \boldsymbol{\mu},\boldsymbol{\Lambda},\boldsymbol{\pi}) &= \left\\{ \prod\_{i=1}^n \mathrm{Cat} (\boldsymbol{z}\_i | \boldsymbol{r}\_i^{(t)}) \right\\} \left\\{ \prod\_{k=1}^K \mathcal{N}(\boldsymbol{\mu}\_k|\boldsymbol{m}\_{n,k}^{(t)},(\kappa\_{n,k}^{(t)} \boldsymbol{\Lambda}\_k)^{-1})\mathcal{W}(\boldsymbol{\Lambda}\_k|\boldsymbol{W}\_{n,k}^{(t)}, \nu\_{n,k}^{(t)}) \right\\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\alpha}\_n^{(t)}) \cr
    &= \Biggl[ \prod\_{i=1}^n \prod\_{k=1}^K (r\_{i,k}^{(t)})^{z\_{i,k}} \Biggr] \Biggl[ \prod\_{k=1}^K \left( \frac{\kappa\_{n,k}^{(t)}}{2\pi} \right)^{D/2} |\boldsymbol{\Lambda}\_k|^{1/2} \exp \left\\{ -\frac{\kappa\_{n,k}^{(t)}}{2}(\boldsymbol{\mu}\_k -\boldsymbol{m}\_{n,k}^{(t)})^\top \boldsymbol{\Lambda}\_k (\boldsymbol{\mu}\_k - \boldsymbol{m}\_{n,k}^{(t)}) \right\\} \cr
    &\qquad \times B(\boldsymbol{W}\_{n,k}^{(t)}, \nu\_{n,k}^{(t)}) | \boldsymbol{\Lambda}\_k |^{(\nu\_{n,k}^{(t)} - D - 1) / 2} \exp \left\\{ -\frac{1}{2} \mathrm{Tr} \\{ ( \boldsymbol{W}\_{n,k}^{(t)} )^{-1} \boldsymbol{\Lambda}\_k \\} \right\\} \Biggr] \cr
    &\qquad \times C(\boldsymbol{\alpha}\_n^{(t)})\prod\_{k=1}^K \pi\_k^{\alpha\_{n,k}^{(t)}-1},\cr
\end{align}
$$

where the updating rule of the hyperparameters is as follows.

$$
\begin{align}
    N\_k^{(t)} &= \sum\_{i=1}^n r\_{i,k}^{(t)} \cr
    \bar{\boldsymbol{x}}\_k^{(t)} &= \frac{1}{N\_k^{(t)}} \sum\_{i=1}^n r\_{i,k}^{(t)} \boldsymbol{x}\_i \cr
    \boldsymbol{m}\_{n,k}^{(t+1)} &= \frac{\kappa\_0\boldsymbol{\mu}\_0 + N\_k^{(t)} \bar{\boldsymbol{x}}\_k^{(t)}}{\kappa\_0 + N\_k^{(t)}}, \cr
    \kappa\_{n,k}^{(t+1)} &= \kappa\_0 + N\_k^{(t)}, \cr
    (\boldsymbol{W}\_{n,k}^{(t+1)})^{-1} &= \boldsymbol{W}\_0^{-1} + \sum\_{i=1}^{n} r\_{i,k}^{(t)} (\boldsymbol{x}\_i-\bar{\boldsymbol{x}}\_k^{(t)})(\boldsymbol{x}\_i-\bar{\boldsymbol{x}}\_k^{(t)})^\top + \frac{\kappa\_0 N\_k^{(t)}}{\kappa\_0 + N\_k^{(t)}}(\bar{\boldsymbol{x}}\_k^{(t)}-\boldsymbol{\mu}\_0)(\bar{\boldsymbol{x}}\_k^{(t)}-\boldsymbol{\mu}\_0)^\top, \cr
    \nu\_{n,k}^{(t+1)} &= \nu\_0 + N\_k^{(t)},\cr
    \alpha\_{n,k}^{(t+1)} &= \alpha\_{0,k} + N\_k^{(t)} \cr
    \ln \rho\_{i,k}^{(t+1)} &= \psi (\alpha\_{n,k}^{(t+1)}) - \psi ( {\textstyle \sum\_{k=1}^K \alpha\_{n,k}^{(t+1)}} ) \nonumber \cr
    &\qquad + \frac{1}{2} \Biggl[ \sum\_{d=1}^D \psi \left( \frac{\nu\_{n,k}^{(t+1)} + 1 - d}{2} \right) + D \ln 2 + \ln | \boldsymbol{W}\_{n,k}^{(t+1)} | \nonumber \cr
    &\qquad - D \ln (2 \pi ) - \frac{D}{\kappa\_{n,k}^{(t+1)}} - \nu\_{n,k}^{(t+1)} (\boldsymbol{x}\_i - \boldsymbol{m}\_{n,k}^{(t+1)})^\top \boldsymbol{W}\_{n,k}^{(t+1)} (\boldsymbol{x}\_i - \boldsymbol{m}\_{n,k}^{(t+1)}) \Biggr] \cr
    r\_{i,k}^{(t+1)} &= \frac{\rho\_{i,k}^{(t+1)}}{\sum\_{k=1}^K \rho\_{i,k}^{(t+1)}}
\end{align}
$$

The approximate predictive distribution is as follows:

* $\boldsymbol{x}\_{n+1} \in \mathbb{R}^D$: a new data point
* $\boldsymbol{\mu}\_{\mathrm{p},k} \in \mathbb{R}^D$: the parameter of the predictive distribution
* $\boldsymbol{\Lambda}\_{\mathrm{p},k} \in \mathbb{R}^{D \times D}$: the parameter of the predictive distribution (a positive definite matrix)
* $\nu\_{\mathrm{p},k} \in \mathbb{R}\_{>0}$: the parameter of the predictive distribution

$$
\begin{align}
    &p(x\_{n+1}|x^n) \cr
    &= \frac{1}{\sum\_{k=1}^K \alpha\_{n,k}^{(t)}} \sum\_{k=1}^K \alpha\_{n,k}^{(t)} \mathrm{St}(x\_{n+1}|\boldsymbol{\mu}\_{\mathrm{p},k},\boldsymbol{\Lambda}\_{\mathrm{p},k}, \nu\_{\mathrm{p},k}) \cr
    &= \frac{1}{\sum\_{k=1}^K \alpha\_{n,k}^{(t)}} \sum\_{k=1}^K \alpha\_{n,k}^{(t)} \Biggl[ \frac{\Gamma (\nu\_{\mathrm{p},k} / 2 + D / 2)}{\Gamma (\nu\_{\mathrm{p},k} / 2)} \frac{|\boldsymbol{\Lambda}\_{\mathrm{p},k}|^{1/2}}{(\nu\_{\mathrm{p},k} \pi)^{D/2}} \nonumber \cr
    &\qquad \qquad \qquad \qquad \qquad \times \left( 1 + \frac{1}{\nu\_{\mathrm{p},k}} (\boldsymbol{x}\_{n+1} - \boldsymbol{\mu}\_{\mathrm{p},k})^\top \boldsymbol{\Lambda}\_{\mathrm{p},k} (\boldsymbol{x}\_{n+1} - \boldsymbol{\mu}\_{\mathrm{p},k}) \right)^{-\nu\_{\mathrm{p},k}/2 - D/2} \Biggr],
\end{align}
$$

where the parameters are obtained from the hyperparameters of the posterior distribution as follows:

$$
\begin{align}
    \boldsymbol{\mu}\_{\mathrm{p},k} &= \boldsymbol{m}\_{n,k}^{(t)} \cr
    \boldsymbol{\Lambda}\_{\mathrm{p},k} &= \frac{\kappa\_{n,k}^{(t)} (\nu\_{n,k}^{(t)} - D + 1)}{\kappa\_{n,k}^{(t)} + 1} \boldsymbol{W}\_{n,k}^{(t)}, \cr
    \nu\_{\mathrm{p},k} &= \nu\_{n,k}^{(t)} - D + 1.
\end{align}
$$
