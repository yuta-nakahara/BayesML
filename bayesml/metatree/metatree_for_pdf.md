<!--
Document Author
Shota Saito <shota.s@gunma-u.ac.jp>
Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
-->

The stochastic data generative model is as follows:

* $\mathcal{X}$ : a space of an explanatory variable (a finite set)
* $\boldsymbol{x}=[x\_1, \ldots, x\_d] \in \mathcal{X}^d$ : an explanatory variable
* $\mathcal{Y}$ : a space of an objective variable
* $y \in \mathcal{Y}$ : an objective variable
* $D\_\mathrm{max} \in \mathbb{N}$ : the maximum depth of trees
* $T$ : $|\mathcal{X}|$-ary regular tree whose depth is smaller than or equal to $D\_\mathrm{max}$, where "regular" means that all inner nodes have $k$ child nodes.
* $\mathcal{T}$ : a set of $T$
* $s$ : a node of a tree
* $\mathcal{S}$ : a set of $s$
* $\mathcal{I}(T)$ : a set of inner nodes of $T$
* $\mathcal{L}(T)$ : a set of leaf nodes of $T$
* $\boldsymbol{k}=(k\_s)\_{s \in \mathcal{S}}$ : feature assign vector where $k\_s \in \\{1,2,\ldots,d\\}$
* $\boldsymbol{\theta}=(\theta\_s)\_{s \in \mathcal{S}}$ : a set of parameter
* $s(\boldsymbol{x}) \in \mathcal{L}(T)$ : a leaf node of $T$ corresponding to $\boldsymbol{x}$

$$p(y | \boldsymbol{x}, \boldsymbol{\theta}, T, \boldsymbol{k})=p(y | \theta\_{s(\boldsymbol{x})})$$

The prior distribution is as follows:

* $g\_s \in [0,1]$ : a hyperparameter assigned to $s \in \mathcal{S}$
* $M\_{T, \boldsymbol{k}}$ : a meta-tree for $(T, \boldsymbol{k})$
* $\mathcal{T}\_{M\_{T, \boldsymbol{k}}}$ : a set of $T$ represented by a meta-tree $M\_{T, \boldsymbol{k}}$
* $B \in \mathbb{N}$ : the number of meta-trees
* $\mathcal{M}=\\{(T\_1, \boldsymbol{k}\_1), (T\_2, \boldsymbol{k}\_2), \ldots, (T\_B, \boldsymbol{k}\_B) \\}$ for $B$ meta-trees $M\_{T\_1, \boldsymbol{k}\_1}, M\_{T\_2, \boldsymbol{k}\_2}, \dots, M\_{T\_B, \boldsymbol{k}\_B}$. (These meta-trees must be given beforehand by some method, e.g., constructed from bootstrap samples similar to the random forest.)

For $T' \in M\_{T, \boldsymbol{k}}$,
$$p(T')=\prod\_{s \in \mathcal{I}(T')} g\_s \prod\_{s' \in \mathcal{L}(T')} (1-g\_{s'}),$$
where $g\_s=0$ for a leaf node $s$ of a meta-tree $M\_{T, \boldsymbol{k}}$.

For $\boldsymbol{k}\_b \in \\{\boldsymbol{k}\_1, \boldsymbol{k}\_2, \ldots, \boldsymbol{k}\_B \\}$,

$$p(\boldsymbol{k}\_b) = \frac{1}{B}.$$

The posterior distribution is as follows:

* $n \in \mathbb{N}$ : a sample size
* $\boldsymbol{x}^n = \\{ \boldsymbol{x}\_1, \boldsymbol{x}\_2, \ldots, \boldsymbol{x}\_n \\}$
* $y^n = \\{ y\_1, y\_2, \ldots, y\_n \\}$
* $g\_{s|\boldsymbol{x}^n, y^n} \in [0,1]$ : a hyperparameter

For $T' \in M\_{T, \boldsymbol{k}}$,
$$p(T' | \boldsymbol{x}^n, y^n, \boldsymbol{k})=\prod\_{s \in \mathcal{I}(T')} g\_{s|\boldsymbol{x}^n, y^n} \prod\_{s' \in \mathcal{L}(T')} (1-g\_{s'|\boldsymbol{x}^n, y^n}),$$
where the updating rules of the hyperparameter are as follows.

$$g\_{s | \boldsymbol{x}^i, y^i} =
\begin{cases}
g\_s & (i = 0),\cr
\frac{g\_{s | \boldsymbol{x}^{i-1}, y^{i-1}} \tilde{q}\_{s\_{\mathrm{child}}}(y\_i | \boldsymbol{x}\_i, \boldsymbol{x}^{i-1}, y^{i-1}, M\_{T, \boldsymbol{k}})}{\tilde{q}\_s(y\_i | \boldsymbol{x}\_i, \boldsymbol{x}^{i-1}, y^{i-1}, M\_{T, \boldsymbol{k}})}  &(\mathrm{otherwise}),
\end{cases}$$
where $s\_{\mathrm{child}}$ is the child node of $s$ on the path corresponding to $\boldsymbol{x}\_{i}$ in $M\_{T, \boldsymbol{k}}$ and

$$
\begin{align*}
&\tilde{q}\_s(y\_{i} | \boldsymbol{x}\_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, M\_{T, \boldsymbol{k}}) \cr
&= \begin{cases}
q\_s(y\_{i} | \boldsymbol{x}\_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, \boldsymbol{k}),& (s \ {\rm is \ the \ leaf \ node \ of} \ M\_{T, \boldsymbol{k}}),\cr
(1-g\_{s | \boldsymbol{x}^{i-1}, y^{i-1}}) q\_s(y\_{i} | \boldsymbol{x}\_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, \boldsymbol{k}) \cr
\qquad + g\_{s | \boldsymbol{x}^{i-1}, y^{i-1}} \tilde{q}\_{s\_{\mathrm{child}}}(y\_{i} | \boldsymbol{x}\_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, M\_{T, \boldsymbol{k}}),& ({\rm otherwise}),
\end{cases}
\end{align*}
$$

$$q\_s(y\_{i} | \boldsymbol{x}\_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, \boldsymbol{k})=\int p(y\_i | \boldsymbol{x}\_i, \boldsymbol{\theta}, T, \boldsymbol{k}) p(\boldsymbol{\theta} | \boldsymbol{x}^{i-1}, y^{i-1}, T, \boldsymbol{k}) \mathrm{d} \boldsymbol{\theta}, \quad s \in \mathcal{L}(T)$$

For $\boldsymbol{k}\_b \in \\{\boldsymbol{k}\_1, \boldsymbol{k}\_2, \ldots, \boldsymbol{k}\_B \\}$,
$$p(\boldsymbol{k}\_b | \boldsymbol{x}^n, y^n)\propto \prod\_{i=1}^n \tilde{q}\_{s\_{\lambda}}(y\_{i}|\boldsymbol{x}\_{i},\boldsymbol{x}^{i-1}, y^{i-1}, M\_{T\_b, \boldsymbol{k}\_b}),$$
where $s\_{\lambda}$ is the root node of $M\_{T, \boldsymbol{k}\_b}$.

The predictive distribution is as follows:

$$p(y\_{n+1}| \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n) = \sum\_{b = 1}^B p(\boldsymbol{k}\_b | \boldsymbol{x}^n, y^n) \tilde{q}\_{s\_{\lambda}}(y\_{n+1}|\boldsymbol{x}\_{n+1},\boldsymbol{x}^n, y^n, M\_{T\_b, \boldsymbol{k}\_b})$$

The expectation of the predictive distribution can be calculated as follows.

$$\mathbb{E}\_{p(y\_{n+1}| \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n)} [Y\_{n+1}| \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n] = \sum\_{b = 1}^B p(\boldsymbol{k}\_b | \boldsymbol{x}^n, y^n) \mathbb{E}\_{\tilde{q}\_{s\_{\lambda}}(y\_{n+1}|\boldsymbol{x}\_{n+1},\boldsymbol{x}^n, y^n, M\_{T\_b, \boldsymbol{k}\_b})} [Y\_{n+1}| \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}\_b],$$
where the expectation for $\tilde{q}$ is recursively given as follows.

$$
\begin{align*}
&\mathbb{E}\_{\tilde{q}\_s(y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, M\_{T\_b, \boldsymbol{k}\_b})} [Y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}\_b] \cr
&= \begin{cases}
\mathbb{E}\_{q\_s(y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}\_b)} [Y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}\_b],& (s \ {\rm is \ the \ leaf \ node \ of} \ M\_{T\_b, \boldsymbol{k}\_b}),\cr
(1-g\_{s | \boldsymbol{x}^n, y^n}) \mathbb{E}\_{q\_s(y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}\_b)} [Y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}\_b] \cr
\qquad + g\_{s | \boldsymbol{x}^n, y^n} \mathbb{E}\_{\tilde{q}\_{s\_{\mathrm{child}}}(y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, M\_{T\_b, \boldsymbol{k}\_b})} [Y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}\_b] ,& ({\rm otherwise}).
\end{cases}
\end{align*}
$$

The maximum value of the predictive distribution can be calculated as follows.

$$\max\_{y\_{n+1}} p(y\_{n+1}| \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n) = \max\_{b = 1, \dots , B} \left\\{ p(\boldsymbol{k}\_b | \boldsymbol{x}^n, y^n) \max\_{y\_{n+1}} \tilde{q}\_{s\_{\lambda}}(y\_{n+1}|\boldsymbol{x}\_{n+1},\boldsymbol{x}^n, y^n, M\_{T\_b, \boldsymbol{k}\_b}) \right\\},$$

where the maximum value of $\tilde{q}$ is recursively given as follows.

$$
\begin{align*}
&\max\_{y\_{n+1}} \tilde{q}\_s(y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, M\_{T\_b, \boldsymbol{k}\_b}) \cr
&= \begin{cases}
\max\_{y\_{n+1}} q\_s(y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}\_b),& (s \ {\rm is \ the \ leaf \ node \ of} \ M\_{T\_b, \boldsymbol{k}\_b}),\cr
\max \\{ (1-g\_{s | \boldsymbol{x}^n, y^n}) \max\_{y\_{n+1}} q\_s(y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}\_b), \cr
\qquad \qquad g\_{s | \boldsymbol{x}^n, y^n} \max\_{y\_{n+1}} \tilde{q}\_{s\_{\mathrm{child}}}(y\_{n+1} | \boldsymbol{x}\_{n+1}, \boldsymbol{x}^n, y^n, M\_{T\_b, \boldsymbol{k}\_b}) \\} ,& ({\rm otherwise}).
\end{cases}
\end{align*}
$$

The mode of the predictive distribution can be also calculated by using the above equation.

References

* Dobashi, N.; Saito, S.; Nakahara, Y.; Matsushima, T. Meta-Tree Random Forest: Probabilistic Data-Generative Model and Bayes Optimal Prediction. Entropy 2021, 23, 768. https://doi.org/10.3390/e23060768
* Nakahara, Y.; Saito, S.; Kamatsuka, A.; Matsushima, T. Probability Distribution on Full Rooted Trees. Entropy 2022, 24, 328. https://doi.org/10.3390/e24030328
