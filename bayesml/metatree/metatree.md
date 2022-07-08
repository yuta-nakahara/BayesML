<!--
Document Author
Shota Saito <shota.s@gunma-u.ac.jp>
-->

The stochastic data generative model is as follows:

* $\mathcal{X}$ : a space of an explanatory variable
* $\boldsymbol{x}=[x_1, \ldots, x_d] \in \mathcal{X}$ : an explanatory variable
* $\mathcal{Y}$ : a space of an objective variable
* $y \in \mathcal{Y}$ : an objective variable
* $T$ : $k$-ary regular tree whose depth is $D_\mathrm{max}$, where "regular" means that all inner nodes have $k$ child nodes.
* $\mathcal{T}$ : a set of $T$
* $s$ : a node of a tree
* $\mathcal{S}$ : a set of $s$
* $\mathcal{I}(T)$ : a set of inner nodes of $T$
* $\mathcal{L}(T)$ : a set of leaf nodes of $T$
* $\boldsymbol{k}=(k_s)_{s \in \mathcal{S}}$ : feature assign vector where $k_s \in \{1,2,\ldots,d\}$
* $\boldsymbol{\theta}=(\theta_s)_{s \in \mathcal{S}}$ : a set of parameter
* $s(\boldsymbol{x}) \in \mathcal{L}(T)$ : a leaf node of $T$ corresponding to $\boldsymbol{x}$

$$p(y | \boldsymbol{x}, \boldsymbol{\theta}, T, \boldsymbol{k})=p(y | \theta_{s(\boldsymbol{x})})$$

The prior distribution is as follows:

* $g_s \in [0,1]$ : a hyperparameter assigned to $s \in \mathcal{S}$
* $M_{T, \boldsymbol{k}}$ : a meta tree for $(T, \boldsymbol{k})$
* $\mathcal{T}_{M_{T, \boldsymbol{k}}}$ : a set of $T$ represented by a meta-tree $M_{T, \boldsymbol{k}}$

For $T' \in M_{T, \boldsymbol{k}}$,
$$p(T')=\prod_{s \in \mathcal{I}(T')} g_s \prod_{s' \in \mathcal{L}(T')} (1-g_{s'}),$$
where $g_s=0$ for a leaf node $s$ of a meta-tree $M_{T, \boldsymbol{k}}$.

The posterior distribution is as follows:

* $n \in \mathbb{N}$ : a sample size
* $\boldsymbol{x}^n = \{ \boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_n \}$
* $y^n = \{ y_1, y_2, \ldots, y_n \}$
* $g_{s|\boldsymbol{x}^n, y^n} \in [0,1]$ : a hyperparameter

For $T' \in M_{T, \boldsymbol{k}}$,
$$p(T' | \boldsymbol{x}^n, y^n, \boldsymbol{k})=\prod_{s \in \mathcal{I}(T')} g_{s|\boldsymbol{x}^n, y^n} \prod_{s' \in \mathcal{L}(T')} (1-g_{s'|\boldsymbol{x}^n, y^n}),$$
where the updating rules of the hyperparameter are as follows.

$$g_{s | \boldsymbol{x}^i, y^i} =
\begin{cases}
g_s & (i = 0),\\
\frac{g_{s | \boldsymbol{x}^{i-1}, y^{i-1}} \tilde{q}_{s_{\mathrm{child}}}(y_i | \boldsymbol{x}_i, \boldsymbol{x}^{i-1}, y^{i-1}, M_{T, \boldsymbol{k}})}{\tilde{q}_s(y_i | \boldsymbol{x}_i, \boldsymbol{x}^{i-1}, y^{i-1}, M_{T, \boldsymbol{k}})}  &(\mathrm{otherwise}),
\end{cases}$$
where $s_{\mathrm{child}}$ is the child node of $s$ on the path corresponding to $\boldsymbol{x}_{i}$ in $M_{T, \boldsymbol{k}}$ and

$$\tilde{q}_s(y_{i} | \boldsymbol{x}_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, M_{T, \boldsymbol{k}}) = \begin{cases}
q_s(y_{i} | \boldsymbol{x}_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, \boldsymbol{k}),& (s \ {\rm is \ the \ leaf \ node \ of} \ M_{T, \boldsymbol{k}}),\\
(1-g_{s | \boldsymbol{x}^{i-1}, y^{i-1}}) q_s(y_{i} | \boldsymbol{x}_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, \boldsymbol{k}) \\
\qquad + g_{s | \boldsymbol{x}^{i-1}, y^{i-1}} \tilde{q}_{s_{\mathrm{child}}}(y_{i} | \boldsymbol{x}_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, M_{T, \boldsymbol{k}}),& ({\rm otherwise}),
\end{cases}$$

$$q_s(y_{i} | \boldsymbol{x}_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, \boldsymbol{k})=\int p(y_i | \boldsymbol{x}_i, \boldsymbol{\theta}, T, \boldsymbol{k}) p(\boldsymbol{\theta} | \boldsymbol{x}^{i-1}, y^{i-1}, T, \boldsymbol{k}) \mathrm{d} \boldsymbol{\theta}, \quad s \in \mathcal{L}(T)$$

For $B$ meta-trees $M_{T_1, \boldsymbol{k}_1}, M_{T_2, \boldsymbol{k}_2}, \dots, M_{T_B, \boldsymbol{k}_B}$, we define $\mathcal{M}=\{(T_1, \boldsymbol{k}_1), (T_2, \boldsymbol{k}_2), \ldots, (T_B, \boldsymbol{k}_B) \}$. Then, an approximate predictive distribution is given as

$$p(y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n) \approx \sum_{(T,\boldsymbol{k} )\in \mathcal{M}} p(\boldsymbol{k}| \boldsymbol{x}^n, y^n) p(T| \boldsymbol{x}^n, y^n, \boldsymbol{k}) \int p(y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{\theta}, T, \boldsymbol{k}) p(\boldsymbol{\theta} | \boldsymbol{x}^{n}, y^{n}, T, \boldsymbol{k}) \mathrm{d} \boldsymbol{\theta},$$
where 
$$p(\boldsymbol{k}| \boldsymbol{x}^n, y^n)\propto \prod_{i=1}^n \tilde{q}_{s_{\lambda}}(y_{i}|\boldsymbol{x}_{i},\boldsymbol{x}^{i-1}, y^{i-1}, M_{T, \boldsymbol{k}}),$$
and $s_{\lambda}$ is the root node of $M_{T, \boldsymbol{k}}$.

References

* Dobashi, N.; Saito, S.; Nakahara, Y.; Matsushima, T. Meta-Tree Random Forest: Probabilistic Data-Generative Model and Bayes Optimal Prediction. Entropy 2021, 23, 768. https://doi.org/10.3390/e23060768
* Nakahara, Y.; Saito, S.; Kamatsuka, A.; Matsushima, T. Probability Distribution on Full Rooted Trees. Entropy 2022, 24, 328. https://doi.org/10.3390/e24030328
