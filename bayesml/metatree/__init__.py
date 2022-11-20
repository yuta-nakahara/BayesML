# Document Author
# Shota Saito <shota.s@gunma-u.ac.jp>
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
r"""
The stochastic data generative model is as follows:

* :math:`\mathcal{X}` : a space of an explanatory variable (a finite set)
* :math:`\boldsymbol{x}=[x_1, \ldots, x_d] \in \mathcal{X}^d` : an explanatory variable
* :math:`\mathcal{Y}` : a space of an objective variable
* :math:`y \in \mathcal{Y}` : an objective variable
* :math:`D_\mathrm{max} \in \mathbb{N}` : the maximum depth of trees
* :math:`T` : :math:`|\mathcal{X}|`-ary regular tree whose depth is smaller than or equal to :math:`D_\mathrm{max}`, where "regular" means that all inner nodes have :math:`k` child nodes.
* :math:`\mathcal{T}` : a set of :math:`T`
* :math:`s` : a node of a tree
* :math:`\mathcal{S}` : a set of :math:`s`
* :math:`\mathcal{I}(T)` : a set of inner nodes of :math:`T`
* :math:`\mathcal{L}(T)` : a set of leaf nodes of :math:`T`
* :math:`\boldsymbol{k}=(k_s)_{s \in \mathcal{S}}` : feature assign vector where :math:`k_s \in \{1,2,\ldots,d\}`
* :math:`\boldsymbol{\theta}=(\theta_s)_{s \in \mathcal{S}}` : a set of parameter
* :math:`s(\boldsymbol{x}) \in \mathcal{L}(T)` : a leaf node of :math:`T` corresponding to :math:`\boldsymbol{x}`

.. math::
    p(y | \boldsymbol{x}, \boldsymbol{\theta}, T, \boldsymbol{k})=p(y | \theta_{s(\boldsymbol{x})})

The prior distribution is as follows:

* :math:`g_{0,s} \in [0,1]` : a hyperparameter assigned to :math:`s \in \mathcal{S}`
* :math:`M_{T, \boldsymbol{k}}` : a meta-tree for :math:`(T, \boldsymbol{k})`
* :math:`\mathcal{T}_{M_{T, \boldsymbol{k}}}` : a set of :math:`T` represented by a meta-tree :math:`M_{T, \boldsymbol{k}}`
* :math:`B \in \mathbb{N}` : the number of meta-trees
* :math:`\mathcal{M}=\{(T_1, \boldsymbol{k}_1), (T_2, \boldsymbol{k}_2), \ldots, (T_B, \boldsymbol{k}_B) \}` for :math:`B` meta-trees :math:`M_{T_1, \boldsymbol{k}_1}, M_{T_2, \boldsymbol{k}_2}, \dots, M_{T_B, \boldsymbol{k}_B}`. (These meta-trees must be given beforehand by some method, e.g., constructed from bootstrap samples similar to the random forest.)

For :math:`T' \in M_{T, \boldsymbol{k}}`,

.. math::
    p(T')=\prod_{s \in \mathcal{I}(T')} g_{0,s} \prod_{s' \in \mathcal{L}(T')} (1-g_{0,s'}),

where :math:`g_{0,s}=0` for a leaf node :math:`s` of a meta-tree :math:`M_{T, \boldsymbol{k}}`.

For :math:`\boldsymbol{k}_b \in \{\boldsymbol{k}_1, \boldsymbol{k}_2, \ldots, \boldsymbol{k}_B \}`,

.. math::
    p(\boldsymbol{k}_b) = \frac{1}{B}.

The posterior distribution is as follows:

* :math:`n \in \mathbb{N}` : a sample size
* :math:`\boldsymbol{x}^n = \{ \boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_n \}`
* :math:`y^n = \{ y_1, y_2, \ldots, y_n \}`
* :math:`g_{n,s} \in [0,1]` : a hyperparameter

For :math:`T' \in M_{T, \boldsymbol{k}}`,

.. math::
    p(T' | \boldsymbol{x}^n, y^n, \boldsymbol{k})=\prod_{s \in \mathcal{I}(T')} g_{n,s} \prod_{s' \in \mathcal{L}(T')} (1-g_{n,s'}),

where the updating rules of the hyperparameter are as follows.

.. math::
    g_{i,s} =
    \begin{cases}
    g_{0,s} & (i = 0),\\
    \frac{g_{i-1,s} \tilde{q}_{s_{\mathrm{child}}}(y_i | \boldsymbol{x}_i, \boldsymbol{x}^{i-1}, y^{i-1}, M_{T, \boldsymbol{k}})}{\tilde{q}_s(y_i | \boldsymbol{x}_i, \boldsymbol{x}^{i-1}, y^{i-1}, M_{T, \boldsymbol{k}})}  &(\mathrm{otherwise}),
    \end{cases}

where :math:`s_{\mathrm{child}}` is the child node of :math:`s` on the path corresponding to :math:`\boldsymbol{x}_{i}` in :math:`M_{T, \boldsymbol{k}}` and

.. math::
    &\tilde{q}_s(y_{i} | \boldsymbol{x}_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, M_{T, \boldsymbol{k}}) \\
    &= \begin{cases}
    q_s(y_{i} | \boldsymbol{x}_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, \boldsymbol{k}),& (s \ {\rm is \ the \ leaf \ node \ of} \ M_{T, \boldsymbol{k}}),\\
    (1-g_{i-1,s}) q_s(y_{i} | \boldsymbol{x}_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, \boldsymbol{k}) \\
    \qquad + g_{i-1,s} \tilde{q}_{s_{\mathrm{child}}}(y_{i} | \boldsymbol{x}_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, M_{T, \boldsymbol{k}}),& ({\rm otherwise}),
    \end{cases}

.. math::
    q_s(y_{i} | \boldsymbol{x}_{i}, \boldsymbol{x}^{i-1}, y^{i-1}, \boldsymbol{k})=\int p(y_i | \boldsymbol{x}_i, \boldsymbol{\theta}, T, \boldsymbol{k}) p(\boldsymbol{\theta} | \boldsymbol{x}^{i-1}, y^{i-1}, T, \boldsymbol{k}) \mathrm{d} \boldsymbol{\theta}, \quad s \in \mathcal{L}(T)

For :math:`\boldsymbol{k}_b \in \{\boldsymbol{k}_1, \boldsymbol{k}_2, \ldots, \boldsymbol{k}_B \}`,

.. math::
    p(\boldsymbol{k}_b | \boldsymbol{x}^n, y^n)\propto \prod_{i=1}^n \tilde{q}_{s_{\lambda}}(y_{i}|\boldsymbol{x}_{i},\boldsymbol{x}^{i-1}, y^{i-1}, M_{T_b, \boldsymbol{k}_b}),

where :math:`s_{\lambda}` is the root node of :math:`M_{T, \boldsymbol{k}_b}`.

The predictive distribution is as follows:

.. math::
    p(y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n) = \sum_{b = 1}^B p(\boldsymbol{k}_b | \boldsymbol{x}^n, y^n) \tilde{q}_{s_{\lambda}}(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, M_{T_b, \boldsymbol{k}_b})

The expectation of the predictive distribution can be calculated as follows.

.. math::
    \mathbb{E}_{p(y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n)} [Y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n] = \sum_{b = 1}^B p(\boldsymbol{k}_b | \boldsymbol{x}^n, y^n) \mathbb{E}_{\tilde{q}_{s_{\lambda}}(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, M_{T_b, \boldsymbol{k}_b})} [Y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}_b],

where the expectation for :math:`\tilde{q}` is recursively given as follows.

.. math::
    &\mathbb{E}_{\tilde{q}_s(y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, M_{T_b, \boldsymbol{k}_b})} [Y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}_b] \\
    &= \begin{cases}
    \mathbb{E}_{q_s(y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}_b)} [Y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}_b],& (s \ {\rm is \ the \ leaf \ node \ of} \ M_{T_b, \boldsymbol{k}_b}),\\
    (1-g_{n,s}) \mathbb{E}_{q_s(y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}_b)} [Y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}_b] \\
    \qquad + g_{n,s} \mathbb{E}_{\tilde{q}_{s_{\mathrm{child}}}(y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, M_{T_b, \boldsymbol{k}_b})} [Y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}_b] ,& ({\rm otherwise}).
    \end{cases}

The maximum value of the predictive distribution can be calculated as follows.

.. math::
    \max_{y_{n+1}} p(y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n) = \max_{b = 1, \dots , B} \left\{ p(\boldsymbol{k}_b | \boldsymbol{x}^n, y^n) \max_{y_{n+1}} \tilde{q}_{s_{\lambda}}(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, M_{T_b, \boldsymbol{k}_b}) \right\},

where the maximum value of :math:`\tilde{q}` is recursively given as follows.

.. math::
    &\max_{y_{n+1}} \tilde{q}_s(y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, M_{T_b, \boldsymbol{k}_b}) \\
    &= \begin{cases}
    \max_{y_{n+1}} q_s(y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}_b),& (s \ {\rm is \ the \ leaf \ node \ of} \ M_{T_b, \boldsymbol{k}_b}),\\
    \max \{ (1-g_{n,s}) \max_{y_{n+1}} q_s(y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}_b), \\
    \qquad \qquad g_{n,s} \max_{y_{n+1}} \tilde{q}_{s_{\mathrm{child}}}(y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, M_{T_b, \boldsymbol{k}_b}) \} ,& ({\rm otherwise}).
    \end{cases}

The mode of the predictive distribution can be also calculated by using the above equation.

References

* Dobashi, N.; Saito, S.; Nakahara, Y.; Matsushima, T. Meta-Tree Random Forest: Probabilistic Data-Generative Model and Bayes Optimal Prediction. *Entropy* 2021, 23, 768. https://doi.org/10.3390/e23060768
* Nakahara, Y.; Saito, S.; Kamatsuka, A.; Matsushima, T. Probability Distribution on Full Rooted Trees. *Entropy* 2022, 24, 328. https://doi.org/10.3390/e24030328
"""
from ._metatree_x_discrete import GenModel
from ._metatree_x_discrete import LearnModel

__all__ = ["GenModel", "LearnModel"]