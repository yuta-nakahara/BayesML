<img src="../logos/BayesML_logo.png" width="200">

# Docstringの運用規則 Ver.2

<div style="text-align:right;float:right">
作成：中原
</div>

* [numpyスタイル](https://numpydoc.readthedocs.io/en/latest/format.html)を採用する
  * 関連するライブラリの多くが採用しており，利用者もそちらに慣れていると思われる．我々も慣れておいたほうが良い．
* VSCodeのautoDocstringという拡張機能が便利
* パッケージの説明は`__init__.py`の冒頭に書く
  * 書くべき項目は以下．
    * データ生成確率モデル（尤度関数）
    * 事前分布
    * 事後分布
    * （事後）予測分布
  * 書き方は以下
    * `r"""` で始めると Raw docstring となり、バックスラッシュを含めることができる。
    * 変数とその取りうる値を箇条書きで書く
      * 例，``` * :math:` x \in \mathcal{X}\` ```
    * 密度関数などの式を別行立てで書く．
      * 例，``` .. math:: e^{i \pi} + 1 = 0 ```
* クラスやメソッドの説明では，
  * 必ずshort summaryを`"""`の直後に概要を1行書く．
  * 引数がある場合は必ずPrametersを書く．その際，引数が理論的に満たすべき条件を書く．
    * 例，分散は正の実数
  * 戻り値がある場合は必ずReturnsを書く
  * `visualize_model`, `visualize_posterior`は必ずExampleを書く
* bernoulliパッケージの`__init__.py`, `_bernoulli.py`を参考にしてほしい

## 例1

```
r"""
The Bernoulli distribution with the beta prior distribution.

The stochastic data generative model is as follows:

* :math:`x \in \{ 0, 1\}`: a data point
* :math:`p \in [0, 1]`: a parameter 

.. math:: \text{Bern}(x|p) = p^x (1-p)^{1-x}.

The prior distribution is as follows:

* :math:`\alpha_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`B(\cdot,\cdot): \mathbb{R}_{>0} \times \mathbb{R}_{>0} \to \times \mathbb{R}_{>0}`: the Beta function

.. math:: \text{Beta}(p|\alpha_0,\beta_0) = \frac{1}{B(\alpha_0, \beta_0)} p^{\alpha_0} (1-p)^{\beta_0}.

The posterior distribution is as follows:

* :math:`x^n = (x_1, x_2, \dots , x_n) \in \{ 0, 1\}^n`: given data
* :math:`\alpha_n \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_n \in \mathbb{R}_{>0}`: a hyperparameter

.. math:: \text{Beta}(p|\alpha_n,\beta_n) = \frac{1}{B(\alpha_n, \beta_n)} p^{\alpha_n} (1-p)^{\beta_n},

where the updating rule of the hyperparameters is

.. math::
    \alpha_n = \alpha_0 + \sum_{i=1}^n I \{ x_i = 1 \},\\
    \beta_n = \beta_0 + \sum_{i=1}^n I \{ x_i = 0 \}.

The predictive distribution is as follows:

* :math:`x_n \in \{ 0, 1\}`: a new data point
* :math:`\alpha_n \in \mathbb{R}_{>0}`: the hyperparameter of the posterior
* :math:`\beta_n \in \mathbb{R}_{>0}`: the hyperparameter of the posterior

.. math::
    p(x|\alpha_n, \beta_n) = \begin{cases}
    \frac{\alpha_n}{\alpha_n + \beta_n}, & x = 1,\\
    \frac{\beta_n}{\alpha_n + \beta_n}, & x = 0.
    \end{cases}
"""
```

## 例2

```
    def set_h_params(self,h_alpha,h_beta):
        """Set the hyperparameters of the prior distribution.
        
        Parameters
        ----------
        h_alpha : float
            a positive real number
        h_beta : float
            a positive real number
        """
```