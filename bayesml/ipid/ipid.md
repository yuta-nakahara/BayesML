<!-- Document Author
Kairi Suzuki <szk8258@gmail.com>
-->
The stochastic data generative model is as follows:

- $d \in \mathbb{N}$ : a dimension of data
- $n \in \mathbb{N}$ : number of data
- $\boldsymbol{x}_t \in \mathbb{R}^d$ : a data point at time $t$ where $t \in [n]$
- $\boldsymbol{x}^n = (\boldsymbol{x}_t)_{t \in [n]}$ : data from time 1 to $n$
- $s^{(m)}_t \in \{0, 1\}$ : the state of data at time $t$
- $m = (s^{(m)}_t)_{t \in [n]} \in \{0, 1\}^{n}$ : the change pattern where $s^{(m)}_1=1$ for all $m$
- $t_j$ : a $j$ th change point
- $\mathcal{T}_m = \{t \in [n] | s^{(m)}_t=1 \}$ : a set of change points for $m$
- $\theta_m = (\theta_t)_{t \in \mathcal{T}_m}$ : a set of parameters

$$
p(\boldsymbol{x}^n | m, \theta_m) = \prod_{j=0}^{|\mathcal{T}_m|-1} \prod_{t=t_j}^{t_{j+1}-1}p(\boldsymbol{x}_t | \theta_{t_j})
$$

The prior distribution is as follows:

- $\alpha_0$ :  a hyperparameter of the change pattarn $m$

$$
p(m | \alpha_0) = \alpha_0^{|\mathcal{T}_m|-1} \cdot (1 - \alpha_0)^{n - |\mathcal{T}_m|}
$$

$$
p(\theta_m | m) = \prod_{j=0}^{|\mathcal{T}_m|-1}p(\theta_{t_j})
$$

The posterior distribution is as follows:

hoge

The predictive distribution is as follows:

hoge
