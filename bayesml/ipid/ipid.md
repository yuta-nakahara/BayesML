<!-- Document Author
Kairi Suzuki <szk8258@gmail.com>
-->
The stochastic data generative model is as follows:

- $d \in \mathbb{N}$ : a dimension of data
- $n \in \mathbb{N}$ : a sample size
- $[n]=\{1, 2, \dots, n\}$
- $\boldsymbol{x}_t \in \mathbb{R}^d$ : a data point at time $t$ where $t \in [n]$
- $\boldsymbol{x}^n = (\boldsymbol{x}_t)_{t \in [n]}$ : data from time 1 to $n$
- $s^{(m)}_t \in \{0, 1\}$ : the state of data at time $t$ and $s^{(m)}_t = 1$ means that time $t$ is the change point
- $m = (s^{(m)}_t)_{t \in [n]} \in \{0, 1\}^{n}$ : the change pattern where $s^{(m)}_1=1$ for all $m$
- $t_j$ : the $j$ th change point, which satisfies $s^{m}_{t_j}=1$
- $\mathcal{T}_m = \{t \in [n] | s^{(m)}_t=1 \}$ : a set of change points for $m$
- $\theta_m = (\theta_t)_{t \in \mathcal{T}_m}$ : a set of parameters

$$
\begin{align}
p(\boldsymbol{x}^n | m, \theta_m) = \prod_{j=0}^{|\mathcal{T}_m|-1} \prod_{t=t_j}^{t_{j+1}-1}p(\boldsymbol{x}_t | \theta_{t_j}).
\end{align}
$$

The prior distribution is as follows:

- $\alpha_0 \in (0, 1)$ :  a hyperparameter of the change pattarn $m$

$$
\begin{align}
\pi(m | \alpha_0) &= \alpha_0^{|\mathcal{T}_m|-1} \cdot (1 - \alpha_0)^{n - |\mathcal{T}_m|}, \\
\pi(\theta_m | m) &= \prod_{j=0}^{|\mathcal{T}_m|-1}\pi(\theta_{t_j}).
\end{align}
$$

The posterior distribution of the last change point less than time $t$ is as follows:

- $\tau_t(m) = \max\{t^{\prime} \in \mathcal{T}_m | t^{\prime} \leq t\}$ : the function that returns the last change point less than time $t$ for $m$
- $T_t \in [t]$ : the random variable on $[t]$, which represents the last change point less than time $t$
- $p(T_t = l) := \sum_{m \in \{m^{\prime} \in \{0, 1\}^n | \tau_t(m^{\prime})=l\}} \pi(m|\alpha_0)$

$$
\begin{equation}
p(T_t = l | x^t) = \frac{p(x_t | T_t=l, x^{t-1}) \cdot p(T_t = l | x^{t-1})}{p(x_t | x^{t-1})},
\end{equation}
$$
where $p(x_t | T_t=l, x^{t-1})$, $p(T_t = l | x^{t-1})$, $p(x_t | x^{t-1})$ can be calculated sequentially as follows.
$$
\begin{align}
p(x_t | T_t=l, x^{t-1}) &= \int p(x_t | \theta_{l}) \cdot \pi(\theta_{l} | T_t=l, x^{t-1}) d\theta_l, \\
\pi(\theta_{l} | T_t=l, x^{t-1}) &=  \left \{
\begin{array}{ll}
\frac{\prod_{i=l}^{t-1}p(x_i | \theta_{l}) \cdot \pi(\theta_{l})}
{\int \prod_{i=l}^{t-1}p(x_i | \theta_{l}) \cdot \pi(\theta_{l}) d\theta_{l}}
& (1 \leq l \leq t-1), \\
\pi(\theta_{l})
& (l = t),
\end{array}
\right. \\
p(T_t = l | x^{t-1}) &= \left \{
\begin{array}{ll}
(1 - \alpha_0) \cdot p(T_{t-1}=l | x^{t-1}) & (1 \leq l \leq t-1), \\
\alpha_0 & (l=t),
\end{array}
\right. \\
% p(T_1 = 1) &= 1, \\
p(x_t | x^{t-1}) &= \sum_{l=1}^{t} p(x_t | T_t=l, x^{t-1}) \cdot p(T_t = l | x^{t-1}).
\end{align}
$$

The posterior distribution of the parameter at time $t$ is as follows:
$$
\begin{align}
\pi(\theta_t | x^t) = \sum_{l=1}^t p(T_t = l | x^t) \cdot \pi(\theta_t | T_t = l, x^t),
\end{align}
$$
where $p(T_t = l | x^t)$ can be updated by (4), (5), (6), (7), (8) and $\pi(\theta_t | T_t = l, x^t)$ can be calculated as follows.
$$
\begin{align}
\pi(\theta_t | T_t = l, x^t) = \frac{\prod_{i=l}^{t}p(x_i | \theta_{t}) \cdot \pi(\theta_{t})}
{\int \prod_{i=l}^{t}p(x_i | \theta_{t}) \cdot \pi(\theta_{t}) d\theta_{t}}.
\end{align}
$$
The predictive distribution is as follows:
