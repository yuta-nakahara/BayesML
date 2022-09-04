<!-- Document Author
Kairi Suzuki <szk8258@gmail.com>
-->
The stochastic data generative model is as follows:

- $n \in \mathbb{N}$ : a sample size
- $[n]=\{1, 2, \dots, n\}$
- $\mathcal{X}$ : a space of a data
- $\boldsymbol{x}_t \in \mathcal{X}$ : a data point at time $t \in [n]$
- $\boldsymbol{x}^n = (\boldsymbol{x}_t)_{t \in [n]}$ : data from time 1 to $n$
- $s^{(m)}_t \in \{0, 1\}$ : the state of data at time $t$. $s^{(m)}_t = 1$ means that time $t$ is the change point.
- $m = (s^{(m)}_t)_{t \in [n]} \in \{0, 1\}^{n}$ : the change pattern ($s^{(m)}_1=1$ for all $m$)
- $t_j$ : the $j$ th change point, which satisfies $s^{m}_{t_j}=1$
- $\mathcal{T}_m = \{t \in [n] \mid s^{(m)}_t=1 \}$ : a set of change points for $m$
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
p(m | \alpha_0) &= \alpha_0^{|\mathcal{T}_m|-1} \cdot (1 - \alpha_0)^{n - |\mathcal{T}_m|}, \\
p(\theta_m | m) &= \prod_{j=0}^{|\mathcal{T}_m|-1}p(\theta_{t_j}).
\end{align}
$$

- $\tau_t \in [t]$ : the last change point less than time $t$

$$
\begin{align}
p(\tau_t) := \sum_{m : \max\{t^{\prime} \in \mathcal{T}_m \mid t^{\prime} \leq t\}=\tau_t} p(m | \alpha_0)
\end{align}
$$

The posterior distribution of the last change point less than time $t$ is as follows:

$$
\begin{equation}
p(\tau_t | x^t) = \frac{p(x_t | \tau_t, x^{t-1}) \cdot p(\tau_t | x^{t-1})}{p(x_t | x^{t-1})},
\end{equation}
$$
where $p(x_t | \tau_t, x^{t-1})$, $p(\tau_t | x^{t-1})$, $p(x_t | x^{t-1})$ can be calculated sequentially as follows.
$$
\begin{align}
p(x_t | \tau_t, x^{t-1}) &= \int p(x_t | \theta_{\tau_t}) \cdot p(\theta_{\tau_t} | \tau_t, x^{t-1}) \mathrm{d}\theta_{\tau_t}, \\
p(\theta_{\tau_t} | \tau_t, x^{t-1}) &=  \left \{
\begin{array}{ll}
\frac{\prod_{i=\tau_t}^{t-1}p(x_i | \theta_{\tau_t}) \cdot p(\theta_{\tau_t})}
{\int \prod_{i=\tau_t}^{t-1}p(x_i | \theta_{\tau_t}) \cdot p(\theta_{\tau_t}) \mathrm{d} \theta_{\tau_t}}
& (1 \leq \tau_t \leq t-1), \\
p(\theta_{\tau_t})
& (\tau_t = t),
\end{array}
\right. \\
p(\tau_t | x^{t-1}) &= \left \{
\begin{array}{ll}
(1 - \alpha_0) \cdot p(\tau_{t-1} | x^{t-1}) & (\tau_t = \tau_{t-1}), \\
\alpha_0 & (\tau_t = t),
\end{array}
\right. \\
% p(T_1 = 1) &= 1, \\
p(x_t | x^{t-1}) &= \sum_{\tau_t=1}^{t} p(x_t | \tau_t, x^{t-1}) \cdot p(\tau_t | x^{t-1}).
\end{align}
$$

The posterior distribution of the parameter at time $t$ is as follows:
$$
\begin{align}
p(\theta_{t} | x^t) = \sum_{\tau_t=1}^t p(\tau_t | x^t) \cdot p(\theta_t | \tau_t, x^t),
\end{align}
$$
where $p(\tau_t | x^t)$ can be updated as above and $p(\theta_t | \tau_t, x^t)$ can be calculated as follows.
$$
\begin{align}
p(\theta_t | \tau_t, x^t) = \frac{\prod_{i=\tau_t}^{t}p(x_i | \theta_{t}) \cdot p(\theta_{t})}
{\int \prod_{i=\tau_t}^{t}p(x_i | \theta_{t}) \cdot p(\theta_{t}) \mathrm{d}\theta_{t}}.
\end{align}
$$
The predictive distribution is as follows:
$$
\begin{align}
p(x_t | x^{t-1}) &= \sum_{\tau_t=1}^t p(x_t | \tau_t, x^{t-1}) \cdot p(\tau_t | x^{t-1}),
\end{align}
$$
