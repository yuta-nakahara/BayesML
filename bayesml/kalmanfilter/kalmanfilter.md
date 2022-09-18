<!--
Document Author
Shota Saito <shota.s@gunma-u.ac.jp>
-->

Bayesian Kalman filter model with Gaussian-gamma prior distribution.

The stochastic data generative models are as follows:

* $D \in \mathbb{N}$ : a dimension of an observed variable
* $\boldsymbol{x}_n \in \mathbb{R}^D$ : an observed variable 
* $L \in \mathbb{N}$ : a dimension of a latent variable
* $\boldsymbol{z}_n \in \mathbb{R}^L$ : a latent variable 
* $\boldsymbol{C} \in \mathbb{R}^{D \times L}$ : a parameter
* $\boldsymbol{A} \in \mathbb{R}^{L \times L}$ : a parameter
* $\boldsymbol{I}_{D \times D}$ : $D \times D$ identity matrix
* $\boldsymbol{I}_{L \times L}$ : $L \times L$ identity matrix
* $\boldsymbol{\mu}_0 \in \mathbb{R}^{L}$ : a parameter
* $\lambda \in \mathbb{R}_{>0}$ : a parameter
* $\gamma \in \mathbb{R}_{>0}$ : a parameter
* $\gamma_0 \in \mathbb{R}_{>0}$ : a parameter


$$p(\boldsymbol{x}_n \mid \boldsymbol{z}_n, \boldsymbol{C}, \lambda)=\mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{C}\boldsymbol{z}_n, (\lambda \boldsymbol{I}_{D \times D})^{-1}),$$
$$p(\boldsymbol{z}_1 \mid \boldsymbol{\mu}_{0}, \gamma_0)=\mathcal{N}(\boldsymbol{z}_1 \mid \boldsymbol{\mu}_{0}, (\gamma_0 \boldsymbol{I}_{L \times L})^{-1}),$$
$$p(\boldsymbol{z}_n \mid \boldsymbol{z}_{n-1}, \boldsymbol{A}, \gamma)=\mathcal{N}(\boldsymbol{z}_n \mid \boldsymbol{A}\boldsymbol{z}_{n-1}, (\gamma \boldsymbol{I}_{L \times L})^{-1})$$

The prior distributions are as follows:

* $\beta_0 \in \mathbb{R}_{>0}$ : a known hyperparameter
* $\boldsymbol{c}_i \in \mathbb{R}^D$ : $i$-th column of $\boldsymbol{C}$

$$p(\boldsymbol{C})=\prod_{i=1}^{L} p(\boldsymbol{c}_i)=\prod_{i=1}^{L} \mathcal{N}(\boldsymbol{c}_i \mid \boldsymbol{0}, (\beta_0 \boldsymbol{I}_{D \times D})^{-1}).$$


* $a_0, b_0 \in \mathbb{R}_{>0}$ : known hyperparameters


$$p(\lambda)=\mathrm{Gam}(\lambda \mid a_0, b_0).$$


* $\delta_0 \in \mathbb{R}_{>0}$ : known hyperparameter
* $\boldsymbol{a}_i \in \mathbb{R}^L$ : $i$-th column of $\boldsymbol{A}$


$$ p(\boldsymbol{A})=\prod_{i=1}^{L} p(\boldsymbol{a}_i)=\prod_{i=1}^{L} \mathcal{N}(\boldsymbol{a}_i \mid \boldsymbol{0}, (\delta_0 \boldsymbol{I}_{L \times L})^{-1}).$$


* $c_0, d_0 \in \mathbb{R}_{>0}$ : known hyperparameters


$$p(\gamma)=\mathrm{Gam}(\gamma \mid c_0, d_0).$$


* $\boldsymbol{\nu}_0 \in \mathbb{R}^{L}$ : a known hyperparameter
* $\tau_0 \in \mathbb{R}_{>0}$ : a known hyperparameter
* $e_0, f_0 \in \mathbb{R}_{>0}$ : known hyperparameters


$$p(\boldsymbol{\mu}_0, \gamma_0)=\mathcal{N}(\boldsymbol{\mu}_0 \mid \boldsymbol{\nu}_0, (\tau_0 \gamma_0 \boldsymbol{I}_{L \times L})^{-1}) \mathrm{Gam}(\gamma_0 \mid e_0, f_0).$$

The approximate posterior distributions $q(\cdot)$ by variational Bayes are as follows:

* $\boldsymbol{m}_{n,i} \in \mathbb{R}^D$ : a hyperparameter ($i=1,2,\ldots,L$)
* $\boldsymbol{\Lambda}_{n,i} \in \mathbb{R}^{D \times D}$ : a hyperparameter ($i=1,2,\ldots,L$)
* $a_n \in \mathbb{R}_{>0}$ : a hyperparameter
* $b_n \in \mathbb{R}_{>0}$ : a hyperparameter
* $\boldsymbol{\eta}_{n,i} \in \mathbb{R}^L$ : a hyperparameter ($i=1,2,\ldots,L$)
* $\boldsymbol{\Xi}_{n,i} \in \mathbb{R}^{L \times L}$ : a hyperparameter ($i=1,2,\ldots,L$)
* $c_n \in \mathbb{R}_{>0}$ : a hyperparameter
* $d_n \in \mathbb{R}_{>0}$ : a hyperparameter
* $\boldsymbol{\nu}_1 \in \mathbb{R}^L$ : a hyperparameter
* $\boldsymbol{\Phi}_1 \in \mathbb{R}^{L \times L}$ : a hyperparameter
* $e_1 \in \mathbb{R}_{>0}$ : a hyperparameter
* $f_1 \in \mathbb{R}_{>0}$ : a hyperparameter
* $\boldsymbol{\mu}_{1} \in \mathbb{R}^L$ : a hyperparameter
* $\boldsymbol{P}_{1} \in \mathbb{R}^{L \times L}$ : a hyperparamete
* $\boldsymbol{\mu}_{n} \in \mathbb{R}^L$: a hyperparameter
* $\boldsymbol{P}_{n} \in \mathbb{R}^{L \times L}$ : a hyperparameter


$$q(\boldsymbol{C})=\prod_{i=1}^{L} q(\boldsymbol{c}_i)=\prod_{i=1}^{L} \mathcal{N}(\boldsymbol{c}_i \mid \boldsymbol{m}_{n,i}, (\boldsymbol{\Lambda}_{n,i})^{-1}),$$
$$q(\lambda)=\mathrm{Gam}(\lambda \mid a_n, b_n),$$
$$q(\boldsymbol{A})=\prod_{i=1}^{L} q(\boldsymbol{a}_i)=\prod_{i=1}^{L} \mathcal{N}(\boldsymbol{a}_i \mid \boldsymbol{\eta}_{n,i}, (\boldsymbol{\Xi}_{n,i})^{-1}),$$
$$q(\gamma)=\mathrm{Gam}(\gamma \mid c_n, d_n),$$
$$q(\boldsymbol{\mu}_0)=\mathcal{N}(\boldsymbol{\mu}_0 \mid \boldsymbol{\nu}_1, (\boldsymbol{\Phi}_1)^{-1}),$$
$$q(\gamma_0)=\mathrm{Gam}(\gamma_0 \mid e_1, f_1),$$
$$q(\boldsymbol{z}_1)=\mathcal{N}(\boldsymbol{z}_1 \mid \boldsymbol{\mu}_1, (\boldsymbol{P}_1)^{-1}),$$
$$q(\boldsymbol{z}_n)=\mathcal{N}(\boldsymbol{z}_n \mid \boldsymbol{\mu}_n, (\boldsymbol{P}_n)^{-1}),$$
where the updating rules of the hyperparameters are as follows:

$$\boldsymbol{\Lambda}_{n,i}=\left(\beta_0 + \frac{a_n}{b_n} \left( \sum_{j=1}^{n} \{(\boldsymbol{P}_j)^{-1}_{ii} + (\boldsymbol{\mu}_j)_i^2\} \right) \right)\boldsymbol{I}_{D \times D},$$
where $(\boldsymbol{P}_j)^{-1}_{ii}$ denotes the $(i,i)$ element of $(\boldsymbol{P}_j)^{-1}$ and $(\boldsymbol{\mu}_j)_i$ denotes the $i$-th component of $\boldsymbol{\mu}_j$.

$$\boldsymbol{m}_{n,i}=(\boldsymbol{\Lambda}_{n,i})^{-1} \frac{a_n}{b_n}  \left(\boldsymbol{X}_n - [\boldsymbol{m}_{n,1}, \ldots, \boldsymbol{m}_{n,i-1}, \boldsymbol{0}, \boldsymbol{m}_{n,i+1}, \ldots, \boldsymbol{m}_{n,L}] [\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_n]\right) [(\boldsymbol{\mu}_1)_i, \ldots, (\boldsymbol{\mu}_n)_i]^\top,$$
where $\boldsymbol{X}_n=[\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n] \in \mathbb{R}^{D \times n}$ and $\boldsymbol{0} \in \mathbb{R}^D$ is zero vector.


$$a_n = a_0 + \frac{nD}{2}$$

$$b_n = b_0 + \frac{1}{2} \sum_{i=1}^{n}(\boldsymbol{x}_i^\top \boldsymbol{x}_i - 2\boldsymbol{x}_i^\top [\boldsymbol{m}_{n,1}, \ldots, \boldsymbol{m}_{n,L}] \boldsymbol{\mu}_i + \boldsymbol{\mu}_i^\top \widetilde{\boldsymbol{C}}_n \boldsymbol{\mu}_i),$$
where $\widetilde{\boldsymbol{C}}_n$ is the matrix whose $(j,j)$ element ($j=1,2,\ldots,L$) is given by
$\mathrm{Tr}\{(\boldsymbol{\Lambda}_{n,j})^{-1}\}+
\boldsymbol{m}_{n,j}^\top \boldsymbol{m}_{n,j}$
and $(j,l)$ element ($j,l=1,2,\ldots,L$) is given by
$\boldsymbol{m}_{n,j}^\top \boldsymbol{m}_{n,l}.$

$$\boldsymbol{\Xi}_{n,i}=\left(\delta_0 + \frac{c_n}{d_n} \left( \sum_{j=1}^{n-1} \{(\boldsymbol{P}_j)^{-1}_{ii} + (\boldsymbol{\mu}_j)_i^2\} \right) \right)\boldsymbol{I}_{L \times L}$$


$$\boldsymbol{\eta}_{n,i}=(\boldsymbol{\Xi}_{n,i})^{-1} \frac{c_n}{d_n}  \left([\boldsymbol{\mu}_2,\ldots, \boldsymbol{\mu}_{n}] - [\boldsymbol{\eta}_{n,1}, \ldots, \boldsymbol{\eta}_{n,i-1}, \boldsymbol{0}, \boldsymbol{\eta}_{n,i+1}, \ldots, \boldsymbol{\eta}_{n,L}] [\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_{n-1}]\right)[(\boldsymbol{\mu}_1)_i, \ldots, (\boldsymbol{\mu}_{n-1})_i]^\top,$$
where $\boldsymbol{0} \in \mathbb{R}^L$ is zero vector.

$$c_n = c_0 + \frac{(n-1)L}{2}$$

$$d_n = d_0 + \frac{1}{2} \sum_{i=2}^{n} \left(\mathrm{Tr}\{(\boldsymbol{P}_i)^{-1}\} +\boldsymbol{\mu}_{i}^\top \boldsymbol{\mu}_{i} - 2\boldsymbol{\mu}_{i}^\top [\boldsymbol{\eta}_{n,1}, \ldots, \boldsymbol{\eta}_{n,L}] \boldsymbol{\mu}_{i-1} + \boldsymbol{\mu}_{i-1}^\top \widetilde{\boldsymbol{A}}_n\boldsymbol{\mu}_{i-1} \right),$$
where $\widetilde{\boldsymbol{A}}_n$ is the matrix whose $(j,j)$ element ($j=1,2,\ldots,L$) is given by
$\mathrm{Tr}\{(\boldsymbol{\Xi}_{n,j})^{-1}\}+
\boldsymbol{\eta}_{n,j}^\top \boldsymbol{\eta}_{n,j}$
and $(j,l)$ element ($j,l=1,2,\ldots,L$) is given by
$\boldsymbol{\eta}_{n,j}^\top \boldsymbol{\eta}_{n,l}.$

$$\boldsymbol{\nu}_1 = \frac{\tau_0 \boldsymbol{\nu}_0 + \boldsymbol{\mu}_1}{\tau_0 + 1}$$

$$\boldsymbol{\Phi}_1 =(\tau_0 + 1)\frac{e_1}{f_1} \boldsymbol{I}_{L \times L}$$

$$e_1 = e_0 + L$$

$$f_1 = f_0 + \frac{\tau_0}{2}(\mathrm{Tr}\{(\boldsymbol{\Phi}_1)^{-1}\}+\boldsymbol{\nu}_1^\top \boldsymbol{\nu}_1 - 2\boldsymbol{\nu}_0^\top \boldsymbol{\nu}_1 +\boldsymbol{\nu}_0^\top \boldsymbol{\nu}_0) + \frac{1}{2}(\mathrm{Tr}\{(\boldsymbol{P}_1)^{-1}\}+\boldsymbol{\mu}_1^\top \boldsymbol{\mu}_1 - 2\boldsymbol{\nu}_1^\top \boldsymbol{\mu}_1 + \mathrm{Tr}\{(\boldsymbol{\Phi}_1)^{-1}\}+\boldsymbol{\nu}_1^\top \boldsymbol{\nu}_1)$$


$$\boldsymbol{P}_1=\frac{e_1}{f_1}\boldsymbol{I}_{L \times L}+\frac{a_1}{b_1} \widetilde{\boldsymbol{C}}_1$$

$$\boldsymbol{\mu}_1=(\boldsymbol{P}_1)^{-1} \left(\frac{e_1}{f_1}\boldsymbol{\nu}_1+\frac{a_1}{b_1} [\boldsymbol{m}_{1,1}, \ldots, \boldsymbol{m}_{1,L}]^\top \boldsymbol{x}_1 \right)$$

$$\boldsymbol{P}_n=\frac{c_n}{d_n}\boldsymbol{I}_{L \times L}+\frac{a_n}{b_n} \widetilde{\boldsymbol{C}}_n$$

$$\boldsymbol{\mu}_n=(\boldsymbol{P}_n)^{-1} \left(\frac{c_n}{d_n}[\boldsymbol{\eta}_{n,1}, \ldots, \boldsymbol{\eta}_{n,L}]\boldsymbol{\mu}_{n-1}+\frac{a_n}{b_n}[\boldsymbol{m}_{n,1}, \ldots, \boldsymbol{m}_{n,L}]^\top \boldsymbol{x}_n \right)$$

Let 
$$\boldsymbol{X}_n = [\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n], $$
$$\boldsymbol{Z}_n = [\boldsymbol{z}_1, \ldots, \boldsymbol{z}_n], $$
$$\boldsymbol{\theta}=\{\boldsymbol{C},\lambda, \boldsymbol{A}, \gamma, \boldsymbol{\mu}_0, \gamma_0\}.$$

The variational lower bound is 

$$\mathbb{E}_{q(\boldsymbol{Z}_n, \boldsymbol{\theta})}[\ln p(\boldsymbol{X}_n, \boldsymbol{Z}_n, \boldsymbol{\theta})] - \mathbb{E}_{q(\boldsymbol{Z}_n, \boldsymbol{\theta})}[\ln q(\boldsymbol{Z}_n, \boldsymbol{\theta})] 
    =\mathbb{E}[\ln p(\boldsymbol{C})]+\mathbb{E}[\ln p(\lambda)]+\mathbb{E}[\ln p(\boldsymbol{A})]+\mathbb{E}[\ln p(\gamma)]+\mathbb{E}[\ln p(\boldsymbol{\mu}_0, \gamma_0)] 
    +\mathbb{E}[\ln p(\boldsymbol{z}_1 \mid \boldsymbol{\mu}_{0}, \gamma_0)]+\sum_{i=2}^n \mathbb{E}[\ln p(\boldsymbol{z}_i \mid \boldsymbol{z}_{i-1}, \boldsymbol{A}, \gamma)]+\sum_{j=1}^n \mathbb{E}[\ln p(\boldsymbol{x}_j \mid \boldsymbol{z}_j, \boldsymbol{C}, \lambda)] 
    -\sum_{k=1}^n \mathbb{E}[\ln q(\boldsymbol{z}_k)]-\mathbb{E}[\ln q(\boldsymbol{C})]-\mathbb{E}[\ln q(\lambda)]-\mathbb{E}[\ln q(\boldsymbol{A})]-\mathbb{E}[\ln q(\gamma)] 
    -\mathbb{E}[\ln q(\boldsymbol{\mu}_0)]-\mathbb{E}[\ln q(\gamma_0)],$$
where each term is given as follows:

$$\mathbb{E}[\ln p(\boldsymbol{C})]=\frac{DL}{2} \ln \left(\frac{\beta_0}{2 \pi} \right) - \frac{\beta_0}{2} \sum_{j=1}^L (\mathrm{Tr}\{(\boldsymbol{\Lambda}_{n,j})^{-1}\}+
\boldsymbol{m}_{n,j}^\top \boldsymbol{m}_{n,j})$$


$$\mathbb{E}[\ln p(\lambda)]=\ln \left(\frac{b_0^{a_0}}{\Gamma(a_0)} \right)+ (a_0 -1)(\psi(a_n) - \ln b_n)-\frac{b_0 a_n}{b_n}, $$
where $\Gamma(\cdot)$ is the gamma function and $\psi(\cdot)$ is the digamma function.

$$\mathbb{E}[\ln p(\boldsymbol{A})]=\frac{L^2}{2} \ln \left(\frac{\delta_0}{2 \pi} \right) - \frac{\delta_0}{2} \sum_{j=1}^L (\mathrm{Tr}\{(\boldsymbol{\Xi}_{n,j})^{-1}\}+
\boldsymbol{\eta}_{n,j}^\top \boldsymbol{\eta}_{n,j})$$

$$\mathbb{E}[\ln p(\gamma)]=\ln \left(\frac{d_0^{c_0}}{\Gamma(c_0)} \right)+ (c_0 -1)(\psi(c_n) - \ln d_n)-\frac{d_0 c_n}{d_n}$$

$$\mathbb{E}[\ln p(\boldsymbol{\mu}_0, \gamma_0)]=\frac{L}{2} \ln \left(\frac{\tau_0}{2 \pi} \right)+\ln \left(\frac{f_0^{e_0}}{\Gamma(e_0)} \right)-\frac{f_0 e_1}{f_1}+\left(\frac{L}{2}+e_0 -1 \right)(\psi(e_1) - \ln f_1) 
     - \frac{\tau_0 e_1}{2 f_1}(\mathrm{Tr}\{(\boldsymbol{\Phi}_1)^{-1}\}+\boldsymbol{\nu}_1^\top \boldsymbol{\nu}_1 - 2\boldsymbol{\nu}_0^\top \boldsymbol{\nu}_1 +\boldsymbol{\nu}_0^\top \boldsymbol{\nu}_0)$$

$$\mathbb{E}[\ln p(\boldsymbol{z}_1 \mid \boldsymbol{\mu}_{0}, \gamma_0)]=\frac{L}{2} \ln \left(\frac{1}{2 \pi} \right)+\frac{L}{2}(\psi(e_1) - \ln f_1)
    -\frac{e_1}{2 f_1}(\mathrm{Tr}\{(\boldsymbol{P}_1)^{-1}\}+\boldsymbol{\mu}_1^\top \boldsymbol{\mu}_1 - 2\boldsymbol{\nu}_1^\top \boldsymbol{\mu}_1 + \mathrm{Tr}\{(\boldsymbol{\Phi}_1)^{-1}\}+\boldsymbol{\nu}_1^\top \boldsymbol{\nu}_1)$$

$$\mathbb{E}[\ln p(\boldsymbol{z}_i \mid \boldsymbol{z}_{i-1}, \boldsymbol{A}, \gamma)]=\frac{L}{2} \ln \left(\frac{1}{2 \pi} \right)+\frac{L}{2}(\psi(c_n) - \ln d_n)
    -\frac{c_n}{2 d_n}\left(\mathrm{Tr}\{(\boldsymbol{P}_i)^{-1}\} +\boldsymbol{\mu}_{i}^\top \boldsymbol{\mu}_{i} - 2\boldsymbol{\mu}_{i}^\top [\boldsymbol{\eta}_{n,1}, \ldots, \boldsymbol{\eta}_{n,L}] \boldsymbol{\mu}_{i-1} + \boldsymbol{\mu}_{i-1}^\top \widetilde{\boldsymbol{A}}_n\boldsymbol{\mu}_{i-1} \right)$$

$$\mathbb{E}[\ln p(\boldsymbol{x}_j \mid \boldsymbol{z}_j, \boldsymbol{C}, \lambda)]=\frac{D}{2} \ln \left(\frac{1}{2 \pi} \right)+\frac{D}{2}(\psi(a_n) - \ln b_n)
   -\frac{a_n}{2 b_n}(\boldsymbol{x}_j^\top \boldsymbol{x}_j - 2\boldsymbol{x}_j^\top [\boldsymbol{m}_{n,1}, \ldots, \boldsymbol{m}_{n,L}] \boldsymbol{\mu}_j + \boldsymbol{\mu}_j^\top \widetilde{\boldsymbol{C}}_n \boldsymbol{\mu}_j)$$

$$\mathbb{E}[\ln q(\boldsymbol{z}_k)]=\frac{1}{2}\ln \frac{|\boldsymbol{P}_k|}{(2 \pi)^L} - \frac{L}{2}$$

$$\mathbb{E}[\ln q(\boldsymbol{C})]=\sum_{i=1}^L \left( \frac{1}{2}\ln \frac{|\boldsymbol{\Lambda}_{n,i}|}{(2 \pi)^D} - \frac{D}{2} \right)$$

$$\mathbb{E}[\ln q(\lambda)]=\ln \left(\frac{b_n^{a_n}}{\Gamma(a_n)} \right)+(a_n -1)(\psi(a_n)-\ln b_n )-a_n$$

$$\mathbb{E}[\ln q(\boldsymbol{A})]=\sum_{i=1}^L \left( \frac{1}{2}\ln \frac{|\boldsymbol{\Xi}_{n,i}|}{(2 \pi)^L} - \frac{L}{2} \right)$$

$$\mathbb{E}[\ln q(\gamma)]=\ln \left(\frac{d_n^{c_n}}{\Gamma(c_n)} \right)+(c_n -1)(\psi(c_n)-\ln d_n )-c_n$$


$$\mathbb{E}[\ln q(\boldsymbol{\mu}_0)]=\frac{1}{2}\ln \frac{|\boldsymbol{\Phi}_1|}{(2 \pi)^L} - \frac{L}{2}$$

$$\mathbb{E}[\ln q(\gamma_0)]=\ln \left(\frac{f_1^{e_1}}{\Gamma(e_1)} \right)+(e_1 -1)(\psi(e_1)-\ln f_1 )-e_1$$
