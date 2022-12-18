# Code Author
# Noboru Namegaya <n.noboru20180403@toki.waseda.jp>
# Koshi Shimada <shimada.koshi.re@gmail.com>
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Document Author
# Noboru Namegaya <n.noboru20180403@toki.waseda.jp>
# Koshi Shimada <shimada.koshi.re@gmail.com>
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
import warnings
import numpy as np
from scipy.stats import norm as ss_norm
from scipy.stats import gamma as ss_gamma 
from scipy.stats import t as ss_t
import matplotlib.pyplot as plt

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning, ParameterFormatWarning
from .. import _check

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution

    Parameters
    ----------
    mu : float, optional
        a real number, by default 0.0
    tau : float, optional
        a positibe real number, by default 1.0
    h_m : float, optional
        a real number, by default 0.0
    h_kappa : float, optional
        a positibe real number, by default 1.0
    h_alpha : float, optional
        a positibe real number, by default 1.0
    h_beta : float, optional
        a positibe real number, by default 1.0
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(self,*,mu=0.0,tau=1.0,h_m=0.0,h_kappa=1.0,h_alpha=1.0,h_beta=1.0,seed=None):
        self.mu = _check.float_(mu,'mu',ParameterFormatError)
        self.tau = _check.pos_float(tau,'tau',ParameterFormatError)

        self.h_m = _check.float_(h_m,'h_m',ParameterFormatError)
        self.h_kappa = _check.pos_float(h_kappa,'h_kappa',ParameterFormatError)
        self.h_alpha = _check.pos_float(h_alpha,'h_alpha',ParameterFormatError)
        self.h_beta = _check.pos_float(h_beta,'h_beta',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

    def set_h_params(self,h_m,h_kappa,h_alpha,h_beta):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_m : float
            a real number
        h_kappa : float
            a positibe real number
        h_alpha : float
            a positibe real number
        h_beta : float
            a positibe real number
        """
        self.h_m = _check.float_(h_m,'h_m',ParameterFormatError)
        self.h_kappa = _check.pos_float(h_kappa,'h_kappa',ParameterFormatError)
        self.h_alpha = _check.pos_float(h_alpha,'h_alpha',ParameterFormatError)
        self.h_beta = _check.pos_float(h_beta,'h_beta',ParameterFormatError)

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float}
            * ``"h_m"`` : The value of ``self.h_m``
            * ``"h_kappa"`` : The value of ``self.h_kappa``
            * ``"h_alpha"`` : The value of ``self.h_alpha``
            * ``"h_beta"`` : The value of ``self.h_beta``
        """
        return {"h_m":self.h_m, "h_kappa":self.h_kappa, "h_alpha":self.h_alpha, "h_beta":self.h_beta}
    
    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        The generated vaule is set at ``self.mu`` and ``self.tau``.
        """
        self.tau = self.rng.gamma(shape=self.h_alpha,scale=1.0/self.h_beta)
        self.mu = self.rng.normal(loc=self.h_m,scale=1.0/np.sqrt(self.tau * self.h_kappa))
        
    def set_params(self,mu,tau):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        mu : float
            a real number :math:`mu \in \mathbb{R}`
        tau : float, optional
            a positibe real number
        """
        self.mu = _check.float_(mu,'mu',ParameterFormatError)
        self.tau = _check.pos_float(tau,'tau',ParameterFormatError)

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:float}
            * ``"mu"`` : The value of ``self.mu``
            * ``"tau"`` : The value of ``self.tau``
        """
        return {"mu":self.mu, "tau":self.tau}

    def gen_sample(self,sample_size):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int
            A positive integer

        Returns
        -------
        x : numpy ndarray
            1 dimensional array whose size is ``sammple_size``.
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        return self.rng.normal(loc=self.mu,scale=1.0/np.sqrt(self.tau),size=sample_size)
        
    def save_sample(self,filename,sample_size):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as a NpzFile with keyword: \"x\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_size : int
            A positive integer
        
        See Also
        --------
        numpy.savez_compressed
        """
        np.savez_compressed(filename,x=self.gen_sample(sample_size))

    def visualize_model(self,sample_size=1000,hist_bins=10):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 20
        hist_bins : float, optional
            A positive float, by default 10

        Examples
        --------
        >>> from bayesml import normal
        >>> model = normal.GenModel()
        >>> model.visualize_model()

        .. image:: ./images/normal_example.png
        """
        
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        _check.pos_int(hist_bins, 'hist_bins', DataFormatError)
        fig, ax = plt.subplots()
        sample = self.gen_sample(sample_size)

        plt.title(f"PDF and normalized histogram")
        ax.hist(sample,density=True,label=f"normalized hist n={sample_size}",bins=hist_bins)

        x = np.linspace(sample.min()-(sample.max()-sample.min())*0.25,
                        sample.max()+(sample.max()-sample.min())*0.25,
                        100)
        y = ss_norm.pdf(x,self.mu,1.0/np.sqrt(self.tau))
        plt.plot(x, y, label=f"Normal PDF mu={self.mu}, tau={self.tau}")
        ax.set_xlabel("Realization")
        ax.set_ylabel("Probability or frequency")
        
        plt.legend()
        plt.show()

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    h0_m : float, optional
        a real number, by default 0.0
    h0_kappa : float, optional
        a positive real number, by default 1.0
    h0_alpha : float, optional
        a positive real number, by default 1.0
    h0_beta : float, optional
        a positive real number, by default 1.0
    
    Attributes
    ----------
    hn_m : float
        a real number
    hn_kappa : float
        a positive real number
    hn_alpha : float
        a positive real number
    hn_beta : float
        a positive real number
    p_mu : float
        a real number
    p_lambda : float
        a positibe real number
    p_nu : float
        a positibe real number
    """

    def __init__(self,h0_m=0.0,h0_kappa=1.0,h0_alpha=1.0,h0_beta=1.0):
        self.h0_m = _check.float_(h0_m,'h0_m',ParameterFormatError)
        self.h0_kappa = _check.pos_float(h0_kappa,'h0_kappa',ParameterFormatError)
        self.h0_alpha = _check.pos_float(h0_alpha,'h0_alpha',ParameterFormatError)
        self.h0_beta = _check.pos_float(h0_beta,'h0_beta',ParameterFormatError)

        self.hn_m = self.h0_m
        self.hn_kappa = self.h0_kappa 
        self.hn_alpha = self.h0_alpha
        self.hn_beta = self.h0_beta

        self.p_mu = self.hn_m
        self.p_nu = 2*self.hn_alpha
        self.p_lambda = self.hn_kappa / (self.hn_kappa+1) * self.hn_alpha / self.hn_beta

    def set_h0_params(self,h0_m,h0_kappa,h0_alpha,h0_beta):
        """Set the hyperparameters of the prior distribution.
        
        Parameters
        ----------
        h0_m : float
            a real number
        h0_kappa : float
            a positive real number
        h0_alpha : float
            a positive real number
        h0_beta : float
            a positive real number
        """
        self.h0_m = _check.float_(h0_m,'h0_m',ParameterFormatError)
        self.h0_kappa = _check.pos_float(h0_kappa,'h0_kappa',ParameterFormatError)
        self.h0_alpha = _check.pos_float(h0_alpha,'h0_alpha',ParameterFormatError)
        self.h0_beta = _check.pos_float(h0_beta,'h0_beta',ParameterFormatError)
        self.reset_hn_params()

    def get_h0_params(self):
        """Get the initial values of the hyperparameters of the posterior distribution.

        Returns
        -------
        h0_params : dict of {str: float}
            * ``"h0_m"`` : The value of ``self.h0_m``
            * ``"h0_kappa"`` : The value of ``self.h0_kappa``
            * ``"h0_alpha"`` : The value of ``self.h0_alpha``
            * ``"h0_beta"`` : The value of ``self.h0_beta``
        """
        return {"h0_m":self.h0_m, "h0_kappa":self.h0_kappa, "h0_alpha":self.h0_alpha, "h0_beta":self.h0_beta}
    
    def set_hn_params(self,hn_m,hn_kappa,hn_alpha,hn_beta):
        """Set updated values of the hyperparameter of the posterior distribution.
        
        Parameters
        ----------
        hn_m : float
            a real number
        hn_kappa : float
            a positive real number
        hn_alpha : float
            a positive real number
        hn_beta : float
            a positive real number
        """
        self.hn_m = _check.float_(hn_m,'hn_m',ParameterFormatError)
        self.hn_kappa = _check.pos_float(hn_kappa,'hn_kappa',ParameterFormatError)
        self.hn_alpha = _check.pos_float(hn_alpha,'hn_alpha',ParameterFormatError)
        self.hn_beta = _check.pos_float(hn_beta,'hn_beta',ParameterFormatError)
        self.calc_pred_dist()

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: float}
            * ``"hn_m"`` : The value of ``self.hn_m``
            * ``"hn_kappa"`` : The value of ``self.hn_kappa``
            * ``"hn_alpha"`` : The value of ``self.hn_alpha``
            * ``"hn_beta"`` : The value of ``self.hn_beta``
        """
        return {"hn_m":self.hn_m, "hn_kappa":self.hn_kappa, "hn_alpha":self.hn_alpha, "hn_beta":self.hn_beta}
    
    def reset_hn_params(self):
        """Reset the hyperparameters of the posterior distribution to their initial values.
        
        They are reset to `self.h0_m`, `self.h0_kappa`, `self.h0_alpha`, and `self.h0_beta`.
        Note that the parameters of the predictive distribution are also calculated from 
        `self.h0_m`, `self.h0_kappa`, `self.h0_alpha`, and `self.h0_beta`.
        """
        self.hn_m = self.h0_m
        self.hn_kappa = self.h0_kappa 
        self.hn_alpha = self.h0_alpha
        self.hn_beta = self.h0_beta
        self.calc_pred_dist()

    def overwrite_h0_params(self):
        """Overwrite the initial values of the hyperparameters of the posterior distribution by the learned values.
        
        They are overwritten by `self.hn_m`, `self.hn_kappa`, `self.hn_alpha`, and `self.hn_beta`.
        Note that the parameters of the predictive distribution are also calculated from 
        `self.hn_m`, `self.hn_kappa`, `self.hn_alpha`, and `self.hn_beta`.
        """
        self.h0_m = self.hn_m
        self.h0_kappa = self.hn_kappa 
        self.h0_alpha = self.hn_alpha
        self.h0_beta = self.hn_beta

    def update_posterior(self,x):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy.ndarray
            All the elements must be natural number.
        """
        _check.floats(x,'x',DataFormatError)
        n=x.size
        x_bar = np.sum(x) / n

        self.hn_beta += (np.sum((x-x_bar)**2) + n*self.hn_kappa / (self.hn_kappa + n) * (x_bar - self.hn_m)**2 ) / 2.0
        self.hn_m =  (self.hn_kappa * self.hn_m + n * x_bar) / (self.hn_kappa + n)
        self.hn_kappa += n
        self.hn_alpha += n*0.5

    def estimate_params(self,loss="squared",dict_out=False):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Note that the criterion is applied to estimating ``mu`` and ``tau`` independently.
        Therefore, a tuple of the student's t-distribution and the gamma distribution will be returned when loss=\"KL\"

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".
        dict_out : bool, optional
            If ``True``, output will be a dict, by default ``False``.
        
        Returns
        -------
        estimates : tuple of {float, None, or rv_frozen}
            * ``mu_hat`` : the estimate for mu
            * ``tau_hat`` : the estimate for tau
            The estimated values under the given loss function. If it is not exist, `None` will be returned.
            If the loss function is \"KL\", the posterior distribution itself will be returned
            as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """
        if loss == "squared":
            if dict_out:
                return {'mu':self.hn_m,'tau':self.hn_alpha/self.hn_beta}
            else:
                return self.hn_m, self.hn_alpha/self.hn_beta
        elif loss == "0-1":
            if self.hn_alpha > 1.0:
                if dict_out:
                    return {'mu':self.hn_m,'tau':(self.hn_alpha - 1.0)  / self.hn_beta}
                else:
                    return self.hn_m, (self.hn_alpha - 1.0)  / self.hn_beta
            else:
                if dict_out:
                    return {'mu':self.hn_m,'tau':0.0}
                else:
                    return self.hn_m, 0.0
        elif loss == "abs":
            if dict_out:
                return {'mu':self.hn_m,'tau':ss_gamma.median(a=self.hn_alpha,scale=1/self.hn_beta)}
            else:
                return self.hn_m, ss_gamma.median(a=self.hn_alpha,scale=1/self.hn_beta)
        elif loss == "KL":
            return ss_t(loc=self.hn_m,scale=np.sqrt(self.hn_beta / self.hn_alpha / self.hn_kappa),df=2*self.hn_alpha),ss_gamma(a=self.hn_alpha,scale=1.0/self.hn_beta)
        else:
            raise(CriteriaError("Unsupported loss function! "
                                "This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))

    def estimate_interval(self,credibility=0.95):
        """Credible interval of the parameter.

        Parameters
        ----------
        credibility : float, optional
            A posterior probability that the interval conitans the paramter, by default 0.95

        Returns
        -------
        (mu_lower, mu_upper),(tau_lower, tau_upper): float
            The lower and the upper bounds of the intervals
        """
        _check.float_in_closed01(credibility,'credibility',CriteriaError)
        return (ss_t.interval(credibility,
                              loc=self.hn_m,
                              scale=np.sqrt(self.hn_beta / self.hn_alpha / self.hn_kappa),
                              df=2*self.hn_alpha),
                ss_gamma.interval(credibility,a=self.hn_alpha,scale=1.0/self.hn_beta))
    
    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import normal
        >>> gen_model = normal.GenModel(mu=1.0,tau=1.0)
        >>> learn_model = normal.LearnModel()
        >>> learn_model.update_posterior(x)
        >>> learn_model.visualize_posterior()
        
        .. image:: ./images/normal_posterior.png
        """
        mu_pdf, tau_pdf = self.estimate_params(loss="KL")
        fig, axes = plt.subplots(1,2)
        # for mu
        x = np.linspace(self.hn_m-4.0*np.sqrt(self.hn_beta / self.hn_alpha / self.hn_kappa),
                        self.hn_m+4.0*np.sqrt(self.hn_beta / self.hn_alpha / self.hn_kappa),
                        100)
        axes[0].plot(x,mu_pdf.pdf(x))
        axes[0].set_xlabel("mu")
        axes[0].set_ylabel("Density")
        #for tau
        x = np.linspace(max(1.0e-8,self.hn_alpha/self.hn_beta-4.0*np.sqrt(self.hn_alpha)/self.hn_beta),
                        self.hn_alpha/self.hn_beta+4.0*np.sqrt(self.hn_alpha)/self.hn_beta,
                        100)
        axes[1].plot(x,tau_pdf.pdf(x))
        axes[1].set_xlabel("tau")
        axes[1].set_ylabel("Density")

        fig.tight_layout()
        plt.show()
    
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: float}
            * ``"p_mu"`` : The value of ``self.p_mu``
            * ``"p_lambda"`` : The value of ``self.p_lambda``
            * ``"p_nu"`` : The value of ``self.p_nu``
        """
        return {"p_mu":self.p_mu, "p_lambda":self.p_lambda, "p_nu":self.p_nu}
    
    def calc_pred_dist(self):
        """Calculate the parameters of the predictive distribution."""
        self.p_mu = self.hn_m
        self.p_nu = 2*self.hn_alpha
        self.p_lambda = self.hn_kappa / (self.hn_kappa+1) * self.hn_alpha / self.hn_beta

    def _calc_pred_density(self,x):
        return ss_t.pdf(x,loc=self.p_mu,scale=1.0/np.sqrt(self.p_lambda),df=self.p_nu)

    def make_prediction(self,loss="squared"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".
        
        Returns
        -------
        Predicted_value : {int, numpy.ndarray}
            The predicted value under the given loss function. 
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as numpy.ndarray.
        """
        if loss == "squared" or loss == "0-1" or loss == "abs":
            return self.p_mu
        elif loss == "KL":
            return ss_t(loc=self.p_mu,scale=1.0/np.sqrt(self.p_lambda),df=self.p_nu)
        else:
            raise(CriteriaError("Unsupported loss function! "
                                "This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))

    def pred_and_update(self,x,loss="squared"):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x : int
            It must be natural number
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".
        
        Returns
        -------
        Predicted_value : {int, numpy.ndarray}
            The predicted value under the given loss function. 
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as numpy.ndarray.
        """
        self.calc_pred_dist()
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(x)
        return prediction
