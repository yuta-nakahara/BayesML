# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Document Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
import warnings
import numpy as np
import os
import sys
from scipy.stats import beta as ss_beta
# from scipy.stats import betabino as ss_betabinom
import matplotlib.pyplot as plt

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning
from .. import _check

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution.

    Parameters
    ----------
    p : float, optional
        a real number in :math:`[0, 1]`, by default 0.5
    h_alpha : float, optional
        a positive real number, by default 0.5
    h_beta : float, optional
        a positibe real number, by default 0.5
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(self,*,p=0.5,h_alpha=0.5,h_beta=0.5,seed=None):
        self.p = _check.float_in_closed01(p,'p',ParameterFormatError)
        self.h_alpha = _check.pos_float(h_alpha,'h_alpha',ParameterFormatError)
        self.h_beta = _check.pos_float(h_beta,'h_beta',ParameterFormatError)
        self.rng = np.random.default_rng(seed)
        self._H_PARAM_KEYS = {'h_alpha','h_beta'}
        self._H0_PARAM_KEYS = {'h0_alpha','h0_beta'}
        self._HN_PARAM_KEYS = {'hn_alpha','hn_beta'}

    def set_h_params(self,**kwargs):
        """Set the hyperparameters of the prior distribution.
        
        Parameters
        ----------
        **kwargs
            a python dictionary {'h_alpha':float, 'h_beta':float} or
            {'h0_alpha':float, 'h0_beta':float} or {'hn_alpha':float, 'hn_beta':float}
            They are obtained by ``get_h_params()`` of GenModel,
            ``get_h0_params`` of LearnModel or ``get_hn_params`` of LearnModel.
        """
        if kwargs.keys() == self._H_PARAM_KEYS:
            self.h_alpha = _check.pos_float(kwargs['h_alpha'],'h_alpha',ParameterFormatError)
            self.h_beta = _check.pos_float(kwargs['h_beta'],'h_beta',ParameterFormatError)
        elif kwargs.keys() == self._H0_PARAM_KEYS:
            self.h_alpha = _check.pos_float(kwargs['h0_alpha'],'h_alpha',ParameterFormatError)
            self.h_beta = _check.pos_float(kwargs['h0_beta'],'h_beta',ParameterFormatError)
        elif kwargs.keys() == self._HN_PARAM_KEYS:
            self.h_alpha = _check.pos_float(kwargs['hn_alpha'],'h_alpha',ParameterFormatError)
            self.h_beta = _check.pos_float(kwargs['hn_beta'],'h_beta',ParameterFormatError)
        else:
            raise(ParameterFormatError(
                "The input of this function must be a python dictionary with keys:"
                +str(self._H_PARAM_KEYS)+" or "
                +str(self._H0_PARAM_KEYS)+" or "
                +str(self._HN_PARAM_KEYS)+".")
                )

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float}
            * ``"h_alpha"`` : The value of ``self.h_alpha``
            * ``"h_beta"`` : The value of ``self.h_beta``
        """
        return {"h_alpha":self.h_alpha, "h_beta":self.h_beta}

    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        The generated vaule is set at ``self.p``.
        """
        self.p = self.rng.beta(self.h_alpha,self.h_beta)
        
    def set_params(self,p):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        p : float
            a real number :math:`p \in [0, 1]`
        """
        self.p = _check.float_in_closed01(p,'p',ParameterFormatError)

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:float}
            * ``"p"`` : The value of ``self.p``.
        """
        return {"p":self.p}

    def gen_sample(self,sample_size):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int
            A positive integer

        Returns
        -------
        x : numpy ndarray
            1 dimensional array whose size is ``sammple_size`` and elements are 0 or 1.
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        return self.rng.binomial(1,self.p,sample_size)
        
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

    def visualize_model(self,sample_size=20,sample_num=5):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 20
        sample_num : int, optional
            A positive integer, by default 5
        
        Examples
        --------
        >>> from bayesml import bernoulli
        >>> model = bernoulli.GenModel()
        >>> model.visualize_model()
        p:0.5
        x0:[0 0 1 1 1 1 1 1 1 0 1 0 1 1 1 0 1 0 1 0]
        x1:[1 0 0 1 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 1]
        x2:[1 1 1 1 0 1 0 1 0 0 0 1 1 0 0 1 1 1 0 1]
        x3:[0 0 1 1 1 0 0 1 1 0 0 1 0 0 1 0 0 1 0 1]
        x4:[0 1 0 1 1 0 1 0 1 1 1 1 1 0 1 0 0 1 1 0]

        .. image:: ./images/bernoulli_example.png
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        _check.pos_int(sample_num,'sample_num',DataFormatError)
        print(f"p:{self.p}")
        fig, ax = plt.subplots(figsize=(5,sample_num))
        for i in range(sample_num):
            x = self.gen_sample(sample_size)
            print(f"x{i}:{x}")
            if i == 0:
                ax.barh(i,x.sum(),label=1,color="C0")
                ax.barh(i,sample_size-x.sum(),left=x.sum(),label=0,color="C1")
            else:
                ax.barh(i,x.sum(),color="C0")
                ax.barh(i,sample_size-x.sum(),left=x.sum(),color="C1")
        ax.legend()
        ax.set_xlabel("Number of occurrences")
        plt.show()

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    h0_alpha : float, optional
        a positive real number, by default 0.5
    h0_beta : float, optional
        a positibe real number, by default 0.5
    
    Attributes
    ----------
    hn_alpha : float
        a positive real number
    hn_beta : float
        a positibe real number
    p_alpha : float
        a positive real number
    p_beta : float
        a positibe real number
    """
    def __init__(self,h0_alpha=0.5,h0_beta=0.5):
        self.h0_alpha = _check.pos_float(h0_alpha,'h0_alpha',ParameterFormatError)
        self.h0_beta = _check.pos_float(h0_beta,'h0_beta',ParameterFormatError)
        self.hn_alpha = self.h0_alpha
        self.hn_beta = self.h0_beta
        self.p_alpha = self.hn_alpha
        self.p_beta = self.hn_beta
        self._H_PARAM_KEYS = {'h_alpha','h_beta'}
        self._H0_PARAM_KEYS = {'h0_alpha','h0_beta'}
        self._HN_PARAM_KEYS = {'hn_alpha','hn_beta'}
    
    def set_h0_params(self,**kwargs):
        """Set initial values of the hyperparameter of the posterior distribution.
        
        Parameters
        ----------
        **kwargs
            a python dictionary {'h_alpha':float, 'h_beta':float} or
            {'h0_alpha':float, 'h0_beta':float} or {'hn_alpha':float, 'hn_beta':float}
            They are obtained by ``get_h_params()`` of GenModel,
            ``get_h0_params`` of LearnModel or ``get_hn_params`` of LearnModel.
        """
        if kwargs.keys() == self._H_PARAM_KEYS:
            self.h0_alpha = _check.pos_float(kwargs['h_alpha'],'h0_alpha',ParameterFormatError)
            self.h0_beta = _check.pos_float(kwargs['h_beta'],'h0_beta',ParameterFormatError)
        elif kwargs.keys() == self._H0_PARAM_KEYS:
            self.h0_alpha = _check.pos_float(kwargs['h0_alpha'],'h0_alpha',ParameterFormatError)
            self.h0_beta = _check.pos_float(kwargs['h0_beta'],'h0_beta',ParameterFormatError)
        elif kwargs.keys() == self._HN_PARAM_KEYS:
            self.h0_alpha = _check.pos_float(kwargs['hn_alpha'],'h0_alpha',ParameterFormatError)
            self.h0_beta = _check.pos_float(kwargs['hn_beta'],'h0_beta',ParameterFormatError)
        else:
            raise(ParameterFormatError(
                "The input of this function must be a python dictionary with keys:"
                +str(self._H_PARAM_KEYS)+" or "
                +str(self._H0_PARAM_KEYS)+" or "
                +str(self._HN_PARAM_KEYS)+".")
                )
        self.reset_hn_params()

    def get_h0_params(self):
        """Get the initial values of the hyperparameters of the posterior distribution.

        Returns
        -------
        h0_params : dict of {str: float}
            * ``"h0_alpha"`` : The value of ``self.h0_alpha``
            * ``"h0_beta"`` : The value of ``self.h0_beta``
        """
        return {"h0_alpha":self.h0_alpha, "h0_beta":self.h0_beta}
    
    def set_hn_params(self,**kwargs):
        """Set updated values of the hyperparameter of the posterior distribution.
        
        Parameters
        ----------
        **kwargs
            a python dictionary {'h_alpha':float, 'h_beta':float} or
            {'h0_alpha':float, 'h0_beta':float} or {'hn_alpha':float, 'hn_beta':float}
            They are obtained by ``get_h_params()`` of GenModel,
            ``get_h0_params`` of LearnModel or ``get_hn_params`` of LearnModel.
        """
        if kwargs.keys() == self._H_PARAM_KEYS:
            self.hn_alpha = _check.pos_float(kwargs['h_alpha'],'hn_alpha',ParameterFormatError)
            self.hn_beta = _check.pos_float(kwargs['h_beta'],'hn_beta',ParameterFormatError)
        elif kwargs.keys() == self._H0_PARAM_KEYS:
            self.hn_alpha = _check.pos_float(kwargs['h0_alpha'],'hn_alpha',ParameterFormatError)
            self.hn_beta = _check.pos_float(kwargs['h0_beta'],'hn_beta',ParameterFormatError)
        elif kwargs.keys() == self._HN_PARAM_KEYS:
            self.hn_alpha = _check.pos_float(kwargs['hn_alpha'],'hn_alpha',ParameterFormatError)
            self.hn_beta = _check.pos_float(kwargs['hn_beta'],'hn_beta',ParameterFormatError)
        else:
            raise(ParameterFormatError(
                "The input of this function must be a python dictionary with keys:"
                +str(self._H_PARAM_KEYS)+" or "
                +str(self._H0_PARAM_KEYS)+" or "
                +str(self._HN_PARAM_KEYS)+".")
                )
        self.calc_pred_dist()

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: float}
            * ``"hn_alpha"`` : The value of ``self.hn_alpha``
            * ``"hn_beta"`` : The value of ``self.hn_beta``
        """
        return {"hn_alpha":self.hn_alpha, "hn_beta":self.hn_beta}
    
    def reset_hn_params(self):
        """Reset the hyperparameters of the posterior distribution to their initial values.
        
        They are reset to `self.h0_alpha` and `self.h0_beta`.
        Note that the parameters of the predictive distribution are also calculated from `self.h0_alpha` and `self.h0_beta`.
        """
        self.hn_alpha = self.h0_alpha
        self.hn_beta = self.h0_beta
        self.calc_pred_dist()

    def overwrite_h0_params(self):
        """Overwrite the initial values of the hyperparameters of the posterior distribution by the learned values.
        
        They are overwritten by `self.hn_alpha` and `self.hn_beta`.
        Note that the parameters of the predictive distribution are also calculated from `self.hn_alpha` and `self.hn_beta`.
        """
        self.h0_alpha = self.hn_alpha
        self.h0_beta = self.hn_beta
        self.calc_pred_dist()

    def update_posterior(self,x):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy.ndarray
            All the elements must be 0 or 1.
        """
        _check.ints_of_01(x,'x',DataFormatError)
        self.hn_alpha += np.sum(x==1)
        self.hn_beta += np.sum(x==0)

    def estimate_params(self,loss="squared"):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".

        Returns
        -------
        Estimator : {float, None, rv_frozen}
            The estimated values under the given loss function. If it is not exist, `None` will be returned.
            If the loss function is \"KL\", the posterior distribution itself will be returned
            as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """
        if loss == "squared":
            return self.hn_alpha / (self.hn_alpha + self.hn_beta)
        elif loss == "0-1":
            if self.hn_alpha > 1.0 and self.hn_beta > 1.0:
                return (self.hn_alpha - 1.0) / (self.hn_alpha + self.hn_beta - 2.0)
            elif self.hn_alpha > 1.0:
                return 1.0
            elif self.hn_beta > 1.0:
                return 0.0
            else:
                warnings.warn("MAP estimate doesn't exist for the current hn_alpha and hn_beta.",ResultWarning)
                return None
        elif loss == "abs":
            return ss_beta.median(self.hn_alpha,self.hn_beta)
        elif loss == "KL":
            return ss_beta(self.hn_alpha,self.hn_beta)
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
        lower, upper: float
            The lower and the upper bound of the interval
        """
        _check.float_in_closed01(credibility,'credibility',CriteriaError)
        return ss_beta.interval(credibility,self.hn_alpha,self.hn_beta)
    
    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import bernoulli
        >>> gen_model = bernoulli.GenModel()
        >>> X = gen_model.gen_sample(20)
        >>> print(X)
        [1 0 1 0 0 0 0 0 0 0 1 1 1 1 0 1 0 0 1 0]
        >>> learn_model = bernoulli.LearnModel()
        >>> learn_model.update_posterior(X)
        >>> learn_model.visualize_posterior()
        
        .. image:: ./images/bernoulli_posterior.png
        """
        p_range = np.linspace(0,1,100,endpoint=False)
        fig, ax = plt.subplots()
        ax.plot(p_range,self.estimate_params(loss="KL").pdf(p_range))
        ax.set_xlabel("p")
        ax.set_ylabel("posterior")
        plt.show()
    
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: float}
            * ``"p_alpha"`` : The value of ``self.p_alpha``
            * ``"p_beta"`` : The value of ``self.p_beta``
        """
        return {"p_alpha":self.p_alpha, "p_beta":self.p_beta}
    
    def set_p_params(self,p_alpha,p_beta):
        """Set the parameters of the predictive distribution.

        Parameters
        ----------
        p_alpha : float
            a positive real number
        p_beta : float
            a positibe real number
        """
        self.p_alpha = _check.pos_float(p_alpha,'p_alpha',ParameterFormatError)
        self.p_beta = _check.pos_float(p_beta,'p_beta',ParameterFormatError)

    def calc_pred_dist(self):
        """Calculate the parameters of the predictive distribution."""
        self.p_alpha = self.hn_alpha
        self.p_beta = self.hn_beta

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
        if loss == "squared":
            return self.p_alpha / (self.p_alpha + self.p_beta)
        elif loss == "0-1" or loss == "abs":
            if self.p_alpha > self.p_beta:
                return 1
            else:
                return 0
        elif loss == "KL":
            return np.array((self.p_beta / (self.p_alpha + self.p_beta),
                             self.p_alpha / (self.p_alpha + self.p_beta)))
            # return ss_betabinom(1,self.p_alpha,self.p_beta)
        else:
            raise(CriteriaError("Unsupported loss function! "
                                "This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))

    def pred_and_update(self,x,loss="squared"):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x : int
            It must be 0 or 1
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
        _check.int_of_01(x,'x',DataFormatError)
        self.calc_pred_dist()
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(x)
        return prediction
