# Code Author
# Kohei Horinouchi <horinochi_18@toki.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Document Author
# Kohei Horinouchi <horinochi_18@toki.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
import warnings
import numpy as np
from scipy.stats import dirichlet as ss_dirichlet
import matplotlib.pyplot as plt

from .. import base
from .._exceptions import (
    ParameterFormatError,
    DataFormatError,
    CriteriaError,
    ResultWarning,
    ParameterFormatWarning
    )
from .. import _check

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution

    Parameters
    ----------
    degree : int, optional
        a positive integer. Default is None, in which case 
        a value consistent with ``theta_vec`` and 
        ``h_alpha_vec`` is used. If all of them are not given, 
        degree is assumed to be 3.
    theta_vec : numpy ndarray, optional
        a real vector in :math:`[0, 1]^d`, by default [1/d. 1/d, ... , 1/d]
    h_alpha_vec : numpy ndarray, optional
        a real vector, by default [1/2, 1/2, ... , 1/2]
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(
        self, *, degree=None, theta_vec=None, h_alpha_vec=None, seed=None,
        ):

        if degree is not None:
            self.degree = _check.pos_int(degree,'degree',ParameterFormatError)
            if theta_vec is None:
                self.theta_vec = np.ones(self.degree) / self.degree
            else:
                self.theta_vec = _check.float_vec_sum_1(theta_vec,'theta_vec',ParameterFormatError)
            
            if h_alpha_vec is None:
                self.h_alpha_vec = np.ones(self.degree) / 2.0
            else:
                self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec,'h_alpha_vec',ParameterFormatError)
        
        elif theta_vec is not None:
            self.theta_vec = _check.float_vec_sum_1(theta_vec,'theta_vec',ParameterFormatError)
            self.degree = self.theta_vec.shape[0]
            if h_alpha_vec is None:
                self.h_alpha_vec = np.ones(self.degree) / 2.0
            else:
                self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec,'h_alpha_vec',ParameterFormatError)
        
        elif h_alpha_vec is not None:
            self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec,'h_alpha_vec',ParameterFormatError)
            self.degree = self.h_alpha_vec.shape[0]
            self.theta_vec = np.ones(self.degree) / self.degree

        else:
            self.degree = 3
            self.theta_vec = np.ones(self.degree) / self.degree
            self.h_alpha_vec = np.ones(self.degree) / 2.0

        if (self.degree != self.theta_vec.shape[0]
            or self.degree != self.h_alpha_vec.shape[0]):
            raise(ParameterFormatError(
                "degree and dimensions of theta_vec and"
                +" h_alpha_mat must be the same,"
                +" if two or more of them are specified."))

        self.rng = np.random.default_rng(seed)
        self._H_PARAM_KEYS = {'h_alpha_vec'}
        self._H0_PARAM_KEYS = {'h0_alpha_vec'}
        self._HN_PARAM_KEYS = {'hn_alpha_vec'}

    def set_h_params(self,**kwargs):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        **kwargs
            a python dictionary {'h_alpha_vec':ndarray}, 
            {'h0_alpha_vec':ndarray}, or {'hn_alpha_vec':ndarray}. 
            They are obtained by ``get_h_params()`` of GenModel,
            ``get_h0_params`` of LearnModel or ``get_hn_params`` of LearnModel.
        """
        if kwargs.keys() == self._H_PARAM_KEYS:
            self.h_alpha_vec = _check.pos_float_vec(kwargs['h_alpha_vec'],'h_alpha_vec',ParameterFormatError)
        elif kwargs.keys() == self._H0_PARAM_KEYS:
            self.h_alpha_vec = _check.pos_float_vec(kwargs['h0_alpha_vec'],'h_alpha_vec',ParameterFormatError)
        elif kwargs.keys() == self._HN_PARAM_KEYS:
            self.h_alpha_vec = _check.pos_float_vec(kwargs['hn_alpha_vec'],'h_alpha_vec',ParameterFormatError)
        else:
            raise(ParameterFormatError(
                "The input of this function must be a python dictionary with keys:"
                +str(self._H_PARAM_KEYS)+" or "
                +str(self._H0_PARAM_KEYS)+" or "
                +str(self._HN_PARAM_KEYS)+".")
                )

        self.degree = self.h_alpha_vec.shape[0]
        if self.degree != self.theta_vec.shape[0]:
            self.theta_vec = np.ones(self.degree) / self.degree
            warnings.warn("theta_vec is reinitialized to [1.0/self.degree, 1.0/self.degree, ..., 1.0/self.degree] because dimension of theta_vec and h_alpha_vec are mismatched.", ParameterFormatWarning)

    def get_h_params(self):
        """
        Returns
        -------
        h_params : {str:numpy ndarray}
            ``{"h_alpha_vec": self.h_alpha_vec}``
        """
        return {"h_alpha_vec": self.h_alpha_vec}

    def gen_params(self):
        """
        Generate the parameter of the stochastic data generative model from the prior distribution.
        
        The generated vaule is set at ``self.theta_vec``.
        """
        self.theta_vec[:] = self.rng.dirichlet(self.h_alpha_vec)

    def set_params(self, theta_vec):
        """
        Parameters
        ----------
        p : numpy ndarray
            a real vector :math:`p \in [0, 1]^d`
        """
        self.theta_vec = _check.float_vec_sum_1(theta_vec,'theta_vec',ParameterFormatError)

        self.degree = self.theta_vec.shape[0]
        if self.degree != self.h_alpha_vec.shape[0]:
            self.h_alpha_vec = np.ones(self.degree) / 2.0
            warnings.warn("h_alpha_vec is reinitialized to [1/2, 1/2, ..., 1/2] because dimension of h_m_vec and mu_vec are mismatched.", ParameterFormatWarning)

    def get_params(self):
        """
        Returns
        -------
        params : {str:numpy ndarray}
            ``{"theta_vec":self.theta_vec}``
        """
        return {"theta_vec": self.theta_vec}

    def gen_sample(self, sample_size):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int
            A positive integer

        Returns
        -------
        x : numpy ndarray
            2-dimensional array whose shape is ``(sample_size,degree)`` whose rows are one-hot vectors.
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        x = np.zeros([sample_size,self.degree],dtype=int)
        for i in range(sample_size):
            x[i,self.rng.choice(self.degree,p=self.theta_vec)] = 1
        return x

    def save_sample(self, filename, sample_size):
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

    def visualize_model(self, sample_size=20, sample_num=5):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 20
        sample_num : int, optional
            A positive integer, by default 5
        
        Examples
        --------
        >>> from bayesml import categorical
        >>> model = categorical.GenModel()
        >>> model.visualize_model()
        theta_vec:[0.33333333 0.33333333 0.33333333]

        .. image:: ./images/categorical_example.png
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        _check.pos_int(sample_num,'sample_num',DataFormatError)
        cmap = plt.get_cmap("hsv")
        print(f"theta_vec:{self.theta_vec}",)
        fig, ax = plt.subplots(2,1,figsize=(5, sample_num+1),gridspec_kw={'height_ratios': [1,sample_num]})
        ax[0].set_title("True distribution")
        for j in range(self.degree):
            ax[0].barh(0,self.theta_vec[j],left=self.theta_vec[:j].sum(),color=cmap(j / self.degree))
        ax[1].set_title("Generated sample")
        for i in range(sample_num):
            x = self.gen_sample(sample_size)
            # print(f"x{i}:{x}")
            tmp_sum = 0
            if i == 0:
                for j in range(self.degree):
                    count = x[:,j].sum()
                    ax[1].barh(
                        i,
                        count,
                        left=tmp_sum,
                        label=j,
                        color=cmap(j / self.degree),
                    )
                    tmp_sum += count
            else:
                for j in range(self.degree):
                    count = x[:,j].sum()
                    ax[1].barh(
                        i,
                        count,
                        left=tmp_sum,
                        color=cmap(j / self.degree),
                    )
                    tmp_sum += count
        ax[1].legend()
        ax[1].set_xlabel("Number of occurrences")
        plt.show()

class LearnModel(base.Posterior, base.PredictiveMixin):
    def __init__(self, degree=None, h0_alpha_vec=None):
        if degree is not None:
            self.degree = _check.pos_int(degree,'degree',ParameterFormatError)
            if h0_alpha_vec is None:
                self.h0_alpha_vec = np.ones(self.degree) / 2.0
            else:
                self.h0_alpha_vec = _check.pos_float_vec(h0_alpha_vec,'h0_alpha_vec',ParameterFormatError)

        elif h0_alpha_vec is not None:
            self.h0_alpha_vec = _check.pos_float_vec(h0_alpha_vec,'h0_alpha_vec',ParameterFormatError)
            self.degree = self.h0_alpha_vec.shape[0]
        
        else:
            self.degree = 3
            self.h0_alpha_vec = np.ones(self.degree) / 2.0

        if self.degree != self.h0_alpha_vec.shape[0]:
            raise(ParameterFormatError(
                "degree and dimensions of h0_alpha_vec and"
                +" must be the same, if both are specified."))

        self.hn_alpha_vec = np.copy(self.h0_alpha_vec)
        self.p_theta_vec = self.hn_alpha_vec / self.hn_alpha_vec.sum()

        self._H_PARAM_KEYS = {'h_alpha_vec'}
        self._H0_PARAM_KEYS = {'h0_alpha_vec'}
        self._HN_PARAM_KEYS = {'hn_alpha_vec'}

    def set_h0_params(self,**kwargs):
        """Set the hyperparameters of the prior distribution.
        
        Parameters
        ----------
        **kwargs
            a python dictionary {'h_alpha_vec':ndarray}, 
            {'h0_alpha_vec':ndarray}, or {'hn_alpha_vec':ndarray}. 
            They are obtained by ``get_h_params()`` of GenModel,
            ``get_h0_params`` of LearnModel or ``get_hn_params`` of LearnModel.
        """
        if kwargs.keys() == self._H_PARAM_KEYS:
            self.h0_alpha_vec = _check.pos_float_vec(kwargs['h_alpha_vec'],'h0_alpha_vec',ParameterFormatError)
        elif kwargs.keys() == self._H0_PARAM_KEYS:
            self.h0_alpha_vec = _check.pos_float_vec(kwargs['h0_alpha_vec'],'h0_alpha_vec',ParameterFormatError)
        elif kwargs.keys() == self._HN_PARAM_KEYS:
            self.h0_alpha_vec = _check.pos_float_vec(kwargs['hn_alpha_vec'],'h0_alpha_vec',ParameterFormatError)
        else:
            raise(ParameterFormatError(
                "The input of this function must be a python dictionary with keys:"
                +str(self._H_PARAM_KEYS)+" or "
                +str(self._H0_PARAM_KEYS)+" or "
                +str(self._HN_PARAM_KEYS)+".")
                )

        self.degree = self.h0_alpha_vec.shape[0]
        self.reset_hn_params()

    def get_h0_params(self):
        """Get the initial values of the hyperparameters of the posterior distribution.

        Returns
        -------
        h0_params : dict of {str: float, numpy.ndarray}
            ``"h0_alpha_vec"`` : The value of ``self.h0_alpha_vec``
        """
        return {"h0_alpha_vec": self.h0_alpha_vec}

    def set_hn_params(self,**kwargs):
        """Set updated values of the hyperparameter of the posterior distribution.
        
        Parameters
        ----------
        **kwargs
            a python dictionary {'h_alpha_vec':ndarray}, 
            {'h0_alpha_vec':ndarray}, or {'hn_alpha_vec':ndarray}. 
            They are obtained by ``get_h_params()`` of GenModel,
            ``get_h0_params`` of LearnModel or ``get_hn_params`` of LearnModel.
        """
        if kwargs.keys() == self._H_PARAM_KEYS:
            self.hn_alpha_vec = _check.pos_float_vec(kwargs['h_alpha_vec'],'hn_alpha_vec',ParameterFormatError)
        elif kwargs.keys() == self._H0_PARAM_KEYS:
            self.hn_alpha_vec = _check.pos_float_vec(kwargs['h0_alpha_vec'],'hn_alpha_vec',ParameterFormatError)
        elif kwargs.keys() == self._HN_PARAM_KEYS:
            self.hn_alpha_vec = _check.pos_float_vec(kwargs['hn_alpha_vec'],'hn_alpha_vec',ParameterFormatError)
        else:
            raise(ParameterFormatError(
                "The input of this function must be a python dictionary with keys:"
                +str(self._H_PARAM_KEYS)+" or "
                +str(self._H0_PARAM_KEYS)+" or "
                +str(self._HN_PARAM_KEYS)+".")
                )

        self.degree = self.hn_alpha_vec.shape[0]
        self.calc_pred_dist()

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: numpy.ndarray}
            ``"hn_alpha_vec"`` : The value of ``self.hn_alpha_vec``
        """
        return {"hn_alpha_vec": self.hn_alpha_vec}

    def reset_hn_params(self):
        """Reset the hyperparameter of the posterior distribution to their initial values.
        
        It is reset to `self.h0_alpha_vec`.
        Note that the parameters of the predictive distribution are also calculated from `self.h0_alpha_vec`.
        """
        self.hn_alpha_vec = np.copy(self.h0_alpha_vec)
        self.calc_pred_dist()

    def overwrite_h0_params(self):
        """Overwrite the initial value of the hyperparameter of the posterior distribution by the learned values.
        
        It is overwitten by `self.hn_alpha_vec`.
        Note that the parameters of the predictive distribution are also calculated from `self.hn_alpha_vec`.
        """
        self.h0_alpha_vec = np.copy(self.hn_alpha_vec)
        self.calc_pred_dist()

    def update_posterior(self, x):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy.ndarray
            2-dimensional array whose shape is ``(sample_size,degree)`` whose rows are one-hot vectors.
        """
        _check.onehot_vecs(x,'x',DataFormatError)
        if self.degree > 1 and x.shape[-1] != self.degree:
            raise(DataFormatError(f"x.shape[-1] must be degree:{self.degree}"))
        x = x.reshape(-1,self.degree)

        for k in range(self.degree):
            self.hn_alpha_vec[k] += x[:,k].sum()

    def estimate_params(self, loss="squared"):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", and \"KL\".

        Returns
        -------
        Estimates : {numpy ndarray, float, None, or rv_frozen}
            The estimated values under the given loss function. If it is not exist, `None` will be returned.
            If the loss function is \"KL\", the posterior distribution itself will be returned
            as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """
        if loss == "squared":
            return self.hn_alpha_vec / np.sum(self.hn_alpha_vec)
        elif loss == "0-1":
            if np.all(self.hn_alpha_vec > 1):
                return (self.hn_alpha_vec - 1) / (np.sum(self.hn_alpha_vec) - self.degree)
            else:
                warnings.warn("MAP estimate of lambda_mat doesn't exist for the current hn_alpha_vec.",ResultWarning)
                return None
        elif loss == "KL":
            return ss_dirichlet(alpha=self.hn_alpha_vec)
        else:
            raise (
                CriteriaError(
                    "Unsupported loss function! "
                    'This function supports "squared", "0-1", and "KL".'
                )
            )

    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import categorical
        >>> gen_model = categorical.GenModel()
        >>> x = gen_model.gen_sample(20)
        >>> learn_model = categorical.LearnModel()
        >>> learn_model.update_posterior(x)
        >>> learn_model.visualize_posterior()
        hn_alpha_vec:[6.5 8.5 6.5]

        .. image:: ./images/categorical_posterior.png
        """
        print(f'hn_alpha_vec:{self.hn_alpha_vec}')
        theta_vec_pdf = self.estimate_params(loss='KL')
        fig, axes = plt.subplots()
        step_num = 200
        if self.degree == 3:
            theta1_range = np.linspace(0.01, 0.99, step_num)
            theta2_range = np.linspace(0.01, 0.99, step_num)
            X, Y = np.meshgrid(theta1_range, theta2_range)
            Z = np.zeros([step_num,step_num])
            for i in range(step_num):
                for j in range(step_num-i-1):
                    Z[i,j] = theta_vec_pdf.logpdf(np.array([X[i,j],
                                                            Y[i,j],
                                                            1.0-X[i,j]-Y[i,j]]))
            Z_min = Z.min()
            for i in range(step_num):
                Z[i,step_num-i-1:] = Z_min
            axes.contourf(X,Y,Z,cmap='Blues')
            theta_hat = self.estimate_params(loss='0-1')
            if theta_hat is not None:
                axes.scatter(theta_hat[0],theta_hat[1],marker="x",color='red',label='MAP estimate')
                plt.legend()
            axes.set_xlabel("theta_vec[0]")
            axes.set_ylabel("theta_vec[1]")

            plt.title(f"Log PDF for theta_vec")
            plt.show()

        elif self.degree == 2:
            theta1_range = np.linspace(0.01, 0.99, step_num)
            Z = np.zeros(step_num)
            for i in range(step_num):
                Z[i] = theta_vec_pdf.pdf(np.array([theta1_range[i],
                                                   1.0-theta1_range[i]]))
            axes.plot(theta1_range,Z)
            plt.show()

        else:
            raise(ParameterFormatError("if degree != 2 or degree != 3, it is impossible to visualize the model by this function."))

    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: numpy.ndarray}
            ``"p_theta_vec"`` : The value of ``self.p_theta_vec``
        """
        return {"p_theta_vec": self.p_theta_vec}

    def calc_pred_dist(self):
        """Calculate the parameters of the predictive distribution."""
        self.p_theta_vec = self.hn_alpha_vec / self.hn_alpha_vec.sum()

    def make_prediction(self, loss="squared"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", and \"KL\".

        Returns
        -------
        Predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
            If the loss function is \"KL\", the predictive distribution will be returned
            as 1-dimensional numpy.ndarray that consists of occurence probabilities.
        """
        if loss == "squared":
            return self.p_theta_vec
        elif loss == "0-1":
            tmp = np.zeros(self.degree,dtype=int)
            tmp[np.argmax(self.p_theta_vec)] = 1
            return tmp
        elif loss == "KL":
            return self.p_theta_vec
        else:
            raise (
                CriteriaError(
                    "Unsupported loss function! "
                    'This function supports "squared", "0-1", and "KL".'
                )
            )

    def pred_and_update(self, x, loss="squared"):
        self.calc_pred_dist()
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(x)
        return prediction
