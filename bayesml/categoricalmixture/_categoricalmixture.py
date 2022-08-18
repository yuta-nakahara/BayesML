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

_EPSILON = np.sqrt(np.finfo(np.float64).eps)

def float_mat_sum_1(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim == 2 and np.all(np.abs(np.sum(val, axis=1) - 1.) <= _EPSILON):
            return val.astype(float)
        if np.issubdtype(val.dtype,np.floating) and val.ndim == 2 and np.all(np.abs(np.sum(val, axis=1) - 1.) <= _EPSILON):
            return val
    raise(exception_class(val_name + " must be a 2-dimensional numpy.ndarray, and the sum of the second dimension must equal to 1."))

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution

    Parameters
    ----------
    m_degree : int, optional
        a positive integer. Default is None, in which case 
        a value consistent with ``pi_vec`` and 
        ``h_alpha_vec`` is used. If all of them are not given, 
        degree is assumed to be 3.
    degree : int, optional
        a positive integer. Default is None, in which case 
        a value consistent with ``theta_mat`` and 
        ``h_beta_vec`` is used. If all of them are not given, 
        degree is assumed to be 3.
    pi_vec : numpy ndarray, optional
        a real vector in :math:`[0, 1]^K`, by default [1/K, 1/K, ... , 1/K]
    theta_mat : numpy ndarray, optional
        a real matrix in :math:`[0, 1]^{K \times d}`, by default [[1/d, 1/d, ... , 1/d]]*K
    h_alpha_vec : numpy ndarray, optional
        a vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
    h_beta_vec : numpy ndarray, optional
        a vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(
        self, *, m_degree=None, degree=None, pi_vec=None, theta_mat=None, h_alpha_vec=None, h_beta_vec=None, seed=None,
        ):
        

        if m_degree is not None:
            self.m_degree = _check.pos_int(m_degree,'m_degree',ParameterFormatError)
            if pi_vec is None:
                self.pi_vec = np.ones(self.m_degree) / self.m_degree
            else:
                self.pi_vec = _check.float_vec_sum_1(pi_vec,'pi_vec',ParameterFormatError)
            
            if h_alpha_vec is None:
                self.h_alpha_vec = np.ones(self.m_degree) / 2.0
            else:
                self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec,'h_alpha_vec',ParameterFormatError)

        elif pi_vec is not None:
            self.pi_vec = _check.float_vec_sum_1(pi_vec,'pi_vec',ParameterFormatError)
            self.m_degree = self.pi_vec.shape[0]
            if h_alpha_vec is None:
                self.h_alpha_vec = np.ones(self.m_degree) / 2.0
            else:
                self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec,'h_alpha_vec',ParameterFormatError)
        
        elif h_alpha_vec is not None:
            self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec,'h_beta_vec',ParameterFormatError)
            self.m_degree = self.h_alpha_vec.shape[0]
            self.pi_vec = np.ones(self.m_degree) / self.m_degree

        else:
            self.m_degree = 3
            self.pi_vec = np.ones(self.m_degree) / self.m_degree
            self.h_alpha_vec = np.ones(self.m_degree) / 2.0

        if (self.m_degree != self.pi_vec.shape[0]
            or self.m_degree != self.h_alpha_vec.shape[0]):
            raise(ParameterFormatError(
                "degree and dimensions of pi_vec and"
                +" h_alpha_vec must be the same,"
                +" if two or more of them are specified."))


        if degree is not None:
            self.degree = _check.pos_int(degree,'degree',ParameterFormatError)

            if theta_mat is None:
                self.theta_mat = np.ones((self.m_degree, self.degree)) / self.degree
            else:
                self.theta_mat = float_mat_sum_1(theta_mat,'theta_mat',ParameterFormatError)
            
            if h_beta_vec is None:
                self.h_beta_vec = np.ones(self.degree) / 2.0
            else:
                self.h_beta_vec = _check.pos_float_vec(h_beta_vec,'h_beta_vec',ParameterFormatError)

        elif theta_mat is not None:
            self.theta_mat = float_mat_sum_1(theta_mat,'theta_mat',ParameterFormatError)

            self.degree = self.theta_mat.shape[1]
            
            if h_beta_vec is None:
                self.h_beta_vec = np.ones(self.degree) / 2.0
            else:
                self.h_beta_vec = _check.pos_float_vec(h_beta_vec,'h_beta_vec',ParameterFormatError)
        
        elif h_beta_vec is not None:
            self.h_beta_vec = _check.pos_float_vec(h_beta_vec,'h_beta_vec',ParameterFormatError)
            self.degree = self.h_beta_vec.shape[0]
            self.theta_mat = np.ones((self.m_degree, self.degree)) / self.degree
        
        else:
            self.degree = 3
            self.theta_mat = np.ones((self.m_degree, self.degree)) / self.degree
            self.h_beta_vec = np.ones(self.degree) / 2.0
        
        if self.m_degree != self.theta_mat.shape[0]:
            raise(ParameterFormatError(
                "degree of the first dimension of theta_mat and degree of pi_vec and h_alpha_vec must be the same,"
                +" if two or more of them are specified."))

        if (self.degree != self.theta_mat.shape[1]
            or self.degree != self.h_beta_vec.shape[0]):
            raise(ParameterFormatError(
                "degree of the second dimension of theta_mat"
                +" and degree of h_beta_vec must be the same,"
                +" if they are specified."))


        self.rng = np.random.default_rng(seed)

    def set_h_params(self, h_alpha_vec, h_beta_vec):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_alpha_vec : numpy ndarray
            a vector of positive real numbers
        h_beta_vec : numpy ndarray
            a vector of positive real numbers
        """
        self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec,'h_alpha_vec',ParameterFormatError)
        self.h_beta_vec = _check.pos_float_vec(h_beta_vec,'h_beta_vec',ParameterFormatError)

        self.m_degree = self.h_alpha_vec.shape[0]
        self.degree = self.h_beta_vec.shape[0]

        if self.m_degree != self.pi_vec.shape[0]:
            self.pi_vec = np.ones(self.degree) / self.degree
            warnings.warn("pi_vec is reinitialized to [1.0/self.m_degree, 1.0/self.m_degree, ... , 1.0/self.m_degree] because dimension of pi_vec and h_alpha_vec are mismatched.", ParameterFormatWarning)

        if self.m_degree != self.theta_mat.shape[0]:
            self.theta_mat = np.ones((self.m_degree, self.degree)) / self.degree
            warnings.warn("theta_mat is reinitialized to [[1.0/self.degree, 1.0/self.degree, ... , 1.0/self.degree]]*self.m_degree because dimension of theta_mat and h_alpha_vec are mismatched.", ParameterFormatWarning)        
        elif self.degree != self.theta_mat.shape[1]:
            self.theta_mat = np.ones((self.m_degree, self.degree)) / self.degree
            warnings.warn("theta_mat is reinitialized to [[1.0/self.degree, 1.0/self.degree, ... , 1.0/self.degree]]*self.m_degree because dimension of theta_mat and h_beta_vec are mismatched.", ParameterFormatWarning)

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str:numpy ndarray}
            ``{"h_alpha_vec": self.h_alpha_vec, "h_beta_vec": self.h_beta_vec}``
        """
        return {"h_alpha_vec": self.h_alpha_vec, "h_beta_vec": self.h_beta_vec}

    def set_params(self, pi_vec, theta_mat):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        p : numpy ndarray
            a real vector :math:`p \in [0, 1]^d`
        """
        self.pi_vec = _check.float_vec_sum_1(pi_vec,'pi_vec',ParameterFormatError)
        self.theta_mat = float_mat_sum_1(theta_mat,'theta_mat',ParameterFormatError)

        self.m_degree = self.pi_vec.shape[0]
        self.degree = self.theta_mat.shape[1]

        if self.m_degree != self.h_alpha_vec.shape[0]:
            self.h_alpha_vec = np.ones(self.degree) / 2.0
            warnings.warn("h_alpha_vec is reinitialized to [1/2, 1/2, ..., 1/2] because dimension of h_alpha_vec and pi_vec are mismatched.", ParameterFormatWarning)

        if self.degree != self.h_beta_vec.shape[0]:
            self.h_beta_vec = np.ones(self.degree) / 2.0
            warnings.warn("h_beta_vec is reinitialized to [1/2, 1/2, ..., 1/2] because dimension of h_beta_vec and theta_mat are mismatched.", ParameterFormatWarning)    


    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:numpy ndarray}
            ``{"pi_vec": self.pi_vec, "theta_mat": self.theta_mat}``
        """
        return {"pi_vec": self.pi_vec, "theta_mat": self.theta_mat}

    
    def gen_params(self):
        return super().gen_params()

    def gen_sample(self):
        return super().gen_sample()

    def save_sample(self):
        return super().save_sample()

    def visualize_model(self):
        return super().visualize_model()        