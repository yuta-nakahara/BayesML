# Code Author
# Jun Nishikawa <Jun.B.Nishikawa@gmail.com>
# Document Author
# Koki Kazama <kokikazama@aoni.waseda.jp>
# Jun Nishikawa <Jun.B.Nishikawa@gmail.com>

import numpy as np
from math import gamma as gamma_func

from .. import base
from .. import _check
from .._exceptions import ParameterFormatError

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution.

    Parameters
    ----------
    n : int, optional
        a positive integer. This is time index.
    degree : int, optional
        a positive integer. Default is 1, in which case 
        a value consistent with ``x_n_vec``, ``h_mu_vec``, ``theta_vecs``, 
        and ``h_Lambda_mat`` is used. If all of them are not given,
        degree is assumed to be 1.
    K : int, optional
        a positive integer. Default is 1, in which case 
        a value consistent with ``z_vec``, ``pi_vec``, ``A_mat``, ``theta_vecs``, 
        ``taus``, ``h_eta_vec`` and ``h_zeta_vec`` is used. If all of them are not given,
        K is assumed to be 1.
    z_vec : numpy ndarray, optional
        a vector of K-dim 0 or 1. Default is None.
    pi_vec : numpy ndarray, optional 
        a vector of K-dim real numbers in each values in [0,1]. Default is (1/K,...,1/K). The sum must be 1.
    A_mat : numpy ndarray, optional
        a matrix of KxK-dim real numbers, each values in [0,1]. Default is (0,...,0).
        a parameter for latent classes.
    x_n_vec : numpy ndarray, optional
        a vector of degree+1-dim real numbers
    theta_vecs : numpy ndarray, optional
        K vectors of degree+1-dim real numbers. A regression coefficient parameter.
        Default is (0,...,0).
    x_n : float, optional
        a real number, a data point at n.
    taus : numpy ndarray, optinal
        K real numbers, a precision parameter of noise. Default is 1.
    h_mu_vec : numpy ndarray, optional
        a vector of degree+1-dim real number. Default is (0,...,0).
    h_lambda_mat : numpy ndarray, optional
        a matrix of (degree+1)x(degree+1) real number. Default is identity matrix.
    h_alpha : float, optional
        a positive real number. Default is 1.
    h_beta : float, optional
        a positive real number.Default is 1.
    h_eta_vec : numpy ndarray, optional
        a vector of positive real numbers. Default is (1/2,...,1/2).
    h_zeta_vecs : numpy ndarray, optional
        K vectors of positive real numbers. Default is (1/2,...,1/2).
    """
    def __init__(
        self, 
        *, 
        # [stochastic data generative model]
        degree: int = None, 
        K: int = None, 
        pi_vec: np.ndarray = None, 
        A_mat: np.ndarray = None, 
        theta_vecs: np.ndarray = None, 
        taus: np.ndarray = None, 
        # [prior distribution]
        h_mu_vec: np.ndarray = None, 
        h_lambda_mat: np.ndarray = None, 
        h_alpha: float = 1, 
        h_beta: float = 1, 
        h_eta_vec: np.ndarray = None, 
        h_zeta_vecs: np.ndarray = None
    ):
        # [Check values]
        # ===== for stochastic data generative model =====
        self.degree = None if degree is None else _check.pos_int(degree, "degree", ParameterFormatError)
        self.K = None if K is None else _check.pos_int(K, "K", ParameterFormatError)
        self.pi_vec = None if pi_vec is None else _check.nonneg_float_vec(pi_vec, "pi_vec", ParameterFormatError)
        self.A_mat = None if A_mat is None else _check.float_vec_sum_1(A_mat, "A_matrix", ParameterFormatError, ndim=2, sum_axis=0)
        self.theta_vecs = None if theta_vecs is None else _check.float_vecs(theta_vecs, "theta_matrix", ParameterFormatError)
        self.taus = None if taus is None else _check.floats(taus, "taus", ParameterFormatError)
        # ===== for prior distribution =====
        self.h_mu_vec = None if h_mu_vec is None else _check.float_vec(h_mu_vec, "h_mu_vec", ParameterFormatError)
        self.h_lambda_mat = None if h_lambda_mat is None else _check.float_vec(h_lambda_mat, "h_lambda_mat", ParameterFormatError)
        self.h_alpha = None if h_alpha is None else _check.float_(h_alpha, "h_alpha", ParameterFormatError)
        self.h_beta = None if h_beta is None else _check.float_(h_beta, "h_beta", ParameterFormatError)
        self.h_eta_vec = None if h_eta_vec is None else _check.pos_float_vec(h_eta_vec, "h_eta_vec", ParameterFormatError)
        self.h_zeta_vecs = None if h_zeta_vecs is None else _check.pos_float_vec(h_zeta_vecs, "h_zeta_vec", ParameterFormatError)

        # [Check value consistency]
        consistency_value_dict_degree = {
            "degree": None if self.degree is None else self.degree, 
            "x_n_vec": None if self.x_n_vec is None else self.x_n_vec.shape[0] - 1, 
            "theta_vecs": None if self.theta_vecs is None else self.theta_vecs.shape[1] - 1, 
            "h_mu_vec": None if self.h_mu_vec is None else self.h_mu_vec.shape[0] - 1, 
            "column of h_lambda_mat": None if self.h_lambda_mat is None else self.h_lambda_mat.shape[0] - 1, 
            "row of h_lambda_mat": None if self.h_lambda_mat is None else self.h_lambda_mat.shape[1] - 1
        }
        consistency_value_dict_K = {
            "K": None if self.K is None else self.K, 
            "z_vec": None if self.z_vec is None else self.z_vec.shape[0], 
            "pi_vec": None if self.pi_vec is None else self.pi_vec.shape[0], 
            "colomn of A_mat": None if self.A_mat is None else self.A_mat.shape[0], 
            "row of A_mat": None if self.A_mat is None else self.A_mat.shape[1], 
            "number of theta_vec": None if self.theta_vecs is None else self.theta_vecs.shape[0], 
            "h_eta_vec": None if self.h_eta_vec is None else self.h_eta_vec.shape[0], 
            "a number of h_zeta_vec": None if self.h_zeta_vecs is None else self.h_zeta_vecs.shape[0], 
            "each dim of h_zeta_vec": None if self.h_zeta_vecs is None else self.h_zeta_vecs.shape[1]
        }
        consistency_values_dict_all = {
            "degree": consistency_value_dict_degree, 
            "K": consistency_value_dict_K
        }
        for key in consistency_values_dict_all:
            value = _check.dim_consistency(consistency_values_dict_all[key], ParameterFormatError)
            exec(f"self.{key} = {value}")

        # [Check values and set default values]
        # ===== for stochastic data generative model =====
        # self.n = _check.pos_int(n, "n", ParameterFormatError)
        # self.x_n = _check.float_(x_n, "x_n", ParameterFormatError)
        # self.z_vec = _check.onehot_vec(z_vec, "z_vec", ParameterFormatError)
        self.pi_vec = np.ones(self.K) / self.K if self.pi_vec is None else self.pi_vec
        self.A_mat = np.broadcast_to(np.reshape(self.pi_vec, [self.K,1]), [self.K,]*2) if self.A_mat is None else self.A_mat
        self.x_n_vec = np.array([1,] + [0,]*self.degree, dtype=float) if self.x_n_vec is None else self.x_n_vec
        self.theta_vecs = np.zeros((self.K, self.degree+1), dtype=float) if self.theta_vecs is None else self.theta_vecs
        self.taus = self.pi_vec.copy() if self.taus is None else self.taus
        # ===== for prior distribution =====
        self.h_mu_vec = np.zeros(self.degree + 1) if self.h_mu_vec is None else self.h_mu_vec
        self.h_lambda_mat = np.identity(self.degree + 1) if self.h_lambda_mat is None else self.h_lambda_mat
        self.h_alpha = _check.float_(h_alpha, "h_alpha", ParameterFormatError)
        self.h_beta = _check.float_(h_beta, "h_beta", ParameterFormatError)
        self.h_eta_vec = np.ones(self.K) / 2. if self.h_eta_vec is None else self.h_eta_vec
        self.h_zeta_vecs = np.ones((self.K, self.K), dtype=float) / 2. if self.h_zeta_vecs is None else self.h_zeta_vecs
        self.det_Lambda = np.linalg.det(self.h_lambda_mat)
        self.gamma_func = np.frompyfunc(gamma_func,1,1)

        # [Check consistency between parameters]

    def set_h_params(
        self, 
        h_mu_vec: np.ndarray, 
        h_lambda_mat: np.ndarray, 
        h_alpha: float, 
        h_beta: float, 
        h_eta_vec: np.ndarray, 
        h_zeta_vecs: np.ndarray
    ):
        """Set the hyperparameters of the prior distribution.
        
        Parameters
        ----------
        h_mu_vec : numpy ndarray, optional
            a vector of degree+1-dim real number. Default is (0,...,0).
        h_lambda_mat : numpy ndarray, optional
            a matrix of (degree+1)x(degree+1) real number. Default is identity matrix.
        h_alpha : float, optional
            a positive real number. Default is 1.
        h_beta : float, optional
            a positive real number.Default is 1.
        h_eta_vec : numpy ndarray, optional
            a vector of positive real numbers. Default is (1/2,...,1/2).
        h_zeta_vecs : numpy ndarray, optional
            K vectors of positive real numbers. Default is (1/2,...,1/2).
        """

        # [Check values]
        self.h_mu_vec = None if h_mu_vec is None else _check.float_vec(h_mu_vec, "h_mu_vec", ParameterFormatError)
        self.h_lambda_mat = None if h_lambda_mat is None else _check.float_vec(h_lambda_mat, "h_lambda_mat", ParameterFormatError)
        self.h_alpha = None if h_alpha is None else _check.float_(h_alpha, "h_alpha", ParameterFormatError)
        self.h_beta = None if h_beta is None else _check.float_(h_beta, "h_beta", ParameterFormatError)
        self.h_eta_vec = None if h_eta_vec is None else _check.pos_float_vec(h_eta_vec, "h_eta_vec", ParameterFormatError)
        self.h_zeta_vecs = None if h_zeta_vecs is None else _check.pos_float_vec(h_zeta_vecs, "h_zeta_vec", ParameterFormatError)
        # [Check value consistency]
        consistency_value_dict_degree = {
            "degree": None if self.degree is None else self.degree, 
            "x_n_vec": None if self.x_n_vec is None else self.x_n_vec.shape[0] - 1, 
            "theta_vecs": None if self.theta_vecs is None else self.theta_vecs.shape[1] - 1, 
            "h_mu_vec": None if self.h_mu_vec is None else self.h_mu_vec.shape[0] - 1, 
            "column of h_lambda_mat": None if self.h_lambda_mat is None else self.h_lambda_mat.shape[0] - 1, 
            "row of h_lambda_mat": None if self.h_lambda_mat is None else self.h_lambda_mat.shape[1] - 1
        }
        consistency_value_dict_K = {
            "K": None if self.K is None else self.K, 
            "z_vec": None if self.z_vec is None else self.z_vec.shape[0], 
            "pi_vec": None if self.pi_vec is None else self.pi_vec.shape[0], 
            "colomn of A_mat": None if self.A_mat is None else self.A_mat.shape[0], 
            "row of A_mat": None if self.A_mat is None else self.A_mat.shape[1], 
            "number of theta_vec": None if self.theta_vecs is None else self.theta_vecs.shape[0], 
            "h_eta_vec": None if self.h_eta_vec is None else self.h_eta_vec.shape[0], 
            "a number of h_zeta_vec": None if self.h_zeta_vecs is None else self.h_zeta_vecs.shape[0], 
            "each dim of h_zeta_vec": None if self.h_zeta_vecs is None else self.h_zeta_vecs.shape[1]
        }
        consistency_values_dict_all = {
            "degree": consistency_value_dict_degree, 
            "K": consistency_value_dict_K
        }
        for key in consistency_values_dict_all:
            value = _check.dim_consistency(consistency_values_dict_all[key], ParameterFormatError)
            if eval(f"self.{key} != {value}"):
                raise(ParameterFormatError(
                    f"The following values must be the same: {list(consistency_values_dict_all[key].keys())}"
                ))

        # [Check values and set default values]
        self.h_mu_vec = np.zeros(self.degree + 1) if self.h_mu_vec is None else self.h_mu_vec
        self.h_lambda_mat = np.identity(self.degree + 1) if self.h_lambda_mat is None else self.h_lambda_mat
        self.h_alpha = _check.float_(h_alpha, "h_alpha", ParameterFormatError)
        self.h_beta = _check.float_(h_beta, "h_beta", ParameterFormatError)
        self.h_eta_vec = np.ones(self.K) / 2. if self.h_eta_vec is None else self.h_eta_vec
        self.h_zeta_vecs = np.ones((self.K, self.K), dtype=float) / 2. if self.h_zeta_vecs is None else self.h_zeta_vecs
        self.det_Lambda = np.linalg.det(self.h_lambda_mat)
        self.gamma_func = np.frompyfunc(gamma_func,1,1)

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float or numpy ndarray}
            * ``"h_mu_vec"`` : The value of ``self.h_mu_vec``
            * ``"h_lambda_mat"`` : The value of ``self.h_lambda_mat``
            * ``"h_alpha"`` : The value of ``self.h_alpha``
            * ``"h_beta"`` : The value of ``self.h_beta``
            * ``"h_eta_vec"`` : The value of ``self.h_eta_vec``
            * ``"h_zeta_vecs"`` : The value of ``self.h_zeta_vecs``
        """
        return {
            "h_mu_vec": self.h_mu_vec, 
            "h_lambda_mat": self.h_lambda_mat, 
            "h_alpha": self.h_alpha, 
            "h_beta": self.h_beta, 
            "h_eta_vec": self.h_eta_vec, 
            "h_zeta_vecs": self.h_zeta_vecs
        }

    def set_params(
        self, 
        z_vec: np.ndarray = None, 
        pi_vec: np.ndarray = None, 
        A_mat: np.ndarray = None, 
        x_n_vec: np.ndarray = None, 
        theta_vecs: np.ndarray = None, 
        x_n: float = 1, 
        taus: np.ndarray = None, 
    ):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        theta_vec : numpy ndarray, optional
            a vector of real numbers, by default [0.0, 0.0, ... , 0.0]
        tau : float, optional
            a positive real number, by default 1.0
        """
        pass

    def get_params(self):
        pass

    def gen_params(self):
        pass

    def gen_sample(self):
        pass

    def save_sample(self):
        pass

    def visualize_model(self):
        pass

