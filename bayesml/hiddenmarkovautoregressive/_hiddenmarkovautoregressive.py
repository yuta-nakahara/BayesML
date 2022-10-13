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
    pi_vec : numpy ndarray, optional 
        a vector of K-dim real numbers in each values in [0,1]. Default is (1/K,...,1/K). The sum must be 1.
    A_mat : numpy ndarray, optional
        a matrix of KxK-dim real numbers, each values in [0,1]. Default is (0,...,0).
        a parameter for latent classes.
    theta_vecs : numpy ndarray, optional
        K vectors of degree+1-dim real numbers. A regression coefficient parameter.
        Default is (0,...,0).
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
            c_num_classes,
            c_degree,
            *, 
            # [stochastic data generative model]
            pi_vec=None,
            a_mat=None,
            theta_vecs=None,
            taus=None,
            # [prior distribution]
            h_eta_vec=None,
            h_zeta_vecs=None,
            h_mu_vecs=None,
            h_lambda_mats=None,
            h_alphas=None,
            h_betas=None,
            seed=None,
            ):
        # constants
        self.c_num_classes = _check.pos_int(c_num_classes,'c_num_classes',ParameterFormatError)
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # params
        self.pi_vec = np.ones(self.c_num_classes) / self.c_num_classes
        self.a_mat = np.ones([self.c_num_classes,self.c_num_classes]) / self.c_num_classes
        self.theta_vecs = np.zeros([self.c_num_classes,self.c_degree+1])
        self.taus = np.ones(self.c_num_classes)

        # h_params
        self.h_eta_vec = np.ones(self.c_num_classes) / 2.0
        self.h_zeta_vecs = np.ones([self.c_num_classes,self.c_num_classes]) / 2.0
        self.h_mu_vecs = np.zeros([self.c_num_classes,self.c_degree+1])
        self.h_lambda_mats = np.tile(np.identity(self.c_degree+1),[self.c_num_classes,1,1])
        self.h_alphas = np.ones(self.c_num_classes)
        self.h_betas = np.ones(self.c_num_classes)

        # [set params]
        self.set_params(
            pi_vec,
            a_mat,
            theta_vecs,
            taus,
        )
        self.set_h_params(
            h_eta_vec,
            h_zeta_vecs,
            h_mu_vecs,
            h_lambda_mats,
            h_alphas,
            h_betas,
        )
        self.det_Lambda = np.linalg.det(self.h_lambda_mats)
        self.gamma_func = np.frompyfunc(gamma_func,1,1)

        # # ===== for stochastic data generative model =====
        # self.pi_vec = None if pi_vec is None else _check.nonneg_float_vec(pi_vec, "pi_vec", ParameterFormatError)
        # self.A_mat = None if A_mat is None else _check.float_vec_sum_1(A_mat, "A_matrix", ParameterFormatError, ndim=2, sum_axis=0)
        # self.theta_vecs = None if theta_vecs is None else _check.float_vecs(theta_vecs, "theta_matrix", ParameterFormatError)
        # self.taus = None if taus is None else _check.floats(taus, "taus", ParameterFormatError)
        # # ===== for prior distribution =====
        # self.h_mu_vec = None if h_mu_vec is None else _check.float_vec(h_mu_vec, "h_mu_vec", ParameterFormatError)
        # self.h_lambda_mat = None if h_lambda_mat is None else _check.float_vec(h_lambda_mat, "h_lambda_mat", ParameterFormatError)
        # self.h_alpha = None if h_alpha is None else _check.float_(h_alpha, "h_alpha", ParameterFormatError)
        # self.h_beta = None if h_beta is None else _check.float_(h_beta, "h_beta", ParameterFormatError)
        # self.h_eta_vec = None if h_eta_vec is None else _check.pos_float_vec(h_eta_vec, "h_eta_vec", ParameterFormatError)
        # self.h_zeta_vecs = None if h_zeta_vecs is None else _check.pos_float_vec(h_zeta_vecs, "h_zeta_vec", ParameterFormatError)

        # # [Check value consistency]
        # consistency_value_dict_degree = {
        #     "degree": None if self.degree is None else self.degree, 
        #     "theta_vecs": None if self.theta_vecs is None else self.theta_vecs.shape[1] - 1, 
        #     "h_mu_vec": None if self.h_mu_vec is None else self.h_mu_vec.shape[0] - 1, 
        #     "column of h_lambda_mat": None if self.h_lambda_mat is None else self.h_lambda_mat.shape[0] - 1, 
        #     "row of h_lambda_mat": None if self.h_lambda_mat is None else self.h_lambda_mat.shape[1] - 1
        # }
        # consistency_value_dict_K = {
        #     "K": None if self.K is None else self.K, 
        #     "pi_vec": None if self.pi_vec is None else self.pi_vec.shape[0], 
        #     "colomn of A_mat": None if self.A_mat is None else self.A_mat.shape[0], 
        #     "row of A_mat": None if self.A_mat is None else self.A_mat.shape[1], 
        #     "number of theta_vec": None if self.theta_vecs is None else self.theta_vecs.shape[0], 
        #     "h_eta_vec": None if self.h_eta_vec is None else self.h_eta_vec.shape[0], 
        #     "a number of h_zeta_vec": None if self.h_zeta_vecs is None else self.h_zeta_vecs.shape[0], 
        #     "each dim of h_zeta_vec": None if self.h_zeta_vecs is None else self.h_zeta_vecs.shape[1]
        # }
        # consistency_values_dict_all = {
        #     "degree": consistency_value_dict_degree, 
        #     "K": consistency_value_dict_K
        # }
        # for key in consistency_values_dict_all:
        #     value = _check.dim_consistency(consistency_values_dict_all[key], ParameterFormatError)
        #     exec(f"self.{key} = {value}")

        # # [Check values and set default values]
        # # ===== for stochastic data generative model =====
        # self.pi_vec = np.ones(self.K) / self.K if self.pi_vec is None else self.pi_vec
        # self.A_mat = np.broadcast_to(np.reshape(self.pi_vec, [self.K,1]), [self.K,]*2) if self.A_mat is None else self.A_mat
        # self.theta_vecs = np.zeros((self.K, self.degree+1), dtype=float) if self.theta_vecs is None else self.theta_vecs
        # self.taus = self.pi_vec.copy() if self.taus is None else self.taus
        # # ===== for prior distribution =====
        # self.h_mu_vec = np.zeros(self.degree + 1) if self.h_mu_vec is None else self.h_mu_vec
        # self.h_lambda_mat = np.identity(self.degree + 1) if self.h_lambda_mat is None else self.h_lambda_mat
        # self.h_alpha = _check.float_(h_alpha, "h_alpha", ParameterFormatError)
        # self.h_beta = _check.float_(h_beta, "h_beta", ParameterFormatError)
        # self.h_eta_vec = np.ones(self.K) / 2. if self.h_eta_vec is None else self.h_eta_vec
        # self.h_zeta_vecs = np.ones((self.K, self.K), dtype=float) / 2. if self.h_zeta_vecs is None else self.h_zeta_vecs
        # self.det_Lambda = np.linalg.det(self.h_lambda_mat)
        # self.gamma_func = np.frompyfunc(gamma_func,1,1)

    def set_h_params(
            self, 
            h_eta_vec=None,
            h_zeta_vecs=None,
            h_mu_vecs=None,
            h_lambda_mats=None,
            h_alphas=None,
            h_betas=None,
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
        if h_eta_vec is not None:
            _check.pos_floats(h_eta_vec,'h_eta_vec',ParameterFormatError)
            self.h_eta_vec[:] = h_eta_vec

        if h_zeta_vecs is not None:
            _check.pos_floats(h_zeta_vecs, 'h_zeta_vecs', ParameterFormatError)
            self.h_zeta_vecs[:] = h_zeta_vecs

        if h_mu_vecs is not None:
            _check.float_vecs(h_mu_vecs,'h_mu_vecs',ParameterFormatError)
            _check.shape_consistency(
                h_mu_vecs.shape[-1],'h_mu_vecs.shape[-1]',
                self.c_degree + 1,'self.c_degree + 1',
                ParameterFormatError
                )
            self.h_mu_vecs[:] = h_mu_vecs

        if h_lambda_mats is not None:
            _check.pos_def_sym_mats(h_lambda_mats,'h_lambda_mats',ParameterFormatError)
            _check.shape_consistency(
                h_lambda_mats.shape[-1],'h_lambda_mats.shape[-1] and h_lambda_mats.shape[-2]',
                self.c_degree+1,'self.c_degree + 1',
                ParameterFormatError
                )
            self.h_lambda_mats[:] = h_lambda_mats

        if h_alphas is not None:
            _check.pos_floats(h_alphas,'h_alphas',ParameterFormatError)
            self.h_alphas[:] = h_alphas

        if h_betas is not None:
            _check.pos_floats(h_betas,'h_betas',ParameterFormatError)
            self.h_betas[:] = h_betas

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float or numpy ndarray}
            * ``"h_eta_vec"`` : The value of ``self.h_eta_vec``
            * ``"h_zeta_vecs"`` : The value of ``self.h_zeta_vecs``
            * ``"h_mu_vecs"`` : The value of ``self.h_mu_vecs``
            * ``"h_lambda_mats"`` : The value of ``self.h_lambda_mats``
            * ``"h_alphas"`` : The value of ``self.h_alphas``
            * ``"h_betas"`` : The value of ``self.h_betas``
        """
        return {
            "h_eta_vec":self.h_eta_vec,
            "h_zeta_vecs":self.h_zeta_vecs,
            "h_mu_vecs":self.h_mu_vecs,
            "h_lambda_mats":self.h_lambda_mats,
            "h_alphas":self.h_alphas,
            "h_betas":self.h_betas,
            }

    def set_params(
            self,
            pi_vec=None,
            a_mat=None,
            theta_vecs=None,
            taus=None,
            ):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        pi_vec : numpy ndarray, optional 
            a vector of K-dim real numbers in each values in [0,1]. Default is (1/K,...,1/K). The sum must be 1.
        A_mat : numpy ndarray, optional
            a matrix of KxK-dim real numbers, each values in [0,1]. Default is (0,...,0).
            a parameter for latent classes.
        theta_vecs : numpy ndarray, optional
            K vectors of degree+1-dim real numbers. A regression coefficient parameter.
            Default is (0,...,0).
        taus : numpy ndarray, optinal
            K real numbers, a precision parameter of noise. Default is 1.
        """
        if pi_vec is not None:
            _check.float_vec_sum_1(pi_vec, "pi_vec", ParameterFormatError)
            _check.shape_consistency(
                pi_vec.shape[0],"pi_vec.shape[0]", 
                self.c_num_classes,"self.c_num_classes", 
                ParameterFormatError
                )
            self.pi_vec[:] = pi_vec

        if a_mat is not None:
            _check.float_vecs_sum_1(a_mat, "a_mat", ParameterFormatError)
            _check.shape_consistency(
                a_mat.shape[-1],"a_mat.shape[-1]", 
                self.c_num_classes,"self.c_num_classes", 
                ParameterFormatError
                )
            self.a_mat[:] = a_mat

        if theta_vecs is not None:
            _check.float_vecs(theta_vecs,'theta_vecs',ParameterFormatError)
            _check.shape_consistency(
                theta_vecs.shape[-1],'theta_vecs.shape[-1]',
                self.c_degree + 1,'self.c_degree + 1',
                ParameterFormatError
                )
            self.theta_vecs[:] = theta_vecs

        if taus is not None:
            _check.pos_floats(taus,'taus',ParameterFormatError)
            self.taus[:] = taus

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str: float or numpy ndarray}
            * ``"pi_vec"`` : The value of ``self.pi_vec``.
            * ``"a_mat"`` : The value of ``self.a_mat``.
            * ``"theta_vecs"`` : The value of ``self.theta_vecs``.
            * ``"taus"`` : The value of ``self.taus``.
        """
        return {
            "pi_vec": self.pi_vec, 
            "a_mat": self.a_mat, 
            "theta_vecs": self.theta_vecs, 
            "taus": self.taus
        }

    def gen_params(self):
        pass

    def gen_sample(self):
        pass

    def save_sample(self):
        pass

    def visualize_model(self):
        pass

