# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
import warnings
import numpy as np
from scipy.stats import multivariate_normal as ss_multivariate_normal
from scipy.stats import wishart as ss_wishart
from scipy.stats import multivariate_t as ss_multivariate_t
from scipy.stats import dirichlet as ss_dirichlet
from scipy.special import gammaln, digamma, xlogy, logsumexp
import matplotlib.pyplot as plt

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning, ParameterFormatWarning
from .. import _check

class GenModel(base.Generative):
    def __init__(
            self,
            c_degree,
            *,
            w_vec=None,
            h_mu_vec=None,
            h_lambda_mat=None,
            seed=None
            ):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # params
        self.w_vec = np.zeros([self.c_degree])

        # h_params
        self.h_mu_vec = np.zeros([self.c_degree])
        self.h_lambda_mat = np.identity(self.c_degree)

        self.set_params(w_vec)
        self.set_h_params(h_mu_vec,h_lambda_mat)

    def set_params(
            self,
            w_vec=None,
            ):
        if w_vec is not None:
            _check.float_vec(w_vec,'w_vec',ParameterFormatError)
            _check.shape_consistency(
                w_vec.shape[0],'w_vec.shape[0]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.w_vec[:] = w_vec

    def set_h_params(
            self,
            h_mu_vec=None,
            h_lambda_mat=None,
            ):
        if h_mu_vec is not None:
            _check.float_vec(h_mu_vec,'h_mu_vec',ParameterFormatError)
            _check.shape_consistency(
                h_mu_vec.shape[0],'h_mu_vec.shape[0]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h_mu_vec[:] = h_mu_vec
        
        if h_lambda_mat is not None:
            _check.pos_def_sym_mat(h_lambda_mat,'h_lambda_mat',ParameterFormatError)
            _check.shape_consistency(
                h_lambda_mat.shape[0],'h_lambda_mat.shape[0] and h_lambda_mat.shape[1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h_lambda_mat[:] = h_lambda_mat

    def get_params(self):
        return {'w_vec':self.w_vec}

    def get_h_params(self):
        return {'h_mu_vec':self.h_mu_vec,'h_lambda_mat':self.h_lambda_mat}

    def gen_params(self):
        pass

    def gen_sample(self):
        pass
    
    def save_sample(self):
        pass
    
    def visualize_model(self):
        pass


class LearnModel(base.Posterior,base.PredictiveMixin):
    def __init__(
            self,
            c_degree,
            *,
            h0_mu_vec=None,
            h0_lambda_mat=None,
            seed = None
            ):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # h0_params
        self.h0_mu_vec = np.zeros([self.c_degree])
        self.h0_lambda_mat = np.identity(self.c_degree)

        # hn_params
        self.hn_mu_vec = np.empty(self.c_degree)
        self.hn_lambda_mat = np.empty([self.c_degree,self.c_degree])

        # p_params
        self.p_sigma_squared = 0.0
        self.p_mu = 0.0
        
        self.set_h0_params(
            h0_mu_vec,
            h0_lambda_mat,
        )

    def set_h0_params(
            self,
            h0_mu_vec=None,
            h0_lambda_mat=None,
            ):
        if h0_mu_vec is not None:
            _check.float_vec(h0_mu_vec,'h0_mu_vec',ParameterFormatError)
            _check.shape_consistency(
                h0_mu_vec.shape[0],'h0_mu_vec.shape[0]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h0_mu_vec[:] = h0_mu_vec
        
        if h0_lambda_mat is not None:
            _check.pos_def_sym_mat(h0_lambda_mat,'h0_lambda_mat',ParameterFormatError)
            _check.shape_consistency(
                h0_lambda_mat.shape[0],'h0_lambda_mat.shape[0] and h0_lambda_mat.shape[1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h0_lambda_mat[:] = h0_lambda_mat
        
        self.reset_hn_params()

    def get_h0_params(self):
        return {'h0_mu_vec':self.h0_mu_vec,'h0_lambda_mat':self.h0_lambda_mat}
    
    def set_hn_params(
            self,
            hn_mu_vec=None,
            hn_lambda_mat=None,
            ):
        if hn_mu_vec is not None:
            _check.float_vec(hn_mu_vec,'hn_mu_vec',ParameterFormatError)
            _check.shape_consistency(
                hn_mu_vec.shape[0],'hn_mu_vec.shape[0]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.hn_mu_vec[:] = hn_mu_vec
        
        if hn_lambda_mat is not None:
            _check.pos_def_sym_mat(hn_lambda_mat,'hn_lambda_mat',ParameterFormatError)
            _check.shape_consistency(
                hn_lambda_mat.shape[0],'hn_lambda_mat.shape[0] and hn_lambda_mat.shape[1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.hn_lambda_mat[:] = hn_lambda_mat

        self.calc_pred_dist()

    def get_hn_params(self):
        return {'hn_mu_vec':self.hn_mu_vec,'hn_lambda_mat':self.hn_lambda_mat}
    
    def reset_hn_params(self):
        self.set_hn_params(self.h0_mu_vec,self.h0_lambda_mat)
    
    def overwrite_h0_params(self):
        self.set_h0_params(self.hn_mu_vec,self.hn_lambda_mat)

    def update_posterior():
        pass

    def estimate_params(self,loss="squared"):
        pass

    def visualize_posterior(self):
        pass
        
    def get_p_params(self):
        return {'p_sigma_squared':self.p_sigma_squared,'p_mu':self.p_mu}

    def calc_pred_dist(self):
        pass

    def make_prediction(self,loss="squared"):
        pass

    def pred_and_update(self,x,loss="squared"):
        pass
