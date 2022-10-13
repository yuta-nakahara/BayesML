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
            c_num_classes,
            c_degree,
            *,
            pi_vec=None,
            theta_vecs=None,
            taus=None,
            h_gamma_vec=None,
            h_mu_vecs=None,
            h_lambda_mats=None,
            h_alphas=None,
            h_betas=None,
            seed=None
            ):
        # constants
        self.c_num_classes = _check.pos_int(c_num_classes,'c_num_classes',ParameterFormatError)
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # params
        self.pi_vec = np.ones(self.c_num_classes) / self.c_num_classes
        self.theta_vecs = np.zeros([self.c_num_classes,self.c_degree])
        self.taus = np.ones(self.c_num_classes)

        # h_params
        self.h_gamma_vec = np.ones(self.c_num_classes) / 2.0
        self.h_mu_vecs = np.zeros([self.c_num_classes,self.c_degree])
        self.h_lambda_mats = np.tile(np.identity(self.c_degree),[self.c_num_classes,1,1])
        self.h_alphas = np.ones(self.c_num_classes)
        self.h_betas = np.ones(self.c_num_classes)

        self.set_params(pi_vec,theta_vecs,taus)
        self.set_h_params(h_gamma_vec,h_mu_vecs,h_lambda_mats,h_alphas,h_betas)

    def set_params(
            self,
            pi_vec=None,
            theta_vecs=None,
            taus=None
            ):
        if pi_vec is not None:
            _check.float_vec_sum_1(pi_vec,'pi_vec',ParameterFormatError)
            _check.shape_consistency(
                pi_vec.shape[-1],'pi_vec.shape[-1]',
                self.c_num_classes,'self.c_num_classes',
                ParameterFormatError
                )
            self.pi_vec[:] = pi_vec

        if theta_vecs is not None:
            _check.float_vecs(theta_vecs,'theta_vecs',ParameterFormatError)
            _check.shape_consistency(
                theta_vecs.shape[-1],'theta_vecs.shape[-1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.theta_vecs[:] = theta_vecs

        if taus is not None:
            _check.pos_floats(taus,'taus',ParameterFormatError)
            self.taus[:] = taus

    def set_h_params(
            self,
            h_gamma_vec=None,
            h_mu_vecs=None,
            h_lambda_mats=None,
            h_alphas=None,
            h_betas=None,
            ):
        if h_gamma_vec is not None:
            _check.pos_floats(h_gamma_vec,'h_gamma_vec',ParameterFormatError)
            self.h_gamma_vec[:] = h_gamma_vec

        if h_mu_vecs is not None:
            _check.float_vecs(h_mu_vecs,'h_mu_vecs',ParameterFormatError)
            _check.shape_consistency(
                h_mu_vecs.shape[-1],'h_mu_vecs.shape[-1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h_mu_vecs[:] = h_mu_vecs

        if h_lambda_mats is not None:
            _check.pos_def_sym_mats(h_lambda_mats,'h_lambda_mats',ParameterFormatError)
            _check.shape_consistency(
                h_lambda_mats.shape[-1],'h_lambda_mats.shape[-1] and h_lambda_mats.shape[-2]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h_lambda_mats[:] = h_lambda_mats

        if h_alphas is not None:
            _check.pos_floats(h_alphas,'h_alphas',ParameterFormatError)
            self.h_alphas[:] = h_alphas

        if h_betas is not None:
            _check.pos_floats(h_betas,'h_betas',ParameterFormatError)
            self.h_betas[:] = h_betas
        
    def get_params(self):
        return {'pi_vec':self.pi_vec,
                'theta_vecs':self.theta_vecs,
                'taus':self.taus}
        
    def get_h_params(self):
        return {'h_gamma_vec':self.h_gamma_vec,
                'h_mu_vecs':self.h_mu_vecs,
                'h_lambda_mats':self.h_lambda_mats,
                'h_alphas':self.h_alphas,
                'h_betas':self.h_betas}
    
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
            c_num_classes,
            c_degree,
            *,
            h0_gamma_vec=None,
            h0_mu_vecs=None,
            h0_lambda_mats=None,
            h0_alphas=None,
            h0_betas=None,
            seed = None
            ):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.c_num_classes = _check.pos_int(c_num_classes,'c_num_classes',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # h0_params
        self.h0_gamma_vec = np.ones(self.c_num_classes) / 2.0
        self.h0_mu_vecs = np.zeros([self.c_num_classes,self.c_degree])
        self.h0_lambda_mats = np.tile(np.identity(self.c_degree),[self.c_num_classes,1,1])
        self.h0_alphas = np.ones(self.c_num_classes)
        self.h0_betas = np.ones(self.c_num_classes)

        # hn_params
        self.hn_gamma_vec = np.empty(self.c_num_classes)
        self.hn_mu_vecs = np.empty([self.c_num_classes,self.c_degree])
        self.hn_lambda_mats = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        self.hn_alphas = np.empty(self.c_num_classes)
        self.hn_betas = np.empty(self.c_num_classes)

        # p_params
        self.p_pi_vec = np.empty(self.c_num_classes)
        self.p_ms = np.empty(self.c_num_classes)        
        self.p_lambdas = np.empty(self.c_num_classes)
        self.p_nus = np.empty(self.c_num_classes)
        
        self.set_h0_params(
            h0_gamma_vec,
            h0_mu_vecs,
            h0_lambda_mats,
            h0_alphas,
            h0_betas,
        )

    def set_h0_params(
            self,
            h0_gamma_vec=None,
            h0_mu_vecs=None,
            h0_lambda_mats=None,
            h0_alphas=None,
            h0_betas=None,
            ):
        if h0_gamma_vec is not None:
            _check.pos_floats(h0_gamma_vec,'h0_gamma_vec',ParameterFormatError)
            self.h0_gamma_vec[:] = h0_gamma_vec

        if h0_mu_vecs is not None:
            _check.float_vecs(h0_mu_vecs,'h0_mu_vecs',ParameterFormatError)
            _check.shape_consistency(
                h0_mu_vecs.shape[-1],'h0_mu_vecs.shape[-1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h0_mu_vecs[:] = h0_mu_vecs

        if h0_lambda_mats is not None:
            _check.pos_def_sym_mats(h0_lambda_mats,'h0_lambda_mats',ParameterFormatError)
            _check.shape_consistency(
                h0_lambda_mats.shape[-1],'h0_lambda_mats.shape[-1] and h0_lambda_mats.shape[-2]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h0_lambda_mats[:] = h0_lambda_mats

        if h0_alphas is not None:
            _check.pos_floats(h0_alphas,'h0_alphas',ParameterFormatError)
            self.h0_alphas[:] = h0_alphas

        if h0_betas is not None:
            _check.pos_floats(h0_betas,'h0_betas',ParameterFormatError)
            self.h0_betas[:] = h0_betas

        self.reset_hn_params()

    def get_h0_params(self):
        return {'h0_gamma_vec':self.h0_gamma_vec,
                'h0_mu_vecs':self.h0_mu_vecs,
                'h0_lambda_mats':self.h0_lambda_mats,
                'h0_alphas':self.h0_alphas,
                'h0_betas':self.h0_betas}
    
    def set_hn_params(
            self,
            hn_gamma_vec=None,
            hn_mu_vecs=None,
            hn_lambda_mats=None,
            hn_alphas=None,
            hn_betas=None,
            ):
        if hn_gamma_vec is not None:
            _check.pos_floats(hn_gamma_vec,'hn_gamma_vec',ParameterFormatError)
            self.hn_gamma_vec[:] = hn_gamma_vec

        if hn_mu_vecs is not None:
            _check.float_vecs(hn_mu_vecs,'hn_mu_vecs',ParameterFormatError)
            _check.shape_consistency(
                hn_mu_vecs.shape[-1],'hn_mu_vecs.shape[-1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.hn_mu_vecs[:] = hn_mu_vecs

        if hn_lambda_mats is not None:
            _check.pos_def_sym_mats(hn_lambda_mats,'hn_lambda_mats',ParameterFormatError)
            _check.shape_consistency(
                hn_lambda_mats.shape[-1],'hn_lambda_mats.shape[-1] and hn_lambda_mats.shape[-2]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.hn_lambda_mats[:] = hn_lambda_mats

        if hn_alphas is not None:
            _check.pos_floats(hn_alphas,'hn_alphas',ParameterFormatError)
            self.hn_alphas[:] = hn_alphas

        if hn_betas is not None:
            _check.pos_floats(hn_betas,'hn_betas',ParameterFormatError)
            self.hn_betas[:] = hn_betas

        self.calc_pred_dist()

    def get_hn_params(self):
        return {'hn_gamma_vec':self.hn_gamma_vec,
                'hn_mu_vecs':self.hn_mu_vecs,
                'hn_lambda_mats':self.hn_lambda_mats,
                'hn_alphas':self.hn_alphas,
                'hn_betas':self.hn_betas}
    
    def update_posterior():
        pass

    def estimate_params(self,loss="squared"):
        pass

    def visualize_posterior(self):
        pass
        
    def get_p_params(self):
        return {'p_pi_vec':self.p_pi_vec,
                'p_ms':self.p_ms,
                'p_lambdas':self.p_lambdas,
                'p_nus':self.p_nus}

    def calc_pred_dist(self):
        pass

    def make_prediction(self,loss="squared"):
        pass

    def pred_and_update(self,x,loss="squared"):
        pass
