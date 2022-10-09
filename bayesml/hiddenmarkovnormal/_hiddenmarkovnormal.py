# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Jun Nishikawa <jun.b.nishikawa@gmail.com>
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
            a_mat=None,
            mu_vecs=None,
            lambda_mats=None,
            h_m_vecs=None,
            h_kappas=None,
            h_nus=None,
            h_w_mats=None,
            h_eta_vec=None,
            h_zeta_vecs=None,
            seed=None
            ):
        # constants
        self.c_num_classes = _check.pos_int(c_num_classes,'c_num_classes',ParameterFormatError)
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # params
        self.pi_vec = np.ones(self.c_num_classes) / self.c_num_classes
        self.a_mat = np.ones([self.c_num_classes,self.c_num_classes]) / self.c_num_classes
        self.mu_vecs = np.zeros([self.c_num_classes,self.c_degree])
        self.lambda_mats = np.tile(np.identity(self.c_degree),[self.c_num_classes,1,1])

        # h_params
        self.h_m_vecs = np.zeros([self.c_num_classes,self.c_degree])
        self.h_kappas = np.ones([self.c_num_classes])
        self.h_nus = np.ones(self.c_num_classes) * self.c_degree
        self.h_w_mats = np.tile(np.identity(self.c_degree),[self.c_num_classes,1,1])
        self.h_eta_vec = np.ones(self.c_num_classes) / 2.0
        self.h_zeta_vecs = np.ones([self.c_num_classes,self.c_num_classes]) / 2.0

        self.set_params(pi_vec,a_mat,mu_vecs,lambda_mats)
        self.set_h_params(h_m_vecs,h_kappas,h_nus,h_w_mats,h_eta_vec,h_zeta_vecs)

    def set_params(
            self,
            pi_vec=None,
            a_mat=None,
            mu_vecs=None,
            lambda_mats=None
            ):
        # Noneでない入力について，以下をチェックする．
        # * それ単体として，モデルの仮定を満たすか（符号，行列の正定値性など）
        # * 配列のサイズなどがconstants（c_で始まる変数）と整合しているか．ただし，ブロードキャスト可能なものは認める
        # 例
        # if h0_m_vecs is not None:
        #     _check.float_vecs(h0_m_vecs,'h0_m_vecs',ParameterFormatError)
        #     if h0_m_vecs.shape[-1] != self.degree:
        #         raise(ParameterFormatError(
        #             "h0_m_vecs.shape[-1] must coincide with self.degree:"
        #             +f"h0_m_vecs.shape[-1]={h0_m_vecs.shape[-1]}, self.degree={self.degree}"))
        #     self.h0_m_vecs[:] = h0_m_vecs
        pass

    def set_h_params(
            self,
            h_m_vecs=None,
            h_kappas=None,
            h_nus=None,
            h_w_mats=None,
            h_eta_vec=None,
            h_zeta_vecs=None
            ):
        # Noneでない入力について，以下をチェックする．
        # * それ単体として，モデルの仮定を満たすか（符号，行列の正定値性など）
        # * 配列のサイズなどがconstants（c_で始まる変数）と整合しているか．ただし，ブロードキャスト可能なものは認める
        # 例
        # if h0_m_vecs is not None:
        #     _check.float_vecs(h0_m_vecs,'h0_m_vecs',ParameterFormatError)
        #     if h0_m_vecs.shape[-1] != self.degree:
        #         raise(ParameterFormatError(
        #             "h0_m_vecs.shape[-1] must coincide with self.degree:"
        #             +f"h0_m_vecs.shape[-1]={h0_m_vecs.shape[-1]}, self.degree={self.degree}"))
        #     self.h0_m_vecs[:] = h0_m_vecs
        pass

    def get_params(self):
        # paramsを辞書として返す関数．
        # 要素の順番はset_paramsの引数の順にそろえる．
        pass

    def get_h_params(self):
        # h_paramsを辞書として返す関数．
        # 要素の順番はset_h_paramsの引数の順にそろえる．
        pass

    # まだ実装しなくてよい
    def gen_params(self):
        pass

    # まだ実装しなくてよい
    def gen_sample(self):
        pass
    
    # まだ実装しなくてよい
    def save_sample(self):
        pass
    
    # まだ実装しなくてよい
    def visualize_model(self):
        pass


class LearnModel(base.Posterior,base.PredictiveMixin):
    def __init__(
            self,
            c_num_classes,
            c_degree,
            *,
            h0_m_vecs=None,
            h0_kappas=None,
            h0_nus=None,
            h0_w_mats=None,
            h0_eta_vec=None,
            h0_zeta_vecs=None,
            seed = None
            ):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.c_num_classes = _check.pos_int(c_num_classes,'c_num_classes',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # h0_params
        self.h0_m_vecs = np.zeros([self.c_num_classes,self.c_degree])
        self.h0_kappas = np.ones([self.c_num_classes])
        self.h0_nus = np.ones(self.c_num_classes) * self.c_degree
        self.h0_w_mats = np.tile(np.identity(self.c_degree),[self.c_num_classes,1,1])
        self.h0_w_mats_inv = np.linalg.inv(self.h0_w_mats)
        self.h0_eta_vec = np.ones(self.c_num_classes) / 2.0
        self.h0_zeta_vecs = np.ones([self.c_num_classes,self.c_num_classes]) / 2.0

        # hn_params
        self.hn_m_vecs = np.empty([self.c_num_classes,self.c_degree])
        self.hn_kappas = np.empty([self.c_num_classes])
        self.hn_nus = np.empty(self.c_num_classes)
        self.hn_w_mats = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        self.hn_w_mats_inv = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        self.hn_eta_vec = np.empty(self.c_num_classes)
        self.hn_zeta_vecs = np.empty([self.c_num_classes,self.c_num_classes])

        # p_params
        self.p_mu_vecs = np.empty([self.c_num_classes,self.c_degree])
        self.p_nus = np.empty([self.c_num_classes])
        self.p_lambda_mats = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        self.p_lambda_mats_inv = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        
        self.set_h0_params(
            h0_m_vecs,
            h0_kappas,
            h0_nus,
            h0_w_mats,
            h0_eta_vec,
            h0_zeta_vecs,
        )

    def set_h0_params(
            self,
            h0_m_vecs = None,
            h0_kappas = None,
            h0_nus = None,
            h0_w_mats = None,
            h0_eta_vec = None,
            h0_zeta_vecs = None,
            ):
        # Noneでない入力について，以下をチェックする．
        # * それ単体として，モデルの仮定を満たすか（符号，行列の正定値性など）
        # * 配列のサイズなどがconstants（c_で始まる変数）と整合しているか．ただし，ブロードキャスト可能なものは認める
        # 例
        # if h0_m_vecs is not None:
        #     _check.float_vecs(h0_m_vecs,'h0_m_vecs',ParameterFormatError)
        #     if h0_m_vecs.shape[-1] != self.degree:
        #         raise(ParameterFormatError(
        #             "h0_m_vecs.shape[-1] must coincide with self.degree:"
        #             +f"h0_m_vecs.shape[-1]={h0_m_vecs.shape[-1]}, self.degree={self.degree}"))
        #     self.h0_m_vecs[:] = h0_m_vecs

        # 最後にreset_hn_params()を呼ぶようにする
        self.reset_hn_params()

    def get_h0_params(self):
        # h0_paramsを辞書として返す関数．
        # 要素の順番はset_h_paramsの引数の順にそろえる．
        pass
    
    def set_hn_params(
            self,
            hn_m_vecs = None,
            hn_kappas = None,
            hn_nus = None,
            hn_w_mats = None,
            hn_eta_vec = None,
            hn_zeta_vecs = None,
            ):
        # Noneでない入力について，以下をチェックする．
        # * それ単体として，モデルの仮定を満たすか（符号，行列の正定値性など）
        # * 配列のサイズなどがconstants（c_で始まる変数）と整合しているか．ただし，ブロードキャスト可能なものは認める
        # 例
        # if h0_m_vecs is not None:
        #     _check.float_vecs(h0_m_vecs,'h0_m_vecs',ParameterFormatError)
        #     if h0_m_vecs.shape[-1] != self.degree:
        #         raise(ParameterFormatError(
        #             "h0_m_vecs.shape[-1] must coincide with self.degree:"
        #             +f"h0_m_vecs.shape[-1]={h0_m_vecs.shape[-1]}, self.degree={self.degree}"))
        #     self.h0_m_vecs[:] = h0_m_vecs

        # 最後にcalc_pred_dist()を呼ぶようにする
        self.calc_pred_dist()

    def get_hn_params(self):
        # hn_paramsを辞書として返す関数．
        # 要素の順番はset_h_paramsの引数の順にそろえる．
        pass
    
    def reset_hn_params(self):
        # h0_paramsの値をhn_paramsの値にそのままコピーする
        # 配列サイズを揃えてあるので，簡単に書けるはず．
        # 例
        # self.hn_alpha_vec[:] = self.h0_alpha_vec
        # self.hn_m_vecs[:] = self.h0_m_vecs
        # self.hn_kappas[:] = self.h0_kappas

        # 最後にcalc_pred_distを呼ぶ．
        self.calc_pred_dist()
    
    def overwrite_h0_params(self):
        # hn_paramsの値をh0_paramsの値にそのままコピーする
        # 配列サイズを揃えてあるので，簡単に書けるはず．
        # 例
        # self.h0_alpha_vec[:] = self.hn_alpha_vec
        # self.h0_m_vecs[:] = self.hn_m_vecs
        # self.h0_kappas[:] = self.hn_kappas

        # 最後にcalc_pred_distを呼ぶ．
        self.calc_pred_dist()

    # まだ実装しなくてよい
    def update_posterior():
        pass

    # まだ実装しなくてよい
    def estimate_params(self,loss="squared"):
        pass

    # まだ実装しなくてよい    
    def visualize_posterior(self):
        pass
        
    def get_p_params(self):
        # p_paramsを辞書として返す関数．
        pass

    # まだ実装しなくてよい    
    def calc_pred_dist(self):
        pass

    # まだ実装しなくてよい    
    def make_prediction(self,loss="squared"):
        pass

    # まだ実装しなくてよい    
    def pred_and_update(self,x,loss="squared"):
        pass
