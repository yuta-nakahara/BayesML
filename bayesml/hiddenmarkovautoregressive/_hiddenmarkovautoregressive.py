# Code Author
# Jun Nishikawa <Jun.B.Nishikawa@gmail.com>
# Document Author
# Koki Kazama <kokikazama@aoni.waseda.jp>
# Jun Nishikawa <Jun.B.Nishikawa@gmail.com>

import numpy as np

from .. import _check
from .._exceptions import ParameterFormatError

class GenModel:
    def __init__(
        self, 
        *, 
        degree: int = 1, 
        K: int = 2, 
        n: int = 1, 
        z_vec: np.ndarray = None, 
        pi_vec: np.ndarray = None, 
        A_matrix: np.ndarray = None, 
        x_n_vec: np.ndarray = None, 
        theta_matrix: np.ndarray = None, 
        x_n: float = 1, 
        tau_vec: np.ndarray = None
    ):
        # TODO: get default values
        # TODO: 境界値テスト

        # [input variables]
        self.degree = _check.pos_int(degree, "degree", ParameterFormatError)
        self.K = _check.pos_int(K, "K", ParameterFormatError)
        self.n = _check.pos_int(n, "n", ParameterFormatError)
        self.x_n = _check.float_(x_n, "x_n", ParameterFormatError)
        # [check values or set default values]
        self.z_vec = np.array([1,] + [0,]*self.K) \
            if z_vec is None else _check.onehot_vec(z_vec, "z_vec", ParameterFormatError)
        self.pi_vec = np.ones(self.K) / self.K  \
            if pi_vec is None else _check.nonneg_float_vec(pi_vec, "pi_vec", ParameterFormatError)
        self.A_matrix = np.broadcast_to(np.reshape(self.pi_vec, [self.K,1]), [self.K,]*2) \
            if A_matrix is None else _check.float_vec_sum_1(A_matrix, "A_matrix", ParameterFormatError, ndim=2, sum_axis=0)
        self.x_n_vec = np.ones(self.K + 1) \
            if x_n_vec is None else _check.float_vec(x_n_vec, "x_n_vec", ParameterFormatError)
        self.theta_matrix = np.zeros(self.K,self.degree+1) \
            if theta_matrix is None else _check.float_vecs(theta_matrix, "theta_matrix", ParameterFormatError)
        self.tau_vec = self.pi_vec.copy() \
            if tau_vec is None else _check.float_vec(tau_vec, "tau_vec", ParameterFormatError)

