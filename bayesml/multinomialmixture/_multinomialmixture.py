import warnings
import numpy as np

from .. import base
from .._exceptions import (ParameterFormatError, ParameterFormatWarning)
from .. import _check


class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution
    Parameters
    ----------
    num_classes : int, optional
        a positive integer. Default is None, in which case
        a value consistent with ``pi_vec`` and
        ``h_alpha_vec`` is used. If all of them are not given,
        degree is assumed to be 3.
    degree : int, optional
        a positive integer. Default is None, in which case
        a value consistent with ``theta_vecs`` and
        ``h_beta_vec`` is used. If all of them are not given,
        degree is assumed to be 3.
    pi_vec : numpy ndarray, optional
        a real vector in :math:`[0, 1]^K`, by default [1/K, 1/K, ... , 1/K]
    theta_vecs : numpy ndarray, optional
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
        self,
        *,
        num_classes=None,
        degree=None,
        pi_vec=None,
        theta_vecs=None,
        h_alpha_vec=None,
        h_beta_vec=None,
        seed=None,
    ):

        if num_classes is not None:
            self.num_classes = _check.pos_int(num_classes, 'num_classes', ParameterFormatError)
            if pi_vec is None:
                self.pi_vec = np.ones(self.num_classes) / self.num_classes
            else:
                self.pi_vec = _check.float_vec_sum_1(pi_vec, 'pi_vec', ParameterFormatError)

            if h_alpha_vec is None:
                self.h_alpha_vec = np.ones(self.num_classes) / 2.0
            else:
                self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec, 'h_alpha_vec', ParameterFormatError)

        elif pi_vec is not None:
            self.pi_vec = _check.float_vec_sum_1(pi_vec, 'pi_vec', ParameterFormatError)
            self.num_classes = self.pi_vec.shape[0]
            if h_alpha_vec is None:
                self.h_alpha_vec = np.ones(self.num_classes) / 2.0
            else:
                self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec, 'h_alpha_vec', ParameterFormatError)

        elif h_alpha_vec is not None:
            self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec, 'h_beta_vec', ParameterFormatError)
            self.num_classes = self.h_alpha_vec.shape[0]
            self.pi_vec = np.ones(self.num_classes) / self.num_classes

        else:
            self.num_classes = 3
            self.pi_vec = np.ones(self.num_classes) / self.num_classes
            self.h_alpha_vec = np.ones(self.num_classes) / 2.0

        if (self.num_classes != self.pi_vec.shape[0] or self.num_classes != self.h_alpha_vec.shape[0]):
            raise (ParameterFormatError("degree and dimensions of pi_vec and" + " h_alpha_vec must be the same," +
                                        " if two or more of them are specified."))

        if degree is not None:
            self.degree = _check.pos_int(degree, 'degree', ParameterFormatError)

            if theta_vecs is None:
                self.theta_vecs = np.ones((self.num_classes, self.degree)) / self.degree
            else:
                self.theta_vecs = _check.float_vecs_sum_1(theta_vecs, 'theta_vecs', ParameterFormatError)

            if h_beta_vec is None:
                self.h_beta_vec = np.ones(self.degree) / 2.0
            else:
                self.h_beta_vec = _check.pos_float_vec(h_beta_vec, 'h_beta_vec', ParameterFormatError)

        elif theta_vecs is not None:
            self.theta_vecs = _check.float_vecs_sum_1(theta_vecs, 'theta_vecs', ParameterFormatError)

            self.degree = self.theta_vecs.shape[1]

            if h_beta_vec is None:
                self.h_beta_vec = np.ones(self.degree) / 2.0
            else:
                self.h_beta_vec = _check.pos_float_vec(h_beta_vec, 'h_beta_vec', ParameterFormatError)

        elif h_beta_vec is not None:
            self.h_beta_vec = _check.pos_float_vec(h_beta_vec, 'h_beta_vec', ParameterFormatError)
            self.degree = self.h_beta_vec.shape[0]
            self.theta_vecs = np.ones((self.num_classes, self.degree)) / self.degree

        else:
            self.degree = 3
            self.theta_vecs = np.ones((self.num_classes, self.degree)) / self.degree
            self.h_beta_vec = np.ones(self.degree) / 2.0

        if self.num_classes != self.theta_vecs.shape[0]:
            raise (ParameterFormatError(
                "degree of the first dimension of theta_vecs and degree of pi_vec and h_alpha_vec must be the same," +
                " if two or more of them are specified."))

        if (self.degree != self.theta_vecs.shape[1] or self.degree != self.h_beta_vec.shape[0]):
            raise (ParameterFormatError("degree of the second dimension of theta_vecs" +
                                        " and degree of h_beta_vec must be the same," + " if they are specified."))

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
        self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec, 'h_alpha_vec', ParameterFormatError)
        self.h_beta_vec = _check.pos_float_vec(h_beta_vec, 'h_beta_vec', ParameterFormatError)

        self.num_classes = self.h_alpha_vec.shape[0]
        self.degree = self.h_beta_vec.shape[0]

        if self.num_classes != self.pi_vec.shape[0]:
            self.pi_vec = np.ones(self.degree) / self.degree
            warnings.warn(
                "pi_vec is reinitialized to [1.0/self.num_classes, 1.0/self.num_classes, ... , 1.0/self.num_classes] \
                     because dimension of pi_vec and h_alpha_vec are mismatched.", ParameterFormatWarning)

        if self.num_classes != self.theta_vecs.shape[0]:
            self.theta_vecs = np.ones((self.num_classes, self.degree)) / self.degree
            warnings.warn(
                "theta_vecs is reinitialized to \
                    [[1.0/self.degree, 1.0/self.degree, ... , 1.0/self.degree]]*self.num_classes \
                     because dimension of theta_vecs and h_alpha_vec are mismatched.", ParameterFormatWarning)
        elif self.degree != self.theta_vecs.shape[1]:
            self.theta_vecs = np.ones((self.num_classes, self.degree)) / self.degree
            warnings.warn(
                "theta_vecs is reinitialized to \
                    [[1.0/self.degree, 1.0/self.degree, ... , 1.0/self.degree]]*self.num_classes  \
                    because dimension of theta_vecs and h_beta_vec are mismatched.", ParameterFormatWarning)

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.
        Returns
        -------
        h_params : dict of {str:numpy ndarray}
            ``{"h_alpha_vec": self.h_alpha_vec, "h_beta_vec": self.h_beta_vec}``
        """
        return {"h_alpha_vec": self.h_alpha_vec, "h_beta_vec": self.h_beta_vec}

    def set_params(self, pi_vec, theta_vecs):
        """Set the parameter of the sthocastic data generative model.
        Parameters
        ----------
        p : numpy ndarray
            a real vector :math:`p \in [0, 1]^d`
        """
        self.pi_vec = _check.float_vec_sum_1(pi_vec, 'pi_vec', ParameterFormatError)
        self.theta_vecs = _check.float_vecs_sum_1(theta_vecs, 'theta_vecs', ParameterFormatError)

        self.num_classes = self.pi_vec.shape[0]
        self.degree = self.theta_vecs.shape[1]

        if self.num_classes != self.h_alpha_vec.shape[0]:
            self.h_alpha_vec = np.ones(self.degree) / 2.0
            warnings.warn(
                "h_alpha_vec is reinitialized to [1/2, 1/2, ..., 1/2] \
                    because dimension of h_alpha_vec and pi_vec are mismatched.", ParameterFormatWarning)

        if self.degree != self.h_beta_vec.shape[0]:
            self.h_beta_vec = np.ones(self.degree) / 2.0
            warnings.warn(
                "h_beta_vec is reinitialized to [1/2, 1/2, ..., 1/2] \
                    because dimension of h_beta_vec and theta_vecs are mismatched.", ParameterFormatWarning)

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.
        Returns
        -------
        params : dict of {str:numpy ndarray}
            ``{"pi_vec": self.pi_vec, "theta_vecs": self.theta_vecs}``
        """
        return {"pi_vec": self.pi_vec, "theta_vecs": self.theta_vecs}

    def gen_params(self):
        return super().gen_params()

    def gen_sample(self):
        return super().gen_sample()

    def save_sample(self):
        return super().save_sample()

    def visualize_model(self):
        return super().visualize_model()
