# Code Author
# Yasushi Esaki <esakiful@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln

from .. import base
from .._exceptions import (ParameterFormatError, DataFormatError)
from .. import _check


class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution.
    Parameters
    ----------
    c_num_classes : int
        A positive integer.
    c_degree : int
        A positive integer.
    pi_vec : numpy.ndarray, optional
        A real vector in :math:`[0, 1]^K`, by default [1/K, 1/K, ... , 1/K].
        The sum of its elements must be 1.
    theta_vecs : numpy ndarray, optional
        real vectors in :math:`[0, 1]^{K \times d}`, by default [[1/d, 1/d, ... , 1/d]]*K.
        The sum of each row must be 1.
    h_alpha_vec : numpy.ndarray, optional
        A vector of positive real numbers, by default [1/2, 1/2, ... , 1/2].
    h_beta_vec : numpy.ndarray, optional
        A vector of positive real numbers, by default [1/2, 1/2, ... , 1/2].
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None.
    """

    def __init__(
        self,
        *,
        c_num_classes,
        c_degree,
        pi_vec=None,
        theta_vecs=None,
        h_alpha_vec=None,
        h_beta_vec=None,
        seed=None,
    ):

        # constants
        self.c_degree = _check.pos_int(c_degree, 'c_degree', ParameterFormatError)
        self.c_num_classes = _check.pos_int(c_num_classes, 'c_num_classes', ParameterFormatError)
        self.c_rng = np.random.default_rng(seed)

        # params
        self.pi_vec = np.ones(self.c_num_classes) / self.c_num_classes
        self.theta_vecs = np.ones((self.c_num_classes, self.c_degree)) / self.c_degree

        # h_params
        self.h_alpha_vec = np.ones(self.c_num_classes) / 2.0
        self.h_beta_vec = np.ones(self.c_degree) / 2.0

        self.set_params(pi_vec, theta_vecs)
        self.set_h_params(h_alpha_vec, h_beta_vec)

    def set_h_params(self, h_alpha_vec=None, h_beta_vec=None):
        """Set the hyperparameters of the prior distribution.
        Parameters
        ----------
        h_alpha_vec : numpy.ndarray, optional
            A vector of positive real numbers.
        h_beta_vec : numpy.ndarray, optional
            A vector of positive real numbers.
        """

        if h_alpha_vec is not None:
            _check.pos_floats(h_alpha_vec, 'h_alpha_vec', ParameterFormatError)
            if h_alpha_vec.shape[0] != self.c_num_classes:
                raise (ParameterFormatError(
                    "h_alpha_vec.shape[0] must coincide with c_num_classes: " +
                    f"h_alpha_vec.shape[0]={h_alpha_vec.shape[0]}, c_num_classes={self.c_num_classes}"))
            self.h_alpha_vec = h_alpha_vec

        if h_beta_vec is not None:
            _check.pos_floats(h_beta_vec, 'h_beta_vec', ParameterFormatError)
            if h_beta_vec.shape[0] != self.c_degree:
                raise (ParameterFormatError("h_beta_vec.shape[0] must coincide with c_degree: " +
                                            f"h_beta_vec.shape[0]={h_beta_vec.shape[0]}, c_degree={self.c_degree}"))
            self.h_beta_vec = h_beta_vec

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.
        Returns
        -------
        h_params : dict of {str:numpy.ndarray}
            ``{"h_alpha_vec": self.h_alpha_vec, "h_beta_vec": self.h_beta_vec}``
        """
        return {"h_alpha_vec": self.h_alpha_vec, "h_beta_vec": self.h_beta_vec}

    def set_params(self, pi_vec=None, theta_vecs=None):
        """Set the parameter of the sthocastic data generative model.
        Parameters
        ----------
        pi_vec : numpy.ndarray, optional
            A real vector in :math:`[0, 1]^K`.
            The sum of its elements must be 1.
        theta_vecs : numpy.ndarray, optional
            A real matrix in :math:`[0, 1]^{K \times d}`.
            The sum of each row must be 1.
        """

        if pi_vec is not None:
            _check.float_vec_sum_1(pi_vec, 'pi_vec', ParameterFormatError)
            if pi_vec.shape[0] != self.c_num_classes:
                raise (ParameterFormatError("pi_vec.shape[0] must coincide with c_num_classes: " +
                                            f"pi_vec.shape[0]={pi_vec.shape[0]}, c_num_classes={self.c_num_classes}"))
            self.pi_vec = pi_vec

        if theta_vecs is not None:
            _check.float_vecs_sum_1(theta_vecs, 'theta_vecs', ParameterFormatError)
            if theta_vecs.shape[0] != self.c_num_classes:
                raise (ParameterFormatError(
                    "theta_vecs.shape[0] must coincide with c_num_classes: " +
                    f"theta_vecs.shape[0]={theta_vecs.shape[0]}, c_num_classes={self.c_num_classes}"))
            elif theta_vecs.shape[1] != self.c_degree:
                raise (ParameterFormatError("theta_vecs.shape[1] must coincide with c_degree: " +
                                            f"theta_vecs.shape[1]={theta_vecs.shape[1]}, c_degree={self.c_degree}"))

            self.theta_vecs = theta_vecs

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.
        Returns
        -------
        params : dict of {str:numpy.ndarray}
            ``{"pi_vec": self.pi_vec, "theta_vecs": self.theta_vecs}``
        """
        return {"pi_vec": self.pi_vec, "theta_vecs": self.theta_vecs}

    def gen_params(self):
        """Generate the parameter from the prior distribution.
        The generated value is set at ``self.pi_vec`` and ``self.theta_vecs``.
        """
        self.pi_vec[:] = self.c_rng.dirichlet(self.h_alpha_vec)
        for k in range(self.c_num_classes):
            self.theta_vecs[k] = self.c_rng.dirichlet(self.h_beta_vec)

    def gen_sample(self, sample_size, num_trials=50):
        """Generate a sample from the stochastic data generative model.
        Parameters
        ----------
        sample_size : int
            A positive integer.
        num_trials : int, optional
            A positive integer, by default 50.
        Returns
        -------
        x : numpy.ndarray
            2-dimensional array whose shape is ``(sammple_size, degree)`` and
            its elements are non-negative integers.
        z : numpy.ndarray
            2-dimensional array whose shape is ``(sample_size, num_classes)`` whose rows are one-hot vectors.
        """
        _check.pos_int(sample_size, 'sample_size', DataFormatError)
        _check.pos_int(num_trials, 'num_trials', ValueError)
        z = np.zeros([sample_size, self.c_num_classes], dtype=int)
        x = np.empty([sample_size, self.c_degree])
        for i in range(sample_size):
            k = self.c_rng.choice(self.c_num_classes, p=self.pi_vec)
            z[i, k] = 1
            x[i] = self.c_rng.multinomial(num_trials, pvals=self.theta_vecs[k])
        return x, z

    def save_sample(self, filename, sample_size, num_trials=50):
        """Save the generated sample as NumPy ``.npz`` format.
        It is saved as a NpzFile with keyword: \"x\", \"z\".
        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_size : int
            A positive integer.
        num_trials : int, optional
            A positive integer, by default 50.
        See Also
        --------
        numpy.savez_compressed
        """
        x, z = self.gen_sample(sample_size, num_trials)
        np.savez_compressed(filename, x=x, z=z)

    def visualize_model(self, save_path, sample_size=100, num_trials=50):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        save_path : str
            The filename to which the figure is saved.
        sample_size : int, optional
            A positive integer, by default 100.
        num_trials : int, optional
            A positive integer, by default 50.
        Examples
        --------
        >>> from bayesml import multinomialmixture
        >>> model = multinomialmixture.GenModel(
        >>>             pi_vec=np.array([0.444, 0.444, 0.112]),
        >>>             theta_vecs=np.array([[0.1, 0.2, 0.7], [0.5, 0.2, 0.3], [0.1, 0.8, 0.1]])
        >>>             )
        >>> model.visualize_model('figure.png')
        pi_vec:
         [0.444 0.444 0.112]
        theta_vecs:
         [[0.1 0.2 0.7]
         [0.5 0.2 0.3]
         [0.1 0.8 0.1]]

        .. image:: ./images/multinomialmixture_example.png
        """
        _check.pos_int(sample_size, 'sample_size', DataFormatError)
        _check.pos_int(num_trials, 'num_trials', ValueError)
        if self.degree == 2:
            print(f"pi_vec:\n {self.pi_vec}")
            print(f"theta_vecs:\n {self.theta_vecs}")

            x, _ = self.gen_sample(sample_size, num_trials)
            x /= np.sum(x, axis=1, keepdims=True)

            _, ax = plt.subplots()
            n, _, _ = ax.hist(x[:, 0], bins=30, label='frequency of samples')
            ax.vlines(self.theta_vecs[:, 0], ymin=0, ymax=n.max() * 1.2, colors='red', label='theta_vecs')
            ax.set_xlabel('x[0]')
            ax.set_ylabel('frequency')
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, n.max() * 1.2)
            ax.legend()
            plt.savefig(save_path)
        elif self.degree == 3:
            print(f"pi_vec:\n {self.pi_vec}")
            print(f"theta_vecs:\n {self.theta_vecs}")

            x, _ = self.gen_sample(sample_size, num_trials)
            x /= np.sum(x, axis=1, keepdims=True)

            _, ax = plt.subplots()
            ax.scatter(x[:, 0], x[:, 1], s=20, label='samples')
            ax.scatter(self.theta_vecs[:, 0], self.theta_vecs[:, 1], s=20, c='red', label='theta_vecs')
            _x = np.linspace(0.0, 1.0, 1000)
            ax.fill_between(
                x=_x,
                y1=1.0 - _x,
                y2=1.0,
                facecolor='gray',
            )
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.legend()
            plt.savefig(save_path)
        else:
            raise (ParameterFormatError("if degree > 3, it is impossible to visualize the model by this function."))


class LearnModel(base.Posterior, base.PredictiveMixin):
    """The posterior distribution.

    Parameters
    ----------
    c_num_classes : int
        A positive integer.
    c_degree : int
        A positive integer.
    h0_alpha_vec : numpy.ndarray, optional
        A vector of positive real numbers, by default [1/2, 1/2, ... , 1/2].
    h0_beta_vec : numpy.ndarray, optional
        A vector of positive real numbers, by default [1/2, 1/2, ... , 1/2].
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(), by default None.

    Attributes
    ----------
    hn_alpha_vec : numpy.ndarray
        A vector of positive real numbers.
    hn_alpha_vec : numpy.ndarray
        A vector of positive real numbers.
    r_vecs : numpy.ndarray
        vectors of real numbers. The sum of its elenemts is 1.
    ns : numpy.ndarray
        A vector of positive real numbers.
    s_mats : numpy.ndarray
        A matric of positive real numbers.
    p_pi_vec : numpy.ndarray
        A real vector in :math:`[0, 1]^K`.
    p_theta_vecs : numpy.ndarray
        real vectors in :math:`[0, 1]^{K \times d}`.
    """

    def __init__(self, *, c_num_classes, c_degree, h0_alpha_vec=None, h0_beta_vec=None, seed=None):

        # constants
        self.c_degree = _check.pos_int(c_degree, 'c_degree', ParameterFormatError)
        self.c_num_classes = _check.pos_int(c_num_classes, 'c_num_classes', ParameterFormatError)
        self.c_rng = np.random.default_rng(seed)

        # h0_params
        self.h0_alpha_vec = np.ones(self.c_num_classes) / 2.0
        self.h0_beta_vec = np.ones(self.c_degree) / 2.0

        self.LN_C_H0_ALPHA = None
        self.LN_C_H0_BETA = None

        # hn_params
        self.hn_alpha_vec = np.empty([self.c_num_classes])
        self.hn_beta_vec = np.empty([self.c_degree])

        self.r_vecs = None
        self.ns = np.empty([self.c_num_classes])
        self.s_mats = np.empty([self.c_num_classes, self.c_degree])

        self.p_pi_vec = np.empty([self.c_num_classes])
        self.p_theta_vecs = np.empty([self.c_num_classes, self.c_degree])

        self.set_h0_params(h0_alpha_vec, h0_beta_vec)

    def set_h0_params(self, h0_alpha_vec=None, h0_beta_vec=None):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h0_alpha_vec : numpy.ndarra, optional
            A vector of positive real numbers.
        h0_beta_vec : numpy.ndarray, optional
            A vector of positive real numbers.
        """

        if h0_alpha_vec is not None:
            _check.pos_float_vec(h0_alpha_vec, 'h0_alpha_vec', ParameterFormatError)
            if h0_alpha_vec.shape[0] != self.c_num_classes:
                raise (ParameterFormatError(
                    "h0_alpha_vec.shape[0] must coincide with c_num_classes: " +
                    f"h0_alpha_vec.shape[0]={h0_alpha_vec.shape[0]}, c_num_classes={self.c_num_classes}"))
            self.h0_alpha_vec[:] = h0_alpha_vec

        if h0_beta_vec is not None:
            _check.pos_float_vec(h0_beta_vec, 'h0_beta_vec', ParameterFormatError)
            if h0_beta_vec.shape[0] != self.c_degree:
                raise (ParameterFormatError("h0_beta_vec.shape[0] must coincide with c_degree: " +
                                            f"h0_beta_vec.shape[0]={h0_beta_vec.shape[0]}, c_degree={self.c_degree}"))
            self.h0_beta_vec[:] = h0_beta_vec

        self.LN_C_H0_ALPHA = gammaln(self.h0_alpha_vec.sum()) - gammaln(self.h0_alpha_vec).sum()
        self.LN_C_H0_BETA = gammaln(self.h0_beta_vec.sum()) - gammaln(self.h0_beta_vec).sum()

        self.reset_hn_params()

    def get_h0_params(self):
        """Get the initial values of the hyperparameters of the posterior distribution.
        Returns
        -------
        h0_params : dict of {str: numpy.ndarray}
            * ``"h0_alpha_vec"`` : The value of ``self.h0_alpha_vec``
            * ``"h0_beta_vec"`` : The value of ``self.h0_beta_vec``
        """
        return {"h0_alpha_vec": self.h0_alpha_vec, "h0_beta_vec": self.h0_beta_vec}

    def set_hn_params(self, hn_alpha_vec=None, hn_beta_vec=None):
        """Set updated values of the hyperparameter of the posterior distribution.
        Parameters
        ----------
        hn_alpha_vec : numpy.ndarray, optional
            A vector of positive real numbers.
        hn_beta_vec : numpy.ndarray, optional
            A vector of positive real numbers.
        """

        if hn_alpha_vec is not None:
            _check.pos_float_vec(hn_alpha_vec, 'hn_alpha_vec', ParameterFormatError)
            if hn_alpha_vec.shape[0] != self.c_num_classes:
                raise (ParameterFormatError(
                    "hn_alpha_vec.shape[0] must coincide with c_num_classes: " +
                    f"hn_alpha_vec.shape[0]={hn_alpha_vec.shape[0]}, c_num_classes={self.c_num_classes}"))
            self.hn_alpha_vec[:] = hn_alpha_vec

        if hn_beta_vec is not None:
            _check.pos_float_vec(hn_beta_vec, 'hn_beta_vec', ParameterFormatError)
            if hn_beta_vec.shape[0] != self.c_degree:
                raise (ParameterFormatError("hn_beta_vec.shape[0] must coincide with c_degree: " +
                                            f"hn_beta_vec.shape[0]={hn_beta_vec.shape[0]}, c_degree={self.c_degree}"))
            self.hn_beta_vec[:] = hn_beta_vec

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.
        Returns
        -------
        hn_params : dict of {str: numpy.ndarray}
            * ``"hn_alpha_vec"`` : The value of ``self.hn_alpha_vec``
            * ``"hn_beta_vecs"`` : The value of ``self.hn_beta_vecs``
        """
        return {"hn_alpha_vec": self.hn_alpha_vec, "hn_beta_vec": self.hn_beta_vec}

    def reset_hn_params(self):
        """Reset the hyperparameters of the posterior distribution to their initial values.

        They are reset to `self.h0_alpha_vec`, and `self.h0_beta_vec`.
        Note that the parameters of the predictive distribution are also calculated from them.
        """

        self.hn_alpha_vec[:] = self.h0_alpha_vec
        self.hn_beta_vec[:] = self.h0_beta_vec
