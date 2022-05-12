# Code Author
# Kohei Horinouchi <horinochi_18@toki.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
# Document Author
# Kohei Horinouchi <horinochi_18@toki.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
import warnings
from matplotlib import projections
from more_itertools import peekable
import numpy as np
from scipy.stats import dirichlet as ss_dirichlet

import math

import matplotlib.pyplot as plt

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from .. import base
from .._exceptions import (
    ParameterFormatError,
    DataFormatError,
    CriteriaError,
    ResultWarning,
)


class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution

    Parameters
    ----------
    d : int, optional
        a positive integer (2 or more), by default 3
    p_vec : numpy ndarray, optional
        a real vector in :math:`[0, 1]^d`, by default an empty vector
    h_alpha_vec : numpy ndarray, optional
        a real vector, by default an empty vector
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """

    def __init__(
        self, *, d=3, p_vec=[], h_alpha_vec=[], seed=None,
    ):
        if d == 0 or d == 1:
            raise (ParameterFormatError("d must be 2 or more."))
        else:
            self.d = d

        if len(p_vec) == 0:
            self.p_vec = np.full(self.d, 1 / self.d)
        else:
            self.p_vec = p_vec
        if len(h_alpha_vec) == 0:
            self.h_alpha_vec = np.full(self.d, 1.0)
        else:
            self.h_alpha_vec = h_alpha_vec
        if d != np.size(self.p_vec):
            raise (ParameterFormatError("the number of elements of p_vec must be d."))
        if d != np.size(self.h_alpha_vec):
            raise (
                ParameterFormatError("the number of elements of h_alpha_vec must be d.")
            )
        for i in range(d):
            if self.p_vec[i] < 0.0 or self.p_vec[i] > 1.0:
                raise (ParameterFormatError("the elements of p_vec must be in [0,1]."))

        for i in range(d):
            if self.h_alpha_vec[i] < 0.0:
                raise (
                    ParameterFormatError(
                        "the elements of h_alpha_vec must be a positive real value."
                    )
                )

        if math.isclose(np.sum(self.p_vec), 1.0) == False:
            raise (ParameterFormatError("sum of the elements of p_vec must be 1."))
        self.rng = np.random.default_rng(seed)

    def set_h_params(self, h_alpha_vec):
        """
        Parameters
        ----------
        h_alpha : numpy ndarray
            a positive real vector
        """
        self.h_alpha_vec = h_alpha_vec

    def get_h_params(self):
        """
        Returns
        -------
        h_params : {str:numpy ndarray}
            ``{"h_alpha_vec": self.h_alpha_vec}``
        """
        return {"h_alpha_vec": self.h_alpha_vec}

    def save_h_params(self, filename):
        """Save the hyperparameters as NumPy ``.npz`` format.

        Parameters
        ----------
        filename : str
            The filename to which the hyperparameters are saved.
            ``.npz`` will be appended if it isn't there.
        
        See Also
        --------
        `numpy.savez_compressed <https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed>`_
        """
        np.savez_compressed(filename, h_alpha_vec=self.h_alpha_vec)

    def load_h_params(self, filename):
        """Load the hyperparameters saved by ``save_h_params``.

        Parameters
        ----------
        filename : str
            The filename to be loaded. It must be a NpzFile with keywords: \"h_alpha\" and \"h_beta\".
        """
        h_params = np.load(filename)
        self.set_h_params(h_params["h_alpha_vec"])

    def gen_params(self):
        """
        Generate the parameter of the stochastic data generative model from the prior distribution.
        
        The generated vaule is set at ``self.p_vec``.
        """
        self.p_vec = self.rng.dirichlet(self.h_alpha_vec)

    def set_params(self, p_vec):
        """
        Parameters
        ----------
        p : numpy ndarray
            a real vector :math:`p \in [0, 1]^d`
        """
        self.p_vec = p_vec

    def get_params(self):
        """
        Returns
        -------
        params : {str:numpy ndarray}
            ``{"p_vec":self.p_vec}``
        """
        return {"p_vec": self.p_vec}

    def save_params(self, filename):
        """Save the parameter as NumPy ``.npz`` format.

        Parameters
        ----------
        filename : str
            The filename to which the hyperparameters are saved.
            ``.npz`` will be appended if it isn't there.
        
        See Also
        --------
        `numpy.savez_compressed <https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed>`_
        """
        np.savez_compressed(filename, p_vec=self.p_vec)

    def load_params(self, filename):
        """Load the parameter saved by ``save_h_params``.

        Parameters
        ----------
        filename : str
            The filename to be loaded. It must be a NpzFile with keywords: \"p\".
        """
        params = np.load(filename)
        self.set_params(params["p_vec"])

    def gen_sample(self, sample_size):
        """

        Parameters
        ----------
        sample_size : int
            A positive integer

        Returns
        -------
        X : numpy ndarray
            1 dimensional array whose size is ``self.d + 1`` and elements are in :math:`[0,sample_size]`.
        """
        if sample_size <= 0:
            raise (DataFormatError("sample_size must be a positive integer."))
        return self.rng.multinomial(sample_size, self.p_vec)

    def save_sample(self, filename, sample_size):
        """Save the generated sample as NumPy ``.npz`` format.

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_size : int
            A positive integer
        
        See Also
        --------
        `numpy.savez_compressed <https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed>`_
        """
        if sample_size <= 0:
            raise (DataFormatError("sample_size must be a positive integer."))
        np.savez_compressed(
            filename, X=self.rng.multinomial(sample_size, self.p_vec),
        )

    def visualize_model(self, sample_size=20, sample_num=5):
        """

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
        p_vec :  [0.33333333 0.33333333 0.33333333]
        X0:[8 7 5]
        X1:[7 9 4]
        X2:[ 5 10  5]
        X3:[10  6  4]
        X4:[6 8 6]

        .. image:: ./images/categorical_example.png
        """
        cmap = plt.get_cmap("hsv")
        print("p_vec : ", self.p_vec)
        fig, ax = plt.subplots(figsize=(5, sample_num))
        for i in range(sample_num):
            X = self.gen_sample(sample_size)
            print(f"X{i}:{X}")
            if i == 0:
                for j in range(self.d):
                    if j == 0:
                        ax.barh(i, X[j], label=j, color=cmap(j / self.d))
                    else:
                        ax.barh(
                            i,
                            X[j],
                            left=np.cumsum(X)[j - 1],
                            label=j,
                            color=cmap(j / self.d),
                        )
            else:
                for j in range(self.d):
                    if j == 0:
                        ax.barh(i, X[j], color=cmap(j / self.d))
                    else:
                        ax.barh(
                            i, X[j], left=np.cumsum(X)[j - 1], color=cmap(j / self.d),
                        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_xlabel("Number of occurrences")
        plt.show()

class LearnModel(base.Posterior, base.PredictiveMixin):
    def __init__(self, d=3, h0_alpha_vec=[]):
        if d == 0 or d == 1:
            raise (ParameterFormatError("d must be 2 or more."))
        else:
            self.d = d

        if len(h0_alpha_vec) == 0:
            self.h0_alpha_vec = np.full(self.d, 5.0)
        else:
            self.h0_alpha_vec = h0_alpha_vec

        for i in range(d):
            if self.h0_alpha_vec[i] < 0.0:
                raise (
                    ParameterFormatError(
                        "the elements of h_alpha_vec must be a positive real value."
                    )
                )

        if d != np.size(self.h0_alpha_vec):
            raise (
                ParameterFormatError(
                    "the number of elements of h0_alpha_vec must be d."
                )
            )
        self.hn_alpha_vec = self.h0_alpha_vec
        self.p_alpha_vec = self.hn_alpha_vec

    def set_h0_params(self, h0_alpha_vec):
        self.h0_alpha_vec = h0_alpha_vec
        self.hn_alpha_vec = self.h0_alpha_vec
        self.p_alpha_vec = self.hn_alpha_vec

    def get_h0_params(self):
        return {"h0_alpha_vec": self.h0_alpha_vec}

    def save_h0_params(self, filename):
        np.savez_compressed(filename, h0_alpha_vec=self.h0_alpha_vec)

    def load_h0_params(self, filename):
        h0_params = np.load(filename)
        if "h0_alpha_vec" not in h0_params.files:
            raise (
                ParameterFormatError(
                    filename + ' must be a NpzFile with keywords: "h0_alpha_vec".'
                )
            )
        self.set_h0_params(h0_params["h0_alpha_vec"])

    def get_hn_params(self):
        return {"hn_alpha_vec": self.hn_alpha_vec}

    def save_hn_params(self, filename):
        np.savez_compressed(filename, hn_alpha_vec=self.hn_alpha_vec)

    def reset_hn_params(self):
        self.hn_alpha_vec = self.h0_alpha_vec
        self.p_alpha_vec = self.hn_alpha_vec

    def update_posterior(self, X=[]):
        if np.size(X) == np.size(self.hn_alpha_vec):  # Xの次元とハイパーパラメータの次元が等しいとき
            self.hn_alpha_vec += X
        else:  # Xの次元とハイパーパラメータの次元が異なるとき
            self.hn_alpha_vec = np.full(np.size(X), 5.0)  # Xの次元に合わせてハイパーパラメータを設定しなおす
            self.hn_alpha_vec += X
            self.d = np.size(X)  # 次元の変更

    def estimate_params(self, loss="squared"):
        if loss == "squared":
            return self.hn_alpha_vec / np.sum(self.hn_alpha_vec)
        elif loss == "0-1":
            if np.all(self.hn_alpha_vec > np.ones(self.d)):  # 全てのハイパーパラメータが1以上
                return (self.hn_alpha_vec - np.ones(self.d)) / (
                    np.sum(self.hn_alpha_vec) - self.d
                )
            else:
                # warningを返す
                return None
        elif loss == "abs":
            return None  # 中央値はない
        elif loss == "KL":
            return ss_dirichlet(self.hn_alpha_vec)
        else:
            raise (
                CriteriaError(
                    "Unsupported loss function! "
                    'This function supports "squared", "0-1", "abs", and "KL".'
                )
            )

    # 無くて良い
    def estimate_interval(self, confidence=0.95):
        return None

    def visualize_posterior(self):
        if self.d == 3:
            p1_range = np.linspace(0, 1, 200)
            p2_range = np.linspace(0, 1, 200)
            X, Y = np.meshgrid(p1_range, p2_range)
            p3_range = []
            X[X + Y >= 1] = 0
            Y[X + Y >= 1] = 0
            for _x, _y, _z in zip(X.flatten(), Y.flatten(), (1 - X - Y).flatten()):
                p3_range.append(
                    ss_dirichlet.pdf(np.array([_x, _y, _z]), self.hn_alpha_vec)
                )
            Z = np.array(p3_range).reshape(X.shape)
            ax3d = plt.axes(projection="3d")
            ax3d.plot_surface(X, Y, Z, cmap="plasma")
            ax3d.set_zlim(None)
            ax3d.set_xlabel("$p_1$", size=16)
            ax3d.set_ylabel("$p_2$", size=16)
            ax3d.set_zlabel("pdf$(p_1,p_2,p_3)$", size=16)
            plt.show()
        else:
            raise (ParameterFormatError("There is not a glaph when d is not 3."))

    def get_p_params(self):
        return {"p_alpha_vec": self.p_alpha_vec}

    def save_p_params(self, filename):
        np.savez_compressed(filename, p_alpha_vec=self.p_alpha_vec)

    def calc_pred_dist(self):
        self.p_alpha_vec = self.hn_alpha_vec

    def make_prediction(self, loss="squared"):
        if loss == "squared":
            return self.p_alpha_vec / (np.sum(self.p_alpha_vec))
        elif loss == "0-1" or loss == "abs":
            return None
        elif loss == "KL":
            return None
        else:
            raise (
                CriteriaError(
                    "Unsupported loss function! "
                    'This function supports "squared", "0-1", "abs", and "KL".'
                )
            )

    def pred_and_update(self, x, loss="squared"):
        self.calc_pred_dist()
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(x)
        return prediction
