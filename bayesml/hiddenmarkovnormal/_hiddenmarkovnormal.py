# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Jun Nishikawa <jun.b.nishikawa@gmail.com>
from email import message
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
    """The stochastic data generative model and the prior distribution.

    Parameters
    ----------
    c_num_classes : int
        a positive integer
    c_degree : int
        a positive integer
    pi_vec : numpy.ndarray, optional
        A vector of real numbers in :math:`[0, 1]`, 
        by default [1/c_num_classes, 1/c_num_classes, ... , 1/c_num_classes].
        Sum of its elements must be 1.0.
    a_mat : numpy.ndarray, optional
        A matrix of real numbers in :math:`[0, 1]`, 
        by default a matrix obtained by stacking 
        [1/c_num_classes, 1/c_num_classes, ... , 1/c_num_classes].
        Sum of the elements of each row vector must be 1.0.
        If a single vector is input, will be broadcasted.
    mu_vecs : numpy.ndarray, optional
        Vectors of real numbers, 
        by default zero vectors.
        If a single vector is input, will be broadcasted.
    lambda_mats : numpy.ndarray, optional
        Positive definite symetric matrices, 
        by default the identity matrices.
        If a single matrix is input, it will be broadcasted.
    h_eta_vec : numpy.ndarray, optional
        A vector of positive real numbers, 
        by default [1/2, 1/2, ... , 1/2]
    h_zeta_vecs : numpy.ndarray, optional
        Vectors of positive numbers, 
        by default vectors whose elements are all 1/2
        If a single vector is input, will be broadcasted.
    h_m_vecs : numpy.ndarray, optional
        Vectors of real numbers, 
        by default zero vectors
        If a single vector is input, will be broadcasted.
    h_kappas : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0].
        If a single real number is input, it will be broadcasted.
    h_nus : float or numpy.ndarray, optional
        Real numbers greater than ``c_degree-1``,  
        by default [c_degree, c_degree, ... , c_degree]
        If a single real number is input, it will be broadcasted.
    h_w_mats : numpy.ndarray, optional
        Positive definite symetric matrices, 
        by default the identity matrices.
        If a single matrix is input, it will be broadcasted.
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(), 
        by default None
    """
    def __init__(
            self,
            c_num_classes,
            c_degree,
            *,
            pi_vec=None,
            a_mat=None,
            mu_vecs=None,
            lambda_mats=None,
            h_eta_vec=None,
            h_zeta_vecs=None,
            h_m_vecs=None,
            h_kappas=None,
            h_nus=None,
            h_w_mats=None,
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
        self.h_eta_vec = np.ones(self.c_num_classes) / 2.0
        self.h_zeta_vecs = np.ones([self.c_num_classes,self.c_num_classes]) / 2.0
        self.h_m_vecs = np.zeros([self.c_num_classes,self.c_degree])
        self.h_kappas = np.ones([self.c_num_classes])
        self.h_nus = np.ones(self.c_num_classes) * self.c_degree
        self.h_w_mats = np.tile(np.identity(self.c_degree),[self.c_num_classes,1,1])

        self.set_params(
            pi_vec,
            a_mat,
            mu_vecs,
            lambda_mats)

        self.set_h_params(
            h_eta_vec,
            h_zeta_vecs,
            h_m_vecs,
            h_kappas,
            h_nus,
            h_w_mats)

    def set_params(
            self,
            pi_vec=None,
            a_mat=None,
            mu_vecs=None,
            lambda_mats=None
            ):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        pi_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]`, 
            by default [1/c_num_classes, 1/c_num_classes, ... , 1/c_num_classes].
            Sum of its elements must be 1.0.
        a_mat : numpy.ndarray, optional
            A matrix of real numbers in :math:`[0, 1]`, 
            by default a matrix obtained by stacking 
            [1/c_num_classes, 1/c_num_classes, ... , 1/c_num_classes].
            Sum of the elements of each row vector must be 1.0.
            If a single vector is input, will be broadcasted.
        mu_vecs : numpy.ndarray, optional
            Vectors of real numbers, 
            by default zero vectors.
            If a single vector is input, will be broadcasted.
        lambda_mats : numpy.ndarray, optional
            Positive definite symetric matrices, 
            by default the identity matrices.
            If a single matrix is input, it will be broadcasted.
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

        if mu_vecs is not None:
            _check.float_vecs(mu_vecs, "mu_vecs", ParameterFormatError)
            _check.shape_consistency(
                mu_vecs.shape[-1],"mu_vecs.shape[-1]", 
                self.c_degree,"self.c_degree", 
                ParameterFormatError
                )
            self.mu_vecs[:] = mu_vecs

        if lambda_mats is not None:
            _check.pos_def_sym_mats(lambda_mats,'lambda_mats',ParameterFormatError)
            _check.shape_consistency(
                lambda_mats.shape[-1],"lambda_mats.shape[-1] and lambda_mats.shape[-2]", 
                self.c_degree,"self.c_degree", 
                ParameterFormatError
                )
            self.lambda_mats[:] = lambda_mats

    def set_h_params(
            self,
            h_eta_vec=None,
            h_zeta_vecs=None,
            h_m_vecs=None,
            h_kappas=None,
            h_nus=None,
            h_w_mats=None,
            ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_eta_vec : numpy.ndarray, optional
            A vector of positive real numbers, 
            by default [1/2, 1/2, ... , 1/2]
        h_zeta_vecs : numpy.ndarray, optional
            Vectors of positive numbers, 
            by default vectors whose elements are all 1/2
            If a single vector is input, will be broadcasted.
        h_m_vecs : numpy.ndarray, optional
            Vectors of real numbers, 
            by default zero vectors
            If a single vector is input, will be broadcasted.
        h_kappas : float or numpy.ndarray, optional
            Positive real numbers, 
            by default [1.0, 1.0, ... , 1.0].
            If a single real number is input, it will be broadcasted.
        h_nus : float or numpy.ndarray, optional
            Real numbers greater than ``c_degree-1``,  
            by default [c_degree, c_degree, ... , c_degree]
            If a single real number is input, it will be broadcasted.
        h_w_mats : numpy.ndarray, optional
            Positive definite symetric matrices, 
            by default the identity matrices.
            If a single matrix is input, it will be broadcasted.
        """
        if h_eta_vec is not None:
            _check.pos_floats(h_eta_vec,'h_eta_vec',ParameterFormatError)
            self.h_eta_vec[:] = h_eta_vec

        if h_zeta_vecs is not None:
            _check.pos_floats(h_zeta_vecs, 'h_zeta_vecs', ParameterFormatError)
            self.h_zeta_vecs[:] = h_zeta_vecs

        if h_m_vecs is not None:
            _check.float_vecs(h_m_vecs, "h_m_vecs", ParameterFormatError)
            _check.shape_consistency(
                h_m_vecs.shape[-1],"h_m_vecs.shape[-1]", 
                self.c_degree,"self.c_degree", 
                ParameterFormatError
                )
            self.h_m_vecs[:] = h_m_vecs

        if h_kappas is not None:
            _check.pos_floats(h_kappas, "h_kappas", ParameterFormatError)
            self.h_kappas[:] = h_kappas

        if h_nus is not None:
            _check.floats(h_nus, "h_nus", ParameterFormatError)
            if np.all(h_nus <= self.c_degree - 1):
                raise(ParameterFormatError(
                    "All the values in h_nus must be greater than self.c_degree - 1: "
                    + f"self.c_degree = {self.c_degree}, h_nus = {h_nus}"))
            self.h_nus[:] = h_nus

        if h_w_mats is not None:
            _check.pos_def_sym_mats(h_w_mats,'h_w_mats',ParameterFormatError)
            _check.shape_consistency(
                h_w_mats.shape[-1],"h_w_mats.shape[-1] and h_w_mats.shape[-2]", 
                self.c_degree,"self.c_degree", 
                ParameterFormatError
                )
            self.h_w_mats[:] = h_w_mats

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : {str:float, numpy.ndarray}
            * ``"pi_vec"`` : The value of ``self.pi_vec``
            * ``"a_mat"`` : The value of ``self.a_mat``
            * ``"mu_vecs"`` : The value of ``self.mu_vecs``
            * ``"lambda_mats"`` : The value of ``self.lambda_mats``
        """
        return {'pi_vec':self.pi_vec,
                'a_mat':self.a_mat,
                'mu_vecs':self.mu_vecs,
                'lambda_mats': self.lambda_mats}

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.
        
        Returns
        -------
        h_params : {str:float, np.ndarray}
            * ``"h_eta_vec"`` : The value of ``self.h_eta_vec``
            * ``"h_zeta_vecs"`` : The value of ``self.h_zeta_vecs``
            * ``"h_m_vecs"`` : The value of ``self.h_m_vecs``
            * ``"h_kappas"`` : The value of ``self.h_kappas``
            * ``"h_nus"`` : The value of ``self.h_nus``
            * ``"h_w_mats"`` : The value of ``self.h_w_mats``
        """
        return {'h_eta_vec':self.h_eta_vec,
                'h_zeta_vecs':self.h_zeta_vecs,
                'h_m_vecs':self.h_m_vecs,
                'h_kappas':self.h_kappas,
                'h_nus':self.h_nus,
                'h_w_mats':self.h_w_mats}

    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        To confirm the generated vaules, use `self.get_params()`.
        """
        self.pi_vec[:] = self.rng.dirichlet(self.h_eta_vec)
        for k in range(self.c_num_classes):
            self.a_mat[k] = self.rng.dirichlet(self.h_zeta_vecs[k])
        for k in range(self.c_num_classes):
            self.lambda_mats[k] = ss_wishart.rvs(df=self.h_nus[k],scale=self.h_w_mats[k],random_state=self.rng)
            self.mu_vecs[k] = self.rng.multivariate_normal(mean=self.h_m_vecs[k],cov=np.linalg.inv(self.h_kappas[k]*self.lambda_mats[k]))

    def gen_sample(self,sample_length):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_length : int
            A positive integer

        Returns
        -------
        x : numpy ndarray
            2-dimensional array whose shape is 
            ``(sample_length,c_degree)`` .
            Its elements are real numbers.
        z : numpy ndarray
            2-dimensional array whose shape is 
            ``(sample_length,c_num_classes)`` 
            whose rows are one-hot vectors.
        """
        _check.pos_int(sample_length,'sample_length',DataFormatError)
        z = np.zeros([sample_length,self.c_num_classes],dtype=int)
        x = np.empty([sample_length,self.c_degree])
        _lambda_mats_inv = np.linalg.inv(self.lambda_mats)
        
        # i=0
        k = self.rng.choice(self.c_num_classes,p=self.pi_vec)
        z[0,k] = 1
        x[0] = self.rng.multivariate_normal(mean=self.mu_vecs[k],cov=_lambda_mats_inv[k])
        # i>0
        for i in range(1,sample_length):
            k = self.rng.choice(self.c_num_classes,p=self.a_mat[np.argmax(z[i-1])])
            z[i,k] = 1
            x[i] = self.rng.multivariate_normal(mean=self.mu_vecs[k],cov=_lambda_mats_inv[k])
        return x,z
    
    def save_sample(self,filename,sample_length):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as a NpzFile with keyword: \"x\", \"z\".
        
        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_length : int
            A positive integer
        
        See Also
        --------
        numpy.savez_compressed
        """
        x,z=self.gen_sample(sample_length)
        np.savez_compressed(filename,x=x,z=z)
    
    def visualize_model(self,sample_length=200):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_length : int, optional
            A positive integer, by default 100
        
        Examples
        --------
        >>> from bayesml import hiddenmarkovnormal
        >>> import numpy as np
        >>> model = hiddenmarkovnormal.GenModel(
                c_num_classes=2,
                c_degree=1,
                mu_vecs=np.array([[5],[-5]]),
                a_mat=np.array([[0.95,0.05],[0.1,0.9]]))
        >>> model.visualize_model()
        pi_vec:
        [0.5 0.5]
        a_mat:
        [[0.95 0.05]
        [0.1  0.9 ]]
        mu_vecs:
        [[ 5.]
        [-5.]]
        lambda_mats:
        [[[1.]]

        [[1.]]]

        .. image:: ./images/hiddenmarkovnormal_example.png
        """
        if self.c_degree == 1:
            print(f"pi_vec:\n {self.pi_vec}")
            print(f"a_mat:\n {self.a_mat}")
            print(f"mu_vecs:\n {self.mu_vecs}")
            print(f"lambda_mats:\n {self.lambda_mats}")
            _lambda_mats_inv = np.linalg.inv(self.lambda_mats)
            fig, axes = plt.subplots()
            sample, latent_vars = self.gen_sample(sample_length)

            change_points = [0]
            for i in range(1,sample_length):
                if np.any(latent_vars[i-1] != latent_vars[i]):
                    change_points.append(i)
            change_points.append(sample_length)

            cm = plt.get_cmap('jet')
            for i in range(1,len(change_points)):
                axes.axvspan(
                    change_points[i-1],
                    change_points[i],
                    color=cm(
                        int((np.argmax(latent_vars[change_points[i-1]])
                             / (self.c_num_classes-1)) * 255)
                        ),
                    alpha=0.3,
                    ls='',
                    )
                axes.plot(
                    np.linspace(change_points[i-1],change_points[i],100),
                    np.ones(100) * self.mu_vecs[np.argmax(latent_vars[change_points[i-1]])],
                    c='red',
                    )
            axes.plot(np.arange(sample.shape[0]),sample)
            axes.set_xlabel("time")
            axes.set_ylabel("x")
            plt.show()
        else:
            raise(ParameterFormatError("if c_degree > 1, it is impossible to visualize the model by this function."))

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    c_num_classes : int
        A positive integer.
    c_degree : int
        A positive integer.
    h0_eta_vec : numpy.ndarray, optional
        A vector of positive real numbers, 
        by default [1/2, 1/2, ... , 1/2].
        If a real number is input, it will be broadcasted.
    h0_zeta_vecs : numpy.ndarray, optional
        Vectors of positive numbers, 
        by default vectors whose elements are all 1.0
        If a real number or a single vector is input, will be broadcasted.
    h0_m_vecs : numpy.ndarray, optional
        Vectors of real numbers, 
        by default zero vectors
        If a single vector is input, will be broadcasted.
    h0_kappas : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
    h0_nus : float or numpy.ndarray, optional
        Real numbers greater than c_degree-1, 
        by default c_degree.
        If a single real number is input, it will be broadcasted.
    h0_w_mats : numpy.ndarray, optional
        Positive definite symetric matrices, 
        by default the identity matrices
        If a single matrix is input, it will be broadcasted.
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None.

    Attributes
    ----------
    h0_w_mats_inv : numpy.ndarray
        the inverse matrices of h0_w_mats
    hn_eta_vec : numpy.ndarray
        A vector of positive real numbers
    hn_zeta_vecs : numpy.ndarray
        Vectors of positive numbers
    hn_m_vecs : numpy.ndarray
        Vectors of real numbers.
    hn_kappas : numpy.ndarray
        Positive real numbers
    hn_nus : numpy.ndarray
        Real numbers greater than c_degree-1.
    hn_w_mats : numpy.ndarray
        Positive definite symetric matrices.
    hn_w_mats_inv : numpy.ndarray
        the inverse matrices of hn_w_mats
    p_mu_vecs : numpy.ndarray
        vectors of real numbers
    p_nus : numpy.ndarray
        positive real numbers
    p_lambda_mats : numpy.ndarray
        positive definite symetric matrices
    """
    def __init__(
            self,
            c_num_classes,
            c_degree,
            *,
            h0_eta_vec=None,
            h0_zeta_vecs=None,
            h0_m_vecs=None,
            h0_kappas=None,
            h0_nus=None,
            h0_w_mats=None,
            seed = None
            ):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.c_num_classes = _check.pos_int(c_num_classes,'c_num_classes',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # h0_params
        self.h0_eta_vec = np.ones(self.c_num_classes) / 2.0
        self.h0_zeta_vecs = np.ones([self.c_num_classes,self.c_num_classes]) / 2.0
        self.h0_m_vecs = np.zeros([self.c_num_classes,self.c_degree])
        self.h0_kappas = np.ones([self.c_num_classes])
        self.h0_nus = np.ones(self.c_num_classes) * self.c_degree
        self.h0_w_mats = np.tile(np.identity(self.c_degree),[self.c_num_classes,1,1])
        self.h0_w_mats_inv = np.linalg.inv(self.h0_w_mats)

        self._ln_c_h0_eta_vec = 0.0
        self._ln_c_h0_zeta_vecs_sum = 0.0
        self._ln_b_h0_w_nus = np.empty(self.c_num_classes)

        # hn_params
        self.hn_eta_vec = np.empty(self.c_num_classes)
        self.hn_zeta_vecs = np.empty([self.c_num_classes,self.c_num_classes])
        self.hn_m_vecs = np.empty([self.c_num_classes,self.c_degree])
        self.hn_kappas = np.empty([self.c_num_classes])
        self.hn_nus = np.empty(self.c_num_classes)
        self.hn_w_mats = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        self.hn_w_mats_inv = np.empty([self.c_num_classes,self.c_degree,self.c_degree])

        self._length = 0
        self._ln_rho = None
        self._rho = None
        self.alpha_vecs = None
        self.beta_vecs = None
        self.gamma_vecs = None
        self.xi_mats = None
        self._cs = None
        self._e_lambda_mats = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        self._e_ln_lambda_dets = np.empty(self.c_num_classes)
        self._ln_b_hn_w_nus = np.empty(self.c_num_classes)
        self._ln_pi_tilde_vec = np.empty(self.c_num_classes)
        self._pi_tilde_vec = np.empty(self.c_num_classes)
        self._ln_a_tilde_mat = np.empty([self.c_num_classes,self.c_num_classes])
        self._a_tilde_mat = np.empty([self.c_num_classes,self.c_num_classes])
        self._ln_c_hn_zeta_vecs_sum = 0.0

        # statistics
        self.x_bar_vecs = np.empty([self.c_num_classes,self.c_degree])
        self.ns = np.empty(self.c_num_classes)
        self.ms = np.empty([self.c_num_classes,self.c_num_classes])
        self.s_mats = np.empty([self.c_num_classes,self.c_degree,self.c_degree])

        # variational lower bound
        self.vl = 0.0
        self._vl_p_x = 0.0
        self._vl_p_z = 0.0
        self._vl_p_pi = 0.0
        self._vl_p_a = 0.0
        self._vl_p_mu_lambda = 0.0
        self._vl_q_z = 0.0
        self._vl_q_pi = 0.0
        self._vl_q_a = 0.0
        self._vl_q_mu_lambda = 0.0

        # p_params
        self.p_a_mat = np.ones([self.c_num_classes,self.c_num_classes]) / self.c_num_classes
        self.p_mu_vecs = np.empty([self.c_num_classes,self.c_degree])
        self.p_nus = np.empty([self.c_num_classes])
        self.p_lambda_mats = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        self.p_lambda_mats_inv = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        
        self.set_h0_params(
            h0_eta_vec,
            h0_zeta_vecs,
            h0_m_vecs,
            h0_kappas,
            h0_nus,
            h0_w_mats,
        )

    def set_h0_params(
            self,
            h0_eta_vec = None,
            h0_zeta_vecs = None,
            h0_m_vecs = None,
            h0_kappas = None,
            h0_nus = None,
            h0_w_mats = None,
            ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h0_eta_vec : numpy.ndarray, optional
            A vector of positive real numbers, 
            by default [1/2, 1/2, ... , 1/2].
            If a real number is input, it will be broadcasted.
        h0_zeta_vecs : numpy.ndarray, optional
            Vectors of positive numbers, 
            by default vectors whose elements are all 1.0
            If a real number or a single vector is input, will be broadcasted.
        h0_m_vecs : numpy.ndarray, optional
            Vectors of real numbers, 
            by default zero vectors
            If a single vector is input, will be broadcasted.
        h0_kappas : float or numpy.ndarray, optional
            Positive real numbers, 
            by default [1.0, 1.0, ... , 1.0]
            If a single real number is input, it will be broadcasted.
        h0_nus : float or numpy.ndarray, optional
            Real numbers greater than c_degree-1, 
            by default c_degree.
            If a single real number is input, it will be broadcasted.
        h0_w_mats : numpy.ndarray, optional
            Positive definite symetric matrices, 
            by default the identity matrices
            If a single matrix is input, it will be broadcasted.
        """
        if h0_eta_vec is not None:
            _check.pos_floats(h0_eta_vec,'h0_eta_vec',ParameterFormatError)
            self.h0_eta_vec[:] = h0_eta_vec

        if h0_zeta_vecs is not None:
            _check.pos_floats(h0_zeta_vecs, 'h0_zeta_vecs', ParameterFormatError)
            self.h0_zeta_vecs[:] = h0_zeta_vecs

        if h0_m_vecs is not None:
            _check.float_vecs(h0_m_vecs, "h0_m_vecs", ParameterFormatError)
            _check.shape_consistency(
                h0_m_vecs.shape[-1],"h0_m_vecs.shape[-1]", 
                self.c_degree,"self.c_degree", 
                ParameterFormatError
                )
            self.h0_m_vecs[:] = h0_m_vecs

        if h0_kappas is not None:
            _check.pos_floats(h0_kappas, "h0_kappas", ParameterFormatError)
            self.h0_kappas[:] = h0_kappas

        if h0_nus is not None:
            _check.floats(h0_nus, "h0_nus", ParameterFormatError)
            if np.all(h0_nus <= self.c_degree - 1):
                raise(ParameterFormatError(
                    "All the values in h0_nus must be greater than self.c_degree - 1: "
                    + f"self.c_degree = {self.c_degree}, h0_nus = {h0_nus}"))
            self.h0_nus[:] = h0_nus

        if h0_w_mats is not None:
            _check.pos_def_sym_mats(h0_w_mats,'h0_w_mats',ParameterFormatError)
            _check.shape_consistency(
                h0_w_mats.shape[-1],"h0_w_mats.shape[-1] and h0_w_mats.shape[-2]", 
                self.c_degree,"self.c_degree", 
                ParameterFormatError
                )
            self.h0_w_mats[:] = h0_w_mats
        
        self.h0_w_mats_inv[:] = np.linalg.inv(self.h0_w_mats)

        self._calc_prior_char()
        self.reset_hn_params()

    def get_h0_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h0_params : dict of {str: numpy.ndarray}
            * ``"h0_eta_vec"`` : the value of ``self.h0_eta_vec``
            * ``"h0_zeta_vecs"`` : the value of ``self.h0_zeta_vecs``
            * ``"h0_m_vecs"`` : the value of ``self.h0_m_vecs``
            * ``"h0_kappas"`` : the value of ``self.h0_kappas``
            * ``"h0_nus"`` : the value of ``self.h0_nus``
            * ``"h0_w_mats"`` : the value of ``self.h0_w_mats``
        """
        return {'h0_eta_vec':self.h0_eta_vec,
                'h0_zeta_vecs':self.h0_zeta_vecs,
                'h0_m_vecs':self.h0_m_vecs,
                'h0_kappas':self.h0_kappas,
                'h0_nus':self.h0_nus,
                'h0_w_mats':self.h0_w_mats}
    
    def set_hn_params(
            self,
            hn_eta_vec = None,
            hn_zeta_vecs = None,
            hn_m_vecs = None,
            hn_kappas = None,
            hn_nus = None,
            hn_w_mats = None,
            ):
        """Set the hyperparameter of the posterior distribution.

        Parameters
        ----------
        hn_eta_vec : numpy.ndarray, optional
            A vector of positive real numbers, 
            by default [1/2, 1/2, ... , 1/2].
            If a real number is input, it will be broadcasted.
        hn_zeta_vecs : numpy.ndarray, optional
            Vectors of positive numbers, 
            by default vectors whose elements are all 1.0
            If a real number or a single vector is input, will be broadcasted.
        hn_m_vecs : numpy.ndarray, optional
            Vectors of real numbers, 
            by default zero vectors
            If a single vector is input, will be broadcasted.
        hn_kappas : float or numpy.ndarray, optional
            Positive real numbers, 
            by default [1.0, 1.0, ... , 1.0]
            If a single real number is input, it will be broadcasted.
        hn_nus : float or numpy.ndarray, optional
            Real numbers greater than c_degree-1, 
            by default c_degree.
            If a single real number is input, it will be broadcasted.
        hn_w_mats : numpy.ndarray, optional
            Positive definite symetric matrices, 
            by default the identity matrices
            If a single matrix is input, it will be broadcasted.
        """
        if hn_eta_vec is not None:
            _check.pos_floats(hn_eta_vec,'hn_eta_vec',ParameterFormatError)
            self.hn_eta_vec[:] = hn_eta_vec

        if hn_zeta_vecs is not None:
            _check.pos_floats(hn_zeta_vecs, 'hn_zeta_vecs', ParameterFormatError)
            self.hn_zeta_vecs[:] = hn_zeta_vecs

        if hn_m_vecs is not None:
            _check.float_vecs(hn_m_vecs, "hn_m_vecs", ParameterFormatError)
            _check.shape_consistency(
                hn_m_vecs.shape[-1],"hn_m_vecs.shape[-1]", 
                self.c_degree,"self.c_degree", 
                ParameterFormatError
                )
            self.hn_m_vecs[:] = hn_m_vecs

        if hn_kappas is not None:
            _check.pos_floats(hn_kappas, "hn_kappas", ParameterFormatError)
            self.hn_kappas[:] = hn_kappas

        if hn_nus is not None:
            _check.floats(hn_nus, "hn_nus", ParameterFormatError)
            if np.all(hn_nus <= self.c_degree - 1):
                raise(ParameterFormatError(
                    "All the values in hn_nus must be greater than self.c_degree - 1: "
                    + f"self.c_degree = {self.c_degree}, hn_nus = {hn_nus}"))
            self.hn_nus[:] = hn_nus

        if hn_w_mats is not None:
            _check.pos_def_sym_mats(hn_w_mats,'hn_w_mats',ParameterFormatError)
            _check.shape_consistency(
                hn_w_mats.shape[-1],"hn_w_mats.shape[-1] and hn_w_mats.shape[-2]", 
                self.c_degree,"self.c_degree", 
                ParameterFormatError
                )
            self.hn_w_mats[:] = hn_w_mats
        
        self.hn_w_mats_inv[:] = np.linalg.inv(self.hn_w_mats)

        self._calc_q_pi_char()
        self._calc_q_a_char()
        self._calc_q_lambda_char()

        self.calc_pred_dist()

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: numpy.ndarray}
            * ``"hn_eta_vec"`` : the value of ``self.hn_eta_vec``
            * ``"hn_zeta_vecs"`` : the value of ``self.hn_zeta_vecs``
            * ``"hn_m_vecs"`` : the value of ``self.hn_m_vecs``
            * ``"hn_kappas"`` : the value of ``self.hn_kappas``
            * ``"hn_nus"`` : the value of ``self.hn_nus``
            * ``"hn_w_mats"`` : the value of ``self.hn_w_mats``
        """
        return {'hn_eta_vec':self.hn_eta_vec,
                'hn_zeta_vecs':self.hn_zeta_vecs,
                'hn_m_vecs':self.hn_m_vecs,
                'hn_kappas':self.hn_kappas,
                'hn_nus':self.hn_nus,
                'hn_w_mats':self.hn_w_mats}
    
    def _calc_prior_char(self):
        self._ln_c_h0_eta_vec = gammaln(self.h0_eta_vec.sum()) - gammaln(self.h0_eta_vec).sum()
        self._ln_c_h0_zeta_vecs_sum = np.sum(gammaln(self.h0_zeta_vecs.sum(axis=1)) - gammaln(self.h0_zeta_vecs).sum(axis=1))
        self._ln_b_h0_w_nus = (
            - self.h0_nus*np.linalg.slogdet(self.h0_w_mats)[1]
            - self.h0_nus*self.c_degree*np.log(2.0)
            - self.c_degree*(self.c_degree-1)/2.0*np.log(np.pi)
            - np.sum(gammaln((self.h0_nus[:,np.newaxis]-np.arange(self.c_degree)) / 2.0),
                     axis=1) * 2.0
            ) / 2.0

    def _calc_n_m_x_bar_s(self,x):
        self.ns[:] = self.gamma_vecs.sum(axis=0)
        self.ms[:] = self.xi_mats.sum(axis=0) # xi must be initialized as a zero matrix
        self.x_bar_vecs[:] = (self.gamma_vecs[:,:,np.newaxis] * x[:,np.newaxis,:]).sum(axis=0) / self.ns[:,np.newaxis]
        self.s_mats[:] = np.sum(self.gamma_vecs[:,:,np.newaxis,np.newaxis]
                                * ((x[:,np.newaxis,:] - self.x_bar_vecs)[:,:,:,np.newaxis]
                                   @ (x[:,np.newaxis,:] - self.x_bar_vecs)[:,:,np.newaxis,:]),
                                axis=0) / self.ns[:,np.newaxis,np.newaxis]

    def _calc_q_pi_char(self):
        self._ln_pi_tilde_vec[:] = digamma(self.hn_eta_vec) - digamma(self.hn_eta_vec.sum())
        self._pi_tilde_vec[:] = np.exp(self._ln_pi_tilde_vec)
        # self._pi_tilde_vec[:] = np.exp(self._ln_pi_tilde_vec - self._ln_pi_tilde_vec.max())
        # self._pi_tilde_vec[:] /= self._pi_tilde_vec.sum()

    def _calc_q_a_char(self):
        self._ln_a_tilde_mat[:] = digamma(self.hn_zeta_vecs) - digamma(self.hn_zeta_vecs.sum(axis=1,keepdims=True))
        self._a_tilde_mat[:] = np.exp(self._ln_a_tilde_mat)
        # self._a_tilde_mat[:] = np.exp(self._ln_a_tilde_mat - self._ln_a_tilde_mat.max(axis=1,keepdims=True))
        # self._a_tilde_mat[:] /= self._a_tilde_mat.sum(axis=1,keepdims=True)
        self._ln_c_hn_zeta_vecs_sum = np.sum(gammaln(self.hn_zeta_vecs.sum(axis=1)) - gammaln(self.hn_zeta_vecs).sum(axis=1))

    def _calc_q_lambda_char(self):
        self._e_lambda_mats[:] = self.hn_nus[:,np.newaxis,np.newaxis] * self.hn_w_mats
        self._e_ln_lambda_dets[:] = (np.sum(digamma((self.hn_nus[:,np.newaxis]-np.arange(self.c_degree)) / 2.0),axis=1)
                            + self.c_degree*np.log(2.0)
                            - np.linalg.slogdet(self.hn_w_mats_inv)[1])
        self._ln_b_hn_w_nus[:] = (
            self.hn_nus*np.linalg.slogdet(self.hn_w_mats_inv)[1]
            - self.hn_nus*self.c_degree*np.log(2.0)
            - self.c_degree*(self.c_degree-1)/2.0*np.log(np.pi)
            - np.sum(gammaln((self.hn_nus[:,np.newaxis]-np.arange(self.c_degree)) / 2.0),
                     axis=1) * 2.0
            ) / 2.0

    def calc_vl(self):
        # E[ln p(X|Z,mu,Lambda)]
        self._vl_p_x = np.sum(
            self.ns
            * (self._e_ln_lambda_dets - self.c_degree / self.hn_kappas
               - (self.s_mats * self._e_lambda_mats).sum(axis=(1,2))
               - ((self.x_bar_vecs - self.hn_m_vecs)[:,np.newaxis,:]
                  @ self._e_lambda_mats
                  @ (self.x_bar_vecs - self.hn_m_vecs)[:,:,np.newaxis]
                  )[:,0,0]
               - self.c_degree * np.log(2*np.pi)
               )
            ) / 2.0

        # E[ln p(Z|pi)]
        self._vl_p_z = (self.gamma_vecs[0] * self._ln_pi_tilde_vec).sum() + (self.ms * self._ln_a_tilde_mat).sum()

        # E[ln p(pi)]
        self._vl_p_pi = self._ln_c_h0_eta_vec + ((self.h0_eta_vec - 1) * self._ln_pi_tilde_vec).sum()

        # E[ln p(A)]
        self._vl_p_a = self._ln_c_h0_zeta_vecs_sum + ((self.h0_zeta_vecs - 1) * self._ln_a_tilde_mat).sum()

        # E[ln p(mu,Lambda)]
        self._vl_p_mu_lambda = np.sum(
            self.c_degree * (np.log(self.h0_kappas) - np.log(2*np.pi)
                           - self.h0_kappas/self.hn_kappas)
            - self.h0_kappas * ((self.hn_m_vecs - self.h0_m_vecs)[:,np.newaxis,:]
                                @ self._e_lambda_mats
                                @ (self.hn_m_vecs - self.h0_m_vecs)[:,:,np.newaxis])[:,0,0]
            + 2.0 * self._ln_b_h0_w_nus
            + (self.h0_nus - self.c_degree) * self._e_ln_lambda_dets
            - np.sum(self.h0_w_mats_inv * self._e_lambda_mats,axis=(1,2))
            ) / 2.0

        # E[ln q(Z|pi)]
        self._vl_q_z = (-(self.gamma_vecs * self._ln_rho).sum()
                        -(self.ms * self._ln_a_tilde_mat).sum()
                        -(self.gamma_vecs[0] * self._ln_pi_tilde_vec).sum()
                        +np.log(self._cs).sum())

        # E[ln q(pi)]
        self._vl_q_pi = ss_dirichlet.entropy(self.hn_eta_vec)

        # E[ln p(A)]
        self._vl_q_a = -self._ln_c_hn_zeta_vecs_sum - ((self.hn_zeta_vecs - 1) * self._ln_a_tilde_mat).sum()

        # E[ln q(mu,Lambda)]
        self._vl_q_mu_lambda =  np.sum(
            + self.c_degree * (1.0 + np.log(2.0*np.pi) - np.log(self.hn_kappas))
            - self._ln_b_hn_w_nus * 2.0
            - (self.hn_nus-self.c_degree)*self._e_ln_lambda_dets
            + self.hn_nus * self.c_degree
            ) / 2.0

        self.vl = (self._vl_p_x
                   + self._vl_p_z
                   + self._vl_p_pi
                   + self._vl_p_a
                   + self._vl_p_mu_lambda
                   + self._vl_q_z
                   + self._vl_q_pi
                   + self._vl_q_a
                   + self._vl_q_mu_lambda)
    
    def _init_fb_params(self):
        self._ln_rho[:] = 0.0
        self._rho[:] = 1.0
        self.alpha_vecs[:] = 1/self.c_num_classes
        self.beta_vecs[:] = 1.0
        self.gamma_vecs[:] = 1/self.c_num_classes
        self.xi_mats[:] = 1/(self.c_num_classes**2)
        self.xi_mats[0] = 0.0
        self._cs[:] = 1.0

    def _init_random_responsibility(self,x):
        self.xi_mats[:] = self.rng.dirichlet(np.ones(self.c_num_classes**2),self.xi_mats.shape[0]).reshape(self.xi_mats.shape)
        self.xi_mats[0] = 0.0
        self.gamma_vecs[:] = self.xi_mats.sum(axis=1)
        self.gamma_vecs[0] = self.xi_mats[1].sum(axis=1)
        self._calc_n_m_x_bar_s(x)

    def _init_subsampling(self,x):
        _size = int(np.sqrt(self._length))
        for k in range(self.c_num_classes):
            _subsample = self.rng.choice(x,size=_size,replace=False,axis=0,shuffle=False)
            self.hn_m_vecs[k] = _subsample.sum(axis=0) / _size
            self.hn_w_mats_inv[k] = ((_subsample - self.hn_m_vecs[k]).T
                                 @ (_subsample - self.hn_m_vecs[k])
                                 / _size * self.hn_nus[k]
                                 + np.identity(self.c_degree) * 1.0E-5) # avoid singular matrix
            self.hn_w_mats[k] = np.linalg.inv(self.hn_w_mats_inv[k])
        self._calc_q_lambda_char()

    def _update_q_mu_lambda(self):
        self.hn_kappas[:] = self.h0_kappas + self.ns
        self.hn_m_vecs[:] = (self.h0_kappas[:,np.newaxis] * self.h0_m_vecs
                             + self.ns[:,np.newaxis] * self.x_bar_vecs) / self.hn_kappas[:,np.newaxis]
        self.hn_nus[:] = self.h0_nus + self.ns
        self.hn_w_mats_inv[:] = (self.h0_w_mats_inv
                                 + self.ns[:,np.newaxis,np.newaxis] * self.s_mats
                                 + (self.h0_kappas * self.ns / self.hn_kappas)[:,np.newaxis,np.newaxis]
                                   * ((self.x_bar_vecs - self.h0_m_vecs)[:,:,np.newaxis]
                                      @ (self.x_bar_vecs - self.h0_m_vecs)[:,np.newaxis,:])
                                 )
        self.hn_w_mats[:] = np.linalg.inv(self.hn_w_mats_inv)
        self._calc_q_lambda_char()

    def _update_q_pi(self):
        self.hn_eta_vec[:] = self.h0_eta_vec + self.ns
        self._calc_q_pi_char()

    def _update_q_a(self):
        self.hn_zeta_vecs[:] = self.h0_zeta_vecs + self.ms
        self._calc_q_a_char()

    def _calc_rho(self,x):
        self._ln_rho[:] = ((self._e_ln_lambda_dets
                            - self.c_degree * np.log(2*np.pi)
                            - self.c_degree / self.hn_kappas
                            - ((x[:,np.newaxis,:]-self.hn_m_vecs)[:,:,np.newaxis,:]
                               @ self._e_lambda_mats
                               @ (x[:,np.newaxis,:]-self.hn_m_vecs)[:,:,:,np.newaxis]
                               )[:,:,0,0]
                            ) / 2.0
                          )
        self._rho[:] = np.exp(self._ln_rho)

    def _forward(self):
        self.alpha_vecs[0] = self._rho[0] * self._pi_tilde_vec
        self._cs[0] = self.alpha_vecs[0].sum()
        self.alpha_vecs[0] /= self._cs[0]
        for i in range(1,self._length):
            self.alpha_vecs[i] = self._rho[i] * (self.alpha_vecs[i-1] @ self._a_tilde_mat)
            self._cs[i] = self.alpha_vecs[i].sum()
            self.alpha_vecs[i] /= self._cs[i]

    def _backward(self):
        for i in range(self._length-2,-1,-1):
            self.beta_vecs[i] = self._a_tilde_mat @ (self._rho[i+1] * self.beta_vecs[i+1])
            self.beta_vecs[i] /= self._cs[i+1]

    def _update_gamma(self):
        self.gamma_vecs[:] = self.alpha_vecs * self.beta_vecs

    def _update_xi(self):
        self.xi_mats[1:,:,:] = self.alpha_vecs[:-1,:,np.newaxis] * self._rho[1:,np.newaxis,:] * self._a_tilde_mat[np.newaxis,:,:] * self.beta_vecs[1:,np.newaxis,:]
        self.xi_mats[1:,:,:] /= self._cs[1:,np.newaxis,np.newaxis]

    def _update_q_z(self,x):
        self._calc_rho(x)
        self._forward()
        self._backward()
        self._update_gamma()
        self._update_xi()
        self._calc_n_m_x_bar_s(x)

    def update_posterior(
            self,
            x,
            max_itr=100,
            num_init=10,
            tolerance=1.0E-8,
            init_type='subsampling'
            ):
        """Update the the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy.ndarray
            All the elements must be real number.
        max_itr : int, optional
            maximum number of iterations, by default 100
        num_init : int, optional
            number of initializations, by default 10
        tolerance : float, optional
            convergence criterion of variational lower bound, by default 1.0E-8
        init_type : str, optional
            type of initialization, by default 'subsampling'
            * 'subsampling': for each latent class, extract a subsample whose size is int(np.sqrt(x.shape[0])).
              and use its mean and covariance matrix as an initial values of hn_m_vecs and hn_lambda_mats.
            * 'random_responsibility': randomly assign responsibility to gamma_vecs
        """
        _check.float_vecs(x,'x',DataFormatError)
        if x.shape[-1] != self.c_degree:
            raise(DataFormatError(
                "x.shape[-1] must be self.c_degree: "
                + f"x.shape[-1]={x.shape[-1]}, self.c_degree={self.c_degree}"))
        x = x.reshape(-1,self.c_degree)
        self._length = x.shape[0]
        self._ln_rho = np.zeros([self._length,self.c_num_classes])
        self._rho = np.ones([self._length,self.c_num_classes])
        self.alpha_vecs = np.ones([self._length,self.c_num_classes])/self.c_num_classes
        self.beta_vecs = np.ones([self._length,self.c_num_classes])
        self.gamma_vecs = np.ones([self._length,self.c_num_classes])/self.c_num_classes
        self.xi_mats = np.zeros([self._length,self.c_num_classes,self.c_num_classes])/(self.c_num_classes**2)
        self._cs = np.ones([self._length])

        tmp_vl = 0.0
        tmp_eta_vec = np.copy(self.hn_eta_vec)
        tmp_zeta_vecs = np.copy(self.hn_zeta_vecs)
        tmp_m_vecs = np.copy(self.hn_m_vecs)
        tmp_kappas = np.copy(self.hn_kappas)
        tmp_nus = np.copy(self.hn_nus)
        tmp_w_mats = np.copy(self.hn_w_mats)
        tmp_w_mats_inv = np.copy(self.hn_w_mats_inv)

        convergence_flag = True
        for i in range(num_init):
            self._init_fb_params()
            self.reset_hn_params()
            if init_type == 'subsampling':
                self._init_subsampling(x)
                self._update_q_z(x)
            elif init_type == 'random_responsibility':
                self._init_random_responsibility(x)
            else:
                raise(ValueError(
                    f'init_type={init_type} is unsupported. '
                    + 'This function supports only '
                    + '"subsampling" and "random_responsibility"'))
            self.calc_vl()
            print(f'\r{i}. VL: {self.vl}',end='')
            for t in range(max_itr):
                vl_before = self.vl
                self._update_q_mu_lambda()
                self._update_q_pi()
                self._update_q_a()
                self._update_q_z(x)
                self.calc_vl()
                print(f'\r{i}. VL: {self.vl} t={t} ',end='')
                if np.abs((self.vl-vl_before)/vl_before) < tolerance:
                    convergence_flag = False
                    print(f'(converged)',end='')
                    break
            if i==0 or self.vl > tmp_vl:
                print('*')
                tmp_vl = self.vl
                tmp_eta_vec[:] = self.hn_eta_vec
                tmp_zeta_vecs[:] = self.hn_zeta_vecs
                tmp_m_vecs[:] = self.hn_m_vecs
                tmp_kappas[:] = self.hn_kappas
                tmp_nus[:] = self.hn_nus
                tmp_w_mats[:] = self.hn_w_mats
                tmp_w_mats_inv[:] = self.hn_w_mats_inv
            else:
                print('')
        if convergence_flag:
            warnings.warn("Algorithm has not converged even once.",ResultWarning)
        
        self.hn_eta_vec[:] = tmp_eta_vec
        self.hn_zeta_vecs[:] = tmp_zeta_vecs
        self.hn_m_vecs[:] = tmp_m_vecs
        self.hn_kappas[:] = tmp_kappas
        self.hn_nus[:] = tmp_nus
        self.hn_w_mats[:] = tmp_w_mats
        self.hn_w_mats_inv[:] = tmp_w_mats_inv
        self._calc_q_pi_char()
        self._calc_q_a_char()
        self._calc_q_lambda_char()
        self._update_q_z(x)

    def estimate_params(self,loss="squared"):
        """Estimate the parameter under the given criterion.

        Note that the criterion is applied to estimating 
        ``xxx``, ``xxx`` and ``xxx`` independently.
        Therefore, a tuple of xxx, xxx and xxx will be returned when loss=\"xxx\"

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"xxx\".
            This function supports \"xxx\", \"xxx\", and \"xxx\".

        Returns
        -------
        Estimates : a tuple of {numpy ndarray, float, None, or rv_frozen}
            * ``xxx`` : the estimate for xxx
            The estimated values under the given loss function. 
            If it is not exist, `np.nan` will be returned.
            If the loss function is \"KL\", the posterior distribution itself 
            will be returned as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """
        pass

    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import xxx
        """
        pass
        
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: numpy.ndarray}
            * ``"xxx"`` : the value of ``self.xxx``
        """
        return {'p_mu_vecs':self.p_mu_vecs,
                'p_nus':self.p_nus,
                'p_lambda_mats':self.p_lambda_mats}

    def calc_pred_dist(self):
        """Calculate the parameters of the predictive distribution."""
        pass

    def make_prediction(self,loss="squared"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"xxx\".
            This function supports \"xxx\" and \"xxx\".

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
        """
        pass

    def pred_and_update(self,x,loss="squared"):
        """Predict a new data point and update the posterior sequentially.

        h0_params will be overwritten by current hn_params 
        before updating hn_params by x.
        
        Parameters
        ----------

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
        """
        pass

    def estimate_latent_vars(self,x,loss=''):
        """Estimate latent variables under the given criterion.

        Note that the criterion is independently applied to each data point.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"xxx\".
            This function supports \"xxx\", \"xxx\", and \"xxx\".

        Returns
        -------
        estimates : numpy.ndarray
            The estimated values under the given loss function. 
            If the loss function is \"xxx\", the posterior distribution will be returned 
            as a numpy.ndarray whose elements consist of occurence probabilities.
        """
        pass

    def estimate_latent_vars_and_update(self):
        """Estimate latent variables and update the posterior sequentially.

        h0_params will be overwritten by current hn_params 
        before updating hn_params by x
        
        Parameters
        ----------

        Returns
        -------
        predicted_value : numpy.ndarray
            The estimated values under the given loss function. 
        """
        pass
