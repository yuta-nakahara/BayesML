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
            axes.plot(np.arange(sample.shape[0]),sample)
            axes.set_xlabel("time")
            axes.set_ylabel("x")
            plt.show()
        else:
            raise(ParameterFormatError("if c_degree > 1, it is impossible to visualize the model by this function."))

class LearnModel(base.Posterior,base.PredictiveMixin):
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

        # hn_params
        self.hn_eta_vec = np.empty(self.c_num_classes)
        self.hn_zeta_vecs = np.empty([self.c_num_classes,self.c_num_classes])
        self.hn_m_vecs = np.empty([self.c_num_classes,self.c_degree])
        self.hn_kappas = np.empty([self.c_num_classes])
        self.hn_nus = np.empty(self.c_num_classes)
        self.hn_w_mats = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        self.hn_w_mats_inv = np.empty([self.c_num_classes,self.c_degree,self.c_degree])

        # p_params
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

        self.reset_hn_params()

    def get_h0_params(self):
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

        self.calc_pred_dist()

    def get_hn_params(self):
        return {'hn_eta_vec':self.hn_eta_vec,
                'hn_zeta_vecs':self.hn_zeta_vecs,
                'hn_m_vecs':self.hn_m_vecs,
                'hn_kappas':self.hn_kappas,
                'hn_nus':self.hn_nus,
                'hn_w_mats':self.hn_w_mats}
    
    def update_posterior():
        pass

    def estimate_params(self,loss="squared"):
        pass

    def visualize_posterior(self):
        pass
        
    def get_p_params(self):
        return {'p_mu_vecs':self.p_mu_vecs,
                'p_nus':self.p_nus,
                'p_lambda_mats':self.p_lambda_mats}

    def calc_pred_dist(self):
        pass

    def make_prediction(self,loss="squared"):
        pass

    def pred_and_update(self,x,loss="squared"):
        pass
