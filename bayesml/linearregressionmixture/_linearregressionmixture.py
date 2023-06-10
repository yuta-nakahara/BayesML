# Code Author
# Yuta Nakahara <y.nakahara@waseda.jp>
# Document Author
# Yuta Nakahara <y.nakahara@waseda.jp>
import warnings
import numpy as np
from scipy.stats import multivariate_normal as ss_multivariate_normal
from scipy.stats import wishart as ss_wishart
from scipy.stats import multivariate_t as ss_multivariate_t
from scipy.stats import dirichlet as ss_dirichlet
from scipy.special import gammaln, digamma, xlogy, logsumexp
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.cm import tab20

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning, ParameterFormatWarning
from .. import _check

_TAB_COLOR_LIST = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution

    Parameters
    ----------
    c_num_classes : int
        A positive integer
    c_degree : int
        A positive integer
    pi_vec : numpy.ndarray, optional
        A vector of real numbers in :math:`[0, 1]`, 
        by default [1/c_num_classes, 1/c_num_classes, ... , 1/c_num_classes]
        Sum of its elements must be 1.0.
    theta_vecs : numpy.ndarray, optional
        Vectors of real numbers, by default zero vectors.
    taus : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
    h_gamma_vec : float or numpy.ndarray, optional
        A vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
        If a single real number is input, it will be broadcasted.
    h_mu_vecs : numpy.ndarray, optional
        Vectors of real numbers, by default zero vectors
    h_lambda_mats : numpy.ndarray, optional
        Positive definite symetric matrices, 
        by default the identity matrices.
        If a single matrix is input, it will be broadcasted.
    h_alphas : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
    h_betas : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
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

    def get_constants(self):
        """Get constants of GenModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_num_classes"`` : the value of ``self.c_num_classes``
            * ``"c_degree"`` : the value of ``self.c_degree``
        """
        return {"c_num_classes":self.c_num_classes, "c_degree":self.c_degree}

    def set_params(
            self,
            pi_vec=None,
            theta_vecs=None,
            taus=None
            ):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        pi_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]`, 
            by default [1/c_num_classes, 1/c_num_classes, ... , 1/c_num_classes]
            Sum of its elements must be 1.0.
        theta_vecs : numpy.ndarray, optional
            Vectors of real numbers, by default zero vectors.
        taus : float or numpy.ndarray, optional
            Positive real numbers, 
            by default [1.0, 1.0, ... , 1.0]
            If a single real number is input, it will be broadcasted.
        """
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
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_gamma_vec : float or numpy.ndarray, optional
            A vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
            If a single real number is input, it will be broadcasted.
        h_mu_vecs : numpy.ndarray, optional
            Vectors of real numbers, by default zero vectors
        h_lambda_mats : numpy.ndarray, optional
            Positive definite symetric matrices, 
            by default the identity matrices.
            If a single matrix is input, it will be broadcasted.
        h_alphas : float or numpy.ndarray, optional
            Positive real numbers, 
            by default [1.0, 1.0, ... , 1.0]
            If a single real number is input, it will be broadcasted.
        h_betas : float or numpy.ndarray, optional
            Positive real numbers, 
            by default [1.0, 1.0, ... , 1.0]
            If a single real number is input, it will be broadcasted.
        """
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
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : {str: numpy.ndarray}
            * ``"pi_vec"`` : The value of ``self.pi_vec``
            * ``"theta_vecs"`` : The value of ``self.theta_vecs``
            * ``"taus"`` : The value of ``self.taus``
        """
        return {'pi_vec':self.pi_vec,
                'theta_vecs':self.theta_vecs,
                'taus':self.taus}
        
    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.
        
        Returns
        -------
        h_params : {str:float, np.ndarray}
            * ``"h_gamma_vec"`` : The value of ``self.h_gamma_vec``
            * ``"h_mu_vecs"`` : The value of ``self.h_mu_vecs``
            * ``"h_lambda_mats"`` : The value of ``self.h_lambda_mats``
            * ``"h_alphas"`` : The value of ``self.h_alphas``
            * ``"h_betas"`` : The value of ``self.h_betas``
        """
        return {'h_gamma_vec':self.h_gamma_vec,
                'h_mu_vecs':self.h_mu_vecs,
                'h_lambda_mats':self.h_lambda_mats,
                'h_alphas':self.h_alphas,
                'h_betas':self.h_betas}
    
    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        The generated vaule is set at ``self.pi_vec``, ``self.theta_vecs`` and ``self.lambda_mats``.
        """
        self.pi_vec[:] = self.rng.dirichlet(self.h_gamma_vec)
        for k in range(self.c_num_classes):
            self.taus[k] =  self.rng.gamma(
                shape=self.h_alphas[k],
                scale=1.0/self.h_betas[k]
            )
            self.theta_vecs[k] = self.rng.multivariate_normal(
                mean=self.h_mu_vecs[k],
                cov=np.linalg.inv(self.taus[k]*self.h_lambda_mats[k])
            )
        return self

    def gen_sample(self,sample_size=None,x=None,constant=True):
        """Generate a sample from the stochastic data generative model.

        If x is given, it will be used for explanatory variables as it is 
        (independent of the other options: sample_size and constant).

        If x is not given, it will be generated from i.i.d. standard normal distribution.
        The size of the generated sample is defined by sample_size.
        If constant is True, the last element of the generated explanatory variables will be overwritten by 1.0.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default ``None``.
        x : numpy ndarray, optional
            float array whose shape is ``(sample_size,c_degree)``, by default ``None``.
        constant : bool, optional
            A boolean value, by default ``True``.

        Returns
        -------
        x : numpy ndarray
            2-dimensional array whose shape is ``(sample_size,c_degree)`` and its elements are real numbers.
        z : numpy ndarray
            2-dimensional array whose shape is ``(sample_size,c_num_classes)`` whose rows are one-hot vectors.
        y : numpy ndarray
            1 dimensional float array whose size is ``sample_size``.
        """
        if x is not None:
            _check.float_vecs(x,'x',DataFormatError)
            _check.shape_consistency(
                x.shape[-1],"x.shape[-1]", 
                self.c_degree,"self.c_degree", 
                ParameterFormatError
                )
            x = x.reshape([-1,self.c_degree])
            sample_size = x.shape[0]
        elif sample_size is not None:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            x = self.rng.multivariate_normal(np.zeros(self.c_degree),np.eye(self.c_degree), size=sample_size)
            if constant:
                x[:,-1] = 1.0
        else:
            raise(DataFormatError("Either of the sample_size and the x must be given as an input."))

        z = np.zeros([sample_size,self.c_num_classes],dtype=int)
        y = np.empty(sample_size)

        for i in range(sample_size):
            k = self.rng.choice(self.c_num_classes,p=self.pi_vec)
            z[i,k] = 1
            y[i] = self.rng.normal(loc = x[i] @ self.theta_vecs[k], scale = 1.0 / np.sqrt(self.taus[k]))
        return x,z,y
    
    def save_sample(self,filename,sample_size=None,x=None,constant=True):
        """Save the generated sample as NumPy ``.npz`` format.

        If x is given, it will be used for explanatory variables as it is 
        (independent of the other options: sample_size and constant).

        If x is not given, it will be generated from i.i.d. standard normal distribution.
        The size of the generated sample is defined by sample_size.
        If constant is True, the last element of the generated explanatory variables will be overwritten by 1.0.
        
        The generated sample is saved as a NpzFile with keyword: \"x\", \"z\", \"y\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_size : int, optional
            A positive integer, by default ``None``.
        x : numpy ndarray, optional
            float array whose shape is ``(sample_size,c_degree)``, by default ``None``.
        constant : bool, optional
            A boolean value, by default ``True``.
        
        See Also
        --------
        numpy.savez_compressed
        """
        x,z,y=self.gen_sample(sample_size,x,constant)
        np.savez_compressed(filename,x=x,z=z,y=y)
    
    def visualize_model(self,sample_size=100,constant=True):
        """Visualize the stochastic data generative model and generated samples.
        
        If x is given, it will be used for explanatory variables as it is 
        (independent of the other options: sample_size and constant).

        If x is not given, it will be generated from i.i.d. standard normal distribution.
        The size of the generated sample is defined by sample_size.
        If constant is True, the last element of the generated explanatory variables will be overwritten by 1.0.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 100
        constant : bool, optional
        
        Examples
        --------
        >>> from bayesml import linearregressionmixture
        >>> import numpy as np
        >>> model = linearregressionmixture.GenModel(
        >>>     c_num_classes=2,
        >>>     c_degree=2,
        >>>     theta_vecs=np.array([[1,3],
        >>>                          [-1,-3]]),
        >>> )
        >>> model.visualize_model()

        theta_vecs:
        [[ 1.  3.]
         [-1. -3.]]
        taus:
        [1. 1.]
        
        .. image:: ./images/linearregressionmixture_example.png
        """
        print(f"theta_vecs:\n{self.theta_vecs}")
        print(f"taus:\n{self.taus}")
        if self.c_degree == 2 and constant==True:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            sample_x, sample_z, sample_y = self.gen_sample(sample_size=sample_size,constant=True)
            fig, ax = plt.subplots()

            x = np.linspace(sample_x[:,0].min()-(sample_x[:,0].max()-sample_x[:,0].min())*0.25,
                            sample_x[:,0].max()+(sample_x[:,0].max()-sample_x[:,0].min())*0.25,
                            100)
            for k in range(self.c_num_classes):
                ax.scatter(
                    sample_x[:,0][sample_z[:,k]==1],
                    sample_y[sample_z[:,k]==1],
                    color=_TAB_COLOR_LIST[k%10],
                )
                ax.plot(
                    x,
                    x*self.theta_vecs[k][0] + self.theta_vecs[k][1],
                    label=f'y={self.theta_vecs[k][0]:.2f}*x + {self.theta_vecs[k][1]:.2f}',
                    color=_TAB_COLOR_LIST[k%10],
                )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            plt.show()
        elif self.c_degree == 1 and constant==False:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            sample_x, sample_z, sample_y = self.gen_sample(sample_size=sample_size,constant=False)
            fig, ax = plt.subplots()
            ax.scatter(sample_x,sample_y)

            x = np.linspace(sample_x.min()-(sample_x.max()-sample_x.min())*0.25,
                            sample_x.max()+(sample_x.max()-sample_x.min())*0.25,
                            100)
            for k in range(self.c_num_classes):
                ax.scatter(
                    sample_x[sample_z[:,k]==1],
                    sample_y[sample_z[:,k]==1],
                    color=_TAB_COLOR_LIST[k%10],
                )
                ax.plot(
                    x,
                    x*self.theta_vecs[k],
                    label=f'y={self.theta_vecs[k][0]:.2f}*x',
                    color=_TAB_COLOR_LIST[k%10],
                )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            plt.show()
        else:
            raise(ParameterFormatError(
                "This function supports only the following cases: "
                +"c_degree = 2 and constant = True; c_degree = 1 "
                +"and constant = False."
                )
            )

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    c_num_classes : int
        a positive integer
    c_degree : int
        a positive integer
    h0_gamma_vec : float or numpy.ndarray, optional
        A vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
        If a single real number is input, it will be broadcasted.
    h0_mu_vecs : numpy.ndarray, optional
        Vectors of real numbers, by default zero vectors
    h0_lambda_mats : numpy.ndarray, optional
        Positive definite symetric matrices, 
        by default the identity matrices.
        If a single matrix is input, it will be broadcasted.
    h0_alphas : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
    h0_betas : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None

    Attributes
    ----------
    hn_gamma_vec : float or numpy.ndarray
        A vector of positive real numbers.
        If a single real number is input, it will be broadcasted.
    hn_mu_vecs : numpy.ndarray
        Vectors of real numbers.
    hn_lambda_mats : numpy.ndarray
        Positive definite symetric matrices. 
    hn_alphas : float or numpy.ndarray
        Positive real numbers. 
    hn_betas : float or numpy.ndarray
        Positive real numbers. 
    r_vecs : numpy.ndarray
        vectors of real numbers. The sum of its elenemts is 1.
    ns : numpy.ndarray
        positive real numbers
    p_pi_vecs : numpy.ndarray
        A vector of real numbers in :math:`[0, 1]`. 
        Sum of its elements must be 1.0.
    p_ms : numpy.ndarray
        Real numbers
    p_lambdas : numpy.ndarray
        Positive real numbers
    p_nus : numpy.ndarray
        Positive real numbers
    """
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

    def get_constants(self):
        """Get constants of LearnModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_num_classes"`` : the value of ``self.c_num_classes``
            * ``"c_degree"`` : the value of ``self.c_degree``
        """
        return {"c_num_classes":self.c_num_classes, "c_degree":self.c_degree}

    def set_h0_params(
            self,
            h0_gamma_vec=None,
            h0_mu_vecs=None,
            h0_lambda_mats=None,
            h0_alphas=None,
            h0_betas=None,
            ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h0_gamma_vec : float or numpy.ndarray, optional
            A vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
            If a single real number is input, it will be broadcasted.
        h0_mu_vecs : numpy.ndarray, optional
            Vectors of real numbers, by default zero vectors
        h0_lambda_mats : numpy.ndarray, optional
            Positive definite symetric matrices, 
            by default the identity matrices.
            If a single matrix is input, it will be broadcasted.
        h0_alphas : float or numpy.ndarray, optional
            Positive real numbers, 
            by default [1.0, 1.0, ... , 1.0]
            If a single real number is input, it will be broadcasted.
        h0_betas : float or numpy.ndarray, optional
            Positive real numbers, 
            by default [1.0, 1.0, ... , 1.0]
            If a single real number is input, it will be broadcasted.
        """
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
