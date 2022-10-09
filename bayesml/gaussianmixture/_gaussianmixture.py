# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Document Author
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
    """The stochastic data generative model and the prior distribution

    Parameters
    ----------
    num_classes : int, optional
        a positive integer. Default is None, in which case 
        a value consistent with ``pi_vec``, ``mu_vecs``, 
        ``lambda_mats``, and ``h_alpha_vec`` is used.
        If all of them are not given, num_classes is assumed to be 2.
    pi_vec : numpy.ndarray, optional
        a real vector in :math:`[0, 1]^K`, by default [1/K, 1/K, ... , 1/K]
    degree : int, optional
        a positive integer. Default is None, in which case 
        a value consistent with ``mu_vecs``, ``lambda_mats``, 
        ``h_m_vec``, ``h_w_mat``, and ``h_nu` is used. 
        If all of them are not given, degree is assumed to be 1.
    mu_vecs : numpy.ndarray, optional
        vectors of real numbers, by default zero vectors.
    lambda_mats : numpy.ndarray, optional
        positive definite symetric matrices, by default the identity matrices
    h_alpha_vec : numpy.ndarray, optional
        a vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
    h_m_vec : numpy.ndarray, optional
        a vector of real numbers, by default [0.0, 0.0, ... , 0.0]
    h_kappa : float, optional
        a positive real number, by default 1.0
    h_nu : float, optional
        a real number > degree-1, by default the value of ``degree``
    h_w_mat : numpy.ndarray, optional
        a positive definite symetric matrix, by default the identity matrix
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(
        self,
        *,
        num_classes=None,
        pi_vec=None,
        degree=None,
        mu_vecs=None,
        lambda_mats=None,
        h_alpha_vec=None,
        h_m_vec=None,
        h_kappa=1.0,
        h_nu=None,
        h_w_mat=None,
        seed=None
        ):
        
        _tmp_degree = np.zeros(5,dtype=int)
        if degree is not None:
            _tmp_degree[0] = _check.pos_int(degree,'degree',ParameterFormatError)
        if mu_vecs is not None:
            _tmp_degree[1] = _check.float_vecs(mu_vecs,'mu_vecs',ParameterFormatError).shape[-1]
        if lambda_mats is not None:
            _tmp_degree[2] = _check.pos_def_sym_mats(lambda_mats,'lambda_mats',ParameterFormatError).shape[-1]
        if h_m_vec is not None:
            _tmp_degree[3] = _check.float_vec(h_m_vec,'h_m_vec',ParameterFormatError).shape[0]
        if h_w_mat is not None:
            _tmp_degree[4] = _check.pos_def_sym_mat(h_w_mat,'h_w_mat',ParameterFormatError).shape[-1]

        _index = (_tmp_degree > 0)
        if np.sum(_index) == 0:
            self.degree = 1 # default value for self.degree
        elif np.all(_tmp_degree[_index] == (_tmp_degree[_index])[0]):
            self.degree = (_tmp_degree[_index])[0]
        else:
            raise(ParameterFormatError(
                "degree and dimensions of mu_vecs, lambda_mats,"
                +" h_m_vec, h_w_mat must be the same,"
                +" if two or more of them are specified."))
        
        _tmp_num_classes = np.zeros(5,dtype=int)
        if num_classes is not None:
            _tmp_num_classes[0] = _check.pos_int(num_classes,'num_classes',ParameterFormatError)
        if mu_vecs is not None:
            _tmp_num_classes[1] = np.prod(_check.float_vecs(mu_vecs,'mu_vecs',ParameterFormatError).shape[:-1])
        if lambda_mats is not None:
            _tmp_num_classes[2] = np.prod(_check.pos_def_sym_mats(lambda_mats,'lambda_mats',ParameterFormatError).shape[:-2])
        if pi_vec is not None:
            _tmp_num_classes[3] = _check.float_vec_sum_1(pi_vec,'pi_vec',ParameterFormatError).shape[0]
        if h_alpha_vec is not None:
            _tmp_num_classes[4] = _check.pos_float_vec(h_alpha_vec,'h_alpha_vec',ParameterFormatError).shape[0]

        _index = (_tmp_num_classes > 0)
        if np.sum(_index) == 0:
            self.num_classes = 2 # default value for self.degree
        elif np.all(_tmp_num_classes[_index] == (_tmp_num_classes[_index])[0]):
            self.num_classes = (_tmp_num_classes[_index])[0]
        else:
            raise(ParameterFormatError(
                "num_classes, dimensions of pi_vec and h_alpha_vec,"
                +" and the first dimension of mu_vecs and lambda_mats"
                +" must be the same, if two or more of them are specified."))

        if mu_vecs is None:
            self.mu_vecs = np.zeros([self.num_classes,self.degree])
        else:
            self.mu_vecs = mu_vecs.reshape(self.num_classes,self.degree)

        if lambda_mats is None:
            self.lambda_mats = np.tile(np.identity(self.degree),(self.num_classes,1,1))
        else:
            self.lambda_mats = lambda_mats.reshape(self.num_classes,self.degree,self.degree)

        if h_m_vec is None:
            self.h_m_vec = np.zeros(self.degree)
        else:
            self.h_m_vec = h_m_vec

        if h_w_mat is None:
            self.h_w_mat = np.identity(self.degree)
        else:
            self.h_w_mat = h_w_mat

        if h_nu is None:
            self.h_nu = float(self.degree)
        else:
            self.h_nu = _check.pos_float(h_nu,'h_nu',ParameterFormatError)
            if self.h_nu <= self.degree - 1:
                raise(ParameterFormatError(
                    "degree must be smaller than h_nu + 1"))

        if pi_vec is None:
            self.pi_vec = np.ones(self.num_classes) / self.num_classes
        else:
            self.pi_vec = pi_vec

        if h_alpha_vec is None:
            self.h_alpha_vec = np.ones(self.num_classes) / 2
        else:
            self.h_alpha_vec = h_alpha_vec
        
        self.h_kappa = _check.pos_float(h_kappa,'h_kappa',ParameterFormatError)
        self.rng = np.random.default_rng(seed)
        
    def set_h_params(self,h_alpha_vec,h_m_vec,h_kappa,h_nu,h_w_mat):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_alpha_vec : numpy.ndarray
            a vector of positive real numbers
        h_m_vec : numpy.ndarray
            a vector of real numbers
        h_kappa : float
            a positive real number
        h_nu : float
            a real number > degree-1
        h_w_mat : numpy.ndarray
            a positive definite symetric matrix
        """
        self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec,'h_alpha_vec',ParameterFormatError)
        self.h_m_vec = _check.float_vec(h_m_vec,'h_m_vec',ParameterFormatError)
        self.h_kappa = _check.pos_float(h_kappa,'h_kappa',ParameterFormatError)
        self.h_nu = _check.pos_float(h_nu,'h_nu',ParameterFormatError)
        self.h_w_mat = _check.pos_def_sym_mat(h_w_mat,'h_w_mat',ParameterFormatError)

        if (self.h_m_vec.shape[0] != self.h_w_mat.shape[0]
            or self.h_nu <= self.h_m_vec.shape[0] - 1):
                raise(ParameterFormatError(
                    "They must be h_m_vec.shape[0] == h_w_mat.shape[0]"
                    +" and h_nu > h_m_vec.shape[0] - 1."))

        self.degree = self.h_m_vec.shape[0]
        self.num_classes = self.h_alpha_vec.shape[0]

        if self.degree != self.mu_vecs.shape[-1]:
            self.mu_vecs = np.zeros([self.num_classes,self.degree])
            warnings.warn("mu_vecs is reinitialized to zero vectors because dimensions of mu_vecs and h_params are mismatched.", ParameterFormatWarning)

        if self.degree != self.lambda_mats.shape[-1]:
            self.lambda_mats = np.tile(np.identity(self.degree),[self.num_classes,1,1])
            warnings.warn("lambda_mats is reinitialized to the identity matrices because dimensions of lambda_mats and h_params are mismatched.", ParameterFormatWarning)
        
        if self.num_classes != self.pi_vec.shape[0]:
            self.pi_vec = np.ones(self.num_classes) / self.num_classes
            warnings.warn("pi_vec is reinitialized to [1/num_classes, 1/num_classes, ..., 1/num_classes] because dimensions of mu_vec and h_params are mismatched.", ParameterFormatWarning)

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.
        
        Returns
        -------
        h_params : {str:float, np.ndarray}
            * ``"h_alpha_vec"`` : The value of ``self.h_alpha_vec``
            * ``"h_m_vec"`` : The value of ``self.h_mu_vec``
            * ``"h_kappa"`` : The value of ``self.h_kappa``
            * ``"h_nu"`` : The value of ``self.h_nu``
            * ``"h_w_mat"`` : The value of ``self.h_w_mat``
        """
        return {"h_alpha_vec":self.h_alpha_vec,
                "h_m_vec":self.h_m_vec, 
                "h_kappa":self.h_kappa, 
                "h_nu":self.h_nu, 
                "h_w_mat":self.h_w_mat}
    
    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        The generated vaule is set at ``self.pi_vec``, ``self.mu_vecs`` and ``self.lambda_mats``.
        """
        self.pi_vec[:] = self.rng.dirichlet(self.h_alpha_vec)
        for k in range(self.num_classes):
            self.lambda_mats[k] = ss_wishart.rvs(df=self.h_nu,scale=self.h_w_mat,random_state=self.rng)
            self.mu_vecs[k] = self.rng.multivariate_normal(mean=self.h_m_vec,cov=np.linalg.inv(self.h_kappa*self.lambda_mats[k]))
    
    def set_params(self,pi_vec,mu_vecs,lambda_mats):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        pi_vec : numpy.ndarray
            a real vector in :math:`[0, 1]^K`. The sum of its elements must be 1.
        mu_vecs : numpy.ndarray
            vectors of real numbers
        lambda_mats : numpy.ndarray
            positive definite symetric matrices
        """
        self.pi_vec = _check.float_vec_sum_1(pi_vec,'pi_vec',ParameterFormatError)
        _tmp_mu_vecs = _check.float_vecs(mu_vecs,'mu_vecs',ParameterFormatError)
        _tmp_shape = _tmp_mu_vecs.shape
        self.mu_vecs = _tmp_mu_vecs.reshape(-1,_tmp_shape[-1])
        _tmp_lambda_mats = _check.pos_def_sym_mats(lambda_mats,'lambda_mats',ParameterFormatError)
        _tmp_shape = _tmp_lambda_mats.shape
        self.lambda_mats = _tmp_lambda_mats.reshape(-1,_tmp_shape[-2],_tmp_shape[-1])
        if (self.pi_vec.shape[0] != self.mu_vecs.shape[0]
            or self.pi_vec.shape[0] != self.lambda_mats.shape[0]):
            raise(ParameterFormatError("The dimension of pi_vec"
                +" and the first dimension of mu_vecs and lambda_mats"
                +" must be the same"))

        if (self.mu_vecs.shape[-1] != self.lambda_mats.shape[-1]):
            raise(ParameterFormatError("The dimensions of mu_vecs and lambda_mats must be the same"))

        self.degree = self.mu_vecs.shape[-1]
        if self.degree != self.h_m_vec.shape[0]:
            self.h_m_vec = np.zeros(self.degree)
            warnings.warn("h_m_vec is reinitialized to [0.0, 0.0, ..., 0.0] because dimension of h_m_vec and mu_vec are mismatched.", ParameterFormatWarning)
        if self.degree != self.h_w_mat.shape[0]:
            self.h_w_mat = np.identity(self.degree)
            warnings.warn("h_w_mat is reinitialized to the identity matrix because dimension of h_w_mat and lambda_mat are mismatched.", ParameterFormatWarning)
        
        self.num_classes = self.pi_vec.shape[0]
        if self.num_classes != self.h_alpha_vec.shape[0]:
            self.h_alpha_vec = np.ones(self.num_classes) / 2
            warnings.warn("h_alpha_vec is reinitialized to [1/2, 1/2, ... , 1/2]] because dimension of h_alpha_vec and pi_vec are mismatched.", ParameterFormatWarning)

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : {str:float, numpy.ndarray}
            * ``"pi_vec"`` : The value of ``self.pi_vec``
            * ``"mu_vecs"`` : The value of ``self.mu_vecs``
            * ``"lambda_mats"`` : The value of ``self.lambda_mats``
        """
        return {"pi_vec":self.pi_vec, "mu_vecs":self.mu_vecs, "lambda_mats":self.lambda_mats}

    def gen_sample(self,sample_size):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int
            A positive integer

        Returns
        -------
        x : numpy ndarray
            2-dimensional array whose shape is ``(sammple_size,degree)`` and its elements are real numbers.
        z : numpy ndarray
            2-dimensional array whose shape is ``(sample_size,num_classes)`` whose rows are one-hot vectors.
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        z = np.zeros([sample_size,self.num_classes],dtype=int)
        x = np.empty([sample_size,self.degree])
        _lambda_mats_inv = np.linalg.inv(self.lambda_mats)
        for i in range(sample_size):
            k = self.rng.choice(self.num_classes,p=self.pi_vec)
            z[i,k] = 1
            x[i] = self.rng.multivariate_normal(mean=self.mu_vecs[k],cov=_lambda_mats_inv[k])
        return x,z
        
    def save_sample(self,filename,sample_size):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as a NpzFile with keyword: \"x\", \"z\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_size : int
            A positive integer
        
        See Also
        --------
        numpy.savez_compressed
        """
        x,z=self.gen_sample(sample_size)
        np.savez_compressed(filename,x=x,z=z)

    def visualize_model(self,sample_size=100):
        """Visualize the stochastic data generative model and generated samples.
        
        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 100
        
        Examples
        --------
        >>> from bayesml import gaussianmixture
        >>> import numpy as np
        >>> model = gaussianmixture.GenModel(
        >>>             pi_vec=np.array([0.444,0.444,0.112]),
        >>>             mu_vecs=np.array([[-2.8],[-0.8],[2]]),
        >>>             lambda_mats=np.array([[[6.25]],[[6.25]],[[100]]])
        >>>             )
        >>> model.visualize_model()
        pi_vec:
         [0.444 0.444 0.112]
        mu_vecs:
         [[-2.8]
         [-0.8]
         [ 2. ]]
        lambda_mats:
         [[[  6.25]]

         [[  6.25]]

         [[100.  ]]]
        
        .. image:: ./images/gaussianmixture_example.png
        """
        if self.degree == 1:
            print(f"pi_vec:\n {self.pi_vec}")
            print(f"mu_vecs:\n {self.mu_vecs}")
            print(f"lambda_mats:\n {self.lambda_mats}")
            _lambda_mats_inv = np.linalg.inv(self.lambda_mats)
            fig, axes = plt.subplots()
            sample, _ = self.gen_sample(sample_size)
            x = np.linspace(sample.min()-(sample.max()-sample.min())*0.25,
                            sample.max()+(sample.max()-sample.min())*0.25,
                            1000)
            y = np.zeros(1000)
            for k in range(self.num_classes):
                y += self.pi_vec[k] * ss_multivariate_normal.pdf(x,self.mu_vecs[k],_lambda_mats_inv[k])
            axes.plot(x,y)
            axes.hist(sample,density=True)
            axes.set_xlabel("x")
            axes.set_ylabel("Density or frequency")
            plt.show()

        elif self.degree == 2:
            print(f"pi_vec:\n {self.pi_vec}")
            print(f"mu_vecs:\n {self.mu_vecs}")
            print(f"lambda_mats:\n {self.lambda_mats}")
            _lambda_mats_inv = np.linalg.inv(self.lambda_mats)
            fig, axes = plt.subplots()
            sample, _ = self.gen_sample(sample_size)
            x = np.linspace(sample[:,0].min()-(sample[:,0].max()-sample[:,0].min())*0.25,
                            sample[:,0].max()+(sample[:,0].max()-sample[:,0].min())*0.25,
                            1000)
            y = np.linspace(sample[:,1].min()-(sample[:,1].max()-sample[:,1].min())*0.25,
                            sample[:,1].max()+(sample[:,1].max()-sample[:,1].min())*0.25,
                            1000)
            xx, yy = np.meshgrid(x,y)
            grid = np.empty((1000,1000,2))
            grid[:,:,0] = xx
            grid[:,:,1] = yy
            z = np.zeros([1000,1000])
            for k in range(self.num_classes):
                z += self.pi_vec[k] * ss_multivariate_normal.pdf(grid,self.mu_vecs[k],_lambda_mats_inv[k])
            axes.contourf(xx,yy,z,cmap='Blues')
            for k in range(self.num_classes):
                axes.plot(self.mu_vecs[k,0],self.mu_vecs[k,1],marker="x",color='red')
            axes.set_xlabel("x[0]")
            axes.set_ylabel("x[1]")
            axes.scatter(sample[:,0],sample[:,1],color='tab:orange')
            plt.show()

        else:
            raise(ParameterFormatError("if degree > 2, it is impossible to visualize the model by this function."))

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    num_classes : int
        a positive integer
    degree : int
        a positive integer
    h0_alpha_vec : numpy.ndarray, optional
        a vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
    h0_m_vecs : numpy.ndarray, optional
        a vector of real numbers, by default [0.0, 0.0, ... , 0.0]
    h0_kappas : float or numpy.ndarray, optional
        a positive real number, by default 1.0
    h0_nus : float or numpy.ndarray, optional
        a real number > degree-1, by default the value of ``degree``
    h0_w_mats : numpy.ndarray, optional
        a positive definite symetric matrix, by default the identity matrix
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None

    Attributes
    ----------
    h0_w_mats_inv : numpy.ndarray
        the inverse matrix of h0_w_mat
    hn_alpha_vec : numpy.ndarray
        a vector of positive real numbers
    hn_m_vecs : numpy.ndarray
        vectors of real numbers
    hn_kappas : numpy.ndarray
        positive real numbers
    hn_nus : numpy.ndarray
        real numbers greater than degree-1
    hn_w_mats : numpy.ndarray
        positive definite symetric matrices
    hn_w_mats_inv : numpy.ndarray
        the inverse matrices of hn_w_mats
    r_vecs : numpy.ndarray
        vectors of real numbers. The sum of its elenemts is 1.
    ns : numpy.ndarray
        positive real numbers
    s_mats : numpy.ndarray
        positive difinite symmetric matrices
    p_mu_vecs : numpy.ndarray
        vectors of real numbers
    p_nus : numpy.ndarray
        positive real numbers
    p_lambda_mats : numpy.ndarray
        positive definite symetric matrices
    """
    def __init__(
            self,
            *,
            num_classes,
            degree,
            h0_alpha_vec = None,
            h0_m_vecs = None,
            h0_kappas = None,
            h0_nus = None,
            h0_w_mats = None,
            seed = None
            ):
        # constants
        self.degree = _check.pos_int(degree,'degree',ParameterFormatError)
        self.num_classes = _check.pos_int(num_classes,'num_classes',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # h0_params
        self.h0_alpha_vec = np.ones(self.num_classes) / 2        
        self.h0_m_vecs = np.zeros([self.num_classes,self.degree])
        self.h0_kappas = np.ones(self.num_classes)
        self.h0_nus = np.ones(self.num_classes) * self.degree
        self.h0_w_mats = np.tile(np.identity(self.degree),[self.num_classes,1,1])
        self.h0_w_mats_inv = np.linalg.inv(self.h0_w_mats)

        self._LN_C_H0_ALPHA = 0.0
        self._LN_B_H0_W_NUS = np.empty(self.num_classes)
        
        # hn_params
        self.hn_alpha_vec = np.empty([self.num_classes])
        self.hn_m_vecs = np.empty([self.num_classes,self.degree])
        self.hn_kappas = np.empty([self.num_classes])
        self.hn_nus = np.empty([self.num_classes])
        self.hn_w_mats = np.empty([self.num_classes,self.degree,self.degree])
        self.hn_w_mats_inv = np.empty([self.num_classes,self.degree,self.degree])

        self._ln_rho = None
        self.r_vecs = None
        self._e_lambda_mats = np.empty([self.num_classes,self.degree,self.degree])
        self._e_ln_lambda_dets = np.empty(self.num_classes)
        self._ln_b_hn_w_nus = np.empty(self.num_classes)
        self._e_ln_pi_vec = np.empty(self.num_classes)

        # statistics
        self.x_bar_vecs = np.empty([self.num_classes,self.degree])
        self.ns = np.empty(self.num_classes)
        self.s_mats = np.empty([self.num_classes,self.degree,self.degree])

        # variational lower bound
        self.vl = 0.0
        self._vl_p_x = 0.0
        self._vl_p_z = 0.0
        self._vl_p_pi = 0.0
        self._vl_p_mu_lambda = 0.0
        self._vl_q_z = 0.0
        self._vl_q_pi = 0.0
        self._vl_q_mu_lambda = 0.0

        # p_params
        self.p_pi_vec = np.empty([self.num_classes])
        self.p_mu_vecs = np.empty([self.num_classes,self.degree])
        self.p_nus = np.empty([self.num_classes])
        self.p_lambda_mats = np.empty([self.num_classes,self.degree,self.degree])
        self.p_lambda_mats_inv = np.empty([self.num_classes,self.degree,self.degree])
        
        self.set_h0_params(
            h0_alpha_vec,
            h0_m_vecs,
            h0_kappas,
            h0_nus,
            h0_w_mats,
        )

    def set_h0_params(
            self,
            h0_alpha_vec = None,
            h0_m_vecs = None,
            h0_kappas = None,
            h0_nus = None,
            h0_w_mats = None,
            ):
        """Set the hyperparameters of the prior distribution.
        
        Parameters
        ----------
        h0_alpha_vec : numpy.ndarray
            a vector of positive real numbers
        h0_m_vecs : numpy.ndarray
            vectors of real numbers
        h0_kappas : float
            positive real numbers
        h0_nus : float
            real numbers greater than degree-1
        h0_w_mats : numpy.ndarray
            positive definite symetric matrices
        """
        if h0_alpha_vec is not None:
            _check.pos_floats(h0_alpha_vec,'h0_alpha_vec',ParameterFormatError)
            self.h0_alpha_vec[:] = h0_alpha_vec

        if h0_m_vecs is not None:
            _check.float_vecs(h0_m_vecs,'h0_m_vecs',ParameterFormatError)
            if h0_m_vecs.shape[-1] != self.degree:
                raise(ParameterFormatError(
                    "h0_m_vecs.shape[-1] must coincide with self.degree:"
                    +f"h0_m_vecs.shape[-1]={h0_m_vecs.shape[-1]}, self.degree={self.degree}"))
            self.h0_m_vecs[:] = h0_m_vecs

        if h0_kappas is not None:
            _check.pos_floats(h0_kappas,'h0_kappas',ParameterFormatError)
            self.h0_kappas[:] = h0_kappas

        if h0_nus is not None:
            _check.pos_floats(h0_nus,'h0_nus',ParameterFormatError)
            if np.any(h0_nus <= self.degree - 1):
                raise(ParameterFormatError(
                    "degree must be smaller than h_nus + 1"))
            self.h0_nus[:] = h0_nus

        if h0_w_mats is not None:
            _check.pos_def_sym_mats(h0_w_mats,'h0_w_mats',ParameterFormatError)
            if h0_w_mats.shape[-1] != self.degree:
                raise(ParameterFormatError(
                    "h0_w_mats.shape[-1] and h0_w_mats.shape[-2] must coincide with self.degree:"
                    +f"h0_w_mats.shape[-1]={h0_w_mats.shape[-1]}, h0_w_mats.shape[-2]={h0_w_mats.shape[-2]}, self.degree={self.degree}"))
            self.h0_w_mats[:] = h0_w_mats
            self.h0_w_mats_inv[:] = np.linalg.inv(self.h0_w_mats)

        self._LN_C_H0_ALPHA = gammaln(self.h0_alpha_vec.sum()) - gammaln(self.h0_alpha_vec).sum()
        self._LN_B_H0_W_NUS = (
            - self.h0_nus*np.linalg.slogdet(self.h0_w_mats)[1]
            - self.h0_nus*self.degree*np.log(2.0)
            - self.degree*(self.degree-1)/2.0*np.log(np.pi)
            - np.sum(gammaln((self.h0_nus[:,np.newaxis]-np.arange(self.degree)) / 2.0),
                     axis=1) * 2.0
            ) / 2.0

        self.reset_hn_params()

    def get_h0_params(self):
        """Get the initial values of the hyperparameters of the posterior distribution.

        Returns
        -------
        h0_params : dict of {str: numpy.ndarray}
            * ``"h0_alpha_vec"`` : The value of ``self.h0_alpha_vec``
            * ``"h0_m_vecs"`` : The value of ``self.h0_m_vecs``
            * ``"h0_kappas"`` : The value of ``self.h0_kappas``
            * ``"h0_nus"`` : The value of ``self.h0_nus``
            * ``"h0_w_mats"`` : The value of ``self.h0_w_mats``
        """
        return {"h0_alpha_vec":self.h0_alpha_vec,
                "h0_m_vecs":self.h0_m_vecs,
                "h0_kappas":self.h0_kappas,
                "h0_nus":self.h0_nus,
                "h0_w_mat":self.h0_w_mats}
    
    def set_hn_params(
            self,
            hn_alpha_vec = None,
            hn_m_vecs = None,
            hn_kappas = None,
            hn_nus = None,
            hn_w_mats = None,
            ):
        """Set updated values of the hyperparameter of the posterior distribution.

        Parameters
        ----------
        hn_alpha_vec : numpy.ndarray
            a vector of positive real numbers
        hn_m_vecs : numpy.ndarray
            vectors of real numbers
        hn_kappas : float
            positive real numbers
        hn_nus : float
            real numbers greater than degree-1
        hn_w_mats : numpy.ndarray
            positive definite symetric matrices
        """
        if hn_alpha_vec is not None:
            _check.pos_floats(hn_alpha_vec,'hn_alpha_vec',ParameterFormatError)
            self.hn_alpha_vec[:] = hn_alpha_vec

        if hn_m_vecs is not None:
            _check.float_vecs(hn_m_vecs,'hn_m_vecs',ParameterFormatError)
            if hn_m_vecs.shape[-1] != self.degree:
                raise(ParameterFormatError(
                    "hn_m_vecs.shape[-1] must coincide with self.degree:"
                    +f"hn_m_vecs.shape[-1]={hn_m_vecs.shape[-1]}, self.degree={self.degree}"))
            self.hn_m_vecs[:] = hn_m_vecs

        if hn_kappas is not None:
            _check.pos_floats(hn_kappas,'hn_kappas',ParameterFormatError)
            self.hn_kappas[:] = hn_kappas

        if hn_nus is not None:
            _check.pos_floats(hn_nus,'hn_nus',ParameterFormatError)
            if np.any(hn_nus <= self.degree - 1):
                raise(ParameterFormatError(
                    "degree must be smaller than h_nus + 1"))
            self.hn_nus[:] = hn_nus

        if hn_w_mats is not None:
            _check.pos_def_sym_mats(hn_w_mats,'hn_w_mats',ParameterFormatError)
            if hn_w_mats.shape[-1] != self.degree:
                raise(ParameterFormatError(
                    "hn_w_mats.shape[-1] and hn_w_mats.shape[-2] must coincide with self.degree:"
                    +f"hn_w_mats.shape[-1]={hn_w_mats.shape[-1]}, hn_w_mats.shape[-2]={hn_w_mats.shape[-2]}, self.degree={self.degree}"))
            self.hn_w_mats[:] = hn_w_mats
            self.hn_w_mats_inv[:] = np.linalg.inv(self.hn_w_mats)

        self.calc_pred_dist()

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: numpy.ndarray}
            * ``"hn_alpha_vec"`` : The value of ``self.hn_alpha_vec``
            * ``"hn_m_vecs"`` : The value of ``self.hn_m_vecs``
            * ``"hn_kappas"`` : The value of ``self.hn_kappas``
            * ``"hn_nus"`` : The value of ``self.hn_nus``
            * ``"hn_w_mats"`` : The value of ``self.hn_w_mats``
        """
        return {"hn_alpha_vec":self.hn_alpha_vec,
                "hn_m_vec":self.hn_m_vecs,
                "hn_kappa":self.hn_kappas,
                "hn_nu":self.hn_nus,
                "hn_w_mat":self.hn_w_mats}
    
    def reset_hn_params(self):
        """Reset the hyperparameters of the posterior distribution to their initial values.
        
        They are reset to `self.h0_alpha_vec`, `self.h0_m_vecs`, `self.h0_kappas`, `self.h0_nus` and `self.h0_w_mats`.
        Note that the parameters of the predictive distribution are also calculated from them.
        """
        self.hn_alpha_vec[:] = self.h0_alpha_vec
        self.hn_m_vecs[:] = self.h0_m_vecs
        self.hn_kappas[:] = self.h0_kappas
        self.hn_nus[:] = self.h0_nus
        self.hn_w_mats[:] = self.h0_w_mats
        self.hn_w_mats_inv = np.linalg.inv(self.hn_w_mats)

        self._calc_q_pi_char()
        self._calc_q_lambda_char()

        self.calc_pred_dist()
    
    def overwrite_h0_params(self):
        """Overwrite the initial values of the hyperparameters of the posterior distribution by the learned values.
        
        They are overwitten by `self.hn_alpha_vec`, `self.hn_m_vecs`, `self.hn_kappas`, `self.hn_nus` and `self.hn_w_mats`.
        Note that the parameters of the predictive distribution are also calculated from them.
        """
        self.h0_alpha_vec[:] = self.hn_alpha_vec
        self.h0_m_vecs[:] = self.hn_m_vecs
        self.h0_kappas[:] = self.hn_kappas
        self.h0_nus[:] = self.hn_nus
        self.h0_w_mats[:] = self.hn_w_mats
        self.h0_w_mats_inv = np.linalg.inv(self.h0_w_mats)

        self.calc_pred_dist()

    def calc_vl(self):
        # E[ln p(X|Z,mu,Lambda)]
        self._vl_p_x = np.sum(
            self.ns
            * (self._e_ln_lambda_dets - self.degree / self.hn_kappas
               - (self.s_mats * self._e_lambda_mats).sum(axis=(1,2))
               - ((self.x_bar_vecs - self.hn_m_vecs)[:,np.newaxis,:]
                  @ self._e_lambda_mats
                  @ (self.x_bar_vecs - self.hn_m_vecs)[:,:,np.newaxis]
                  )[:,0,0]
               - self.degree * np.log(2*np.pi)
               )
            ) / 2.0

        # E[ln p(Z|pi)]
        self._vl_p_z = (self.ns * self._e_ln_pi_vec).sum()

        # E[ln p(pi)]
        self._vl_p_pi = self._LN_C_H0_ALPHA + ((self.h0_alpha_vec - 1) * self._e_ln_pi_vec).sum()

        # E[ln p(mu,Lambda)]
        self._vl_p_mu_lambda = np.sum(
            self.degree * (np.log(self.h0_kappas) - np.log(2*np.pi)
                           - self.h0_kappas/self.hn_kappas)
            - self.h0_kappas * ((self.hn_m_vecs - self.h0_m_vecs)[:,np.newaxis,:]
                                @ self._e_lambda_mats
                                @ (self.hn_m_vecs - self.h0_m_vecs)[:,:,np.newaxis])[:,0,0]
            + 2.0 * self._LN_B_H0_W_NUS
            + (self.h0_nus - self.degree) * self._e_ln_lambda_dets
            - np.sum(self.h0_w_mats_inv * self._e_lambda_mats,axis=(1,2))
            ) / 2.0

        # E[ln q(Z|pi)]
        self._vl_q_z = -np.sum(xlogy(self.r_vecs,self.r_vecs))

        # E[ln q(pi)]
        self._vl_q_pi = ss_dirichlet.entropy(self.hn_alpha_vec)

        # E[ln q(mu,Lambda)]
        self._vl_q_mu_lambda =  np.sum(
            + self.degree * (1.0 + np.log(2.0*np.pi) - np.log(self.hn_kappas))
            - self._ln_b_hn_w_nus * 2.0
            - (self.hn_nus-self.degree)*self._e_ln_lambda_dets
            + self.hn_nus * self.degree
            ) / 2.0

        self.vl = (self._vl_p_x
                   + self._vl_p_z
                   + self._vl_p_pi
                   + self._vl_p_mu_lambda
                   + self._vl_q_z
                   + self._vl_q_pi
                   + self._vl_q_mu_lambda)

    def _calc_n_x_bar_s(self,x):
        self.ns[:] = self.r_vecs.sum(axis=0)
        self.x_bar_vecs[:] = (self.r_vecs[:,:,np.newaxis] * x[:,np.newaxis,:]).sum(axis=0) / self.ns[:,np.newaxis]
        self.s_mats[:] = np.sum(self.r_vecs[:,:,np.newaxis,np.newaxis]
                                * ((x[:,np.newaxis,:] - self.x_bar_vecs)[:,:,:,np.newaxis]
                                   @ (x[:,np.newaxis,:] - self.x_bar_vecs)[:,:,np.newaxis,:]),
                                axis=0) / self.ns[:,np.newaxis,np.newaxis]

    def _init_random_responsibility(self,x):
        self.r_vecs[:] = self.rng.dirichlet(np.ones(self.num_classes),self.r_vecs.shape[0])
        self._calc_n_x_bar_s(x)

    def _calc_q_pi_char(self):
        self._e_ln_pi_vec[:] = digamma(self.hn_alpha_vec) - digamma(self.hn_alpha_vec.sum())

    def _update_q_pi(self):
        self.hn_alpha_vec[:] = self.h0_alpha_vec + self.ns
        self._calc_q_pi_char()

    def _calc_q_lambda_char(self):
        self._e_lambda_mats[:] = self.hn_nus[:,np.newaxis,np.newaxis] * self.hn_w_mats
        self._e_ln_lambda_dets[:] = (np.sum(digamma((self.hn_nus[:,np.newaxis]-np.arange(self.degree)) / 2.0),axis=1)
                            + self.degree*np.log(2.0)
                            - np.linalg.slogdet(self.hn_w_mats_inv)[1])
        self._ln_b_hn_w_nus[:] = (
            self.hn_nus*np.linalg.slogdet(self.hn_w_mats_inv)[1]
            - self.hn_nus*self.degree*np.log(2.0)
            - self.degree*(self.degree-1)/2.0*np.log(np.pi)
            - np.sum(gammaln((self.hn_nus[:,np.newaxis]-np.arange(self.degree)) / 2.0),
                     axis=1) * 2.0
            ) / 2.0

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

    def _update_q_z(self,x):
        self._ln_rho[:] = (self._e_ln_pi_vec
                          + (self._e_ln_lambda_dets
                             - self.degree * np.log(2*np.pi)
                             - self.degree / self.hn_kappas
                             - ((x[:,np.newaxis,:]-self.hn_m_vecs)[:,:,np.newaxis,:]
                                @ self._e_lambda_mats
                                @ (x[:,np.newaxis,:]-self.hn_m_vecs)[:,:,:,np.newaxis]
                                )[:,:,0,0]
                             ) / 2.0
                          )
        self.r_vecs[:] = np.exp(self._ln_rho - self._ln_rho.max(axis=1,keepdims=True))
        self.r_vecs[:] /= self.r_vecs.sum(axis=1,keepdims=True)
        self._calc_n_x_bar_s(x)

    def _init_subsampling(self,x):
        _size = int(np.sqrt(x.shape[0])) + 1
        for k in range(self.num_classes):
            _subsample = self.rng.choice(x,size=_size,replace=False,axis=0,shuffle=False)
            self.hn_m_vecs[k] = _subsample.sum(axis=0) / _size
            self.hn_w_mats[k] = ((_subsample - self.hn_m_vecs[k]).T
                                 @ (_subsample - self.hn_m_vecs[k])
                                 / _size / self.hn_nus[k]
                                 + np.identity(self.degree) * 1.0E-5) # avoid singular matrix
            self.hn_w_mats_inv[k] = np.linalg.inv(self.hn_w_mats[k])
        self._calc_q_pi_char()
        self._calc_q_lambda_char()

    def update_posterior(self,x,max_itr=100,num_init=10,tolerance=1.0E-8,init_type='subsampling'):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy.ndarray
            All the elements must be real number.
        max_itr : int, optional
            maximum number of iterations, by default 100
        num_init : int, optional
            number of initializations, by default 10
        tolerance : float, optional
            convergence croterion of variational lower bound, by default 1.0E-8
        """
        _check.float_vecs(x,'x',DataFormatError)
        if self.degree > 1 and x.shape[-1] != self.degree:
            raise(DataFormatError(
                "x.shape[-1] must be self.degree: "
                + f"x.shape[-1]={x.shape[-1]}, self.degree={self.degree}"))
        x = x.reshape(-1,self.degree)
        self._ln_rho = np.empty([x.shape[0],self.num_classes])
        self.r_vecs = np.empty([x.shape[0],self.num_classes])

        tmp_vl = 0.0
        tmp_alpha_vec = np.copy(self.hn_alpha_vec)
        tmp_m_vecs = np.copy(self.hn_m_vecs)
        tmp_kappas = np.copy(self.hn_kappas)
        tmp_nus = np.copy(self.hn_nus)
        tmp_w_mats = np.copy(self.hn_w_mats)
        tmp_w_mats_inv = np.copy(self.hn_w_mats_inv)

        convergence_flag = True
        for i in range(num_init):
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
                tmp_alpha_vec[:] = self.hn_alpha_vec
                tmp_m_vecs[:] = self.hn_m_vecs
                tmp_kappas[:] = self.hn_kappas
                tmp_nus[:] = self.hn_nus
                tmp_w_mats[:] = self.hn_w_mats
                tmp_w_mats_inv[:] = self.hn_w_mats_inv
            else:
                print('')
        if convergence_flag:
            warnings.warn("Algorithm has not converged even once.",ResultWarning)
        
        self.hn_alpha_vec[:] = tmp_alpha_vec
        self.hn_m_vecs[:] = tmp_m_vecs
        self.hn_kappas[:] = tmp_kappas
        self.hn_nus[:] = tmp_nus
        self.hn_w_mats[:] = tmp_w_mats
        self.hn_w_mats_inv[:] = tmp_w_mats_inv
        self._calc_q_pi_char()
        self._calc_q_lambda_char()
        self._update_q_z(x)


    def estimate_params(self,loss="squared"):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Note that the criterion is applied to estimating 
        ``pi_vec``, ``mu_vecs`` and ``lambda_mats`` independently.
        Therefore, a tuple of the dirichlet distribution, 
        the student's t-distributions and 
        the wishart distributions will be returned when loss=\"KL\"

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", and \"KL\".

        Returns
        -------
        Estimates : a tuple of {numpy ndarray, float, None, or rv_frozen}
            * ``pi_vec_hat`` : the estimate for pi_vec
            * ``mu_vecs_hat`` : the estimate for mu_vecs
            * ``lambda_mats_hat`` : the estimate for lambda_mats
            The estimated values under the given loss function. 
            If it is not exist, `np.nan` will be returned.
            If the loss function is \"KL\", the posterior distribution itself 
            will be returned as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """

        if loss == "squared":
            return self.hn_alpha_vec/self.hn_alpha_vec.sum(), self.hn_m_vecs, self._e_lambda_mats
        elif loss == "0-1":
            pi_vec_hat = np.empty(self.num_classes)
            if np.all(self.hn_alpha_vec > 1):
                pi_vec_hat[:] = (self.hn_alpha_vec - 1) / (np.sum(self.hn_alpha_vec) - self.degree)
            else:
                warnings.warn("MAP estimate of lambda_mat doesn't exist for the current hn_alpha_vec.",ResultWarning)
                pi_vec_hat[:] = np.nan

            lambda_mats_hat = np.empty([self.num_classes,self.degree,self.degree])
            for k in range(self.num_classes):
                if self.hn_nus[k] >= self.degree + 1:
                    lambda_mats_hat[k] = (self.hn_nus[k] - self.degree - 1) * self.hn_w_mats[k]
                else:
                    warnings.warn(f"MAP estimate of lambda_mat doesn't exist for the current hn_nus[{k}].",ResultWarning)
                    lambda_mats_hat[k] = np.nan
            return pi_vec_hat, self.hn_m_vecs, lambda_mats_hat
        elif loss == "KL":
            mu_vec_pdfs = []
            lambda_mat_pdfs = []
            for k in range(self.num_classes):
                mu_vec_pdfs.append(ss_multivariate_t(loc=self.hn_m_vecs[k],
                                                 shape=self.hn_w_mats_inv[k] / self.hn_kappas[k] / (self.hn_nus[k] - self.degree + 1),
                                                 df=self.hn_nus[k] - self.degree + 1))
                lambda_mat_pdfs.append(ss_wishart(df=self.hn_nus[k],scale=self.hn_w_mats[k]))
            return (ss_dirichlet(self.hn_alpha_vec),
                    mu_vec_pdfs,
                    lambda_mat_pdfs)
        else:
            raise(CriteriaError(f"loss={loss} is unsupported. "
                                +"This function supports \"squared\", \"0-1\", and \"KL\"."))
    
    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import gaussianmixture
        >>> gen_model = gaussianmixture.GenModel(
        >>>     num_classes=2,
        >>>     degree=1,
        >>>     mu_vecs=np.array([[-2],[2]]),
        >>>     )
        >>> x,z = gen_model.gen_sample(100)
        >>> learn_model = gaussianmixture.LearnModel(num_classes=2, degree=1)
        >>> learn_model.update_posterior(x)
        >>> learn_model.visualize_posterior()
        hn_m_vecs:
        [[ 2.09365933]
        [-1.97862429]]
        hn_kappas:
        [47.68878373 54.31121627]
        hn_nus:
        [47.68878373 54.31121627]
        hn_w_mats:
        [[[0.02226992]]

        [[0.01575793]]]
        E[lambda_mats]=
        [[[1.06202546]]

        [[0.85583258]]]

        .. image:: ./images/gaussianmixture_posterior.png
        """
        print("hn_m_vecs:")
        print(f"{self.hn_m_vecs}")
        print("hn_kappas:")
        print(f"{self.hn_kappas}")
        print("hn_nus:")
        print(f"{self.hn_nus}")
        print("hn_w_mats:")
        print(f"{self.hn_w_mats}")
        print("E[lambda_mats]=")
        print(f"{self._e_lambda_mats}")
        _, mu_vec_pdfs, lambda_mat_pdfs = self.estimate_params(loss="KL")
        if self.degree == 1:
            fig, axes = plt.subplots(1,2)
            axes[0].set_xlabel("mu_vecs")
            axes[0].set_ylabel("Density")
            axes[1].set_xlabel("lambda_mats")
            axes[1].set_ylabel("Log density")
            for k in range(self.num_classes):
                # for mu_vecs
                x = np.linspace(self.hn_m_vecs[k,0]-4.0*np.sqrt((self.hn_w_mats_inv[k] / self.hn_kappas[k] / self.hn_nus[k])[0,0]),
                                self.hn_m_vecs[k,0]+4.0*np.sqrt((self.hn_w_mats_inv[k] / self.hn_kappas[k] / self.hn_nus[k])[0,0]),
                                100)
                axes[0].plot(x,mu_vec_pdfs[k].pdf(x))
                # for lambda_mats
                x = np.linspace(max(1.0e-8,self.hn_nus[k]*self.hn_w_mats[k]-4.0*np.sqrt(self.hn_nus[k]/2.0)*(2.0*self.hn_w_mats[k])),
                                self.hn_nus[k]*self.hn_w_mats[k]+4.0*np.sqrt(self.hn_nus[k]/2.0)*(2.0*self.hn_w_mats[k]),
                                500)
                axes[1].plot(x[:,0,0],lambda_mat_pdfs[k].logpdf(x[:,0,0]))

            fig.tight_layout()
            plt.show()

        elif self.degree == 2:
            fig, axes = plt.subplots()
            for k in range(self.num_classes):
                x = np.linspace(self.hn_m_vecs[k,0]-3.0*np.sqrt((self.hn_w_mats_inv[k] / self.hn_kappas[k] / self.hn_nus[k])[0,0]),
                                self.hn_m_vecs[k,0]+3.0*np.sqrt((self.hn_w_mats_inv[k] / self.hn_kappas[k] / self.hn_nus[k])[0,0]),
                                100)
                y = np.linspace(self.hn_m_vecs[k,1]-3.0*np.sqrt((self.hn_w_mats_inv[k] / self.hn_kappas[k] / self.hn_nus[k])[1,1]),
                                self.hn_m_vecs[k,1]+3.0*np.sqrt((self.hn_w_mats_inv[k] / self.hn_kappas[k] / self.hn_nus[k])[1,1]),
                                100)
                xx, yy = np.meshgrid(x,y)
                grid = np.empty((100,100,2))
                grid[:,:,0] = xx
                grid[:,:,1] = yy
                axes.contour(xx,yy,mu_vec_pdfs[k].pdf(grid),cmap='Blues',alpha=self.hn_alpha_vec[k]/self.hn_alpha_vec.sum())
                axes.plot(self.hn_m_vecs[k,0],self.hn_m_vecs[k,1],marker="x",color='red')
            axes.set_xlabel("mu_vec[0]")
            axes.set_ylabel("mu_vec[1]")
            plt.show()

        else:
            raise(ParameterFormatError("if degree > 2, it is impossible to visualize the model by this function."))
        
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: numpy.ndarray}
            * ``"p_mu_vecs"`` : The value of ``self.p_mu_vecs``
            * ``"p_nus"`` : The value of ``self.p_nus``
            * ``"p_lambda_mats"`` : The value of ``self.p_lambda_mats``
        """
        return {"p_mu_vecs":self.p_mu_vecs, "p_nus":self.p_nus, "p_lambda_mats":self.p_lambda_mats}

    def calc_pred_dist(self):
        """Calculate the parameters of the predictive distribution."""
        self.p_pi_vec[:] = self.hn_alpha_vec / self.hn_alpha_vec.sum()
        self.p_mu_vecs[:] = self.hn_m_vecs
        self.p_nus[:] = self.hn_nus - self.degree + 1
        self.p_lambda_mats[:] = (self.hn_kappas * self.p_nus / (self.hn_kappas + 1))[:,np.newaxis,np.newaxis] * self.hn_w_mats
        self.p_lambda_mats_inv[:] = np.linalg.inv(self.p_lambda_mats)

    def make_prediction(self,loss="squared"):
        pass
        # """Predict a new data point under the given criterion.

        # Parameters
        # ----------
        # loss : str, optional
        #     Loss function underlying the Bayes risk function, by default \"squared\".
        #     This function supports \"squared\" and \"0-1\".

        # Returns
        # -------
        # Predicted_value : {float, numpy.ndarray}
        #     The predicted value under the given loss function. 
        # """
        # if loss == "squared":
        #     return np.sum(self.p_pi_vec[:,np.newaxis] * self.p_mu_vecs, axis=0)
        # elif loss == "0-1":
        #     tmp_max = -1.0
        #     tmp_argmax = np.empty([self.degree])
        #     for k in range(self.num_classes):
        #         val = ss_multivariate_t.pdf(x=self.p_mu_vecs[k],
        #                                     loc=self.p_mu_vecs[k],
        #                                     shape=self.p_lambda_mats_inv[k],
        #                                     df=self.p_nus[k])
        #         if val * self.p_pi_vec[k] > tmp_max:
        #             tmp_argmax[:] = self.p_mu_vecs[k]
        #             tmp_max = val * self.p_pi_vec[k]
        #     return tmp_argmax
        # else:
        #     raise(CriteriaError("Unsupported loss function! "
        #                         +"This function supports \"squared\", \"0-1\", and \"KL\"."))

    def pred_and_update(self,x,loss="squared"):
        pass
#         """Predict a new data point and update the posterior sequentially.

#         Parameters
#         ----------
#         x : numpy.ndarray
#             It must be a degree-dimensional vector
#         loss : str, optional
#             Loss function underlying the Bayes risk function, by default \"squared\".
#             This function supports \"squared\", \"0-1\", and \"KL\".

#         Returns
#         -------
#         Predicted_value : {float, numpy.ndarray}
#             The predicted value under the given loss function. 
#         """
#         _check.float_vec(x,'x',DataFormatError)
#         if x.shape != (self.degree,):
#             raise(DataFormatError(f"x must be a 1-dimensional float array whose size is degree: {self.degree}."))
#         self.calc_pred_dist()
#         prediction = self.make_prediction(loss=loss)
#         self.update_posterior(x[np.newaxis,:])
#         return prediction
