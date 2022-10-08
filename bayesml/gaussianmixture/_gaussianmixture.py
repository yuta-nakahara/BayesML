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
from scipy.special import gammaln, digamma, xlogy
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
        a positive integer. Default is None, in which case 
        a value consistent with ``pi_vec``, ``mu_vecs``, 
        ``lambda_mats``, and ``h_alpha_vec`` is used.
        If all of them are not given, num_classes is assumed to be 2.
    degree : int
        a positive integer. Default is None, in which case 
        a value consistent with ``mu_vecs``, ``lambda_mats``, 
        ``h_m_vec``, ``h_w_mat``, and ``h_nu` is used. 
        If all of them are not given, degree is assumed to be 1.
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

        self.LN_C_H0_ALPHA = 0.0
        self.LN_B_H0_W_NUS = np.empty(self.num_classes)
        
        # hn_params
        self.hn_alpha_vec = np.empty([self.num_classes])
        self.hn_m_vecs = np.empty([self.num_classes,self.degree])
        self.hn_kappas = np.empty([self.num_classes])
        self.hn_nus = np.empty([self.num_classes])
        self.hn_w_mats = np.empty([self.num_classes,self.degree,self.degree])
        self.hn_w_mats_inv = np.empty([self.num_classes,self.degree,self.degree])

        # statistics
        self.r_vecs = None
        self.x_bar_vecs = np.empty([self.num_classes,self.degree])
        self.ns = np.empty(self.num_classes)
        self.s_mats = np.empty([self.num_classes,self.degree,self.degree])
        self.e_lambda_mats = np.empty([self.num_classes,self.degree,self.degree])
        self.e_ln_lambda_dets = np.empty(self.num_classes)
        self.e_ln_pi_vec = np.empty(self.num_classes)

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

        self.LN_C_H0_ALPHA = gammaln(self.h0_alpha_vec.sum()) - gammaln(self.h0_alpha_vec).sum()
        self.LN_B_H0_W_NUS = (
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
        self.e_lambda_mats = self.hn_nus[:,np.newaxis,np.newaxis] * self.hn_w_mats
        self.e_ln_lambda_dets = (np.sum(digamma((self.hn_nus[:,np.newaxis]-np.arange(self.degree)) / 2.0),axis=1)
                            + self.degree*np.log(2.0)
                            - np.linalg.slogdet(self.hn_w_mats_inv)[1])
        self.e_ln_pi_vec = digamma(self.hn_alpha_vec) - digamma(self.hn_alpha_vec.sum())
        
        # tentative
        self.ns = np.ones(self.num_classes) * 10
        self.s_mats = np.tile(np.identity(self.degree),[self.num_classes,1,1]) * 5
        self.r_vecs = np.ones([20,self.degree])/self.degree
        self.x_bar_vecs = np.ones([self.num_classes,self.degree])

        vl = 0.0

        # E[ln p(X|Z,mu,Lambda)]
        vl += np.sum(
            self.ns
            * (self.e_ln_lambda_dets - self.degree / self.hn_kappas
               - (self.s_mats * self.e_lambda_mats).sum(axis=(1,2))
               - ((self.x_bar_vecs - self.hn_m_vecs)[:,np.newaxis,:]
                  @ self.e_lambda_mats
                  @ (self.x_bar_vecs - self.hn_m_vecs)[:,:,np.newaxis]
                  )[:,0,0]
               - self.degree * np.log(2*np.pi)
               )
            ) / 2.0

        # E[ln p(Z|pi)]
        vl += (self.ns * self.e_ln_pi_vec).sum()

        # E[ln p(pi)]
        vl += self.LN_C_H0_ALPHA + ((self.h0_alpha_vec - 1) * self.e_ln_pi_vec).sum()

        # E[ln p(mu,Lambda)]
        vl += np.sum(
            self.degree * (np.log(self.h0_kappas) - np.log(2*np.pi) - self.h0_kappas/self.hn_kappas)
            - ((self.hn_m_vecs - self.h0_m_vecs)[:,np.newaxis,:]
               @ self.e_lambda_mats
               @ (self.hn_m_vecs - self.h0_m_vecs)[:,:,np.newaxis])[:,0,0]
            + 2.0 * self.LN_B_H0_W_NUS
            + (self.h0_nus - self.degree) / 2.0 * self.e_ln_lambda_dets
            - np.sum(self.h0_w_mats_inv * self.hn_w_mats,axis=(1,2))
            ) / 2.0

        # E[ln q(Z|pi)]
        vl -= np.sum(xlogy(self.r_vecs,self.r_vecs))

        # E[ln q(pi)]
        vl += ss_dirichlet.entropy(self.hn_alpha_vec)

        # E[ln q(mu,Lambda)]
        vl +=  np.sum(
            + self.degree * (1.0 + np.log(2.0*np.pi) - np.log(self.hn_kappas))
            - self.LN_B_H0_W_NUS * 2.0
            - (self.hn_nus-self.degree)*self.e_ln_lambda_dets
            + self.hn_nus * self.degree
            ) / 2.0

        return vl

    def update_posterior(self,x):
        pass
#         """Update the hyperparameters of the posterior distribution using traning data.

#         Parameters
#         ----------
#         x : numpy.ndarray
#             All the elements must be real number.
#         """
#         _check.float_vecs(x,'x',DataFormatError)
#         if self.degree > 1 and x.shape[-1] != self.degree:
#             raise(DataFormatError(f"x.shape[-1] must be degree:{self.degree}"))
#         x = x.reshape(-1,self.degree)

#         n = x.shape[0]
#         x_bar = x.sum(axis=0)/n

#         self.hn_w_mat_inv[:] = (self.hn_w_mat_inv + (x-x_bar).T @ (x-x_bar)
#                                 + (x_bar - self.hn_m_vec)[:,np.newaxis] @ (x_bar - self.hn_m_vec)[np.newaxis,:]
#                                   * self.hn_kappa * n / (self.hn_kappa + n))
#         self.hn_m_vec[:] = (self.hn_kappa*self.hn_m_vec + n*x_bar) / (self.hn_kappa+n)
#         self.hn_kappa += n
#         self.hn_nu += n

#         self.hn_w_mat[:] = np.linalg.inv(self.hn_w_mat_inv) 

    def estimate_params(self,loss="squared"):
        pass
#         """Estimate the parameter of the stochastic data generative model under the given criterion.

#         Note that the criterion is applied to estimating ``mu_vec`` and ``lambda_mat`` independently.
#         Therefore, a tuple of the student's t-distribution and the wishart distribution will be returned when loss=\"KL\"

#         Parameters
#         ----------
#         loss : str, optional
#             Loss function underlying the Bayes risk function, by default \"squared\".
#             This function supports \"squared\", \"0-1\", and \"KL\".

#         Returns
#         -------
#         Estimates : tuple of {numpy ndarray, float, None, or rv_frozen}
#             * ``mu_vec_hat`` : the estimate for mu_vec
#             * ``lambda_mat_hat`` : the estimate for lambda_mat
#             The estimated values under the given loss function. If it is not exist, `None` will be returned.
#             If the loss function is \"KL\", the posterior distribution itself will be returned
#             as rv_frozen object of scipy.stats.

#         See Also
#         --------
#         scipy.stats.rv_continuous
#         scipy.stats.rv_discrete
#         """

#         if loss == "squared":
#             return self.hn_m_vec, self.hn_nu * self.hn_w_mat
#         elif loss == "0-1":
#             if self.hn_nu >= self.degree + 1:
#                 return self.hn_m_vec, (self.hn_nu - self.degree - 1) * self.hn_w_mat
#             else:
#                 warnings.warn("MAP estimate of lambda_mat doesn't exist for the current hn_nu.",ResultWarning)
#                 return self.hn_m_vec, None
#         elif loss == "KL":
#             return (ss_multivariate_t(loc=self.hn_m_vec,
#                                         shape=self.hn_w_mat_inv / self.hn_kappa / (self.hn_nu - self.degree + 1),
#                                         df=self.hn_nu - self.degree + 1),
#                     ss_wishart(df=self.hn_nu,scale=self.hn_w_mat))
#         else:
#             raise(CriteriaError("Unsupported loss function! "
#                                 +"This function supports \"squared\", \"0-1\", and \"KL\"."))
    
    def visualize_posterior(self):
        pass
#         """Visualize the posterior distribution for the parameter.
        
#         Examples
#         --------
#         >>> from bayesml import multivariate_normal
#         >>> gen_model = multivariate_normal.GenModel()
#         >>> x = gen_model.gen_sample(100)
#         >>> learn_model = multivariate_normal.LearnModel()
#         >>> learn_model.update_posterior(x)
#         >>> learn_model.visualize_posterior()
#         hn_m_vec:
#         [-0.06924909  0.08126454]
#         hn_kappa:
#         101.0
#         hn_nu:
#         102.0
#         hn_w_mat:
#         [[ 0.00983415 -0.00059828]
#         [-0.00059828  0.00741698]]
#         E[lambda_mat]=
#         [[ 1.0030838  -0.06102455]
#         [-0.06102455  0.7565315 ]]

#         .. image:: ./images/multivariate_normal_posterior.png
#         """
#         print("hn_m_vec:")
#         print(f"{self.hn_m_vec}")
#         print("hn_kappa:")
#         print(f"{self.hn_kappa}")
#         print("hn_nu:")
#         print(f"{self.hn_nu}")
#         print("hn_w_mat:")
#         print(f"{self.hn_w_mat}")
#         print("E[lambda_mat]=")
#         print(f"{self.hn_nu * self.hn_w_mat}")
#         mu_vec_pdf, w_mat_pdf = self.estimate_params(loss="KL")
#         if self.degree == 1:
#             fig, axes = plt.subplots(1,2)
#             # for mu_vec
#             x = np.linspace(self.hn_m_vec[0]-4.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[0,0]),
#                             self.hn_m_vec[0]+4.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[0,0]),
#                             100)
#             axes[0].plot(x,mu_vec_pdf.pdf(x))
#             axes[0].set_xlabel("mu_vec")
#             axes[0].set_ylabel("Density")
#             # for lambda_mat
#             x = np.linspace(max(1.0e-8,self.hn_nu*self.hn_w_mat)-4.0*np.sqrt(self.hn_nu/2.0)*(2.0*self.hn_w_mat),
#                             self.hn_nu*self.hn_w_mat+4.0*np.sqrt(self.hn_nu/2.0)*(2.0*self.hn_w_mat),
#                             100)
#             print(self.hn_w_mat)
#             axes[1].plot(x[:,0,0],w_mat_pdf.pdf(x[:,0,0]))
#             axes[1].set_xlabel("w_mat")
#             axes[1].set_ylabel("Density")

#             fig.tight_layout()
#             plt.show()

#         elif self.degree == 2:
#             fig, axes = plt.subplots()
#             x = np.linspace(self.hn_m_vec[0]-3.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[0,0]),
#                             self.hn_m_vec[0]+3.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[0,0]),
#                             100)
#             y = np.linspace(self.hn_m_vec[1]-3.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[1,1]),
#                             self.hn_m_vec[1]+3.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[1,1]),
#                             100)
#             xx, yy = np.meshgrid(x,y)
#             grid = np.empty((100,100,2))
#             grid[:,:,0] = xx
#             grid[:,:,1] = yy
#             axes.contourf(xx,yy,mu_vec_pdf.pdf(grid),cmap='Blues')
#             axes.plot(self.hn_m_vec[0],self.hn_m_vec[1],marker="x",color='red')
#             axes.set_xlabel("mu_vec[0]")
#             axes.set_ylabel("mu_vec[1]")
#             plt.show()

#         else:
#             raise(ParameterFormatError("if degree > 2, it is impossible to visualize the model by this function."))
        
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
