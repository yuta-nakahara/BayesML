# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Document Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
import warnings
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning, ParameterFormatWarning
from .. import _check

_CMAP = plt.get_cmap("Blues")

class _GenNode:
    """ The node class used by generative model and the prior distribution

    Parameters
    ----------
    depth : int
            a non-negetive integer :math:' >= 0'
    h_g : float
            a positive real number  in \[0 , 1 \], by default 0.5
    """
    def __init__(self,
                 depth,
                 c_k,
                 h_g,
                 ):
        self.depth = depth
        self.children = [None for i in range(c_k)]  # child nodes
        self.h_g = h_g
        self.h_beta_vec = np.ones(c_k) / 2
        self.theta_vec = np.ones(c_k) / c_k
        self.leaf = False

class GenModel(base.Generative):
    """ The stochastice data generative model and the prior distribution

    Parameters
    ----------
    c_k : int
        A positive integer
    c_d_max : int, optional
        A positive integer, by default 10
    theta_vec : numpy.ndarray, optional
        A vector of real numbers in :math:`[0, 1]`, 
        by default [1/c_k, 1/c_k, ... , 1/c_k]
        Sum of its elements must be 1.0.
    root : contexttree._GenNode, optional
        A root node of a context tree, 
        by default a tree consists of only one node.
    h_g : float, optional
        A real number in :math:`[0, 1]`, by default 0.5
    h_beta_vec : numpy.ndarray, optional
        A vector of positive real numbers, 
        by default [1/2, 1/2, ... , 1/2].
        If a single real number is input, it will be broadcasted.
    h_root : contexttree._GenNode, optional
        A root node of a superposed tree for hyperparameters 
        by default ``None``
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default ``None``
    """
    def __init__(
            self,
            c_k,
            c_d_max=10,
            *,
            theta_vec=None,
            root=None,
            h_g=0.5,
            h_beta_vec=None,
            h_root=None,
            seed=None,
            ):
        # constants
        self.c_d_max = _check.pos_int(c_d_max,'c_d_max',ParameterFormatError)
        self.c_k = _check.pos_int(c_k,'c_k',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # h_params
        self.h_g = 0.5
        self.h_beta_vec = np.ones(self.c_k) / 2
        self.h_root = None

        self.set_h_params(
            h_g,
            h_beta_vec,
            h_root,
        )

        # params
        self.theta_vec = np.ones(self.c_k) / self.c_k
        self.root = _GenNode(0,self.c_k,self.h_g)
        self.root.leaf = True

        self.set_params(
            theta_vec,
            root,
        )

    def _gen_params_recursion(self,node,h_node):
        """ generate parameters recursively"""
        if node.depth == self.c_d_max:
            node.h_g = 0
        if h_node is None:
            if node.depth == self.c_d_max or self.rng.random() > self.h_g:  # 葉ノード
                node.theta_vec[:] = self.rng.dirichlet(self.h_beta_vec)
                node.leaf = True
            else:  # 内部ノード
                node.leaf = False
                for i in range(self.c_k):
                    if node.children[i] is None:
                        node.children[i] = _GenNode(node.depth+1,self.c_k,self.h_g)
                    self._gen_params_recursion(node.children[i],None)
        else:
            if node.depth == self.c_d_max or self.rng.random() > h_node.h_g:  # 葉ノード
                node.theta_vec[:] = self.rng.dirichlet(h_node.h_beta_vec)
                node.leaf = True
            else:  # 内部ノード
                node.leaf = False
                for i in range(self.c_k):
                    if node.children[i] is None:
                        node.children[i] = _GenNode(node.depth+1,self.c_k,self.h_g)
                    self._gen_params_recursion(node.children[i],h_node.children[i])

    def _gen_params_recursion_tree_fix(self,node,h_node):
        """ generate parameters recursively for fixed tree"""
        if h_node is None:
            if node.leaf:  # 葉ノード
                node.theta_vec[:] = self.rng.dirichlet(self.h_beta_vec)
            else:  # 内部ノード
                for i in range(self.c_k):
                    if node.children[i] is not None:
                        self._gen_params_recursion_tree_fix(node.children[i],None)
        else:
            if node.leaf:  # 葉ノード
                node.theta_vec[:] = self.rng.dirichlet(h_node.h_beta_vec)
            else:  # 内部ノード
                for i in range(self.c_k):
                    if node.children[i] is not None:
                        self._gen_params_recursion_tree_fix(node.children[i],h_node.children[i])

    def _set_recursion(self,node,original_tree_node):
        """ copy parameters from a fixed tree

        Parameters
        ----------
        node : object
                a object form GenNode class
        original_tree_node : object
                a object form GenNode class
        """
        node.h_g = original_tree_node.h_g
        node.h_beta_vec[:] = original_tree_node.h_beta_vec
        node.theta_vec[:] = original_tree_node.theta_vec
        if original_tree_node.leaf or node.depth == self.c_d_max:  # 葉ノード
            node.leaf = True
            if node.depth == self.c_d_max:
                node.h_g = 0
        else:
            node.leaf = False
            for i in range(self.c_k):
                node.children[i] = _GenNode(node.depth+1,self.c_k,self.h_g)
                self._set_recursion(node.children[i],original_tree_node.children[i])
    
    def _gen_sample_recursion(self,node,x):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        node : object
                a object form GenNode class

        x : numpy ndarray
            1 dimensional array whose elements are 0 or 1.
        """
        if node.leaf:  # 葉ノード
            print('')
            return self.rng.choice(self.c_k,p=node.theta_vec)
        else:
            print(f'{x[-node.depth-1]}',end=',')
            return self._gen_sample_recursion(node.children[x[-node.depth-1]],x)
    
    def _visualize_model_recursion(self,tree_graph,node,node_id,parent_id,sibling_num,p_v):
        """Visualize the stochastic data generative model and generated samples.

        """
        tmp_id = node_id
        tmp_p_v = p_v
        
        # add node information
        label_string = f'h_g={node.h_g:.2f}\\lp_v={tmp_p_v:.2f}\\ltheta_vec='
        if node.leaf:
            label_string += '['
            for i in range(self.c_k):
                label_string += f'{node.theta_vec[i]:.2f}'
                if i < self.c_k-1:
                    label_string += ','
            label_string += ']'
        else:
            label_string += 'None'
            
        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_v))}')
        if tmp_p_v > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        if node.leaf != True:
            for i in range(self.c_k):
                node_id = self._visualize_model_recursion(tree_graph,node.children[i],node_id+1,tmp_id,i,tmp_p_v*node.h_g)
        
        return node_id

    def set_h_params(self,
            h_g=None,
            h_beta_vec=None,
            h_root=None,
            ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_g : float, optional
            A real number in :math:`[0, 1]`, by default ``None``
        h_beta_vec : numpy.ndarray, optional
            A vector of positive real numbers, 
            by default ``None``
        h_root : contexttree._GenNode, optional
            A root node of a superposed tree for hyperparameters 
            by default ``None``
        """
        if h_g is not None:
            self.h_g = _check.float_in_closed01(h_g,'h_g',ParameterFormatError)

        if h_beta_vec is not None:
            _check.pos_floats(h_beta_vec,'h_beta_vec',ParameterFormatError)
            self.h_beta_vec[:] = h_beta_vec
        
        if h_root is not None:
            if type(h_root) is not _GenNode:
                raise(ParameterFormatError(
                    "h_root must be an instance of contexttree._GenNode"
                ))
            self.h_root = _GenNode(0,self.c_k,self.h_g)
            self._set_recursion(self.h_root,h_root)

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        hn_params : dict of {str: float, list, dict, numpy.ndarray}
            * ``"h_g"`` : the value of ``self.h_g``
            * ``"h_beta_vec"`` : the value of ``self.h_beta_vec``
            * ``"h_root"`` : the value of ``self.h_root``
        """
        return {"h_g":self.h_g, 
                "h_beta_vec":self.h_beta_vec, 
                "h_root":self.h_root}
            
    def gen_params(self,tree_fix=False):
        """Generate the parameter from the prior distribution.

        The generated vaule is set at ``self.root``.

        Parameters
        ----------
        tree_fix : bool
            If ``True``, tree shape will be fixed, by default ``False``.
        """
        if tree_fix:
            self._gen_params_recursion_tree_fix(self.root,self.h_root)
        else:
            self._gen_params_recursion(self.root,self.h_root)
    
    def set_params(self,theta_vec=None,root=None):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        theta_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]`, 
            by default None.
            Sum of its elements must be 1.0.
        root : contexttree._GenNode, optional
            A root node of a contexttree, by default None.
        """
        if theta_vec is not None:
            _check.float_vec_sum_1(theta_vec, "theta_vec", ParameterFormatError)
            _check.shape_consistency(
                theta_vec.shape[0],"theta_vec.shape[0]",
                self.c_k,"self.c_k",
                ParameterFormatError
                )
            self.theta_vec[:] = theta_vec
        if root is not None:
            if type(root) is not _GenNode:
                raise(ParameterFormatError(
                    "root must be an instance of metatree._GenNode"
                ))
            self._set_recursion(self.root,root)

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:float}
            * ``"theta_vec"`` : The value of ``self.theta_vec``.
            * ``"root"`` : The value of ``self.root``.
        """
        return {"theta_vec":self.theta_vec,"root":self.root}

    def gen_sample(self,sample_length,initial_values=None):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_length : int
            A positive integer
        initial_valules : numpy ndarray, optional
            1 dimensional int array whose size coincide with ``self.c_d_max``,
            by default ``None``. Its elements must be in [0,c_k-1]

        Returns
        -------
        x : numpy ndarray
            1 dimensional int array whose size is ``sammple_length``.
        """
        _check.pos_int(sample_length,'sample_length',DataFormatError)
        x = np.zeros(sample_length+self.c_d_max,dtype=int)
        if initial_values is not None:
            _check.nonneg_ints(initial_values,'initial_values',DataFormatError)
            _check.shape_consistency(
                initial_values.shape[0],'initial_values.shape[0]',
                self.c_d_max,'self.c_d_max',
                DataFormatError
            )
            if initial_values.max() >= self.c_k:
                raise(DataFormatError(f"initial_values.max() must smaller than c_k:{self.c_k}"))
            x[:self.c_d_max] = initial_values
        
        for i in range(self.c_d_max,sample_length+self.c_d_max):
            x[i] = self._gen_sample_recursion(self.root,x[:i])

        return x[self.c_d_max:]
        
    def save_sample(self,filename,sample_length,initial_values=None):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as a NpzFile with keyword: \"x\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_length : int
            A positive integer
        initial_valules : numpy ndarray, optional
            1 dimensional int array whose size coincide with ``self.c_d_max``,
            by default ``None``. Its elements must be in [0,c_k-1]
        
        See Also
        --------
        numpy.savez_compressed
        """
        x = self.gen_sample(sample_length,initial_values)
        np.savez_compressed(filename,x)

    def visualize_model(self,filename=None,format=None,sample_length=10):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        filename : str, optional
            Filename for saving the figure, by default ``None``
        format : str, optional
            Rendering output format (``\"pdf\"``, ``\"png\"``, ...).
        sample_length : int, optional
            A positive integer, by default 10

        Examples
        --------
        >>> from bayesml import contexttree
        >>> model = contexttree.GenModel(c_k=3,c_d_max=3,h_g=0.75)
        >>> gen_model.gen_params()
        >>> model.visualize_model()
        [1 1 1 1 1 1 0 0 0 1]

        .. image:: ./images/contexttree_example.png

        See Also
        --------
        graphbiz.Digraph
        """
        #例外処理
        _check.pos_int(sample_length,'sample_length',DataFormatError)

        try:
            import graphviz
            tree_graph = graphviz.Digraph(filename=filename,format=format)
            tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
            self._visualize_model_recursion(tree_graph, self.root, 0, None, None, 1.0)        
            # コンソール上で表示できるようにした方がいいかもしれない．
            tree_graph.view()
        except ImportError as e:
            print(e)
        except graphviz.CalledProcessError as e:
            print(e)
        x = self.gen_sample(sample_length)
        print(x)

class _LearnNode():
    """ The node class used by the posterior distribution

    Parameters
    ----------
    depth : int
            a non-negetive integer :math:' >= 0'
    hn_g : float
            a positive real number  in \[0 , 1 \], by default 0.5
    k : int
            a positive integer, by default None
    sub_model : class
            a class of generative model used by MT-Model 
    """
    def __init__(self,
                 depth,
                 c_k = 2,
                 hn_g = 0.5,
                 ):
        self.depth = depth
        self.children = [None for i in range(c_num_children)]  # child nodes
        self.hn_g = hn_g
        self.k = k
        self.sub_model = sub_model
        self.leaf = False
        self.map_leaf = False

# class LearnModel(base.Posterior,base.PredictiveMixin):
#     """The posterior distribution and the predictive distribution.

#     Parameters
#     ----------
#     c_k : int
#         A positive integer
#     c_d_max : int, optional
#         A positive integer, by default 10
#     c_num_children : int, optional
#         A positive integer, by default 2
#     SubModel : class, optional
#         LearnModel of bernoulli, categorical, 
#         poisson, normal, multivariate_normal, 
#         exponential, linearregression, 
#         by default bernoulli.LearnModel
#     h0_k_prob_vec : numpy.ndarray, optional
#         A vector of real numbers in :math:`[0, 1]`, 
#         by default [1/c_k, 1/c_k, ... , 1/c_k]
#         Sum of its elements must be 1.0.
#     h0_g : float, optional
#         A real number in :math:`[0, 1]`, by default 0.5
#     sub_h0_params : dict, optional
#         h0_params for self.SubModel, by default {}
#     h0_metatree_list : list of metatree._LearnNode, optional
#         Root nodes of meta-trees, by default []
#     h0_metatree_prob_vec : numpy.ndarray, optional
#         A vector of real numbers in :math:`[0, 1]` 
#         that represents prior distribution of h0_metatree_list, 
#         by default uniform distribution
#         Sum of its elements must be 1.0.

#     Attributes
#     ----------
#     hn_k_prob_vec : numpy.ndarray
#         A vector of real numbers in :math:`[0, 1]`. 
#         Sum of its elements is 1.
#     hn_g : float
#         A real number in :math:`[0, 1]`
#     sub_hn_params : dict
#         hn_params for self.SubModel
#     hn_metatree_list : list of metatree._LearnNode
#         Root nodes of meta-trees
#     hn_metatree_prob_vec : numpy.ndarray
#         A vector of real numbers in :math:`[0, 1]` 
#         that represents prior distribution of h0_metatree_list.
#         Sum of its elements is 1.0.
#     """
#     def __init__(
#             self,
#             c_k,
#             c_d_max=10,
#             c_num_children=2,
#             *,
#             SubModel=bernoulli.LearnModel,
#             h0_k_prob_vec = None,
#             h0_g=0.5,
#             sub_h0_params={},
#             h0_metatree_list=[],
#             h0_metatree_prob_vec=None
#             ):
#         # constants
#         self.c_d_max = _check.pos_int(c_d_max,'c_d_max',ParameterFormatError)
#         self.c_num_children = _check.pos_int(c_num_children,'c_num_children',ParameterFormatError)
#         self.c_k = _check.pos_int(c_k,'c_k',ParameterFormatError)
#         if SubModel not in LEARN_MODELS:
#             raise(ParameterFormatError(
#                 "SubModel must be a LearnModel of bernoulli, "
#                 +"poisson, normal, exponential."
#             ))
#         self.SubModel = SubModel

#         # h0_params
#         self.h0_k_prob_vec = np.ones(self.c_k) / self.c_k
#         self.h0_g = 0.5
#         self.sub_h0_params = {}
#         self.h0_metatree_list = []
#         self.h0_metatree_prob_vec = None

#         # hn_params
#         self.hn_k_prob_vec = np.ones(self.c_k) / self.c_k
#         self.hn_g = 0.5
#         self.sub_hn_params = {}
#         self.hn_metatree_list = []
#         self.hn_metatree_prob_vec = None

#         self._tmp_x = np.zeros(self.c_k,dtype=int)

#         self.set_h0_params(
#             h0_k_prob_vec,
#             h0_g,
#             sub_h0_params,
#             h0_metatree_list,
#             h0_metatree_prob_vec,
#         )

#     def set_h0_params(self,
#         h0_k_prob_vec = None,
#         h0_g=None,
#         sub_h0_params=None,
#         h0_metatree_list=None,
#         h0_metatree_prob_vec=None
#         ):
#         """Set the hyperparameters of the prior distribution.

#         Parameters
#         ----------
#         h0_k_prob_vec : numpy.ndarray, optional
#             A vector of real numbers in :math:`[0, 1]`, 
#             by default None
#             Sum of its elements must be 1.
#         h0_g : float, optional
#             A real number in :math:`[0, 1]`, by default None
#         sub_h0_params : dict, optional
#             h0_params for self.SubModel, by default None
#         h0_metatree_list : list of metatree._LearnNode, optional
#             Root nodes of meta-trees, by default None
#         h0_metatree_prob_vec : numpy.ndarray, optional
#             A vector of real numbers in :math:`[0, 1]` 
#             that represents prior distribution of h0_metatree_list, 
#             by default None.
#             Sum of its elements must be 1.0.
#         """
#         if h0_k_prob_vec is not None:
#             _check.float_vec_sum_1(h0_k_prob_vec,'h0_k_prob_vec',ParameterFormatError)
#             _check.shape_consistency(
#                 h0_k_prob_vec.shape[0],'h0_k_prob_vec',
#                 self.c_k,'self.c_k',
#                 ParameterFormatError
#                 )
#             self.h0_k_prob_vec[:] = h0_k_prob_vec

#         if h0_g is not None:
#             self.h0_g = _check.float_in_closed01(h0_g,'h0_g',ParameterFormatError)

#         if sub_h0_params is not None:
#             self.sub_h0_params = copy.deepcopy(sub_h0_params)
#             self.SubModel(**self.sub_h0_params)

#         if h0_metatree_list is not None:
#             self.h0_metatree_list = copy.deepcopy(h0_metatree_list)
#             if h0_metatree_prob_vec is not None:
#                 self.h0_metatree_prob_vec = np.copy(
#                     _check.float_vec_sum_1(
#                         h0_metatree_prob_vec,
#                         'h0_metatree_prob_vec',
#                         ParameterFormatError
#                     )
#                 )
#             elif len(self.h0_metatree_list) > 0:
#                 metatree_num = len(self.h0_metatree_list)
#                 self.h0_metatree_prob_vec = np.ones(metatree_num) / metatree_num
#         elif h0_metatree_prob_vec is not None:
#             self.h0_metatree_prob_vec = np.copy(
#                 _check.float_vec_sum_1(
#                     h0_metatree_prob_vec,
#                     'h0_metatree_prob_vec',
#                     ParameterFormatError
#                 )
#             )

#         if type(self.h0_metatree_prob_vec) is np.ndarray:             
#             if self.h0_metatree_prob_vec.shape[0] != len(self.h0_metatree_list):
#                 raise(ParameterFormatError(
#                     "Length of h0_metatree_list and dimension of h0_metatree_prob_vec must be the same."
#                 ))
#         else:
#             if len(self.h0_metatree_list) > 0:
#                 raise(ParameterFormatError(
#                     "Length of h0_metatree_list must be zero when self.h0_metatree_prob_vec is None."
#                 ))

#         self.reset_hn_params()

#     def get_h0_params(self):
#         """Get the hyperparameters of the prior distribution.

#         Returns
#         -------
#         h0_params : dict of {str: float, list, dict, numpy.ndarray}
#             * ``"h0_k_prob_vec"`` : the value of ``self.h0_k_prob_vec``
#             * ``"h0_g"`` : the value of ``self.h0_g``
#             * ``"sub_h0_params"`` : the value of ``self.sub_h0_params``
#             * ``"h0_metatree_list"`` : the value of ``self.h0_metatree_list``
#             * ``"h0_metatree_prob_vec"`` : the value of ``self.h0_metatree_prob_vec``
#         """
#         return {"h0_k_prob_vec":self.h0_k_prob_vec,
#                 "h0_g":self.h0_g, 
#                 "sub_h0_params":self.sub_h0_params, 
#                 "h0_metatree_list":self.h0_metatree_list,
#                 "h0_metatree_prob_vec":self.h0_metatree_prob_vec}
    
#     def set_hn_params(self,
#         hn_k_prob_vec = None,
#         hn_g=None,
#         sub_hn_params=None,
#         hn_metatree_list=None,
#         hn_metatree_prob_vec=None
#         ):
#         """Set the hyperparameter of the posterior distribution.

#         Parameters
#         ----------
#         hn_k_prob_vec : numpy.ndarray, optional
#             A vector of real numbers in :math:`[0, 1]`, 
#             by default None
#             Sum of its elements must be 1.
#         hn_g : float, optional
#             A real number in :math:`[0, 1]`, by default None
#         sub_hn_params : dict, optional
#             hn_params for self.SubModel, by default None
#         hn_metatree_list : list of metatree._LearnNode, optional
#             Root nodes of meta-trees, by default None
#         hn_metatree_prob_vec : numpy.ndarray, optional
#             A vector of real numbers in :math:`[0, 1]` 
#             that represents prior distribution of hn_metatree_list, 
#             by default None.
#             Sum of its elements must be 1.0.
#         """
#         if hn_k_prob_vec is not None:
#             _check.float_vec_sum_1(hn_k_prob_vec,'hn_k_prob_vec',ParameterFormatError)
#             _check.shape_consistency(
#                 hn_k_prob_vec.shape[0],'hn_k_prob_vec',
#                 self.c_k,'self.c_k',
#                 ParameterFormatError
#                 )
#             self.hn_k_prob_vec[:] = hn_k_prob_vec

#         if hn_g is not None:
#             self.hn_g = _check.float_in_closed01(hn_g,'hn_g',ParameterFormatError)

#         if sub_hn_params is not None:
#             self.sub_hn_params = copy.deepcopy(sub_hn_params)
#             self.SubModel(**self.sub_hn_params)

#         if hn_metatree_list is not None:
#             self.hn_metatree_list = copy.deepcopy(hn_metatree_list)
#             if hn_metatree_prob_vec is not None:
#                 self.hn_metatree_prob_vec = np.copy(
#                     _check.float_vec_sum_1(
#                         hn_metatree_prob_vec,
#                         'hn_metatree_prob_vec',
#                         ParameterFormatError
#                     )
#                 )
#             elif len(self.hn_metatree_list) > 0:
#                 metatree_num = len(self.hn_metatree_list)
#                 self.hn_metatree_prob_vec = np.ones(metatree_num) / metatree_num
#         elif hn_metatree_prob_vec is not None:
#             self.hn_metatree_prob_vec = np.copy(
#                 _check.float_vec_sum_1(
#                     hn_metatree_prob_vec,
#                     'hn_metatree_prob_vec',
#                     ParameterFormatError
#                 )
#             )

#         if type(self.hn_metatree_prob_vec) is np.ndarray:             
#             if self.hn_metatree_prob_vec.shape[0] != len(self.hn_metatree_list):
#                 raise(ParameterFormatError(
#                     "Length of hn_metatree_list and dimension of hn_metatree_prob_vec must be the same."
#                 ))
#         else:
#             if len(self.hn_metatree_list) > 0:
#                 raise(ParameterFormatError(
#                     "Length of hn_metatree_list must be zero when self.hn_metatree_prob_vec is None."
#                 ))

#         self.calc_pred_dist(np.zeros(self.c_k,dtype=int))

#     def get_hn_params(self):
#         """Get the hyperparameters of the posterior distribution.

#         Returns
#         -------
#         hn_params : dict of {str: float, list, dict, numpy.ndarray}
#             * ``"hn_k_prob_vec"`` : the value of ``self.hn_k_prob_vec``
#             * ``"hn_g"`` : the value of ``self.hn_g``
#             * ``"sub_hn_params"`` : the value of ``self.sub_hn_params``
#             * ``"hn_metatree_list"`` : the value of ``self.hn_metatree_list``
#             * ``"hn_metatree_prob_vec"`` : the value of ``self.hn_metatree_prob_vec``
#         """
#         return {"hn_k_prob_vec":self.hn_k_prob_vec,
#                 "hn_g":self.hn_g, 
#                 "sub_hn_params":self.sub_hn_params, 
#                 "hn_metatree_list":self.hn_metatree_list,
#                 "hn_metatree_prob_vec":self.hn_metatree_prob_vec}
    
#     # def reset_hn_params(self):
#     #     """Reset the hyperparameters of the posterior distribution to their initial values.
        
#     #     They are reset to `self.h0_k_prob_vec`, `self.h0_g`, `self.sub_h0_params`, 
#     #     `self.h0_metatree_list` and `self.h0_metatree_prob_vec`.
#     #     Note that the parameters of the predictive distribution are also calculated from them.
#     #     """
#     #     self.hn_k_prob_vec = np.copy(self.h0_k_prob_vec)
#     #     self.hn_g = np.copy(self.h0_g)
#     #     self.sub_hn_params = copy.deepcopy(self.sub_h0_params)
#     #     self.hn_metatree_list = copy.deepcopy(self.h0_metatree_list)
#     #     self.hn_metatree_prob_vec = copy.deepcopy(self.h0_metatree_prob_vec)

#     #     self.calc_pred_dist(np.zeros(self.c_k,dtype=int))
    
#     # def overwrite_h0_params(self):
#     #     """Overwrite the initial values of the hyperparameters of the posterior distribution by the learned values.
        
#     #     They are overwitten by `self.hn_k_prob_vec`, `self.hn_g`, `self.sub_hn_params`, 
#     #     `self.hn_metatree_list` and `self.hn_metatree_prob_vec`.
#     #     Note that the parameters of the predictive distribution are also calculated from them.
#     #     """
#     #     self.h0_k_prob_vec = np.copy(self.hn_k_prob_vec)
#     #     self.h0_g = np.copy(self.hn_g)
#     #     self.sub_h0_params = copy.deepcopy(self.sub_hn_params)
#     #     self.h0_metatree_list = copy.deepcopy(self.hn_metatree_list)
#     #     self.h0_metatree_prob_vec = np.copy(self.hn_metatree_prob_vec)

#     #     self.calc_pred_dist(np.zeros(self.c_k))

#     def _copy_tree_from_sklearn_tree(self,new_node, original_tree,node_id):
#         if original_tree.children_left[node_id] != sklearn_tree._tree.TREE_LEAF:  # 内部ノード
#             new_node.k = original_tree.feature[node_id]
#             new_node.children[0] = _LearnNode(depth=new_node.depth+1,
#                                               c_num_children=2,
#                                               hn_g=self.h0_g,
#                                               k=None,
#                                               sub_model=self.SubModel(**self.sub_h0_params))
#             self._copy_tree_from_sklearn_tree(new_node.children[0],original_tree,original_tree.children_left[node_id])
#             new_node.children[1] = _LearnNode(depth=new_node.depth+1,
#                                               c_num_children=2,
#                                               hn_g=self.h0_g,
#                                               k=None,
#                                               sub_model=self.SubModel(**self.sub_h0_params))
#             self._copy_tree_from_sklearn_tree(new_node.children[1],original_tree,original_tree.children_right[node_id])
#         else:
#             new_node.hn_g = 0.0
#             new_node.leaf = True

#     def _update_posterior_leaf(self,node,x,y):
#             try:
#                 node.sub_model.calc_pred_dist(x)
#             except:
#                 node.sub_model.calc_pred_dist()
#             pred_dist = node.sub_model.make_prediction(loss='KL') # Futurework: direct method to get marginal likelihood is better

#             try:
#                 node.sub_model.update_posterior(x,y)
#             except:
#                 node.sub_model.update_posterior(y)

#             if type(pred_dist) is np.ndarray:
#                 return pred_dist[y]
#             try:
#                 return pred_dist.pdf(y)
#             except:
#                 return pred_dist.pmf(y)

#     def _update_posterior_recursion(self,node,x,y):
#         if node.leaf == False:  # 内部ノード
#             tmp1 = self._update_posterior_recursion(node.children[x[node.k]],x,y)
#             tmp2 = (1 - node.hn_g) * self._update_posterior_leaf(node,x,y) + node.hn_g * tmp1
#             node.hn_g = node.hn_g * tmp1 / tmp2
#             return tmp2
#         else:  # 葉ノード
#             return self._update_posterior_leaf(node,x,y)

#     def _compare_metatree_recursion(self,node1,node2):
#         if node1.leaf == True and node2.leaf == True:
#             return True
#         elif node1.k == node2.k:
#             for i in range(self.c_num_children):
#                 if self._compare_metatree_recursion(node1.children[i],node2.children[i]) == False:
#                     return False
#             return True
#         else:
#             return False
    
#     def _marge_metatrees(self,metatree_list,metatree_prob_vec):
#         num_metatrees = len(metatree_list)
#         for i in range(num_metatrees):
#             for j in range(i+1,num_metatrees):
#                 if self._compare_metatree_recursion(metatree_list[i],metatree_list[j]):
#                     metatree_list[i] = None
#                     metatree_prob_vec[j] += metatree_prob_vec[i]
#                     metatree_prob_vec[i] = -1
#                     break
#         metatree_list = [tmp for tmp in metatree_list if tmp != None]
#         metatree_prob_vec = metatree_prob_vec[metatree_prob_vec > -0.5]
#         return metatree_list,metatree_prob_vec

#     def _MTRF(self,x,y,n_estimators=100,**kwargs):
#         """make metatrees

#         Parameters
#         ----------
#         x : numpy ndarray
#             values of explanatory variables whose dtype is int
#         y : numpy ndarray
#             values of objective variable whose dtype may be int or float
#         n_estimators : int, optional
#             number of trees in sklearn.RandomForestClassifier, by default 100

#         Returns
#         -------
#         metatree_list : list of metatree._LearnNode
#             Each element is a root node of metatree.
#         metatree_prob_vec : numpy ndarray
#         """
#         if self.c_num_children != 2:
#             raise(ParameterFormatError("MTRF is supported only when c_num_children == 2."))
#         if self.SubModel in DISCRETE_LEARN_MODELS:
#             randomforest = RandomForestClassifier(n_estimators=n_estimators,max_depth=self.c_d_max)
#         if self.SubModel in CONTINUOUS_LEARN_MODELS:
#             randomforest = RandomForestRegressor(n_estimators=n_estimators,max_depth=self.c_d_max)
#         randomforest.fit(x,y)
#         tmp_metatree_list = [_LearnNode(0,2,self.h0_g,None,self.SubModel(**self.sub_h0_params)) for i in range(n_estimators)]
#         tmp_metatree_prob_vec = np.ones(n_estimators) / n_estimators
#         for i in range(n_estimators):
#             self._copy_tree_from_sklearn_tree(tmp_metatree_list[i],randomforest.estimators_[i].tree_, 0)

#         tmp_metatree_list,tmp_metatree_prob_vec = self._marge_metatrees(tmp_metatree_list,tmp_metatree_prob_vec)

#         log_metatree_posteriors = np.log(tmp_metatree_prob_vec)
#         for i,metatree in enumerate(tmp_metatree_list):
#             for j in range(x.shape[0]):
#                 log_metatree_posteriors[i] += np.log(self._update_posterior_recursion(metatree,x[j],y[j]))
#         tmp_metatree_prob_vec[:] = np.exp(log_metatree_posteriors - log_metatree_posteriors.max())
#         tmp_metatree_prob_vec[:] /= tmp_metatree_prob_vec.sum()
#         return tmp_metatree_list,tmp_metatree_prob_vec

#     def _given_MT(self,x,y):
#         """make metatrees

#         Parameters
#         ----------
#         x : numpy ndarray
#             values of explanatory variables whose dtype is int
#         y : numpy ndarray
#             values of objective variable whose dtype may be int or float

#         Returns
#         -------
#         metatree_list : list of metatree._LearnNode
#             Each element is a root node of metatree.
#         metatree_prob_vec : numpy ndarray
#         """
#         if len(self.hn_metatree_list) == 0:
#             raise(ParameterFormatError("given_MT is supported only when len(self.hn_metatree_list) > 0."))
#         log_metatree_posteriors = np.log(self.hn_metatree_prob_vec)
#         for i,metatree in enumerate(self.hn_metatree_list):
#             for j in range(x.shape[0]):
#                 log_metatree_posteriors[i] += np.log(self._update_posterior_recursion(metatree,x[j],y[j]))
#         self.hn_metatree_prob_vec[:] = np.exp(log_metatree_posteriors - log_metatree_posteriors.max())
#         self.hn_metatree_prob_vec[:] /= self.hn_metatree_prob_vec.sum()
#         return self.hn_metatree_list,self.hn_metatree_prob_vec

#     def update_posterior(self,x,y,alg_type='MTRF',**kwargs):
#         """Update the hyperparameters of the posterior distribution using traning data.

#         Parameters
#         ----------
#         x : numpy ndarray
#             values of explanatory variables whose dtype is int
#         y : numpy ndarray
#             values of objective variable whose dtype may be int or float
#         alg_type : {'MTRF', 'given_MT'}, optional
#             type of algorithm, by default 'MTRF'
#         **kwargs : dict, optional
#             optional parameters of algorithms, by default {}
#         """
#         _check.nonneg_int_vecs(x,'x',DataFormatError)
#         if x.shape[-1] != self.c_k:
#             raise(DataFormatError(f"x.shape[-1] must equal to c_k:{self.c_k}"))
#         if x.max() >= self.c_num_children:
#             raise(DataFormatError(f"x.max() must smaller than c_num_children:{self.c_num_children}"))
                
#         if type(y) is np.ndarray:
#             if x.shape[:-1] != y.shape: 
#                 raise(DataFormatError(f"x.shape[:-1] and y.shape must be same."))
#         elif x.shape[:-1] != ():
#             raise(DataFormatError(f"If y is a scaler, x.shape[:-1] must be the empty tuple ()."))

#         x = x.reshape(-1,self.c_k)
#         y = np.ravel(y)

#         if alg_type == 'MTRF':
#             self.hn_metatree_list, self.hn_metatree_prob_vec = self._MTRF(x,y,**kwargs)
#         elif alg_type == 'given_MT':
#             self.hn_metatree_list, self.hn_metatree_prob_vec = self._given_MT(x,y)

#     def _map_recursion_add_nodes(self,node):
#         if node.depth == self.c_d_max or node.depth == self.c_k:  # 葉ノード
#             node.hn_g = 0.0
#             node.leaf = True
#             node.map_leaf = True
#         else:  # 内部ノード
#             for i in range(self.c_num_children):
#                 node.children[i] = _LearnNode(depth=node.depth+1,
#                                               c_num_children=self.c_num_children,
#                                               hn_g=self.h0_g,
#                                               k=None,
#                                               sub_model=self.SubModel(**self.sub_h0_params))
#                 self._map_recursion_add_nodes(node.children[i])

#     def _map_recursion(self,node):
#         if node.leaf:
#             if node.depth == self.c_d_max or node.depth == self.c_k:
#                 node.map_leaf = True
#                 return 1.0
#             elif 1.0 - node.hn_g > node.hn_g * self.h0_g ** (self.c_k ** (self.c_d_max - node.depth)-2):
#                 node.map_leaf = True
#                 return 1.0 - node.hn_g
#             else:
#                 self._map_recursion_add_nodes(node)
#                 return node.hn_g * self.h0_g ** (self.c_k ** (self.c_d_max - node.depth)-2)
#         else:
#             tmp1 = 1.0-node.hn_g
#             tmp_vec = np.empty(self.c_num_children)
#             for i in range(self.c_num_children):
#                 tmp_vec[i] = self._map_recursion(node.children[i])
#             if tmp1 > node.hn_g*tmp_vec.prod():
#                 node.map_leaf = True
#                 return tmp1
#             else:
#                 node.map_leaf = False
#                 return node.hn_g*tmp_vec.prod()

#     def _copy_map_tree_recursion(self,copyed_node,original_node):
#         copyed_node.hn_g = original_node.hn_g
#         if original_node.map_leaf == False:
#             copyed_node.k = original_node.k
#             for i in range(self.c_num_children):
#                 copyed_node.children[i] = _LearnNode(copyed_node.depth+1,self.c_num_children)
#                 self._copy_map_tree_recursion(copyed_node.children[i],original_node.children[i])
#         else:
#             copyed_node.sub_model = copy.deepcopy(original_node.sub_model)
#             copyed_node.leaf = True

#     def estimate_params(self,loss="0-1",visualize=True,filename=None,format=None):
#         """Estimate the parameter under the given criterion.

#         Parameters
#         ----------
#         loss : str, optional
#             Loss function underlying the Bayes risk function, by default ``\"0-1\"``.
#             This function supports only ``\"0-1\"``.
#         visualize : bool, optional
#             If ``True``, the estimated metatree will be visualized, by default ``True``.
#             This visualization requires ``graphviz``.
#         filename : str, optional
#             Filename for saving the figure, by default ``None``
#         format : str, optional
#             Rendering output format (``\"pdf\"``, ``\"png\"``, ...).


#         Returns
#         -------
#         map_root : metatree._LearnNode
#             The root node of the estimated meta-tree 
#             that also contains the estimated parameters in each node.

#         See Also
#         --------
#         graphbiz.Digraph
#         """

#         if loss == "0-1":
#             map_index = 0
#             map_prob = 0.0
#             for i,metatree in enumerate(self.hn_metatree_list):
#                 prob = self.hn_metatree_prob_vec[i] * self._map_recursion(metatree)
#                 if prob > map_prob:
#                     map_index = i
#                     map_prob = prob
#             map_root = _LearnNode(0,self.c_num_children)
#             self._copy_map_tree_recursion(map_root,self.hn_metatree_list[map_index])
#             if visualize:
#                 import graphviz
#                 tree_graph = graphviz.Digraph(filename=filename,format=format)
#                 tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
#                 self._visualize_model_recursion(tree_graph, map_root, 0, None, None, 1.0)
#                 tree_graph.view()
#             return map_root
#         else:
#             raise(CriteriaError("Unsupported loss function! "
#                                 +"This function supports only \"0-1\"."))
    
#     def _visualize_model_recursion(self,tree_graph,node,node_id,parent_id,sibling_num,p_v):
#         tmp_id = node_id
#         tmp_p_v = p_v
        
#         # add node information
#         label_string = f'k={node.k}\\lhn_g={node.hn_g:.2f}\\lp_v={tmp_p_v:.2f}\\lsub_params={{'
#         if node.sub_model is not None:
#             try:
#                 sub_params = node.sub_model.estimate_params(loss='0-1',dict_out=True)
#             except:
#                 sub_params = node.sub_model.estimate_params(dict_out=True)
            
#             for key,value in sub_params.items():
#                 try:
#                     label_string += f'\\l{key}:{value:.2f}'
#                 except:
#                     label_string += f'\\l{key}:{value}'
#                 label_string += '}'
#         else:
#             label_string += '\\lNone}'

#         tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_v))}')
#         if tmp_p_v > 0.65:
#             tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
#         # add edge information
#         if parent_id is not None:
#             tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
#         if node.leaf != True:
#             for i in range(self.c_num_children):
#                 node_id = self._visualize_model_recursion(tree_graph,node.children[i],node_id+1,tmp_id,i,tmp_p_v*node.hn_g)
        
#         return node_id

#     def visualize_posterior(self,filename=None,format=None):
#         """Visualize the posterior distribution for the parameter.
        
#         This method requires ``graphviz``.

#         Parameters
#         ----------
#         filename : str, optional
#             Filename for saving the figure, by default ``None``
#         format : str, optional
#             Rendering output format (``\"pdf\"``, ``\"png\"``, ...).

#         Examples
#         --------
#         >>> from bayesml import metatree
#         >>> gen_model = metatree.GenModel(c_k=3,h_g=0.75)
#         >>> x,y = gen_model.gen_sample(500)
#         >>> learn_model = metatree.LearnModel(c_k=3)
#         >>> learn_model.update_posterior(x,y)
#         >>> learn_model.visualize_posterior()

#         .. image:: ./images/metatree_posterior.png

#         See Also
#         --------
#         graphbiz.Digraph
#         """
#         MAP_index = np.argmax(self.hn_metatree_prob_vec)
#         print(f'MAP probability of metatree:{self.hn_metatree_prob_vec[MAP_index]}')
#         try:
#             import graphviz
#             tree_graph = graphviz.Digraph(filename=filename,format=format)
#             tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
#             self._visualize_model_recursion(tree_graph, self.hn_metatree_list[MAP_index], 0, None, None, 1.0)        
#             # コンソール上で表示できるようにした方がいいかもしれない．
#             tree_graph.view()
#         except ImportError as e:
#             print(e)
#         except graphviz.CalledProcessError as e:
#             print(e)
    
#     def get_p_params(self):
#         """Get the parameters of the predictive distribution.

#         This model does not have a simple parametric expression of the predictive distribution.
#         Therefore, this function returns ``None``.

#         Returns
#         -------
#         ``None``
#         """
#         return None
    
#     def _calc_pred_dist_leaf(self,node,x):
#             try:
#                 node.sub_model.calc_pred_dist(x)
#             except:
#                 node.sub_model.calc_pred_dist()

#     def _calc_pred_dist_recursion(self,node,x):
#         if node.leaf == False:  # 内部ノード
#             self._calc_pred_dist_recursion(node.children[x[node.k]],x)
#         else:  # 葉ノード
#             return self._calc_pred_dist_leaf(node,x)

#     def calc_pred_dist(self,x):
#         """Calculate the parameters of the predictive distribution.
        
#         Parameters
#         ----------
#         x : numpy ndarray
#             values of explanatory variables whose dtype is int
#         """
#         _check.nonneg_int_vec(x,'x',DataFormatError)
#         if x.shape[0] != self.c_k:
#             raise(DataFormatError(f"x.shape[0] must equal to c_k:{self.c_k}"))
#         if x.max() >= self.c_num_children:
#             raise(DataFormatError(f"x.max() must smaller than c_num_children:{self.c_num_children}"))
#         self._tmp_x[:] = x
#         for root in self.hn_metatree_list:
#             self._calc_pred_dist_recursion(root,self._tmp_x)

#     def _make_prediction_recursion_squared(self,node):
#             if node.leaf == False:  # 内部ノード
#                 return ((1 - node.hn_g) * node.sub_model.make_prediction(loss='squared')
#                         + node.hn_g * self._make_prediction_recursion_squared(node.children[self._tmp_x[node.k]]))
#             else:  # 葉ノード
#                 return node.sub_model.make_prediction(loss='squared')

#     def _make_prediction_leaf_01(self,node):
#         mode = node.sub_model.make_prediction(loss='0-1')
#         pred_dist = node.sub_model.make_prediction(loss='KL')
#         if type(pred_dist) is np.ndarray:
#             mode_prob = pred_dist[mode]
#         elif hasattr(pred_dist,'pdf'):
#             mode_prob = pred_dist.pdf(mode)
#         elif hasattr(pred_dist,'pmf'):
#             mode_prob = pred_dist.pmf(mode)
#         else:
#             mode_prob = None
#         return mode, mode_prob

#     def _make_prediction_recursion_01(self,node):
#         if node.leaf == False:  # 内部ノード
#             mode1,mode_prob1 = self._make_prediction_leaf_01(node)
#             mode2,mode_prob2 = self._make_prediction_recursion_01(node.children[self._tmp_x[node.k]])
#             if (1 - node.hn_g) * mode_prob1 > node.hn_g * mode_prob2:
#                 return mode1,mode_prob1
#             else:
#                 return mode2,mode_prob2
#         else:  # 葉ノード
#             return self._make_prediction_leaf_01(node)

#     def make_prediction(self,loss="0-1"):
#         """Predict a new data point under the given criterion.

#         Parameters
#         ----------
#         loss : str, optional
#             Loss function underlying the Bayes risk function, by default \"0-1\".
#             This function supports \"squared\", \"0-1\".

#         Returns
#         -------
#         predicted_value : {float, numpy.ndarray}
#             The predicted value under the given loss function. 
#         """
#         if loss == "squared":
#             tmp_pred_vec = np.empty(len(self.hn_metatree_list))
#             for i,metatree in enumerate(self.hn_metatree_list):
#                 tmp_pred_vec[i] = self._make_prediction_recursion_squared(metatree)
#             return self.hn_metatree_prob_vec @ tmp_pred_vec
#         elif loss == "0-1":
#             tmp_mode = np.empty(len(self.hn_metatree_list))
#             tmp_mode_prob_vec = np.empty(len(self.hn_metatree_list))
#             for i,metatree in enumerate(self.hn_metatree_list):
#                 tmp_mode[i],tmp_mode_prob_vec[i] = self._make_prediction_recursion_01(metatree)
#             return tmp_mode[np.argmax(self.hn_metatree_prob_vec * tmp_mode_prob_vec)]
#         else:
#             raise(CriteriaError("Unsupported loss function! "
#                                 +"This function supports \"squared\" and \"0-1\"."))

#     def pred_and_update(self,x,y,loss="0-1"):
#         """Predict a new data point and update the posterior sequentially.

#         Parameters
#         ----------
#         x : numpy.ndarray
#             It must be a degree-dimensional vector
#         y : numpy ndarray
#             values of objective variable whose dtype may be int or float
#         loss : str, optional
#             Loss function underlying the Bayes risk function, by default \"0-1\".
#             This function supports \"squared\", \"0-1\", and \"KL\".

#         Returns
#         -------
#         predicted_value : {float, numpy.ndarray}
#             The predicted value under the given loss function. 
#         """
#         _check.nonneg_int_vec(x,'x',DataFormatError)
#         if x.shape[-1] != self.c_k:
#             raise(DataFormatError(f"x.shape[-1] must equal to c_k:{self.c_k}"))
#         if x.max() >= self.c_num_children:
#             raise(DataFormatError(f"x.max() must smaller than c_num_children:{self.c_num_children}"))
#         self.calc_pred_dist(x)
#         prediction = self.make_prediction(loss=loss)
#         self.update_posterior(x,y,alg_type='given_MT')
#         return prediction