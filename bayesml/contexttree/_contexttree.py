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
            return self.rng.choice(self.c_k,p=node.theta_vec)
        else:
            return self._gen_sample_recursion(node.children[x[-node.depth-1]],x)
    
    def _visualize_model_recursion(self,tree_graph,node,node_id,parent_id,sibling_num,p_v):
        """Visualize the stochastic data generative model and generated samples."""
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
        h_params : dict of {str: float, numpy.ndarray, contexttree._GenNode}
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
        >>> model = contexttree.GenModel(c_k=2,c_d_max=3,h_g=0.75)
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
                 c_k,
                 h0_g,
                 hn_g,
                 h0_beta_vec,
                 hn_beta_vec,
                 ):
        self.depth = depth
        self.children = [None for i in range(c_k)]  # child nodes
        self.h0_g = h0_g
        self.hn_g = hn_g
        self.h0_beta_vec = np.copy(h0_beta_vec)
        self.hn_beta_vec = np.copy(hn_beta_vec)
        self.leaf = False
        self.map_leaf = False

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    c_k : int
        A positive integer
    c_d_max : int, optional
        A positive integer, by default 10
    h0_g : float, optional
        A real number in :math:`[0, 1]`, by default 0.5
    h0_beta_vec : numpy.ndarray, optional
        A vector of positive real numbers, 
        by default [1/2, 1/2, ... , 1/2].
        If a single real number is input, it will be broadcasted.
    h0_root : contexttree._LearnNode, optional
        A root node of a superposed tree for hyperparameters 
        by default ``None``

    Attributes
    ----------
    hn_g : float
        A real number in :math:`[0, 1]`
    hn_beta_vec : numpy.ndarray
        A vector of positive real numbers.
    hn_root : contexttree._LearnNode
        A root node of a superposed tree for hyperparameters.
    """
    def __init__(
            self,
            c_k,
            c_d_max=10,
            *,
            h0_g=0.5,
            h0_beta_vec=None,
            h0_root=None,
            ):
        # constants
        self.c_d_max = _check.pos_int(c_d_max,'c_d_max',ParameterFormatError)
        self.c_k = _check.pos_int(c_k,'c_k',ParameterFormatError)

        # h0_params
        self.h0_g = h0_g
        self.h0_beta_vec = np.ones(self.c_k) / 2
        self.h0_root = None

        # hn_params
        self.hn_g = h0_g
        self.hn_beta_vec = np.ones(self.c_k) / 2
        self.hn_root = None

        # p_params
        self.p_theta_vec = np.ones(self.c_k) / self.c_k

        self.set_h0_params(
            h0_g,
            h0_beta_vec,
            h0_root,
        )

    def _set_recursion(self,node:_LearnNode,original_tree_node:_LearnNode):
        """ copy parameters from a fixed tree

        Parameters
        ----------
        node : object
                a object form LearnNode class
        original_tree_node : object
                a object form LearnNode class
        """
        node.h0_g = original_tree_node.h0_g
        node.h0_beta_vec[:] = original_tree_node.h0_beta_vec
        node.hn_g = original_tree_node.hn_g
        node.hn_beta_vec[:] = original_tree_node.hn_beta_vec
        if original_tree_node.leaf or node.depth == self.c_d_max:  # 葉ノード
            node.leaf = True
            if node.depth == self.c_d_max:
                node.h0_g = 0
                node.hn_g = 0
        else:
            node.leaf = False
            for i in range(self.c_k):
                node.children[i] = _LearnNode(
                    node.depth+1,
                    self.c_k,
                    self.h0_g,
                    self.hn_g,
                    self.h0_beta_vec,
                    self.hn_beta_vec,
                )
                self._set_recursion(node.children[i],original_tree_node.children[i])

    def set_h0_params(self,
        h0_g=None,
        h0_beta_vec=None,
        h0_root=None,
        ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h0_g : float, optional
            A real number in :math:`[0, 1]`, by default ``None``
        h0_beta_vec : numpy.ndarray, optional
            A vector of positive real numbers, 
            by default ``None``
        h0_root : contexttree._GenNode, optional
            A root node of a superposed tree for hyperparameters 
            by default ``None``
        """
        if h0_g is not None:
            self.h0_g = _check.float_in_closed01(h0_g,'h0_g',ParameterFormatError)

        if h0_beta_vec is not None:
            _check.pos_floats(h0_beta_vec,'h0_beta_vec',ParameterFormatError)
            self.h0_beta_vec[:] = h0_beta_vec
        
        if h0_root is not None:
            if type(h0_root) is not _LearnNode:
                raise(ParameterFormatError(
                    "h0_root must be an instance of contexttree._LearnNode"
                ))
            self._set_recursion(self.h0_root,h0_root)

        self.reset_hn_params()

    def get_h0_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h0_params : dict of {str: float, numpy.ndarray, contexttre._LearnNode}
            * ``"h0_g"`` : the value of ``self.h0_g``
            * ``"h0_beta_vec"`` : the value of ``self.h0_beta_vec``
            * ``"h0_root"`` : the value of ``self.h0_root``
        """
        return {"h0_g":self.h0_g, 
                "h0_beta_vec":self.h0_beta_vec, 
                "h0_root":self.h0_root}
    
    def set_hn_params(self,
        hn_g=None,
        hn_beta_vec=None,
        hn_root=None,
        ):
        """Set the hyperparameter of the posterior distribution.

        Parameters
        ----------
        hn_g : float, optional
            A real number in :math:`[0, 1]`, by default ``None``
        hn_beta_vec : numpy.ndarray, optional
            A vector of positive real numbers, 
            by default ``None``
        hn_root : contexttree._GenNode, optional
            A root node of a superposed tree for hyperparameters 
            by default ``None``
        """
        if hn_g is not None:
            self.hn_g = _check.float_in_closed01(hn_g,'hn_g',ParameterFormatError)

        if hn_beta_vec is not None:
            _check.pos_floats(hn_beta_vec,'hn_beta_vec',ParameterFormatError)
            self.hn_beta_vec[:] = hn_beta_vec
        
        if hn_root is not None:
            if type(hn_root) is not _LearnNode:
                raise(ParameterFormatError(
                    "hn_root must be an instance of contexttree._LearnNode"
                ))
            self._set_recursion(self.hn_root,hn_root)

        self.calc_pred_dist(np.zeros(self.c_d_max,dtype=int))

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: float, numpy.ndarray, contexttre._LearnNode}
            * ``"hn_g"`` : the value of ``self.hn_g``
            * ``"hn_beta_vec"`` : the value of ``self.hn_beta_vec``
            * ``"hn_root"`` : the value of ``self.hn_root``
        """
        return {"hn_g":self.hn_g, 
                "hn_beta_vec":self.hn_beta_vec, 
                "hn_root":self.hn_root}
    
    def _update_posterior_leaf(self,node:_LearnNode,x,i):
            tmp = node.hn_beta_vec[x[i]] / node.hn_beta_vec.sum()
            node.hn_beta_vec[x[i]] += 1
            return tmp

    def _update_posterior_recursion(self,node:_LearnNode,x,i):
        if node.depth < self.c_d_max and i-1-node.depth >= 0:  # 内部ノード
            if node.children[x[i-node.depth-1]] is None:
                node.children[x[i-node.depth-1]] = _LearnNode(
                    node.depth+1,
                    self.c_k,
                    self.h0_g,
                    self.hn_g,
                    self.h0_beta_vec,
                    self.hn_beta_vec,
                )
                if node.depth + 1 == self.c_d_max:
                    node.children[x[i-node.depth-1]].h0_g = 0.0
                    node.children[x[i-node.depth-1]].hn_g = 0.0
                    node.children[x[i-node.depth-1]].leaf = True
            tmp1 = self._update_posterior_recursion(node.children[x[i-node.depth-1]],x,i)
            tmp2 = (1 - node.hn_g) * self._update_posterior_leaf(node,x,i) + node.hn_g * tmp1
            node.hn_g = node.hn_g * tmp1 / tmp2
            return tmp2
        else:  # 葉ノード
            return self._update_posterior_leaf(node,x,i)
    
    def update_posterior(self,x):
        """Update the hyperparameters using traning data.

        Parameters
        ----------
        x : numpy ndarray
            1-dimensional int array
        """
        _check.nonneg_ints(x,'x',DataFormatError)
        if x.max() >= self.c_k:
            raise(DataFormatError(f"x.max() must smaller than c_k:{self.c_k}"))
        x = np.ravel(x)

        if self.hn_root is None:
            self.hn_root = _LearnNode(
                0,
                self.c_k,
                self.hn_g,
                self.hn_g,
                self.h0_beta_vec,
                self.hn_beta_vec,
            )

        for i in range(x.shape[0]):
            self._update_posterior_recursion(self.hn_root,x,i)

    def _map_recursion_add_nodes(self,node:_LearnNode):
        if node.depth == self.c_d_max or node.depth == self.c_k:  # 葉ノード
            node.h0_g = 0.0
            node.hn_g = 0.0
            node.leaf = True
            node.map_leaf = True
        else:  # 内部ノード
            for i in range(self.c_k):
                node.children[i] = _LearnNode(depth=node.depth+1,
                                              c_k=self.c_k,
                                              h0_g=self.h0_g,
                                              hn_g=self.hn_g,
                                              h0_beta_vec=self.h0_beta_vec,
                                              hn_beta_vec=self.hn_beta_vec,
                                              )
                self._map_recursion_add_nodes(node.children[i])

    def _map_recursion(self,node:_LearnNode):
        if node.depth == self.c_d_max:
            node.map_leaf = True
            return 1.0
        else:
            tmp1 = 1.0-node.hn_g
            tmp_vec = np.empty(self.c_k)
            for i in range(self.c_k):
                if node.children[i] is not None:
                    tmp_vec[i] = self._map_recursion(node.children[i])
                else:
                    node.children[i] = _LearnNode(
                        node.depth+1,
                        self.c_k,
                        self.h0_g,
                        self.hn_g,
                        self.h0_beta_vec,
                        self.hn_beta_vec,
                    )
                    if 1.0 - node.h0_g > self.h0_g ** ((self.c_k ** (self.c_d_max - node.depth - 1) - 1)/(self.c_k-1)):
                        node.children[i].map_leaf = True
                        tmp_vec[i] = 1.0 - node.hn_g
                    else:
                        self._map_recursion_add_nodes(node.children[i])
                        tmp_vec[i] = self.h0_g ** ((self.c_k ** (self.c_d_max - node.depth) - 1)/(self.c_k-1))
            if tmp1 > node.hn_g*tmp_vec.prod():
                node.map_leaf = True
                return tmp1
            else:
                node.map_leaf = False
                return node.hn_g*tmp_vec.prod()

    def _copy_map_tree_recursion(self,copyed_node:_LearnNode,original_node:_LearnNode):
        copyed_node.h0_g = original_node.h0_g
        copyed_node.hn_g = original_node.hn_g
        copyed_node.h0_beta_vec[:] = original_node.h0_beta_vec
        copyed_node.hn_beta_vec[:] = original_node.hn_beta_vec
        if original_node.map_leaf == False:
            for i in range(self.c_k):
                copyed_node.children[i] = _LearnNode(
                    copyed_node.depth+1,
                    self.c_k,
                    self.h0_g,
                    self.hn_g,
                    self.h0_beta_vec,
                    self.hn_beta_vec,
                    )
                self._copy_map_tree_recursion(copyed_node.children[i],original_node.children[i])
        else:
            copyed_node.leaf = True

    def estimate_params(self,loss="0-1",visualize=True,filename=None,format=None):
        """Estimate the parameter under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default ``\"0-1\"``.
            This function supports only ``\"0-1\"``.
        visualize : bool, optional
            If ``True``, the estimated metatree will be visualized, by default ``True``.
            This visualization requires ``graphviz``.
        filename : str, optional
            Filename for saving the figure, by default ``None``
        format : str, optional
            Rendering output format (``\"pdf\"``, ``\"png\"``, ...).


        Returns
        -------
        map_root : metatree._LearnNode
            The root node of the estimated meta-tree 
            that also contains the estimated parameters in each node.

        See Also
        --------
        graphbiz.Digraph
        """

        if loss == "0-1":
            if self.hn_root is None:
                self.hn_root = _LearnNode(
                    0,
                    self.c_k,
                    self.hn_g,
                    self.hn_g,
                    self.h0_beta_vec,
                    self.hn_beta_vec,
                )
            self._map_recursion(self.hn_root)
            map_root = _LearnNode(
                0,
                self.c_k,
                self.h0_g,
                self.hn_g,
                self.h0_beta_vec,
                self.hn_beta_vec,
                )
            self._copy_map_tree_recursion(map_root,self.hn_root)
            if visualize:
                import graphviz
                tree_graph = graphviz.Digraph(filename=filename,format=format)
                tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
                self._visualize_model_recursion(tree_graph, map_root, 0, None, None, 1.0)
                tree_graph.view()
            return map_root
        else:
            raise(CriteriaError("Unsupported loss function! "
                                +"This function supports only \"0-1\"."))
    
    def _visualize_model_recursion(self,tree_graph,node:_LearnNode,node_id,parent_id,sibling_num,p_v):
        tmp_id = node_id
        tmp_p_v = p_v
        
        # add node information
        label_string = f'hn_g={node.hn_g:.2f}\\lp_v={tmp_p_v:.2f}\\ltheta_vec='
        label_string += '['
        for i in range(self.c_k):
            theta_vec_hat = node.hn_beta_vec / node.hn_beta_vec.sum()
            label_string += f'{theta_vec_hat[i]:.2f}'
            if i < self.c_k-1:
                label_string += ','
        label_string += ']'
            
        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_v))}')
        if tmp_p_v > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        for i in range(self.c_k):
            if node.children[i] is not None:
                node_id = self._visualize_model_recursion(tree_graph,node.children[i],node_id+1,tmp_id,i,tmp_p_v*node.hn_g)
        
        return node_id

    def visualize_posterior(self,filename=None,format=None):
        """Visualize the posterior distribution for the parameter.
        
        This method requires ``graphviz``.

        Parameters
        ----------
        filename : str, optional
            Filename for saving the figure, by default ``None``
        format : str, optional
            Rendering output format (``\"pdf\"``, ``\"png\"``, ...).

        Examples
        --------
        >>> from bayesml import contexttree
        >>> gen_model = contexttree.GenModel(c_k=2,c_d_max=3,h_g=0.75)
        >>> x = gen_model.gen_sample(50)
        >>> learn_model = contexttree.LearnModel(c_k=2,c_d_max=3,h0_g=0.75)
        >>> learn_model.update_posterior(x)
        >>> learn_model.visualize_posterior()

        .. image:: ./images/contexttree_posterior.png

        See Also
        --------
        graphbiz.Digraph
        """
        try:
            import graphviz
            tree_graph = graphviz.Digraph(filename=filename,format=format)
            tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
            self._visualize_model_recursion(tree_graph, self.hn_root, 0, None, None, 1.0)
            # コンソール上で表示できるようにした方がいいかもしれない．
            tree_graph.view()
        except ImportError as e:
            print(e)
        except graphviz.CalledProcessError as e:
            print(e)
    
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: numpy.ndarray}
            * ``"p_theta_vec"`` : the value of ``self.p_theta_vec``
        """
        return {"p_theta_vec":self.p_theta_vec}
    
    def _calc_pred_dist_leaf(self,node:_LearnNode):
            return node.hn_beta_vec / node.hn_beta_vec.sum()

    def _calc_pred_dist_recursion(self,node:_LearnNode,x,i):
        if node.depth < self.c_d_max and i-1-node.depth >= 0:  # 内部ノード
            if node.children[x[i-node.depth-1]] is None:
                node.children[x[i-node.depth-1]] = _LearnNode(
                    node.depth+1,
                    self.c_k,
                    self.h0_g,
                    self.hn_g,
                    self.h0_beta_vec,
                    self.hn_beta_vec,
                )
                if node.depth + 1 == self.c_d_max:
                    node.children[x[i-node.depth-1]].h0_g = 0.0
                    node.children[x[i-node.depth-1]].hn_g = 0.0
                    node.children[x[i-node.depth-1]].leaf = True
            tmp1 = self._calc_pred_dist_recursion(node.children[x[i-node.depth-1]],x,i)
            tmp2 = (1 - node.hn_g) * self._calc_pred_dist_leaf(node) + node.hn_g * tmp1
            return tmp2
        else:  # 葉ノード
            return self._calc_pred_dist_leaf(node)

    def calc_pred_dist(self,x):
        """Calculate the parameters of the predictive distribution.
        
        Parameters
        ----------
        x : numpy ndarray
            1-dimensional int array
        """
        _check.nonneg_int_vec(x,'x',DataFormatError)
        if x.max() >= self.c_k:
            raise(DataFormatError(f"x.max() must smaller than c_k:{self.c_k}"))
        i = x.shape[0] - 1

        if self.hn_root is None:
            self.hn_root = _LearnNode(
                0,
                self.c_k,
                self.hn_g,
                self.hn_g,
                self.h0_beta_vec,
                self.hn_beta_vec,
            )

        self.p_theta_vec[:] = self._calc_pred_dist_recursion(self.hn_root,x,i)

    def make_prediction(self,loss="KL"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"KL\".
            This function supports \"KL\" and \"0-1\".

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
            If the loss function is \"KL\", the predictive 
            distribution will be returned as a 1-dimensional 
            numpy.ndarray that consists of occurence probabilities.
        """
        if loss == "KL":
            return self.p_theta_vec
        elif loss == "0-1":
            return np.argmax(self.p_theta_vec)
        else:
            raise(CriteriaError("Unsupported loss function! "
                                +"This function supports \"0-1\" and \"KL\"."))

    def _pred_and_update_leaf(self,node:_LearnNode,x,i):
            tmp = node.hn_beta_vec / node.hn_beta_vec.sum()
            node.hn_beta_vec[x[i]] += 1
            return tmp

    def _pred_and_update_recursion(self,node:_LearnNode,x,i):
        if node.depth < self.c_d_max and i-1-node.depth >= 0:  # 内部ノード
            if node.children[x[i-node.depth-1]] is None:
                node.children[x[i-node.depth-1]] = _LearnNode(
                    node.depth+1,
                    self.c_k,
                    self.h0_g,
                    self.hn_g,
                    self.h0_beta_vec,
                    self.hn_beta_vec,
                )
                if node.depth + 1 == self.c_d_max:
                    node.children[x[i-node.depth-1]].h0_g = 0.0
                    node.children[x[i-node.depth-1]].hn_g = 0.0
                    node.children[x[i-node.depth-1]].leaf = True
            tmp1 = self._pred_and_update_recursion(node.children[x[i-node.depth-1]],x,i)
            tmp2 = (1 - node.hn_g) * self._pred_and_update_leaf(node,x,i) + node.hn_g * tmp1
            node.hn_g = node.hn_g * tmp1[x[i]] / tmp2[x[i]]
            return tmp2
        else:  # 葉ノード
            return self._pred_and_update_leaf(node,x,i)

    def pred_and_update(self,x,loss="KL"):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x : numpy.ndarray
            1-dimensional int array
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"KL\".
            This function supports \"KL\", and \"0-1\".

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
        """
        _check.nonneg_int_vec(x,'x',DataFormatError)
        if x.max() >= self.c_k:
            raise(DataFormatError(f"x.max() must smaller than c_k:{self.c_k}"))
        i = x.shape[0] - 1

        if self.hn_root is None:
            self.hn_root = _LearnNode(
                0,
                self.c_k,
                self.hn_g,
                self.hn_g,
                self.h0_beta_vec,
                self.hn_beta_vec,
            )

        self.p_theta_vec[:] = self._pred_and_update_recursion(self.hn_root,x,i)
        prediction = self.make_prediction(loss=loss)
        return prediction
