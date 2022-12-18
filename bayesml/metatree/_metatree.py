# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Document Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Wenbin Yu <ywb827748728@163.com>
import warnings
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from sklearn import tree as sklearn_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning, ParameterFormatWarning
from .. import _check
from .. import bernoulli, categorical, normal, multivariate_normal, linearregression, poisson, exponential

_CMAP = plt.get_cmap("Blues")
MODELS = {
    bernoulli,
    # categorical,
    normal,
    # multivariate_normal,
    # linearregression,
    poisson,
    exponential,
    }
DISCRETE_MODELS = {
    bernoulli,
    # categorical,
    poisson,
    }
CONTINUOUS_MODELS = {
    normal,
    # multivariate_normal,
    # linearregression,
    exponential,
    }

class _Node:
    def __init__(self,
                 depth,
                 k_candidates=None,
                 h_g=0.5,
                 k=None,
                 sub_model=None,
                 children=None,
                 ranges=None,
                 thresholds=None,
                 leaf=False,
                 map_leaf=False
                 ):
        self.depth = depth
        self.children = children
        self.k_candidates = k_candidates
        self.h_g = h_g
        self.k = k
        self.sub_model = sub_model
        self.ranges = ranges
        self.thresholds = thresholds
        self.leaf = leaf
        self.map_leaf = map_leaf

class GenModel(base.Generative):
    """ The stochastice data generative model and the prior distribution

    Parameters
    ----------
    c_dim_continuous : int
        A non-negative integer
    c_dim_categorical : int
        A non-negative integer
    c_num_children_vec : numpy.ndarray, optional
        A vector of positive integers whose length is 
        ``c_dim_continuous+c_dim_categorical``, by default [2,2,...,2].
        The first ``c_dim_continuous`` elements represent 
        the numbers of children of continuous features at 
        inner nodes. The rest ``c_dim_categorial`` elements 
        represent those of categorical features.
        If a single integer is input, it will be broadcasted.
    c_max_depth : int, optional
        A positive integer, by default 2
    c_num_assignment_vec : numpy.ndarray, optional
        A vector of positive integers whose length is 
        ``c_dim_continuous+c_dim_categorical``. 
        The first ``c_dim_continuous`` elements represent 
        the maximum assignment numbers of continuous features 
        on a path. The rest ``c_dim_categorial`` elements 
        represent those of categorical features.
        By default [c_max_depth,...,c_max_depth,1,...,1].
    c_ranges : numpy.ndarray, optional
        A numpy.ndarray whose size is (c_dim_continuous,2).
        A threshold for the ``k``-th continuous feature will be 
        generated between ``c_ranges[k,0]`` and ``c_ranges[k,1]``. 
        By default, [[-3,3],[-3,3],...,[-3,3]].
    SubModel : class, optional
        bernoulli, poisson, normal, or exponential, 
        by default bernoulli
    root : metatree._Node, optional
        A root node of a meta-tree, 
        by default a tree consists of only one node.
    h_k_weight_vec : numpy.ndarray, optional
        A vector of positive real numbers whose length is 
        ``c_dim_continuous+c_dim_categorical``, 
        by default [1/c_num_assignment_vec.sum(),...,1/c_num_assignment_vec.sum()].
    h_g : float, optional
        A real number in :math:`[0, 1]`, by default 0.5
    sub_h_params : dict, optional
        h_params for self.SubModel.GenModel, by default {}
    h_metatree_list : list of metatree._Node, optional
        Root nodes of meta-trees, by default []
    h_metatree_prob_vec : numpy.ndarray, optional
        A vector of real numbers in :math:`[0, 1]` 
        that represents prior distribution of h_metatree_list, 
        by default uniform distribution
        Sum of its elements must be 1.0.
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None

    Attributes
    ----------
    c_dim_features: int
        c_dim_continuous + c_dim_categorical
    """
    def __init__(
            self,
            c_dim_continuous,
            c_dim_categorical,
            c_max_depth=2,
            c_num_children_vec=2,
            c_num_assignment_vec=None,
            c_ranges=None,
            *,
            SubModel=bernoulli,
            root=None,
            h_k_weight_vec = None,
            h_g=0.5,
            sub_h_params={},
            h_metatree_list=[],
            h_metatree_prob_vec=None,
            seed=None,
            ):
        # constants
        self.c_dim_continuous = _check.nonneg_int(c_dim_continuous,'c_dim_continuous',ParameterFormatError)
        self.c_dim_categorical = _check.nonneg_int(c_dim_categorical,'c_dim_categorical',ParameterFormatError)
        _check.pos_int(
            self.c_dim_continuous+self.c_dim_categorical,
            'c_dim_continuous+c_dim_categorical',
            ParameterFormatError)
        self.c_dim_features = self.c_dim_continuous+self.c_dim_categorical
        
        self.c_max_depth = _check.pos_int(c_max_depth,'c_max_depth',ParameterFormatError)
        
        _check.pos_ints(c_num_children_vec,'c_num_children_vec',ParameterFormatError)
        if np.any(c_num_children_vec<2):
            raise(ParameterFormatError(
                'All the elements of c_num_children_vec must be greater than or equal to 2: '
                +f'c_num_children_vec={c_num_children_vec}.'
            ))
        self.c_num_children_vec = np.ones(self.c_dim_continuous+self.c_dim_categorical,dtype=int)*2
        self.c_num_children_vec[:] = c_num_children_vec
        
        self.c_num_assignment_vec = np.ones(self.c_dim_features,dtype=int)
        self.c_num_assignment_vec[:self.c_dim_continuous] *= self.c_max_depth
        if c_num_assignment_vec is not None:
            _check.pos_ints(c_num_assignment_vec,'c_num_assignment_vec',ParameterFormatError)
            if np.any(c_num_assignment_vec>self.c_max_depth):
                raise(ParameterFormatError(
                    'All the elements of c_num_assignment_vec must be less than or equal to self.c_max_depth: '
                    +f'c_num_assignment_vec={c_num_assignment_vec}.'
                ))
            self.c_num_assignment_vec[:] = c_num_assignment_vec
        
        self.c_ranges = np.zeros([self.c_dim_continuous,2])
        self.c_ranges[:,0] -= 3
        self.c_ranges[:,1] += 3
        if c_ranges is not None:
            _check.float_vecs(c_ranges,'c_ranges',ParameterFormatError)
            self.c_ranges[:] = c_ranges
            if np.any(self.c_ranges[:,0] > self.c_ranges[:,1]):
                raise(ParameterFormatError(
                    'self.c_ranges[:,1] must be greater than or equal to self.c_ranges[:,0]'
                ))
        
        if SubModel not in MODELS:
            raise(ParameterFormatError(
                "SubModel must be bernoulli, "
                +"poisson, normal, or exponential."
            ))
        self.SubModel = SubModel
        
        self.rng = np.random.default_rng(seed)

        # h_params
        self.h_k_weight_vec = np.ones(self.c_dim_features) / self.c_num_assignment_vec.sum()
        self.h_g = 0.5
        self.sub_h_params = {}
        self.h_metatree_list = []
        self.h_metatree_prob_vec = None

        self.set_h_params(
            h_k_weight_vec,
            h_g,
            sub_h_params,
            h_metatree_list,
            h_metatree_prob_vec,
        )

        # params
        self._root_k_candidates = []
        for i in range(self.c_dim_features):
            for j in range(self.c_num_assignment_vec[i]):
                self._root_k_candidates.append(i)
        self.root = _Node(
            0,
            self._root_k_candidates,
            self.h_g,
            sub_model=self.SubModel.GenModel(seed=self.rng,**self.sub_h_params),
            ranges=self.c_ranges,
            leaf=True
            )

        self.set_params(root)

    def get_constants(self):
        """Get constants of GenModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_dim_continuous"`` : the value of ``self.c_dim_continuous``
            * ``"c_dim_categorical"`` : the value of ``self.c_dim_categorical``
            * ``"c_num_children_vec"`` : the value of ``self.c_num_children_vec``
            * ``"c_max_depth"`` : the value of ``self.c_max_depth``
            * ``"c_num_assignment_vec"`` : the value of ``self.c_num_assignment_vec``
            * ``"c_ranges"`` : the value of ``self.c_ranges``
        """
        return {"c_dim_continuous":self.c_dim_continuous,
                "c_dim_categorical":self.c_dim_categorical,
                "c_num_children_vec":self.c_num_children_vec,
                "c_max_depth":self.c_max_depth,
                "c_num_assignment_vec":self.c_num_assignment_vec,
                "c_ranges":self.c_ranges}

    def _gen_params_recursion(self,node:_Node,h_node:_Node,feature_fix,threshold_fix,threshold_type):
        if h_node is None:
            if node.depth == self.c_max_depth:
                node.h_g = 0
            else:
                node.h_g = self.h_g
            # node.sub_model.set_h_params(**self.sub_h_params)
            node.sub_model = self.SubModel.GenModel(seed=self.rng,**self.sub_h_params)
            if node.depth == self.c_max_depth or not node.k_candidates or self.rng.random() > self.h_g:  # leaf node
                node.sub_model.gen_params()
                node.leaf = True
            else:  # inner node
                flag = node.k is None
                if not feature_fix or flag:
                    node.k = self.rng.choice(node.k_candidates,
                                             p=self.h_k_weight_vec[node.k_candidates]/self.h_k_weight_vec[node.k_candidates].sum())
                    node.children = [None for i in range(self.c_num_children_vec[node.k])]
                if node.k < self.c_dim_continuous:
                    if not threshold_fix or flag:
                        node.thresholds = np.empty(self.c_num_children_vec[node.k]+1)
                        if threshold_type == 'random':
                            tmp = self.rng.dirichlet(np.ones(self.c_num_children_vec[node.k]))
                            tmp *= (node.ranges[node.k,1] - node.ranges[node.k,0])
                            node.thresholds[0] = node.ranges[node.k,0]
                            for i in range(self.c_num_children_vec[node.k]):
                                node.thresholds[i+1] = node.thresholds[i]+tmp[i]
                        if threshold_type == 'even':
                            node.thresholds[:] = np.linspace(node.ranges[node.k,0],node.ranges[node.k,1],self.c_num_children_vec[node.k]+1)
                else:
                    node.thresholds = None
                child_k_candidates = node.k_candidates.copy()
                child_k_candidates.remove(node.k)
                node.leaf = False
                for i in range(self.c_num_children_vec[node.k]):
                    if node.children[i] is None:
                        node.children[i] = _Node(
                            node.depth+1,
                            sub_model=self.SubModel.GenModel(seed=self.rng,**self.sub_h_params),
                            )
                    node.children[i].k_candidates = child_k_candidates
                    node.children[i].ranges = np.array(node.ranges)
                    if node.thresholds is not None:
                        node.children[i].ranges[node.k,0] = node.thresholds[i]
                        node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                    self._gen_params_recursion(node.children[i],None,feature_fix,threshold_fix,threshold_type)
        else:
            if node.depth == self.c_max_depth:
                node.h_g = 0
            else:
                node.h_g = h_node.h_g
            try:
                sub_h_params = h_node.sub_model.get_h_params()
            except:
                sub_h_params = h_node.sub_model.get_hn_params()
            node.sub_model.set_h_params(*sub_h_params.values())
            if node.depth == self.c_max_depth or not node.k_candidates or self.rng.random() > h_node.h_g:  # leaf node
                node.sub_model.gen_params()
                node.leaf = True
            else:  # inner node
                node.k = h_node.k
                node.children = [None for i in range(self.c_num_children_vec[node.k])]
                if node.k < self.c_dim_continuous:
                    node.thresholds = np.array(h_node.thresholds)
                else:
                    node.thresholds = None
                child_k_candidates = node.k_candidates.copy()
                child_k_candidates.remove(node.k)
                node.leaf = False
                for i in range(self.c_num_children_vec[node.k]):
                    if node.children[i] is None:
                        node.children[i] = _Node(
                            node.depth+1,
                            sub_model=self.SubModel.GenModel(seed=self.rng,**self.sub_h_params),
                            )
                    node.children[i].k_candidates = child_k_candidates
                    node.children[i].ranges = np.array(node.ranges)
                    if node.thresholds is not None:
                        node.children[i].ranges[node.k,0] = node.thresholds[i]
                        node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                    self._gen_params_recursion(node.children[i],h_node.children[i],feature_fix,threshold_fix,threshold_type)

    def _gen_params_recursion_feature_and_tree_fix(self,node:_Node,threshold_fix,threshold_type):
        if node.depth == self.c_max_depth:
            node.h_g = 0
        else:
            node.h_g = self.h_g
        # node.sub_model.set_h_params(**self.sub_h_params)
        node.sub_model = self.SubModel.GenModel(seed=self.rng,**self.sub_h_params)
        if node.leaf:  # leaf node
            node.sub_model.gen_params()
            node.leaf = True
        else:  # inner node
            if node.k < self.c_dim_continuous:
                if not threshold_fix:
                    node.thresholds = np.empty(self.c_num_children_vec[node.k]+1)
                    if threshold_type == 'random':
                        tmp = self.rng.dirichlet(np.ones(self.c_num_children_vec[node.k]))
                        tmp *= (node.ranges[node.k,1] - node.ranges[node.k,0])
                        node.thresholds[0] = node.ranges[node.k,0]
                        for i in range(self.c_num_children_vec[node.k]):
                            node.thresholds[i+1] = node.thresholds[i]+tmp[i]
                    if threshold_type == 'even':
                        node.thresholds[:] = np.linspace(node.ranges[node.k,0],node.ranges[node.k,1],self.c_num_children_vec[node.k]+1)
            else:
                node.thresholds = None
            child_k_candidates = node.k_candidates.copy()
            child_k_candidates.remove(node.k)
            node.leaf = False
            for i in range(self.c_num_children_vec[node.k]):
                if node.children[i] is not None:
                    node.children[i].k_candidates = child_k_candidates
                    node.children[i].ranges = np.array(node.ranges)
                    if node.thresholds is not None:
                        node.children[i].ranges[node.k,0] = node.thresholds[i]
                        node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                    self._gen_params_recursion_feature_and_tree_fix(node.children[i],threshold_fix,threshold_type)

    def _set_params_recursion(self,node:_Node,original_tree_node:_Node):
        if original_tree_node.leaf:  # leaf node
            try:
                sub_params = original_tree_node.sub_model.get_params()
                node.sub_model.set_params(**sub_params)
            except:
                try:
                    sub_params = original_tree_node.sub_model.estimate_params(loss='0-1',dict_out=True)
                    node.sub_model.set_params(**sub_params)
                except:
                    sub_params = original_tree_node.sub_model.estimate_params(dict_out=True)
                    node.sub_model.set_params(**sub_params)

            if node.depth == self.c_max_depth:
                node.h_g = 0
            node.leaf = True
        else:
            node.k = original_tree_node.k
            node.children = [None for i in range(self.c_num_children_vec[node.k])]
            if node.k < self.c_dim_continuous:
                node.thresholds = np.array(original_tree_node.thresholds)
            else:
                node.thresholds = None
            child_k_candidates = node.k_candidates.copy()
            child_k_candidates.remove(node.k)
            node.leaf = False
            for i in range(self.c_num_children_vec[node.k]):
                if node.children[i] is None:
                    node.children[i] = _Node(
                        node.depth+1,
                        h_g=self.h_g,
                        sub_model=self.SubModel.GenModel(seed=self.rng,**self.sub_h_params)
                        )
                node.children[i].k_candidates = child_k_candidates
                node.children[i].ranges = np.array(node.ranges)
                if node.thresholds is not None:
                    node.children[i].ranges[node.k,0] = node.thresholds[i]
                    node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                self._set_params_recursion(node.children[i],original_tree_node.children[i])
    
    def _gen_sample_recursion(self,node:_Node,x_continuous,x_categorical):
        if node.leaf:  # leaf node
            try:
                y = node.sub_model.gen_sample(sample_size=1,x=x_continuous)
            except:
                y = node.sub_model.gen_sample(sample_size=1)
            return y
        else:
            if node.k < self.c_dim_continuous:
                for i in range(self.c_num_children_vec[node.k]):
                    if node.thresholds[i] < x_continuous[node.k] and x_continuous[node.k] < node.thresholds[i+1]:
                        index = i
                        break
            else:
                index = x_categorical[node.k-self.c_dim_continuous]
            return self._gen_sample_recursion(node.children[index],x_continuous,x_categorical)
    
    def _visualize_model_recursion(self,tree_graph,node:_Node,node_id,parent_id,parent_k,sibling_num,p_v):
        tmp_id = node_id
        tmp_p_v = p_v

        # add node information
        if node.leaf:
            label_string = 'k=None\\l'
        else:
            label_string = f'k={node.k}\\l'
            if node.k < self.c_dim_continuous:
                label_string += 'thresholds=\\l{'
                for i in range(self.c_num_children_vec[node.k]-1):
                    if i == 0:
                        label_string += f'{node.thresholds[i+1]:.2f}'
                    else:
                        label_string += f',{node.thresholds[i+1]:.2f}'
                label_string += '}\\l'
        label_string += f'h_g={node.h_g:.2f}\\lp_v={tmp_p_v:.2f}\\lsub_params={{'
        if node.leaf:
            sub_params = node.sub_model.get_params()
            for key,value in sub_params.items():
                try:
                    label_string += f'\\l{key}:{value:.2f}'
                except:
                    label_string += f'\\l{key}:{value}'
            label_string += '}'
        else:
            label_string += '\\lNone}'
            
        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_v))}')
        if tmp_p_v > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            if parent_k < self.c_dim_continuous:
                if node.ranges[parent_k,0] <= self.c_ranges[parent_k,0] + 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[*,{node.ranges[parent_k,1]:.2f})')
                elif node.ranges[parent_k,1] >= self.c_ranges[parent_k,1] - 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{node.ranges[parent_k,0]:.2f},*)')
                else:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{node.ranges[parent_k,0]:.2f},{node.ranges[parent_k,1]:.2f})')
            else:
                tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        if node.leaf != True:
            for i in range(self.c_num_children_vec[node.k]):
                node_id = self._visualize_model_recursion(tree_graph,node.children[i],node_id+1,tmp_id,node.k,i,tmp_p_v*node.h_g)
        
        return node_id

    def _set_h_g_recursion(self,node:_Node):
        if node.depth == self.c_max_depth:
            node.h_g = 0
        else:
            node.h_g = self.h_g
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_h_g_recursion(node.children[i])

    def _set_sub_h_params_recursion(self,node:_Node):
        # node.sub_model.set_h_params(**self.sub_h_params)
        node.sub_model = self.SubModel.GenModel(seed=self.rng,**self.sub_h_params)
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_sub_h_params_recursion(node.children[i])

    def _set_h_params_recursion(self,node:_Node,original_tree_node:_Node):
        if original_tree_node is None:
            if node.depth == self.c_max_depth:
                node.h_g = 0
            else:
                node.h_g = self.h_g
            # node.sub_model.set_h_params(**self.sub_h_params)
            node.sub_model = self.SubModel.GenModel(seed=self.rng,**self.sub_h_params)
            if not node.leaf:
                for i in range(self.c_num_children_vec[node.k]):
                    self._set_h_params_recursion(node.children[i],None)
        else:
            if node.depth == self.c_max_depth:
                node.h_g = 0
            else:
                node.h_g = original_tree_node.h_g
            try:
                sub_h_params = node.sub_model.get_h_params()
            except:
                sub_h_params = node.sub_model.get_hn_params()
            node.sub_model.set_h_params(*sub_h_params.values())
            if original_tree_node.leaf or node.depth == self.c_max_depth:  # leaf node
                node.leaf = True
            else:
                node.k = original_tree_node.k
                node.children = [None for i in range(self.c_num_children_vec[node.k])]
                if node.k < self.c_dim_continuous:
                    node.thresholds = np.array(original_tree_node.thresholds)
                else:
                    node.thresholds = None
                child_k_candidates = node.k_candidates.copy()
                child_k_candidates.remove(node.k)
                node.leaf = False
                for i in range(self.c_num_children_vec[node.k]):
                    if node.children[i] is None:
                        node.children[i] = _Node(
                            node.depth+1,
                            sub_model=self.SubModel.GenModel(seed=self.rng,**self.sub_h_params),
                            )
                    node.children[i].k_candidates = child_k_candidates
                    node.children[i].ranges = np.array(node.ranges)
                    if node.thresholds is not None:
                        node.children[i].ranges[node.k,0] = node.thresholds[i]
                        node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                    self._set_h_params_recursion(node.children[i],original_tree_node.children[i])

    def set_h_params(self,
            h_k_weight_vec = None,
            h_g=None,
            sub_h_params=None,
            h_metatree_list=None,
            h_metatree_prob_vec=None
            ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_k_weight_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]`, 
            by default None
            Sum of its elements must be 1.
        h_g : float, optional
            A real number in :math:`[0, 1]`, by default None
        sub_h_params : dict, optional
            h_params for self.SubModel.GenModel, by default None
        h_metatree_list : list of metatree._Node, optional
            Root nodes of meta-trees, by default None
        h_metatree_prob_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]` 
            that represents prior distribution of h_metatree_list, 
            by default None.
            Sum of its elements must be 1.0.
        """
        if h_k_weight_vec is not None:
            _check.nonneg_float_vec(h_k_weight_vec,'h_k_weight_vec',ParameterFormatError)
            _check.shape_consistency(
                h_k_weight_vec.shape[0],'h_k_weight_vec.shape[0]',
                self.c_dim_features,'self.c_dim_features',
                ParameterFormatError
                )
            self.h_k_weight_vec[:] = h_k_weight_vec

        if h_g is not None:
            self.h_g = _check.float_in_closed01(h_g,'h_g',ParameterFormatError)
            if self.h_metatree_list:
                for h_root in self.h_metatree_list:
                    self._set_h_g_recursion(h_root)

        if sub_h_params is not None:
            self.SubModel.GenModel(seed=self.rng,**sub_h_params)
            self.sub_h_params = copy.deepcopy(sub_h_params)
            if self.h_metatree_list:
                for h_root in self.h_metatree_list:
                    self._set_sub_h_params_recursion(h_root)

        if h_metatree_list is not None:
            if not isinstance(h_metatree_list,list):
                raise(ParameterFormatError(
                    "h_metatree_list must be a list"
                ))
            if h_metatree_list:
                for h_root in h_metatree_list:
                    if type(h_root) is not _Node:
                        raise(ParameterFormatError(
                            "all elements of h_metatree_list must be instances of metatree._Node or empty"
                        ))
            diff = len(h_metatree_list) - len(self.h_metatree_list)
            if diff < 0:
                del self.h_metatree_list[diff:]
            elif diff > 0:
                for i in range(diff):
                    self.h_metatree_list.append(
                        _Node(
                            0,
                            self._root_k_candidates,
                            self.h_g,
                            sub_model=self.SubModel.GenModel(seed=self.rng,**self.sub_h_params),
                            ranges=self.c_ranges,
                            )
                    )
            for i in range(len(self.h_metatree_list)):
                self._set_h_params_recursion(self.h_metatree_list[i],h_metatree_list[i])
            if h_metatree_prob_vec is not None:
                self.h_metatree_prob_vec = np.array(
                    _check.float_vec_sum_1(
                        h_metatree_prob_vec,
                        'h_metatree_prob_vec',
                        ParameterFormatError
                    )
                )
            else:
                if h_metatree_list:
                    metatree_num = len(self.h_metatree_list)
                    self.h_metatree_prob_vec = np.ones(metatree_num) / metatree_num
                else:
                    self.h_metatree_prob_vec = None
        elif h_metatree_prob_vec is not None:
            self.h_metatree_prob_vec = np.array(
                _check.float_vec_sum_1(
                    h_metatree_prob_vec,
                    'h_metatree_prob_vec',
                    ParameterFormatError
                )
            )

        if type(self.h_metatree_prob_vec) is np.ndarray:             
            if self.h_metatree_prob_vec.shape[0] != len(self.h_metatree_list):
                raise(ParameterFormatError(
                    "Length of h_metatree_list and dimension of h_metatree_prob_vec must be the same."
                ))
        elif self.h_metatree_prob_vec is None:
            if len(self.h_metatree_list) > 0:
                raise(ParameterFormatError(
                    "Length of h_metatree_list must be zero when self.h_metatree_prob_vec is None."
                ))
        else:
            raise(ParameterFormatError(
                "self.h_metatree_prob_vec must be None or a numpy.ndarray."
            ))

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float, list, dict, numpy.ndarray}
            * ``"h_k_weight_vec"`` : the value of ``self.h_k_weight_vec``
            * ``"h_g"`` : the value of ``self.h_g``
            * ``"sub_h_params"`` : the value of ``self.sub_h_params``
            * ``"h_metatree_list"`` : the value of ``self.h_metatree_list``
            * ``"h_metatree_prob_vec"`` : the value of ``self.h_metatree_prob_vec``
        """
        return {"h_k_weight_vec":self.h_k_weight_vec,
                "h_g":self.h_g, 
                "sub_h_params":self.sub_h_params, 
                "h_metatree_list":self.h_metatree_list,
                "h_metatree_prob_vec":self.h_metatree_prob_vec}
    
    def gen_params(self,feature_fix=False,threshold_fix=False,tree_fix=False,threshold_type='even'):
        """Generate the parameter from the prior distribution.

        The generated vaule is set at ``self.root``.

        Parameters
        ----------
        feature_fix : bool, optional
            If ``True``, feature assignment indices will be fixed, by default ``False``.
        threshold_fix : bool, optional
            If ``True``, thresholds for continuous features will be fixed, by default ``False``. 
            If ``feature_fix`` is ``False``, ``threshold_fix`` must be ``False``. 
        tree_fix : bool, optional
            If ``True``, tree shape will be fixed, by default ``False``. 
            If ``feature_fix`` is ``False``, ``tree_fix`` must be ``False``.
        threshold_type : {'even', 'random'}, optional
            A type of threshold generating procedure, by default ``'even'``
            If ``'even'``, self.c_ranges will be recursively divided by equal intervals. 
            if ``'random'``, self.c_ranges will be recursively divided by at random intervals.
        """
        if feature_fix:
            warnings.warn(
                "If feature_fix=True, tree will be generated according to "
                +"self.h_g not any element of self.h_metatree_list.",ResultWarning)
            if tree_fix:
                self._gen_params_recursion_feature_and_tree_fix(self.root,threshold_fix,threshold_type)
            else:
                self._gen_params_recursion(self.root,None,True,threshold_fix,threshold_type)
        else:
            if threshold_fix or tree_fix:
                warnings.warn("If feature_fix=False, threshold and tree cannot be fixed.",ResultWarning)
            if self.h_metatree_list:
                tmp_root = self.rng.choice(self.h_metatree_list,p=self.h_metatree_prob_vec)
                self._gen_params_recursion(self.root,tmp_root,False,False,threshold_type)
            else:
                self._gen_params_recursion(self.root,None,False,False,threshold_type)
    
    def set_params(self,root=None):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        root : metatree._Node, optional
            A root node of a meta-tree, by default None.
        """
        if root is not None:
            if type(root) is not _Node:
                raise(ParameterFormatError(
                    "root must be an instance of metatree._Node"
                ))
            self._set_params_recursion(self.root,root)

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:metatree._Node}
            * ``"root"`` : The value of ``self.root``.
        """
        return {"root":self.root}

    def gen_sample(self,sample_size=None,x_continuous=None,x_categorical=None):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default ``None``
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x[i,j] must satisfy 
            0 <= x[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].

        Returns
        -------
        x_continuous : numpy ndarray
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``.
            Each element x[i,j] must satisfies 
            0 <= x[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y : numpy ndarray
            1 dimensional array whose size is ``sample_size``.
        """
        if sample_size is not None:
            sample_size = _check.pos_int(sample_size,'sample_size',DataFormatError)

            if x_continuous is not None:
                _check.float_vecs(x_continuous,'x_continuous',DataFormatError)
                _check.shape_consistency(
                    x_continuous.shape[-1],'x_continuous.shape[-1]',
                    self.c_dim_continuous,'self.c_dim_continuous',
                    ParameterFormatError
                    )
                _check.shape_consistency(
                    x_continuous.shape[0],'x_continuous.shape[0]',
                    sample_size,'sample_size',
                    ParameterFormatError
                    )
            else:
                x_continuous = np.empty([sample_size,self.c_dim_continuous],dtype=float)
                for i in range(self.c_dim_continuous):
                    x_continuous[:,i] = ((self.c_ranges[i,1]-self.c_ranges[i,0])
                                         * self.rng.random(sample_size)
                                         + self.c_ranges[i,0])
            if x_categorical is not None:
                _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
                _check.shape_consistency(
                    x_categorical.shape[-1],'x_categorical.shape[-1]',
                    self.c_dim_categorical,'self.c_dim_categorical',
                    ParameterFormatError
                    )
                _check.shape_consistency(
                    x_categorical.shape[0],'x_categorical.shape[0]',
                    sample_size,'sample_size',
                    ParameterFormatError
                    )
                for i in range(self.c_dim_categorical):
                    if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                        raise(DataFormatError(
                            f"x_categorical[{i}].max() must smaller than "
                            +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                            +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))
            else:
                x_categorical = np.empty([sample_size,self.c_dim_categorical],dtype=int)
                for i in range(self.c_dim_categorical):
                    x_categorical[:,i] = self.rng.choice(
                        self.c_num_children_vec[self.c_dim_continuous+i],
                        sample_size)
                        
        elif x_continuous is not None:
            _check.float_vecs(x_continuous,'x_continuous',DataFormatError)
            _check.shape_consistency(
                x_continuous.shape[-1],'x_continuous.shape[-1]',
                self.c_dim_continuous,'self.c_dim_continuous',
                ParameterFormatError
                )
            x_continuous = x_continuous.reshape(-1,self.c_dim_continuous)

            sample_size = x_continuous.shape[0]

            if x_categorical is not None:
                _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
                _check.shape_consistency(
                    x_categorical.shape[-1],'x_categorical.shape[-1]',
                    self.c_dim_categorical,'self.c_dim_categorical',
                    ParameterFormatError
                    )
                _check.shape_consistency(
                    x_categorical.shape[0],'x_categorical.shape[0]',
                    x_continuous.shape[0],'x_continuous.shape[0]',
                    ParameterFormatError
                    )
                for i in range(self.c_dim_categorical):
                    if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                        raise(DataFormatError(
                            f"x_categorical[{i}].max() must smaller than "
                            +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                            +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))
            else:
                x_categorical = np.empty([sample_size,self.c_dim_categorical],dtype=int)
                for i in range(self.c_dim_categorical):
                    x_categorical[:,i] = self.rng.choice(
                        self.c_num_children_vec[self.c_dim_continuous+i],
                        sample_size)

        elif x_categorical is not None:
            _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
            _check.shape_consistency(
                x_categorical.shape[-1],'x_categorical.shape[-1]',
                self.c_dim_categorical,'self.c_dim_categorical',
                ParameterFormatError
                )
            x_categorical = x_categorical.reshape(-1,self.c_dim_categorical)
            for i in range(self.c_dim_categorical):
                if x_categorical[i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                    raise(DataFormatError(
                        f"x_categorical[{i}].max() must smaller than "
                        +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                        +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))

            sample_size = x_categorical.shape[0]
            
            x_continuous = np.empty([sample_size,self.c_dim_continuous],dtype=float)
            for i in range(self.c_dim_continuous):
                x_continuous[:,i] = ((self.c_ranges[i,1]-self.c_ranges[i,0])
                                        * self.rng.random(sample_size)
                                        + self.c_ranges[i,0])
        else:
            raise(DataFormatError("Either of sample_size, x_continuous, and x_categorical must be given as a input."))

        if self.SubModel in DISCRETE_MODELS:
            y = np.empty(sample_size,dtype=int)
        elif self.SubModel in CONTINUOUS_MODELS:
            y = np.empty(sample_size,dtype=float)
        
        for i in range(sample_size):
            y[i] = self._gen_sample_recursion(self.root,x_continuous[i],x_categorical[i])

        return x_continuous,x_categorical,y
        
    def save_sample(self,filename,sample_size,x=None):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as a NpzFile with keyword: \"x\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_size : int, optional
            A positive integer, by default ``None``
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x[i,j] must satisfy 
            0 <= x[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        
        See Also
        --------
        numpy.savez_compressed
        """
        x_continuous,x_categorical,y = self.gen_sample(sample_size,x_continuous,x_categorical)
        np.savez_compressed(filename,x_continuous=x_continuous,x_categorical=x_categorical,y=y)

    def _plot_2d_threshold_recursion_continuous(self,ax,node:_Node):
        if not node.leaf:
            if node.k == 0:
                ax.vlines(x=node.thresholds[1:-1],ymin=node.ranges[1,0],ymax=node.ranges[1,1],colors='red')
            else:
                ax.hlines(y=node.thresholds[1:-1],xmin=node.ranges[0,0],xmax=node.ranges[0,1],colors='red')
            for i in range(self.c_num_children_vec[node.k]):
                if node.children[i] is not None:
                    self._plot_2d_threshold_recursion_continuous(ax,node.children[i])

    def _plot_1d_threshold_recursion_continuous(self,ax,node:_Node,ymin,ymax):
        if not node.leaf:
            ax.vlines(x=node.thresholds[1:-1],ymin=ymin,ymax=ymax,colors='red')
            for i in range(self.c_num_children_vec[node.k]):
                if node.children[i] is not None:
                    self._plot_1d_threshold_recursion_continuous(ax,node.children[i],ymin,ymax)

    def _plot_1d_threshold_recursion_categorical(self,ax,node:_Node,ymin,ymax):
        if not node.leaf:
            ax.vlines(
                x=np.linspace(0,
                    self.c_num_children_vec[node.k]-1,
                    2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                ymin=ymin,
                ymax=ymax,
                colors='red')
            for i in range(self.c_num_children_vec[node.k]):
                if node.children[i] is not None:
                    self._plot_1d_threshold_recursion_categorical(ax,node.children[i],ymin,ymax)

    def _plot_2d_threshold_recursion_mix(self,ax,node:_Node,categorical_index):
        if not node.leaf:
            if node.k == 0:
                if categorical_index is None:
                    ax.vlines(x=node.thresholds[1:-1],ymin=0-0.2,ymax=self.c_num_children_vec[1]-1+0.2,colors='red')
                else:
                    ax.vlines(x=node.thresholds[1:-1],ymin=categorical_index-0.2,ymax=categorical_index+0.2,colors='red')
                for i in range(self.c_num_children_vec[node.k]):
                    if node.children[i] is not None:
                        self._plot_2d_threshold_recursion_mix(ax,node.children[i],categorical_index)
            else:
                ax.hlines(
                    y=np.linspace(0,
                        self.c_num_children_vec[node.k]-1,
                        2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                    xmin=node.ranges[0,0],
                    xmax=node.ranges[0,1],
                    colors='red')
                for i in range(self.c_num_children_vec[node.k]):
                    if node.children[i] is not None:
                        self._plot_2d_threshold_recursion_mix(ax,node.children[i],i)

    def _plot_2d_threshold_recursion_categorical(self,ax,node:_Node,categorical_index):
        if not node.leaf:
            if node.k == 0:
                if categorical_index is None:
                    ax.vlines(
                        x=np.linspace(0,
                            self.c_num_children_vec[node.k]-1,
                            2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                        ymin=-0.2,
                        ymax=self.c_num_children_vec[1]-1+0.2,
                        colors='red')
                else:
                    ax.vlines(
                        x=np.linspace(0,
                            self.c_num_children_vec[node.k]-1,
                            2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                        ymin=max(categorical_index-0.5,-0.2),
                        ymax=min(categorical_index+0.5,self.c_num_children_vec[1]-1+0.2),
                        colors='red')
            else:
                if categorical_index is None:
                    ax.hlines(
                        y=np.linspace(0,
                            self.c_num_children_vec[node.k]-1,
                            2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                        xmin=-0.2,
                        xmax=self.c_num_children_vec[0]-1+0.2,
                        colors='red')
                else:
                    ax.hlines(
                        y=np.linspace(0,
                            self.c_num_children_vec[node.k]-1,
                            2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                        xmin=max(categorical_index-0.5,-0.2),
                        xmax=min(categorical_index+0.5,self.c_num_children_vec[0]-1+0.2),
                        colors='red')
            for i in range(self.c_num_children_vec[node.k]):
                if node.children[i] is not None:
                    self._plot_2d_threshold_recursion_categorical(ax,node.children[i],i)

    def visualize_model(self,filename=None,format=None,sample_size=100):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        filename : str, optional
            Filename for saving the figure, by default ``None``
        format : str, optional
            Rendering output format (``\"pdf\"``, ``\"png\"``, ...).
        sample_size : int, optional
            A positive integer, by default 100

        Examples
        --------
        >>> from bayesml import metatree
        >>> model = metatree.GenModel(
        >>>     c_dim_continuous=1,
        >>>     c_dim_categorical=1)
        >>> model.gen_params(threshold_type='random')
        >>> model.visualize_model()

        .. image:: ./images/metatree_example1.png

        .. image:: ./images/metatree_example2.png

        See Also
        --------
        graphviz.Digraph
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)

        try:
            import graphviz
            tree_graph = graphviz.Digraph(filename=filename,format=format)
            tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
            self._visualize_model_recursion(tree_graph, self.root, 0, None, None, None, 1.0)
            # Can we show the image on the console without saving the file?
            tree_graph.view()
        except ImportError as e:
            print(e)
        except graphviz.CalledProcessError as e:
            print(e)

        fig, ax = plt.subplots()
        x_continuous,x_categorical,y = self.gen_sample(sample_size)
        if self.c_dim_categorical > 0:
            x_categorical_jitter = x_categorical + 0.2*(self.rng.random(x_categorical.shape)-0.5)

        if self.SubModel in DISCRETE_MODELS:
            y_jitter = y + 0.2*(self.rng.random(y.shape)-0.5)

        if self.c_dim_features == 1:
            if self.c_dim_categorical == 1:
                if self.SubModel in DISCRETE_MODELS:
                    ax.scatter(x_categorical_jitter,y_jitter)
                else:
                    ax.scatter(x_categorical_jitter,y)
                ymin, ymax = ax.get_ylim()
                self._plot_1d_threshold_recursion_categorical(ax,self.root,ymin,ymax)
                ax.set_xlabel('x_categorical[0]')
                ax.set_ylabel('y')
            else:
                if self.SubModel in DISCRETE_MODELS:
                    ax.scatter(x_continuous,y_jitter)
                else:
                    ax.scatter(x_continuous,y)
                ymin, ymax = ax.get_ylim()
                self._plot_1d_threshold_recursion_continuous(ax,self.root,ymin,ymax)
                ax.set_xlabel('x_continuous[0]')
                ax.set_ylabel('y')
            plt.show()
        elif self.c_dim_features == 2:
            if self.c_dim_categorical == 2:
                mappable = ax.scatter(x_categorical_jitter[:,0],x_categorical_jitter[:,1],c=y)
                self._plot_2d_threshold_recursion_categorical(ax,self.root,None)
                ax.set_xlabel('x_categorical[0]')
                ax.set_ylabel('x_categorical[1]')
            elif self.c_dim_categorical == 1:
                mappable = ax.scatter(x_continuous,x_categorical_jitter,c=y)
                self._plot_2d_threshold_recursion_mix(ax,self.root,None)
                ax.set_xlabel('x_continuous[0]')
                ax.set_ylabel('x_categorical[0]')
            else:
                mappable = ax.scatter(x_continuous[:,0],x_continuous[:,1],c=y)
                self._plot_2d_threshold_recursion_continuous(ax,self.root)
                ax.set_xlabel('x_continuous[0]')
                ax.set_ylabel('x_continuous[1]')
            fig.colorbar(mappable,label='y')
            plt.show()
        else:
            print(x_continuous)
            print(x_categorical)
            print(y)

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    c_dim_continuous : int
        A non-negative integer
    c_dim_categorical : int
        A non-negative integer
    c_num_children_vec : numpy.ndarray
        A vector of positive integers whose length is 
        ``c_dim_continuous+c_dim_categorical``.
        The first ``c_dim_continuous`` elements represent 
        the numbers of children of continuous features at 
        inner nodes. The rest ``c_dim_categorial`` elements 
        represent those of categorical features.
        If a single integer is input, it will be broadcasted.
    c_max_depth : int, optional
        A positive integer, by default 2
    c_num_assignment_vec : numpy.ndarray, optional
        A vector of positive integers whose length is 
        ``c_dim_continuous+c_dim_categorical``. 
        The first ``c_dim_continuous`` elements represent 
        the maximum assignment numbers of continuous features 
        on a path. The rest ``c_dim_categorial`` elements 
        represent those of categorical features.
        By default [c_max_depth,...,c_max_depth,1,...,1].
    c_ranges : numpy.ndarray, optional
        A numpy.ndarray whose size is (c_dim_continuous,2).
        A threshold for the ``k``-th continuous feature will be 
        generated between ``c_ranges[k,0]`` and ``c_ranges[k,1]``. 
        By default, [[-3,3],[-3,3],...,[-3,3]].
    SubModel : class, optional
        bernoulli, poisson, normal, or exponential, 
        by default bernoulli
    h0_k_weight_vec : numpy.ndarray, optional
        A vector of positive real numbers whose length is 
        ``c_dim_continuous+c_dim_categorical``, 
        by default [1/c_num_assignment_vec.sum(),...,1/c_num_assignment_vec.sum()].
    h0_g : float, optional
        A real number in :math:`[0, 1]`, by default 0.5
    sub_h0_params : dict, optional
        h0_params for self.SubModel.LearnModel, by default {}
    h0_metatree_list : list of metatree._Node, optional
        Root nodes of meta-trees, by default []
    h0_metatree_prob_vec : numpy.ndarray, optional
        A vector of real numbers in :math:`[0, 1]` 
        that represents prior distribution of h0_metatree_list, 
        by default uniform distribution
        Sum of its elements must be 1.0.

    Attributes
    ----------
    c_dim_features: int
        c_dim_continuous + c_dim_categorical
    hn_k_weight_vec : numpy.ndarray
        A vector of positive real numbers whose length is 
        ``c_dim_continuous+c_dim_categorical``
    hn_g : float
        A real number in :math:`[0, 1]`
    sub_hn_params : dict
        hn_params for self.SubModel.LearnModel
    hn_metatree_list : list of metatree._Node
        Root nodes of meta-trees
    hn_metatree_prob_vec : numpy.ndarray
        A vector of real numbers in :math:`[0, 1]` 
        that represents prior distribution of h0_metatree_list.
        Sum of its elements is 1.0.
    """
    def __init__(
            self,
            c_dim_continuous,
            c_dim_categorical,
            c_num_children_vec,
            c_max_depth=2,
            c_num_assignment_vec=None,
            c_ranges=None,
            *,
            SubModel=bernoulli,
            h0_k_weight_vec = None,
            h0_g=0.5,
            sub_h0_params={},
            h0_metatree_list=[],
            h0_metatree_prob_vec=None
            ):
        # constants
        self.c_dim_continuous = _check.nonneg_int(c_dim_continuous,'c_dim_continuous',ParameterFormatError)
        self.c_dim_categorical = _check.nonneg_int(c_dim_categorical,'c_dim_categorical',ParameterFormatError)
        _check.pos_int(
            self.c_dim_continuous+self.c_dim_categorical,
            'c_dim_continuous+c_dim_categorical',
            ParameterFormatError)
        self.c_dim_features = self.c_dim_continuous+self.c_dim_categorical
        
        self.c_max_depth = _check.pos_int(c_max_depth,'c_max_depth',ParameterFormatError)
        
        _check.pos_ints(c_num_children_vec,'c_num_children_vec',ParameterFormatError)
        if np.any(c_num_children_vec<2):
            raise(ParameterFormatError(
                'All the elements of c_num_children_vec must be greater than or equal to 2: '
                +f'c_num_children_vec={c_num_children_vec}.'
            ))
        self.c_num_children_vec = np.ones(self.c_dim_continuous+self.c_dim_categorical,dtype=int)*2
        self.c_num_children_vec[:] = c_num_children_vec
        
        self.c_num_assignment_vec = np.ones(self.c_dim_features,dtype=int)
        self.c_num_assignment_vec[:self.c_dim_continuous] *= self.c_max_depth
        if c_num_assignment_vec is not None:
            _check.pos_ints(c_num_assignment_vec,'c_num_assignment_vec',ParameterFormatError)
            if np.any(c_num_assignment_vec>self.c_max_depth):
                raise(ParameterFormatError(
                    'All the elements of c_num_assignment_vec must be less than or equal to self.c_max_depth: '
                    +f'c_num_assignment_vec={c_num_assignment_vec}.'
                ))
            self.c_num_assignment_vec[:] = c_num_assignment_vec
        
        self.c_ranges = np.zeros([self.c_dim_continuous,2])
        self.c_ranges[:,0] -= 3
        self.c_ranges[:,1] += 3
        if c_ranges is not None:
            _check.float_vecs(c_ranges,'c_ranges',ParameterFormatError)
            self.c_ranges[:] = c_ranges
            if np.any(self.c_ranges[:,0] > self.c_ranges[:,1]):
                raise(ParameterFormatError(
                    'self.c_ranges[:,1] must be greater than or equal to self.c_ranges[:,0]'
                ))
        
        if SubModel not in MODELS:
            raise(ParameterFormatError(
                "SubModel must be bernoulli, "
                +"poisson, normal, or exponential."
            ))
        self.SubModel = SubModel

        self._root_k_candidates = []
        for i in range(self.c_dim_features):
            for j in range(self.c_num_assignment_vec[i]):
                self._root_k_candidates.append(i)

        # h0_params
        self.h0_k_weight_vec = np.ones(self.c_dim_features) / self.c_num_assignment_vec.sum()
        self.h0_g = 0.5
        self.sub_h0_params = {}
        self.h0_metatree_list = []
        self.h0_metatree_prob_vec = None

        # hn_params
        self.hn_k_weight_vec = np.ones(self.c_dim_features) / self.c_num_assignment_vec.sum()
        self.hn_g = 0.5
        self.sub_hn_params = {}
        self.hn_metatree_list = []
        self.hn_metatree_prob_vec = None

        self._tmp_x_continuous = np.zeros(self.c_dim_continuous,dtype=float)
        self._tmp_x_categorical = np.zeros(self.c_dim_categorical,dtype=int)

        self.set_h0_params(
            h0_k_weight_vec,
            h0_g,
            sub_h0_params,
            h0_metatree_list,
            h0_metatree_prob_vec,
        )

    def get_constants(self):
        """Get constants of LearnModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_dim_continuous"`` : the value of ``self.c_dim_continuous``
            * ``"c_dim_categorical"`` : the value of ``self.c_dim_categorical``
            * ``"c_num_children_vec"`` : the value of ``self.c_num_children_vec``
            * ``"c_max_depth"`` : the value of ``self.c_max_depth``
            * ``"c_num_assignment_vec"`` : the value of ``self.c_num_assignment_vec``
            * ``"c_ranges"`` : the value of ``self.c_ranges``
        """
        return {"c_dim_continuous":self.c_dim_continuous,
                "c_dim_categorical":self.c_dim_categorical,
                "c_num_children_vec":self.c_num_children_vec,
                "c_max_depth":self.c_max_depth,
                "c_num_assignment_vec":self.c_num_assignment_vec,
                "c_ranges":self.c_ranges}

    def _set_h0_g_recursion(self,node:_Node):
        if node.depth == self.c_max_depth:
            node.h_g = 0
        else:
            node.h_g = self.h0_g
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_h0_g_recursion(node.children[i])

    def _set_sub_h0_params_recursion(self,node:_Node):
        # node.sub_model.set_h0_params(**self.sub_h0_params)
        node.sub_model = self.SubModel.LearnModel(**self.sub_h0_params)
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_sub_h0_params_recursion(node.children[i])

    def _set_h0_params_recursion(self,node:_Node,original_tree_node:_Node):
        if original_tree_node is None:
            if node.depth == self.c_max_depth:
                node.h_g = 0
            else:
                node.h_g = self.h0_g
            # node.sub_model.set_h0_params(**self.sub_h0_params)
            node.sub_model = self.SubModel.LearnModel(**self.sub_h0_params)
            if not node.leaf:
                for i in range(self.c_num_children_vec[node.k]):
                    self._set_h0_params_recursion(node.children[i],None)
        else:
            if node.depth == self.c_max_depth:
                node.h_g = 0
            else:
                node.h_g = original_tree_node.h_g
            try:
                sub_h0_params = node.sub_model.get_h_params()
            except:
                sub_h0_params = node.sub_model.get_h0_params()
            node.sub_model.set_h0_params(*sub_h0_params.values())
            if original_tree_node.leaf or node.depth == self.c_max_depth:  # leaf node
                node.leaf = True
            else:
                node.k = original_tree_node.k
                node.children = [None for i in range(self.c_num_children_vec[node.k])]
                if node.k < self.c_dim_continuous:
                    node.thresholds = np.array(original_tree_node.thresholds)
                else:
                    node.thresholds = None
                child_k_candidates = node.k_candidates.copy()
                child_k_candidates.remove(node.k)
                node.leaf = False
                for i in range(self.c_num_children_vec[node.k]):
                    if node.children[i] is None:
                        node.children[i] = _Node(
                            node.depth+1,
                            sub_model=self.SubModel.LearnModel(**self.sub_h0_params),
                            )
                    node.children[i].k_candidates = child_k_candidates
                    node.children[i].ranges = np.array(node.ranges)
                    if node.thresholds is not None:
                        node.children[i].ranges[node.k,0] = node.thresholds[i]
                        node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                    self._set_h0_params_recursion(node.children[i],original_tree_node.children[i])

    def _set_hn_g_recursion(self,node:_Node):
        if node.depth == self.c_max_depth:
            node.h_g = 0
        else:
            node.h_g = self.hn_g
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_hn_g_recursion(node.children[i])

    def _set_sub_hn_params_recursion(self,node:_Node):
        # node.sub_model.set_hn_params(**self.sub_hn_params)
        node.sub_model = self.SubModel.LearnModel(**self.sub_hn_params)
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_sub_hn_params_recursion(node.children[i])

    def _set_hn_params_recursion(self,node:_Node,original_tree_node:_Node):
        if original_tree_node is None:
            if node.depth == self.c_max_depth:
                node.h_g = 0
            else:
                node.h_g = self.hn_g
            # node.sub_model.set_hn_params(**self.sub_hn_params)
            node.sub_model = self.SubModel.LearnModel(**self.sub_hn_params)
            if not node.leaf:
                for i in range(self.c_num_children_vec[node.k]):
                    self._set_hn_params_recursion(node.children[i],None)
        else:
            if node.depth == self.c_max_depth:
                node.h_g = 0
            else:
                node.h_g = original_tree_node.h_g
            try:
                sub_hn_params = node.sub_model.get_h_params()
            except:
                sub_hn_params = node.sub_model.get_hn_params()
            node.sub_model.set_hn_params(*sub_hn_params.values())
            if original_tree_node.leaf or node.depth == self.c_max_depth:  # leaf node
                node.leaf = True
            else:
                node.k = original_tree_node.k
                node.children = [None for i in range(self.c_num_children_vec[node.k])]
                if node.k < self.c_dim_continuous:
                    node.thresholds = np.array(original_tree_node.thresholds)
                else:
                    node.thresholds = None
                child_k_candidates = node.k_candidates.copy()
                child_k_candidates.remove(node.k)
                node.leaf = False
                for i in range(self.c_num_children_vec[node.k]):
                    if node.children[i] is None:
                        node.children[i] = _Node(
                            node.depth+1,
                            sub_model=self.SubModel.LearnModel(**self.sub_hn_params),
                            )
                    node.children[i].k_candidates = child_k_candidates
                    node.children[i].ranges = np.array(node.ranges)
                    if node.thresholds is not None:
                        node.children[i].ranges[node.k,0] = node.thresholds[i]
                        node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                    self._set_hn_params_recursion(node.children[i],original_tree_node.children[i])

    def set_h0_params(self,
        h0_k_weight_vec = None,
        h0_g=None,
        sub_h0_params=None,
        h0_metatree_list=None,
        h0_metatree_prob_vec=None
        ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h0_k_weight_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]`, 
            by default None
            Sum of its elements must be 1.
        h0_g : float, optional
            A real number in :math:`[0, 1]`, by default None
        sub_h0_params : dict, optional
            h0_params for self.SubModel.LearnModel, by default None
        h0_metatree_list : list of metatree._Node, optional
            Root nodes of meta-trees, by default None
        h0_metatree_prob_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]` 
            that represents prior distribution of h0_metatree_list, 
            by default None.
            Sum of its elements must be 1.0.
        """
        if h0_k_weight_vec is not None:
            _check.nonneg_float_vec(h0_k_weight_vec,'h0_k_weight_vec',ParameterFormatError)
            _check.shape_consistency(
                h0_k_weight_vec.shape[0],'h0_k_weight_vec.shape[0]',
                self.c_dim_features,'self.c_dim_features',
                ParameterFormatError
                )
            self.h0_k_weight_vec[:] = h0_k_weight_vec

        if h0_g is not None:
            self.h0_g = _check.float_in_closed01(h0_g,'h0_g',ParameterFormatError)
            if self.h0_metatree_list:
                for h0_root in self.h0_metatree_list:
                    self._set_h0_g_recursion(h0_root)

        if sub_h0_params is not None:
            self.SubModel.LearnModel(**sub_h0_params)
            self.sub_h0_params = copy.deepcopy(sub_h0_params)
            if self.h0_metatree_list:
                for h0_root in self.h0_metatree_list:
                    self._set_sub_h0_params_recursion(h0_root)

        if h0_metatree_list is not None:
            if not isinstance(h0_metatree_list,list):
                raise(ParameterFormatError(
                    "h0_metatree_list must be a list"
                ))
            if h0_metatree_list:
                for h0_root in h0_metatree_list:
                    if type(h0_root) is not _Node:
                        raise(ParameterFormatError(
                            "all elements of h0_metatree_list must be instances of metatree._Node or empty"
                        ))
            diff = len(h0_metatree_list) - len(self.h0_metatree_list)
            if diff < 0:
                del self.h0_metatree_list[diff:]
            elif diff > 0:
                for i in range(diff):
                    self.h0_metatree_list.append(
                        _Node(
                            0,
                            self._root_k_candidates,
                            self.h0_g,
                            sub_model=self.SubModel.LearnModel(**self.sub_h0_params),
                            ranges=self.c_ranges,
                            )
                    )
            for i in range(len(self.h0_metatree_list)):
                self._set_h0_params_recursion(self.h0_metatree_list[i],h0_metatree_list[i])
            if h0_metatree_prob_vec is not None:
                self.h0_metatree_prob_vec = np.array(
                    _check.float_vec_sum_1(
                        h0_metatree_prob_vec,
                        'h0_metatree_prob_vec',
                        ParameterFormatError
                    )
                )
            else:
                if h0_metatree_list:
                    metatree_num = len(self.h0_metatree_list)
                    self.h0_metatree_prob_vec = np.ones(metatree_num) / metatree_num
                else:
                    self.h0_metatree_prob_vec = None
        elif h0_metatree_prob_vec is not None:
            self.h0_metatree_prob_vec = np.array(
                _check.float_vec_sum_1(
                    h0_metatree_prob_vec,
                    'h0_metatree_prob_vec',
                    ParameterFormatError
                )
            )

        if type(self.h0_metatree_prob_vec) is np.ndarray:             
            if self.h0_metatree_prob_vec.shape[0] != len(self.h0_metatree_list):
                raise(ParameterFormatError(
                    "Length of h0_metatree_list and dimension of h0_metatree_prob_vec must be the same."
                ))
        elif self.h0_metatree_prob_vec is None:
            if len(self.h0_metatree_list) > 0:
                raise(ParameterFormatError(
                    "Length of h0_metatree_list must be zero when self.h0_metatree_prob_vec is None."
                ))
        else:
            raise(ParameterFormatError(
                "self.h0_metatree_prob_vec must be None or a numpy.ndarray."
            ))

        self.reset_hn_params()

    def get_h0_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h0_params : dict of {str: float, list, dict, numpy.ndarray}
            * ``"h0_k_weight_vec"`` : the value of ``self.h0_k_weight_vec``
            * ``"h0_g"`` : the value of ``self.h0_g``
            * ``"sub_h0_params"`` : the value of ``self.sub_h0_params``
            * ``"h0_metatree_list"`` : the value of ``self.h0_metatree_list``
            * ``"h0_metatree_prob_vec"`` : the value of ``self.h0_metatree_prob_vec``
        """
        return {"h0_k_weight_vec":self.h0_k_weight_vec,
                "h0_g":self.h0_g, 
                "sub_h0_params":self.sub_h0_params, 
                "h0_metatree_list":self.h0_metatree_list,
                "h0_metatree_prob_vec":self.h0_metatree_prob_vec}
    
    def set_hn_params(self,
        hn_k_weight_vec = None,
        hn_g=None,
        sub_hn_params=None,
        hn_metatree_list=None,
        hn_metatree_prob_vec=None
        ):
        """Set the hyperparameters of the posterior distribution.

        Parameters
        ----------
        hn_k_weight_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]`, 
            by default None
            Sum of its elements must be 1.
        hn_g : float, optional
            A real number in :math:`[0, 1]`, by default None
        sub_hn_params : dict, optional
            hn_params for self.SubModel.LearnModel, by default None
        hn_metatree_list : list of metatree._Node, optional
            Root nodes of meta-trees, by default None
        hn_metatree_prob_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]` 
            that represents prior distribution of hn_metatree_list, 
            by default None.
            Sum of its elements must be 1.0.
        """
        if hn_k_weight_vec is not None:
            _check.nonneg_float_vec(hn_k_weight_vec,'hn_k_weight_vec',ParameterFormatError)
            _check.shape_consistency(
                hn_k_weight_vec.shape[0],'hn_k_weight_vec.shape[0]',
                self.c_dim_features,'self.c_dim_features',
                ParameterFormatError
                )
            self.hn_k_weight_vec[:] = hn_k_weight_vec

        if hn_g is not None:
            self.hn_g = _check.float_in_closed01(hn_g,'hn_g',ParameterFormatError)
            if self.hn_metatree_list:
                for hn_root in self.hn_metatree_list:
                    self._set_hn_g_recursion(hn_root)

        if sub_hn_params is not None:
            self.SubModel.LearnModel(**sub_hn_params)
            self.sub_hn_params = copy.deepcopy(sub_hn_params)
            if self.hn_metatree_list:
                for hn_root in self.hn_metatree_list:
                    self._set_sub_hn_params_recursion(hn_root)

        if hn_metatree_list is not None:
            if not isinstance(hn_metatree_list,list):
                raise(ParameterFormatError(
                    "hn_metatree_list must be a list"
                ))
            if hn_metatree_list:
                for hn_root in hn_metatree_list:
                    if type(hn_root) is not _Node:
                        raise(ParameterFormatError(
                            "all elements of hn_metatree_list must be instances of metatree._Node or empty"
                        ))
            diff = len(hn_metatree_list) - len(self.hn_metatree_list)
            if diff < 0:
                del self.hn_metatree_list[diff:]
            elif diff > 0:
                for i in range(diff):
                    self.hn_metatree_list.append(
                        _Node(
                            0,
                            self._root_k_candidates,
                            self.hn_g,
                            sub_model=self.SubModel.LearnModel(**self.sub_hn_params),
                            ranges=self.c_ranges,
                            )
                    )
            for i in range(len(self.hn_metatree_list)):
                self._set_hn_params_recursion(self.hn_metatree_list[i],hn_metatree_list[i])
            if hn_metatree_prob_vec is not None:
                self.hn_metatree_prob_vec = np.array(
                    _check.float_vec_sum_1(
                        hn_metatree_prob_vec,
                        'hn_metatree_prob_vec',
                        ParameterFormatError
                    )
                )
            else:
                if hn_metatree_list:
                    metatree_num = len(self.hn_metatree_list)
                    self.hn_metatree_prob_vec = np.ones(metatree_num) / metatree_num
                else:
                    self.hn_metatree_prob_vec = None
        elif hn_metatree_prob_vec is not None:
            self.hn_metatree_prob_vec = np.array(
                _check.float_vec_sum_1(
                    hn_metatree_prob_vec,
                    'hn_metatree_prob_vec',
                    ParameterFormatError
                )
            )

        if type(self.hn_metatree_prob_vec) is np.ndarray:             
            if self.hn_metatree_prob_vec.shape[0] != len(self.hn_metatree_list):
                raise(ParameterFormatError(
                    "Length of hn_metatree_list and dimension of hn_metatree_prob_vec must be the same."
                ))
        elif self.hn_metatree_prob_vec is None:
            if len(self.hn_metatree_list) > 0:
                raise(ParameterFormatError(
                    "Length of hn_metatree_list must be zero when self.hn_metatree_prob_vec is None."
                ))
        else:
            raise(ParameterFormatError(
                "self.hn_metatree_prob_vec must be None or a numpy.ndarray."
            ))

        self.calc_pred_dist(
            np.zeros(self.c_dim_continuous,dtype=float),
            np.zeros(self.c_dim_categorical,dtype=int))

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: float, list, dict, numpy.ndarray}
            * ``"hn_k_weight_vec"`` : the value of ``self.hn_k_weight_vec``
            * ``"hn_g"`` : the value of ``self.hn_g``
            * ``"sub_hn_params"`` : the value of ``self.sub_hn_params``
            * ``"hn_metatree_list"`` : the value of ``self.hn_metatree_list``
            * ``"hn_metatree_prob_vec"`` : the value of ``self.hn_metatree_prob_vec``
        """
        return {"hn_k_weight_vec":self.hn_k_weight_vec,
                "hn_g":self.hn_g, 
                "sub_hn_params":self.sub_hn_params, 
                "hn_metatree_list":self.hn_metatree_list,
                "hn_metatree_prob_vec":self.hn_metatree_prob_vec}
    
    def _copy_tree_from_sklearn_tree(self,new_node:_Node, original_tree,node_id):
        if original_tree.children_left[node_id] != sklearn_tree._tree.TREE_LEAF:  # inner node
            new_node.k = original_tree.feature[node_id]
            new_node.children = [None,None]
            if new_node.k < self.c_dim_continuous:
                new_node.thresholds = np.array(
                    [new_node.ranges[new_node.k,0],
                     original_tree.threshold[node_id],
                     new_node.ranges[new_node.k,1]])
            else:
                new_node.thresholds = None
            child_k_candidates = new_node.k_candidates.copy()
            child_k_candidates.remove(new_node.k)
            new_node.children[0] = _Node(
                new_node.depth+1,
                child_k_candidates,
                h_g=self.h0_g,
                sub_model=self.SubModel.LearnModel(**self.sub_h0_params),
                ranges=np.array(new_node.ranges)
                )
            if new_node.thresholds is not None:
                new_node.children[0].ranges[new_node.k,1] = new_node.thresholds[1]
            self._copy_tree_from_sklearn_tree(new_node.children[0],original_tree,original_tree.children_left[node_id])
            new_node.children[1] = _Node(
                new_node.depth+1,
                child_k_candidates,
                h_g=self.h0_g,
                sub_model=self.SubModel.LearnModel(**self.sub_h0_params),
                ranges=np.array(new_node.ranges)
                )
            if new_node.thresholds is not None:
                new_node.children[1].ranges[new_node.k,0] = new_node.thresholds[1]
            self._copy_tree_from_sklearn_tree(new_node.children[1],original_tree,original_tree.children_right[node_id])
        else:
            new_node.h_g = 0.0
            new_node.leaf = True

    def _update_posterior_leaf(self,node:_Node,x_continuous,y):
            try:
                node.sub_model.calc_pred_dist(x_continuous)
            except:
                node.sub_model.calc_pred_dist()
            pred_dist = node.sub_model.make_prediction(loss='KL') # Futurework: direct method to get marginal likelihood is better

            try:
                node.sub_model.update_posterior(x_continuous,y)
            except:
                node.sub_model.update_posterior(y)

            if type(pred_dist) is np.ndarray:
                return pred_dist[y]
            try:
                return pred_dist.pdf(y)
            except:
                return pred_dist.pmf(y)

    def _update_posterior_recursion(self,node:_Node,x_continuous,x_categorical,y):
        if not node.leaf:  # inner node
            if node.k < self.c_dim_continuous:
                for i in range(self.c_num_children_vec[node.k]):
                    if node.thresholds[i] < x_continuous[node.k] and x_continuous[node.k] < node.thresholds[i+1]:
                        index = i
                        break
            else:
                index = x_categorical[node.k-self.c_dim_continuous]
            tmp1 = self._update_posterior_recursion(node.children[index],x_continuous,x_categorical,y)
            tmp2 = (1 - node.h_g) * self._update_posterior_leaf(node,x_continuous,y) + node.h_g * tmp1
            node.h_g = node.h_g * tmp1 / tmp2
            return tmp2
        else:  # leaf node
            return self._update_posterior_leaf(node,x_continuous,y)

    def _compare_metatree_recursion(self,node1:_Node,node2:_Node):
        if node1.leaf:
            if node2.leaf:
                return True
            else:
                return False
        else:
            if node2.leaf:
                return False
            elif node1.k < self.c_dim_continuous:
                if node1.k != node2.k or not np.allclose(node1.thresholds,node2.thresholds):
                    return False
                else:
                    for i in range(self.c_num_children_vec[node1.k]):
                        if not self._compare_metatree_recursion(node1.children[i],node2.children[i]):
                            return False
                    return True
            else:
                if node1.k != node2.k:
                    return False
                else:
                    for i in range(self.c_num_children_vec[node1.k]):
                        if not self._compare_metatree_recursion(node1.children[i],node2.children[i]):
                            return False
                    return True
    
    def _marge_metatrees(self,metatree_list,metatree_prob_vec):
        num_metatrees = len(metatree_list)
        for i in range(num_metatrees):
            for j in range(i+1,num_metatrees):
                if self._compare_metatree_recursion(metatree_list[i],metatree_list[j]):
                    metatree_list[i] = None
                    metatree_prob_vec[j] += metatree_prob_vec[i]
                    metatree_prob_vec[i] = -1
                    break
        metatree_list = [tmp for tmp in metatree_list if tmp != None]
        metatree_prob_vec = metatree_prob_vec[metatree_prob_vec > -0.5]
        return metatree_list,metatree_prob_vec

    def _MTRF(self,x_continuous,x_categorical,y,n_estimators=100,**kwargs):
        """make metatrees

        Parameters
        ----------
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x[i,j] must satisfy 
            0 <= x[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y : numpy ndarray
            values of objective variable whose dtype may be int or float
        n_estimators : int, optional
            number of trees in sklearn.RandomForestClassifier or 
            sklearn.RandomForestRegressor, by default 100

        Returns
        -------
        metatree_list : list of metatree._Node
            Each element is a root node of metatree.
        metatree_prob_vec : numpy ndarray
        """
        if np.any(self.c_num_children_vec != 2):
            raise(ParameterFormatError("MTRF is supported only when all the elements of c_num_children_vec is 2."))
        if self.SubModel in DISCRETE_MODELS:
            randomforest = RandomForestClassifier(n_estimators=n_estimators,max_depth=self.c_max_depth,**kwargs)
        if self.SubModel in CONTINUOUS_MODELS:
            randomforest = RandomForestRegressor(n_estimators=n_estimators,max_depth=self.c_max_depth,**kwargs)

        x = np.empty([y.shape[0],self.c_dim_features])
        x[:,:self.c_dim_continuous] = x_continuous
        x[:,self.c_dim_continuous:] = x_categorical

        randomforest.fit(x,y)
        
        tmp_metatree_list = [
            _Node(
                0,
                self._root_k_candidates,
                self.hn_g,
                sub_model=self.SubModel.LearnModel(**self.sub_hn_params),
                ranges=self.c_ranges,
                )
            for i in range(n_estimators)
            ]
        tmp_metatree_prob_vec = np.ones(n_estimators) / n_estimators
        for i in range(n_estimators):
            self._copy_tree_from_sklearn_tree(tmp_metatree_list[i],randomforest.estimators_[i].tree_, 0)

        tmp_metatree_list,tmp_metatree_prob_vec = self._marge_metatrees(tmp_metatree_list,tmp_metatree_prob_vec)

        log_metatree_posteriors = np.log(tmp_metatree_prob_vec)
        for i,metatree in enumerate(tmp_metatree_list):
            for j in range(y.shape[0]):
                log_metatree_posteriors[i] += np.log(self._update_posterior_recursion(metatree,x_continuous[j],x_categorical[j],y[j]))
        tmp_metatree_prob_vec[:] = np.exp(log_metatree_posteriors - log_metatree_posteriors.max())
        tmp_metatree_prob_vec[:] /= tmp_metatree_prob_vec.sum()
        return tmp_metatree_list,tmp_metatree_prob_vec

    def _given_MT(self,x_continuous,x_categorical,y):
        """make metatrees

        Parameters
        ----------
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x[i,j] must satisfy 
            0 <= x[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y : numpy ndarray
            values of objective variable whose dtype may be int or float

        Returns
        -------
        metatree_list : list of metatree._Node
            Each element is a root node of metatree.
        metatree_prob_vec : numpy ndarray
        """
        if not self.hn_metatree_list:
            raise(ParameterFormatError("given_MT is supported only when len(self.hn_metatree_list) > 0."))
        log_metatree_posteriors = np.log(self.hn_metatree_prob_vec)
        for i,metatree in enumerate(self.hn_metatree_list):
            for j in range(y.shape[0]):
                log_metatree_posteriors[i] += np.log(self._update_posterior_recursion(metatree,x_continuous[j],x_categorical[j],y[j]))
        self.hn_metatree_prob_vec[:] = np.exp(log_metatree_posteriors - log_metatree_posteriors.max())
        self.hn_metatree_prob_vec[:] /= self.hn_metatree_prob_vec.sum()
        return self.hn_metatree_list,self.hn_metatree_prob_vec

    def update_posterior(self,x_continuous=None,x_categorical=None,y=None,alg_type='MTRF',**kwargs):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x[i,j] must satisfy 
            0 <= x[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y : numpy ndarray
            values of objective variable whose dtype may be int or float
        alg_type : {'MTRF', 'given_MT'}, optional
            type of algorithm, by default 'MTRF'
        **kwargs : dict, optional
            optional parameters of algorithms, by default {}
        """
        if self.SubModel in DISCRETE_MODELS:
            _check.nonneg_ints(y,'y',ParameterFormatError)
        elif self.SubModel in CONTINUOUS_MODELS:
            _check.floats(y,'y',ParameterFormatError)
        y = np.ravel(y)

        if self.c_dim_continuous > 0 and self.c_dim_categorical > 0:
            _check.float_vecs(x_continuous,'x_continuous',DataFormatError)
            _check.shape_consistency(
                x_continuous.shape[-1],'x_continuous.shape[-1]',
                self.c_dim_continuous,'self.c_dim_continuous',
                ParameterFormatError
                )
            x_continuous.reshape([-1,self.c_dim_continuous])

            _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
            _check.shape_consistency(
                x_categorical.shape[-1],'x_categorical.shape[-1]',
                self.c_dim_categorical,'self.c_dim_categorical',
                ParameterFormatError
                )
            for i in range(self.c_dim_categorical):
                if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                    raise(DataFormatError(
                        f"x_categorical[{i}].max() must smaller than "
                        +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                        +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))
            x_categorical.reshape([-1,self.c_dim_categorical])

            _check.shape_consistency(
                x_continuous.shape[0],'x_continuous.shape[0]',
                x_categorical.shape[0],'x_categorical.shape[0]',
                ParameterFormatError
                )
            _check.shape_consistency(
                x_categorical.shape[0],'x_categorical.shape[0]',
                y.shape[0],'y.shape[0]',
                ParameterFormatError
                )

        elif self.c_dim_continuous > 0:
            _check.float_vecs(x_continuous,'x_continuous',DataFormatError)
            _check.shape_consistency(
                x_continuous.shape[-1],'x_continuous.shape[-1]',
                self.c_dim_continuous,'self.c_dim_continuous',
                ParameterFormatError
                )
            x_continuous.reshape([-1,self.c_dim_continuous])

            _check.shape_consistency(
                x_continuous.shape[0],'x_continuous.shape[0]',
                y.shape[0],'y.shape[0]',
                ParameterFormatError
                )
            
            x_categorical = np.empty([y.shape[0],0]) # dummy

        elif self.c_dim_categorical > 0:
            _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
            _check.shape_consistency(
                x_categorical.shape[-1],'x_categorical.shape[-1]',
                self.c_dim_categorical,'self.c_dim_categorical',
                ParameterFormatError
                )
            for i in range(self.c_dim_categorical):
                if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                    raise(DataFormatError(
                        f"x_categorical[{i}].max() must smaller than "
                        +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                        +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))
            x_categorical.reshape([-1,self.c_dim_categorical])

            _check.shape_consistency(
                x_categorical.shape[0],'x_categorical.shape[0]',
                y.shape[0],'y.shape[0]',
                ParameterFormatError
                )
            
            x_continuous = np.empty([y.shape[0],0]) # dummy

        if alg_type == 'MTRF':
            self.hn_metatree_list, self.hn_metatree_prob_vec = self._MTRF(x_continuous,x_categorical,y,**kwargs)
        elif alg_type == 'given_MT':
            self.hn_metatree_list, self.hn_metatree_prob_vec = self._given_MT(x_continuous,x_categorical,y)

    def _map_recursion_add_nodes(self,node:_Node):
        if node.depth == self.c_max_depth or not node.k_candidates:  # leaf node
            node.h_g = 0.0
            node.sub_model = self.SubModel.LearnModel(**self.sub_hn_params)
            node.leaf = True
            node.map_leaf = True
        else:  # inner node
            node.k = node.k_candidates[self.hn_k_weight_vec[node.k_candidates].argmax()]
            node.children = [None for i in range(self.c_num_children_vec[node.k])]
            if node.k < self.c_dim_continuous:
                node.thresholds = np.linspace(
                    node.ranges[node.k,0],
                    node.ranges[node.k,1],
                    self.c_num_children_vec[node.k]+1
                    )
            else:
                node.thresholds = None
            child_k_candidates = node.k_candidates.copy()
            child_k_candidates.remove(node.k)
            node.leaf = False
            for i in range(self.c_num_children_vec[node.k]):
                node.children[i] = _Node(
                    node.depth+1,
                    child_k_candidates,
                    self.hn_g,
                    ranges=np.array(node.ranges)
                    )
                if node.thresholds is not None:
                    node.children[i].ranges[node.k,0] = node.thresholds[i]
                    node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                self._map_recursion_add_nodes(node.children[i])

    def _map_recursion(self,node:_Node):
        if node.leaf:
            if node.depth == self.c_max_depth or not node.k_candidates:
                node.map_leaf = True
                return 1.0
            else:
                sum_nodes = 0
                num_nodes = 1
                rest_num_children_vec = np.sort(self.c_num_children_vec[node.k_candidates])
                for i in range(min(self.c_max_depth-node.depth,len(node.k_candidates))):
                    sum_nodes += num_nodes
                    num_nodes *= rest_num_children_vec[i]
                if 1.0 - node.h_g > node.h_g * self.hn_g ** (sum_nodes-1):
                    node.map_leaf = True
                    return 1.0 - node.h_g
                else:
                    self._map_recursion_add_nodes(node)
                    return node.h_g * self.hn_g ** (sum_nodes-1)
        else:
            tmp1 = 1.0-node.h_g
            tmp_vec = np.empty(self.c_num_children_vec[node.k])
            for i in range(self.c_num_children_vec[node.k]):
                tmp_vec[i] = self._map_recursion(node.children[i])
            if tmp1 > node.h_g*tmp_vec.prod():
                node.map_leaf = True
                return tmp1
            else:
                node.map_leaf = False
                return node.h_g*tmp_vec.prod()

    def _copy_map_tree_recursion(self,copied_node:_Node,original_node:_Node):
        copied_node.h_g = original_node.h_g
        if original_node.map_leaf:
            copied_node.sub_model = copy.deepcopy(original_node.sub_model)
            copied_node.leaf = True
        else:
            copied_node.k = original_node.k
            copied_node.children = [None for i in range(self.c_num_children_vec[copied_node.k])]
            if copied_node.k < self.c_dim_continuous:
                copied_node.thresholds = np.array(original_node.thresholds)
            else:
                copied_node.thresholds = None
            child_k_candidates = copied_node.k_candidates.copy()
            child_k_candidates.remove(copied_node.k)
            copied_node.leaf = False
            for i in range(self.c_num_children_vec[copied_node.k]):
                copied_node.children[i] = _Node(
                    copied_node.depth+1,
                    child_k_candidates,
                    ranges=np.array(copied_node.ranges),
                    )
                if copied_node.thresholds is not None:
                    copied_node.children[i].ranges[copied_node.k,0] = copied_node.thresholds[i]
                    copied_node.children[i].ranges[copied_node.k,1] = copied_node.thresholds[i+1]
                self._copy_map_tree_recursion(copied_node.children[i],original_node.children[i])

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
        map_root : metatree._Node
            The root node of the estimated meta-tree 
            that also contains the estimated parameters in each node.

        See Also
        --------
        graphviz.Digraph

        Warnings
        --------
        Multiple metatrees can represent equivalent model classes. 
        This function does not take such duplication into account.
        """

        if loss == "0-1":
            map_index = 0
            map_prob = 0.0
            for i,metatree in enumerate(self.hn_metatree_list):
                prob = self.hn_metatree_prob_vec[i] * self._map_recursion(metatree)
                if prob > map_prob:
                    map_index = i
                    map_prob = prob
            map_root = _Node(
                0,
                self._root_k_candidates,
                self.hn_g,
                ranges=self.c_ranges,
                )
            self._copy_map_tree_recursion(map_root,self.hn_metatree_list[map_index])
            if visualize:
                import graphviz
                tree_graph = graphviz.Digraph(filename=filename,format=format)
                tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
                self._visualize_model_recursion(tree_graph, map_root, 0, None, None, None, 1.0)
                tree_graph.view()
            return {'root':map_root}
        else:
            raise(CriteriaError("Unsupported loss function! "
                                +"This function supports only \"0-1\"."))
    
    def _visualize_model_recursion(self,tree_graph,node:_Node,node_id,parent_id,parent_k,sibling_num,p_v):
        tmp_id = node_id
        tmp_p_v = p_v

        # add node information
        if node.leaf:
            label_string = 'k=None\\l'
        else:
            label_string = f'k={node.k}\\l'
            if node.k < self.c_dim_continuous:
                label_string += 'thresholds=\\l{'
                for i in range(self.c_num_children_vec[node.k]-1):
                    if i == 0:
                        label_string += f'{node.thresholds[i+1]:.2f}'
                    else:
                        label_string += f',{node.thresholds[i+1]:.2f}'
                label_string += '}\\l'
        label_string += f'hn_g={node.h_g:.2f}\\lp_v={tmp_p_v:.2f}\\lsub_params={{'
        if node.sub_model is not None:
            try:
                sub_params = node.sub_model.estimate_params(loss='0-1',dict_out=True)
            except:
                sub_params = node.sub_model.estimate_params(dict_out=True)

            for key,value in sub_params.items():
                try:
                    label_string += f'\\l{key}:{value:.2f}'
                except:
                    label_string += f'\\l{key}:{value}'
            label_string += '}'
        else:
            label_string += '\\lNone}'
            
        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_v))}')
        if tmp_p_v > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            if parent_k < self.c_dim_continuous:
                if node.ranges[parent_k,0] <= self.c_ranges[parent_k,0] + 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[*,{node.ranges[parent_k,1]:.2f})')
                elif node.ranges[parent_k,1] >= self.c_ranges[parent_k,1] - 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{node.ranges[parent_k,0]:.2f},*)')
                else:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{node.ranges[parent_k,0]:.2f},{node.ranges[parent_k,1]:.2f})')
            else:
                tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                node_id = self._visualize_model_recursion(tree_graph,node.children[i],node_id+1,tmp_id,node.k,i,tmp_p_v*node.h_g)
        
        return node_id

    def _visualize_model_recursion_none(self,tree_graph,depth,k_candidates,ranges,node_id,parent_id,parent_k,sibling_num,p_v):
        tmp_id = node_id
        tmp_p_v = p_v
        
        # add node information
        if depth == self.c_max_depth or not k_candidates:
            label_string = 'k=None\\l'
        else:
            k = k_candidates[self.hn_k_weight_vec[k_candidates].argmax()]
            label_string = f'k={k}\\l'
            if k < self.c_dim_continuous:
                thresholds = np.linspace(ranges[k,0],ranges[k,1],self.c_num_children_vec[k]+1)
                label_string += 'thresholds=\\l{'
                for i in range(self.c_num_children_vec[k]-1):
                    if i == 0:
                        label_string += f'{thresholds[i+1]:.2f}'
                    else:
                        label_string += f',{thresholds[i+1]:.2f}'
                label_string += '}\\l'
            else:
                thresholds = None
            child_k_candidates = k_candidates.copy()
            child_k_candidates.remove(k)
        label_string += f'hn_g={self.hn_g:.2f}\\lp_v={tmp_p_v:.2f}\\lsub_params={{'

        sub_model = self.SubModel.LearnModel(**self.sub_hn_params)
        try:
            sub_params = sub_model.estimate_params(loss='0-1',dict_out=True)
        except:
            sub_params = sub_model.estimate_params(dict_out=True)

        for key,value in sub_params.items():
            try:
                label_string += f'\\l{key}:{value:.2f}'
            except:
                label_string += f'\\l{key}:{value}'
        label_string += '}'

        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_v))}')
        if tmp_p_v > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            if parent_k < self.c_dim_continuous:
                if ranges[parent_k,0] <= self.c_ranges[parent_k,0] + 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[*,{ranges[parent_k,1]:.2f})')
                elif ranges[parent_k,1] >= self.c_ranges[parent_k,1] - 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{ranges[parent_k,0]:.2f},*)')
                else:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{ranges[parent_k,0]:.2f},{ranges[parent_k,1]:.2f})')
            else:
                tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        if depth < self.c_max_depth and k_candidates:
            for i in range(self.c_num_children_vec[k]):
                child_ranges = np.array(ranges)
                if thresholds is not None:
                    child_ranges[k,0] = thresholds[i]
                    child_ranges[k,1] = thresholds[i+1]
                node_id = self._visualize_model_recursion_none(tree_graph,depth+1,child_k_candidates,child_ranges,node_id+1,tmp_id,k,i,tmp_p_v*self.hn_g)
        
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
        >>> from bayesml import metatree
        >>> gen_model = metatree.GenModel(c_k=3,h_g=0.75)
        >>> gen_model.gen_params()
        >>> x,y = gen_model.gen_sample(500)
        >>> learn_model = metatree.LearnModel(c_k=3)
        >>> learn_model.update_posterior(x,y)
        >>> learn_model.visualize_posterior()

        .. image:: ./images/metatree_posterior.png

        See Also
        --------
        graphviz.Digraph
        """
        try:
            import graphviz
            tree_graph = graphviz.Digraph(filename=filename,format=format)
            tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
            if not self.hn_metatree_list:
                self._visualize_model_recursion_none(tree_graph, 0, self._root_k_candidates, self.c_ranges, 0, None, None, None, 1.0)
            else:
                MAP_index = np.argmax(self.hn_metatree_prob_vec)
                print(f'Approximate MAP probability of metatree:{self.hn_metatree_prob_vec[MAP_index]}')
                self._visualize_model_recursion(tree_graph, self.hn_metatree_list[MAP_index], 0, None, None, None, 1.0)
            # Can we show the image on the console without saving the file?
            tree_graph.view()
        except ImportError as e:
            print(e)
        except graphviz.CalledProcessError as e:
            print(e)
    
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        This model does not have a simple parametric expression of the predictive distribution.
        Therefore, this function returns ``None``.

        Returns
        -------
        ``None``
        """
        return None
    
    def _calc_pred_dist_leaf(self,node:_Node,x):
            try:
                node.sub_model.calc_pred_dist(x)
            except:
                node.sub_model.calc_pred_dist()

    def _calc_pred_dist_recursion(self,node:_Node,x):
        self._calc_pred_dist_leaf(node,x)
        if not node.leaf:  # inner node
            self._calc_pred_dist_recursion(node.children[x[node.k]],x)

    def calc_pred_dist(self,x_continuous,x_categorical):
        """Calculate the parameters of the predictive distribution.
        
        Parameters
        ----------
        x : numpy ndarray
            values of explanatory variables whose dtype is int
        """
        pass
        # _check.nonneg_int_vec(x,'x',DataFormatError)
        # if x.shape[0] != self.c_k:
        #     raise(DataFormatError(f"x.shape[0] must equal to c_k:{self.c_k}"))
        # if x.max() >= self.c_num_children:
        #     raise(DataFormatError(f"x.max() must smaller than c_num_children:{self.c_num_children}"))
        # self._tmp_x[:] = x
        # for root in self.hn_metatree_list:
        #     self._calc_pred_dist_recursion(root,self._tmp_x)

    def _make_prediction_recursion_squared(self,node:_Node):
        if node.leaf == False:  # inner node
            return ((1 - node.h_g) * node.sub_model.make_prediction(loss='squared')
                    + node.h_g * self._make_prediction_recursion_squared(node.children[self._tmp_x[node.k]]))
        else:  # leaf node
            return node.sub_model.make_prediction(loss='squared')

    def _make_prediction_leaf_01(self,node:_Node):
        mode = node.sub_model.make_prediction(loss='0-1')
        pred_dist = node.sub_model.make_prediction(loss='KL')
        if type(pred_dist) is np.ndarray:
            mode_prob = pred_dist[mode]
        else:
            try:
                mode_prob = pred_dist.pdf(mode)
            except:
                try:
                    mode_prob = pred_dist.pmf(mode)
                except:
                    mode_prob = None
        # elif hasattr(pred_dist,'pdf'):
        #     mode_prob = pred_dist.pdf(mode)
        # elif hasattr(pred_dist,'pmf'):
        #     mode_prob = pred_dist.pmf(mode)
        # else:
        #     mode_prob = None
        return mode, mode_prob

    def _make_prediction_recursion_01(self,node:_Node):
        if node.leaf == False:  # inner node
            mode1,mode_prob1 = self._make_prediction_leaf_01(node)
            mode2,mode_prob2 = self._make_prediction_recursion_01(node.children[self._tmp_x[node.k]])
            if (1 - node.h_g) * mode_prob1 > node.h_g * mode_prob2:
                return mode1,mode_prob1
            else:
                return mode2,mode_prob2
        else:  # leaf node
            return self._make_prediction_leaf_01(node)

    def make_prediction(self,loss="0-1"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"0-1\".
            This function supports \"squared\", \"0-1\".

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
        """
        if loss == "squared":
            tmp_pred_vec = np.empty(len(self.hn_metatree_list))
            for i,metatree in enumerate(self.hn_metatree_list):
                tmp_pred_vec[i] = self._make_prediction_recursion_squared(metatree)
            return self.hn_metatree_prob_vec @ tmp_pred_vec
        elif loss == "0-1":
            tmp_mode = np.empty(len(self.hn_metatree_list))
            tmp_mode_prob_vec = np.empty(len(self.hn_metatree_list))
            for i,metatree in enumerate(self.hn_metatree_list):
                tmp_mode[i],tmp_mode_prob_vec[i] = self._make_prediction_recursion_01(metatree)
            return tmp_mode[np.argmax(self.hn_metatree_prob_vec * tmp_mode_prob_vec)]
        else:
            raise(CriteriaError("Unsupported loss function! "
                                +"This function supports \"squared\" and \"0-1\"."))

    def pred_and_update(self,x,y,loss="0-1"):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x : numpy.ndarray
            It must be a degree-dimensional vector
        y : numpy ndarray
            values of objective variable whose dtype may be int or float
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"0-1\".
            This function supports \"squared\", \"0-1\", and \"KL\".

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
        """
        _check.nonneg_int_vec(x,'x',DataFormatError)
        if x.shape[-1] != self.c_k:
            raise(DataFormatError(f"x.shape[-1] must equal to c_k:{self.c_k}"))
        if x.max() >= self.c_num_children:
            raise(DataFormatError(f"x.max() must smaller than c_num_children:{self.c_num_children}"))
        self.calc_pred_dist(x)
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(x,y,alg_type='given_MT')
        return prediction
