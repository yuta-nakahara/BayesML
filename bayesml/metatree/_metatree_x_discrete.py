# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Document Author
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

class _GenNode:
    """ The node class used by generative model and the prior distribution

    Parameters
    ----------
    depth : int
            a non-negetive integer :math:' >= 0'
    k_candidates : list
            feature value which used to split node
    h_g : float
            a positive real number  in \[0 , 1 \], by default 0.5
    k : int
            a positive integer, by default None
    sub_model : class
            a class of generative model used by MT-Model 
    """
    def __init__(self,
                 depth,
                 k_candidates,
                 NUM_CHILDREN = 2,
                 h_g = 0.5,
                 k = None,
                 sub_model = None
                 ):
        self.depth = depth
        self.children = [None for i in range(NUM_CHILDREN)]  # child nodes
        self.k_candidates = k_candidates
        self.h_g = h_g
        self.k = k
        self.sub_model = sub_model
        self.leaf = False

class GenModel(base.Generative):
    """ The stochastice data generative model and the prior distribution

    Parameters
    ----------
    D_MAX : int 
            a positive integer, by default 3
    K : int
            a positive integer, by default 3
    NUM_CHILDREN : int
            a positive integer, by default 2
    h_k_prob : list of float
            a list of positive real number witch in \[0 , 1 \], by default None
    h_g : float 
            a positive real number  in \[0 , 1 \], by default 0.5
    sub_model : class
            a class of generative model used by MT-Model 
    sub_h_params : dictionary
            a empty dcitionary to record parameters of sub_model, by default empty
    """
    def __init__(
        self,
        *,
        D_MAX=3,
        K=3,
        NUM_CHILDREN=2,
        h_k_prob = None,
        h_g=0.5,
        SubModel=bernoulli.GenModel,
        sub_h_params={},
        seed=None
        ):
        # 例外処理
        if not isinstance(D_MAX, int) or D_MAX <= 0: #numpy 1次元の場合は要検討
            raise(ParameterFormatError("D_MAX must be an int type positive number."))
        if not isinstance(K, int) or K <= 0:
            raise(ParameterFormatError("K must be a int type positive number."))
        if not isinstance(NUM_CHILDREN, int) or NUM_CHILDREN <= 0:
            raise(ParameterFormatError("NUM_CHILDREN must be a int type positive number."))
        if h_k_prob is not None and (not isinstance(h_k_prob,list) or len(h_k_prob) != K):
            raise(ParameterFormatError("h_k_prob must be a K-dimentional list type."))
        if not isinstance(h_g, float) or h_g < 0.0 or h_g > 1.0:
            raise(ParameterFormatError("h_g must be a float type between [0.0 , 1.0]."))
        #

      
        self.D_MAX = D_MAX
        self.K = K
        self.NUM_CHILDREN = NUM_CHILDREN
        if h_k_prob is not None:
            self.h_k_prob = h_k_prob
        else:
            self.h_k_prob = np.ones(self.K) / self.K
        self.h_g = h_g
        self.SubModel = SubModel
        self.sub_h_params = sub_h_params
        self.rng = np.random.default_rng(seed)
        self.root = _GenNode(0,list(range(self.K)),self.NUM_CHILDREN,self.h_g,None,None)

    def _gen_params_recursion(self,node,feature_fix):
        """ generate parameters recursively

        Parameters
        ----------
        node : object
                a object form GenNode class
        sub_h_params : bool
                a bool parameter show the feature is fixed or not
        """
        if node.depth == self.D_MAX or self.rng.random() > node.h_g:  # 葉ノード
            node.sub_model = self.SubModel(**self.sub_h_params)
            node.sub_model.gen_params()
            if node.depth == self.D_MAX:
                node.h_g = 0
            node.leaf = True
        else:  # 内部ノード
            if feature_fix == False or node.k is None:
                node.k = self.rng.choice(node.k_candidates,
                                         p=self.h_k_prob[node.k_candidates]/self.h_k_prob[node.k_candidates].sum())
            child_k_candidates = copy.copy(node.k_candidates)
            child_k_candidates.remove(node.k)
            node.leaf = False
            for i in range(self.NUM_CHILDREN):
                if feature_fix == False or node.children[i] is None:
                    node.children[i] = _GenNode(node.depth+1,child_k_candidates,self.NUM_CHILDREN,self.h_g,None,None)
                else:
                    node.children[i].k_candidates = child_k_candidates
                self._gen_params_recursion(node.children[i],feature_fix)

    def _gen_params_recursion_tree_fix(self,node,feature_fix):
        """ generate parameters recursively for fixed tree

        Parameters
        ----------
        node : object
                a object form GenNode class
        sub_h_params : bool
                a bool parameter show the feature is fixed or not
        """
        if self.leaf:  # 葉ノード
            node.sub_model = self.SubModel(**self.sub_h_params)
            node.sub_model.gen_params()
            if node.depth == self.D_MAX:
                node.h_g = 0
            node.leaf = True
        else:  # 内部ノード
            if feature_fix == False or node.k is None:
                node.k = self.rng.choice(node.k_candidates,
                                         p=self.h_k_prob[node.k_candidates]/self.h_k_prob[node.k_candidates].sum())
            child_k_candidates = copy.copy(node.k_candidates)
            child_k_candidates.remove(node.k)
            node.leaf = False
            for i in range(self.NUM_CHILDREN):
                if feature_fix == False or node.children[i] is None:
                    node.children[i] = _GenNode(node.depth+1,child_k_candidates,self.NUM_CHILDREN,self.h_g,None,None)
                else:
                    node.children[i].k_candidates = child_k_candidates
                self._gen_params_recursion_tree_fix(node.children[i],feature_fix)

    def _set_params_recursion(self,node,original_tree_node):
        """ copy parameters from a fixed tree

        Parameters
        ----------
        node : object
                a object form GenNode class
        original_tree_node : object
                a object form GenNode class
        """
        if original_tree_node.leaf:  # 葉ノード
            node.sub_model = copy.deepcopy(original_tree_node.sub_model)
            if node.depth == self.D_MAX:
                node.h_g = 0
            node.leaf = True
        else:
            node.k = original_tree_node.k
            child_k_candidates = copy.copy(node.k_candidates)
            child_k_candidates.remove(node.k)
            for i in range(self.NUM_CHILDREN):
                node.children[i] = _GenNode(node.depth+1,child_k_candidates,self.NUM_CHILDREN,self.h_g,None,None)
                self._set_params_recursion(node.children[i],original_tree_node.children[i])
    
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
            try:
                y = node.sub_model.gen_sample(sample_size=1,X=x)
            except:
                y = node.sub_model.gen_sample(sample_size=1)
            return y
        else:
            return self._gen_sample_recursion(node.children[x[node.k]],x)
    
    def _visualize_model_recursion(self,tree_graph,node,node_id,parent_id,sibling_num,p_v):
        """Visualize the stochastic data generative model and generated samples.

        """
        tmp_id = node_id
        tmp_p_v = p_v
        
        # add node information
        label_string = f'k={node.k}\\lh_g={node.h_g:.2f}\\lp_v={tmp_p_v:.2f}\\lsub_params={{'
        if node.sub_model is not None:
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
            tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        if node.leaf != True:
            for i in range(self.NUM_CHILDREN):
                node_id = self._visualize_model_recursion(tree_graph,node.children[i],node_id+1,tmp_id,i,tmp_p_v*node.h_g)
        
        return node_id

    def set_h_params(self,h_k_prob=None,h_g=0.5,sub_h_params={}):
        """Set the parameter of the sthocastic data generative tree model.

        Parameters
        ----------
        h_k_prob : list 
            K real numbers : math:`p \in [0, 1]` 
        h_g : float
            a real number :math:`p \in [0, 1]`
        sub_h_params :  dict of {str:float}
            a dictionary include hyper parameters for sub model 
        """
        #例外処理
        if h_k_prob is not None and (not isinstance(h_k_prob,list) or len(h_k_prob) != K):
            raise(ParameterFormatError("h_k_prob must be a K-dimentional list type."))
        if h_k_prob is not None:
            self.h_k_prob = h_k_prob
        else:
            self.h_k_prob = np.ones(self.K) / self.K

        if not isinstance(h_g, float) or h_g < 0.0 or h_g > 1.0:
            raise(ParameterFormatError("h_g must be a float type between [0.0 , 1.0]."))
        self.h_g = h_g
        #
        self.sub_h_params = sub_h_params

    def get_h_params(self):
        """Get the parameter of the sthocastic data generative tree model.

        Returns
        -------
        params : dict of {str: float or list or dict}
            * ``"h_k_prob"`` : The value of ``self.h_k_prob``.
            * ``"h_g"`` : The value of ``self.h_g``.
            * ``"sub_h_params"`` : The value of ``self.sub_h_params``.
        """
        return {"h_k_prob":self.h_k_prob, "h_g":self.h_g, "sub_h_params":self.sub_h_params}
    
    def save_h_params(self,filename):
        """Save the parameter with pickle format.

        Parameters
        ----------
        filename : str
            The filename to which the hyperparameters are saved.

        """

        with open(filename,'wb') as f:
            pickle.dump(self.get_h_params(), f)

    def load_h_params(self,filename):
        """Load the parameter saved by ``save_h_params``.

        Parameters
        ----------
        filename : str
            The filename to be loaded. 
        """
        with open(filename, 'rb') as f:
            h_params = pickle.load(f)
        self.set_h_params(**h_params)
        
    def gen_params(self,feature_fix=False,tree_fix=False):
        """Generate a sample from the stochastic data generative tree model.

        Parameters
        ----------
        feature_fix : bool
            A bool integer by default False
        tree_fix : bool
            A bool integer by default False
        """
        if tree_fix:
            self._gen_params_recursion_tree_fix(self.root,feature_fix)
        else:
            self._gen_params_recursion(self.root,feature_fix)
    
    def set_params(self,root):
        """Set the parameter of the sthocastic data generative tree model.
        """
        self._set_params_recursion(self.root,root)

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:float}
            * ``"root"`` : The value of ``self.root``.
        """
        return {"root":self.root}

    def save_params(self,filename):
        """Save the parameter with pickle

        Parameters
        ----------
        filename : str
            The filename to which the hyperparameters are saved.
        ----------
        numpy.savez_compressed
        """
        with open(filename,'wb') as f:
            pickle.dump(self.get_params(), f)

    def load_params(self,filename):
        """Load the parameter saved by ``save_h_params``.

        Parameters
        ----------
        filename : str
            The filename to be loaded. 
        """
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        if "h_alpha" not in h_params.files or "h_beta" not in h_params.files:
            raise(ParameterFormatError(filename+" must be a NpzFile with keywords: \"h_alpha\" and \"h_beta\"."))
        self.set_params(**params)

    def gen_sample(self,sample_size,X=None):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int
            A positive integer

        Returns
        -------
        X : numpy ndarray
            K dimensional array whose size is ``sammple_size`` and elements are 0 or 1.
        Y : numpy ndarray
            1 dimensional array whose size is ``sammple_size`` and elements are real number.
        """
        #例外処理
        if sample_size <= 0:
            raise(DataFormatError("sample_size must be a positive integer."))

        if X is None:
            X = self.rng.choice(self.NUM_CHILDREN,(sample_size,self.K))
        if self.SubModel == bernoulli.GenModel:
            Y = np.empty(sample_size,dtype=int)
        if self.SubModel == normal.GenModel:
            Y = np.empty(sample_size,dtype=float)
        for i in range(sample_size):
            Y[i] = self._gen_sample_recursion(self.root,X[i])
        return X,Y
        
    def save_sample(self,filename,sample_size,X=None):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as aNpzFile with keyword: \"X\".

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
        X,Y = self.gen_sample(sample_size,X)
        np.savez_compressed(filename,X=X,Y=Y)

    def visualize_model(self,filename=None,format=None,sample_size=10):
        """Visualize the stochastic data generative tree model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 10

        Examples
        --------
        >>> from bayesml import metatree
        >>> model = metatree.GenModel()
        >>> model.visualize_model()
        p:0.5
        [[1 0 1]
         [0 0 0]
         [1 0 1]
         [1 1 0]
         [0 0 0]
         [0 0 0]
         [0 1 0]
         [1 0 0]
         [1 0 0]
         [1 1 1]]
        [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]
        .. image:: ./metatree/Digraph.gv
        """
        #例外処理
        if sample_size <= 0:
            raise(DataFormatError("sample_size must be a positive integer."))

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
        # 以下のサンプルの可視化は要改善．jitter plotで3次元散布図を書くのが良いか．
        X,Y = self.gen_sample(sample_size)
        print(X)
        print(Y)

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
                 NUM_CHILDREN = 2,
                 hn_g = 0.5,
                 k = None,
                 sub_model = None
                 ):
        self.depth = depth
        self.children = [None for i in range(NUM_CHILDREN)]  # child nodes
        self.hn_g = hn_g
        self.k = k
        self.sub_model = sub_model
        self.leaf = False
        self.map_leaf = False

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.
    """
    def __init__(
        self,
        *,
        D_MAX=3,
        K=None,
        NUM_CHILDREN=2,
        h0_k_prob_vec = None,
        h0_g=0.5,
        SubModel=bernoulli.LearnModel,
        sub_h0_params={},
        h0_metatree_list=[],
        h0_metatree_prob_vec=None
        ):

        self.D_MAX = _check.pos_int(D_MAX,'D_MAX',ParameterFormatError)
        self.NUM_CHILDREN = _check.pos_int(NUM_CHILDREN,'NUM_CHILDREN',ParameterFormatError)
        if K is not None:
            self.K = _check.pos_int(K,'K',ParameterFormatError)
            if h0_k_prob_vec is not None:
                self.h0_k_prob_vec = _check.float_vec_sum_1(h0_k_prob_vec,'h0_k_prob_vec',ParameterFormatError)
            else:
                self.h0_k_prob_vec = np.ones(self.K) / self.K
        elif h0_k_prob_vec is not None:
            self.h0_k_prob_vec = _check.float_vec_sum_1(h0_k_prob_vec,'h0_k_prob_vec',ParameterFormatError)
            self.K = self.h0_k_prob_vec.shape[0]
        else:
            self.K = 3
            self.h0_k_prob_vec = np.ones(self.K) / self.K
        if self.h0_k_prob_vec.shape[0] != self.K:
            raise(ParameterFormatError(
                "K and dimension of h0_k_prob_vec must be the same."
            ))

        self.h0_g = _check.float_in_closed01(h0_g,'h0_g',ParameterFormatError)
        self.SubModel = SubModel
        self.sub_h0_params = sub_h0_params
        self.h0_metatree_list = h0_metatree_list
        if h0_metatree_prob_vec is not None:
            self.h0_metatree_prob_vec = _check.float_vec_sum_1(h0_metatree_prob_vec,'h0_metatree_prob_vec',ParameterFormatError)
            if self.h0_metatree_prob_vec.shape[0] != len(self.h0_metatree_list):
                raise(ParameterFormatError(
                    "Length of h0_metatree_list and dimension of h0_metatree_prob_vec must be the same."
                ))
        else:
            metatree_num = len(self.h0_metatree_list)
            self.h0_metatree_prob_vec = np.ones(metatree_num) / metatree_num

        self._tmp_x = np.zeros(self.K,dtype=int)
        self.reset_hn_params()

    def set_h0_params(self,
        h0_k_prob_vec = None,
        h0_g=None,
        sub_h0_params=None,
        h0_metatree_list=None,
        h0_metatree_prob_vec=None
        ):
        """Set initial values of the hyperparameter of the posterior distribution.

        Parameters
        ----------
        h0_k_prob_vec : _type_
            _description_
        h0_g : _type_
            _description_
        sub_h0_params : _type_
            _description_
        h0_metatree_list : _type_
            _description_
        h0_metatree_prob_vec : _type_
            _description_
        """
        if h0_k_prob_vec is not None:
            self.h0_k_prob_vec = _check.float_vec_sum_1(h0_k_prob_vec,'h0_k_prob_vec',ParameterFormatError)
            if self.h0_k_prob_vec.shape[0] != self.K:
                raise(ParameterFormatError(
                    "K and dimension of h0_k_prob_vec must be the same. "
                    +"If you want to change K, you should re-construct a new instance of GenModel."
                ))

        if h0_g is not None:
            self.h0_g = _check.float_in_closed01(h0_g,'h0_g',ParameterFormatError)

        if sub_h0_params is not None:
            self.sub_h0_params = sub_h0_params

        if h0_metatree_list is not None:
            self.h0_metatree_list = h0_metatree_list
            if h0_metatree_prob_vec is not None:
                self.h0_metatree_prob_vec = _check.float_vec_sum_1(h0_metatree_prob_vec,'h0_metatree_prob_vec',ParameterFormatError)
            else:
                metatree_num = len(self.h0_metatree_list)
                self.h0_metatree_prob_vec = np.ones(metatree_num) / metatree_num
        elif h0_metatree_prob_vec is not None:
            self.h0_metatree_prob_vec = _check.float_vec_sum_1(h0_metatree_prob_vec,'h0_metatree_prob_vec',ParameterFormatError)

        if type(self.h0_metatree_prob_vec) is np.ndarray:             
            if self.h0_metatree_prob_vec.shape[0] != len(self.h0_metatree_list):
                raise(ParameterFormatError(
                    "Length of h0_metatree_list and dimension of h0_metatree_prob_vec must be the same."
                ))
        else:
            if len(self.h0_metatree_list) > 0:
                raise(ParameterFormatError(
                    "Length of h0_metatree_list must be zero when self.h0_metatree_prob_vec is None."
                ))

        self.reset_hn_params()

    def get_h0_params(self):
        """Get the initial values of the hyperparameters of the posterior distribution.

        Returns
        -------
        h0_params : dict of {str: float, numpy.ndarray}
            * ``"h0_k_prob_vec"`` : the value of ``self.h0_k_prob_vec``
            * ``"h0_g"`` : the value of ``self.h0_g``
            * ``"sub_h0_params"`` : the value of ``self.sub_h0_params``
            * ``"h0_metatree_list"`` : the value of ``self.h0_metatree_list``
            * ``"h0_metatree_prob_vec"`` : the value of ``self.h0_metatree_prob_vec``
        """
        return {"h0_k_prob_vec":self.h0_k_prob_vec,
                "h0_g":self.h0_g, 
                "sub_h0_params":self.sub_h0_params, 
                "h0_metatree_list":self.h0_metatree_list,
                "h0_metatree_prob_vec":self.h0_metatree_prob_vec}
    
    def set_hn_params(self,
        hn_k_prob_vec = None,
        hn_g=None,
        sub_hn_params=None,
        hn_metatree_list=None,
        hn_metatree_prob_vec=None
        ):
        """Set updated values of the hyperparameter of the posterior distribution.

        Parameters
        ----------
        hn_k_prob_vec : _type_
            _description_
        hn_g : _type_
            _description_
        sub_hn_params : _type_
            _description_
        hn_metatree_list : _type_
            _description_
        hn_metatree_prob_vec : _type_
            _description_
        """
        if hn_k_prob_vec is not None:
            self.hn_k_prob_vec = _check.float_vec_sum_1(hn_k_prob_vec,'hn_k_prob_vec',ParameterFormatError)
            if self.hn_k_prob_vec.shape[0] != self.K:
                raise(ParameterFormatError(
                    "K and dimension of hn_k_prob_vec must be the same."
                    +"If you want to change K, you should re-construct a new instance of GenModel."
                ))

        if hn_g is not None:
            self.hn_g = _check.float_in_closed01(hn_g,'hn_g',ParameterFormatError)

        if sub_hn_params is not None:
            self.sub_hn_params = sub_hn_params

        if hn_metatree_list is not None:
            self.hn_metatree_list = hn_metatree_list
            if hn_metatree_prob_vec is not None:
                self.hn_metatree_prob_vec = _check.float_vec_sum_1(hn_metatree_prob_vec,'hn_metatree_prob_vec',ParameterFormatError)
            else:
                metatree_num = len(self.hn_metatree_list)
                self.hn_metatree_prob_vec = np.ones(metatree_num) / metatree_num
        elif hn_metatree_prob_vec is not None:
            self.hn_metatree_prob_vec = _check.float_vec_sum_1(hn_metatree_prob_vec,'hn_metatree_prob_vec',ParameterFormatError)

        if type(self.hn_metatree_prob_vec) is np.ndarray:             
            if self.hn_metatree_prob_vec.shape[0] != len(self.hn_metatree_list):
                raise(ParameterFormatError(
                    "Length of hn_metatree_list and dimension of hn_metatree_prob_vec must be the same."
                ))
        else:
            if len(self.hn_metatree_list) > 0:
                raise(ParameterFormatError(
                    "Length of hn_metatree_list must be zero when self.hn_metatree_prob_vec is None."
                ))

        self.calc_pred_dist(np.zeros(self.K))

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: float, numpy.ndarray}
            * ``"hn_k_prob_vec"`` : the value of ``self.hn_k_prob_vec``
            * ``"hn_g"`` : the value of ``self.hn_g``
            * ``"sub_hn_params"`` : the value of ``self.sub_hn_params``
            * ``"hn_metatree_list"`` : the value of ``self.hn_metatree_list``
            * ``"hn_metatree_prob_vec"`` : the value of ``self.hn_metatree_prob_vec``
        """
        return {"hn_k_prob_vec":self.hn_k_prob_vec,
                "hn_g":self.hn_g, 
                "sub_hn_params":self.sub_hn_params, 
                "hn_metatree_list":self.hn_metatree_list,
                "hn_metatree_prob_vec":self.hn_metatree_prob_vec}
    
    def reset_hn_params(self):
        """Reset the hyperparameters of the posterior distribution to their initial values.
        
        They are reset to `self.h0_k_prob_vec`, `self.h0_g`, `self.sub_h0_params`, 
        `self.h0_metatree_list` and `self.h0_metatree_prob_vec`.
        Note that the parameters of the predictive distribution are also calculated from them.
        """
        self.hn_k_prob_vec = np.copy(self.h0_k_prob_vec)
        self.hn_g = np.copy(self.h0_g)
        self.sub_hn_params = copy.copy(self.sub_h0_params)
        self.hn_metatree_list = copy.copy(self.h0_metatree_list)
        self.hn_metatree_prob_vec = copy.copy(self.h0_metatree_prob_vec)

        self.calc_pred_dist(np.zeros(self.K))
    
    def overwrite_h0_params(self):
        """Overwrite the initial values of the hyperparameters of the posterior distribution by the learned values.
        
        They are overwitten by `self.hn_k_prob_vec`, `self.hn_g`, `self.sub_hn_params`, 
        `self.hn_metatree_list` and `self.hn_metatree_prob_vec`.
        Note that the parameters of the predictive distribution are also calculated from them.
        """
        self.h0_k_prob_vec = np.copy(self.hn_k_prob_vec)
        self.h0_g = np.copy(self.hn_g)
        self.sub_h0_params = copy.copy(self.sub_hn_params)
        self.h0_metatree_list = copy.copy(self.hn_metatree_list)
        self.h0_metatree_prob_vec = np.copy(self.hn_metatree_prob_vec)

        self.calc_pred_dist(np.zeros(self.K))

    def _copy_tree_from_sklearn_tree(self,new_node, original_tree,node_id):
        if original_tree.children_left[node_id] != sklearn_tree._tree.TREE_LEAF:  # 内部ノード
            new_node.k = original_tree.feature[node_id]
            new_node.children[0] = _LearnNode(depth=new_node.depth+1,
                                              NUM_CHILDREN=2,
                                              hn_g=self.h0_g,
                                              k=None,
                                              sub_model=self.SubModel(**self.sub_h0_params))
            self._copy_tree_from_sklearn_tree(new_node.children[0],original_tree,original_tree.children_left[node_id])
            new_node.children[1] = _LearnNode(depth=new_node.depth+1,
                                              NUM_CHILDREN=2,
                                              hn_g=self.h0_g,
                                              k=None,
                                              sub_model=self.SubModel(**self.sub_h0_params))
            self._copy_tree_from_sklearn_tree(new_node.children[1],original_tree,original_tree.children_right[node_id])
        else:
            new_node.hn_g = 0.0
            new_node.leaf = True

    def _update_posterior_leaf(self,node,x,y):
            try:
                node.sub_model.calc_pred_dist(x)
            except:
                node.sub_model.calc_pred_dist()
            pred_dist = node.sub_model.make_prediction(loss='KL')

            try:
                node.sub_model.update_posterior(x,y)
            except:
                node.sub_model.update_posterior(y)

            if type(pred_dist) is np.ndarray:
                return pred_dist[y]
            elif hasattr(pred_dist,'pdf'):
                return pred_dist.pdf(y)
            elif hasattr(pred_dist,'pmf'):
                return pred_dist.pmf(y)
            else:
                warnings.warn("Marginal likelyhood could not be calculated.", ResultWarning)
                return 0.0

    def _update_posterior_recursion(self,node,x,y):
        if node.leaf == False:  # 内部ノード
            tmp1 = self._update_posterior_recursion(node.children[x[node.k]],x,y)
            tmp2 = (1 - node.hn_g) * self._update_posterior_leaf(node,x,y) + node.hn_g * tmp1
            node.hn_g = node.hn_g * tmp1 / tmp2
            return tmp2
        else:  # 葉ノード
            return self._update_posterior_leaf(node,x,y)

    def _compare_metatree_recursion(self,node1,node2):
        if node1.leaf == True and node2.leaf == True:
            return True
        elif node1.k == node2.k:
            for i in range(self.NUM_CHILDREN):
                if self._compare_metatree_recursion(node1.children[i],node2.children[i]) == False:
                    return False
            return True
        else:
            return False
    
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

    def _MTRF(self,x,y,n_estimators=100,**kwargs):
        """make metatrees

        Parameters
        ----------
        x : numpy ndarray
            values of explanatory variables whose dtype must be int
        y : numpy ndarray
            values of objective variable whose dtype may be int or float
        n_estimators : int, optional
            number of trees in sklearn.RandomForestClassifier, by default 100

        Returns
        -------
        metatree_list : list of metatree._LearnNode
            Each element is a root node of metatree.
        metatree_prob_vec : numpy ndarray
        """
        if self.NUM_CHILDREN != 2:
            raise(ParameterFormatError("MTRF is supported only when NUM_CHILDREN == 2."))
        if self.SubModel == bernoulli.LearnModel:
            randomforest = RandomForestClassifier(n_estimators=n_estimators,max_depth=self.D_MAX)
        if self.SubModel == normal.LearnModel:
            randomforest = RandomForestRegressor(n_estimators=n_estimators,max_depth=self.D_MAX)
        randomforest.fit(x,y)
        tmp_metatree_list = [_LearnNode(0,2,self.h0_g,None,self.SubModel(**self.sub_h0_params)) for i in range(n_estimators)]
        tmp_metatree_prob_vec = np.ones(n_estimators) / n_estimators
        for i in range(n_estimators):
            self._copy_tree_from_sklearn_tree(tmp_metatree_list[i],randomforest.estimators_[i].tree_, 0)

        tmp_metatree_list,tmp_metatree_prob_vec = self._marge_metatrees(tmp_metatree_list,tmp_metatree_prob_vec)

        log_metatree_posteriors = np.log(tmp_metatree_prob_vec)
        for i,metatree in enumerate(tmp_metatree_list):
            for j in range(x.shape[0]):
                log_metatree_posteriors[i] += np.log(self._update_posterior_recursion(metatree,x[j],y[j]))
        tmp_metatree_prob_vec[:] = np.exp(log_metatree_posteriors - log_metatree_posteriors.max())
        tmp_metatree_prob_vec[:] /= tmp_metatree_prob_vec.sum()
        return tmp_metatree_list,tmp_metatree_prob_vec

    def _given_MT(self,x,y):
        """make metatrees

        Parameters
        ----------
        x : numpy ndarray
            values of explanatory variables whose dtype must be int
        y : numpy ndarray
            values of objective variable whose dtype may be int or float

        Returns
        -------
        metatree_list : list of metatree._LearnNode
            Each element is a root node of metatree.
        metatree_prob_vec : numpy ndarray
        """
        log_metatree_posteriors = np.log(self.hn_metatree_prob_vec)
        for i,metatree in enumerate(self.hn_metatree_list):
            for j in range(x.shape[0]):
                log_metatree_posteriors[i] += np.log(self._update_posterior_recursion(metatree,x[j],y[j]))
        self.hn_metatree_prob_vec[:] = np.exp(log_metatree_posteriors - log_metatree_posteriors.max())
        self.hn_metatree_prob_vec[:] /= self.hn_metatree_prob_vec.sum()
        return self.hn_metatree_list,self.hn_metatree_prob_vec

    def update_posterior(self,x,y,alg_type='MTRF',**kwargs):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy ndarray
            values of explanatory variables whose dtype must be int
        y : numpy ndarray
            values of objective variable whose dtype may be int or float
        alg_type : {'MTRF'}, optional
            type of algorithm, by default 'MTRF'
        **kwargs : dict, optional
            optional parameters of algorithms, by default {}
        """
        _check.int_vecs(x,'x',DataFormatError)
        if x.shape[-1] != self.K:
            raise(DataFormatError(f"x.shape[-1] must equal to K:{self.K}"))
        if x.max() >= self.NUM_CHILDREN:
            raise(DataFormatError(f"x.max() must smaller than NUM_CHILDREN:{self.NUM_CHILDREN}"))

        if self.SubModel == bernoulli.LearnModel: # acceptable data type of y depends on SubModel
            _check.ints_of_01(y,'y',DataFormatError)
                
        if type(y) is np.ndarray:
            if x.shape[:-1] != y.shape: 
                raise(DataFormatError(f"x.shape[:-1] and y.shape must be same."))
        elif x.shape[:-1] != ():
            raise(DataFormatError(f"If y is a scaler, x.shape[:-1] must be the empty tuple ()."))

        x = x.reshape(-1,self.K)
        y = np.ravel(y)

        if alg_type == 'MTRF':
            self.hn_metatree_list, self.hn_metatree_prob_vec = self._MTRF(x,y,**kwargs)
        elif alg_type == 'given_MT':
            self.hn_metatree_list, self.hn_metatree_prob_vec = self._given_MT(x,y)

    def _map_recursion(self,node):
        if node.leaf:
            node.map_leaf = True
            return 1.0
        else:
            tmp1 = 1.0-node.hn_g
            tmp_vec = np.empty(self.NUM_CHILDREN)
            for i in range(self.NUM_CHILDREN):
                tmp_vec[i] = self._map_recursion(node.children[i])
            if tmp1 > node.hn_g*tmp_vec.prod():
                node.map_leaf = True
                return tmp1
            else:
                node.map_leaf = False
                return node.hn_g*tmp_vec.prod()

    def _copy_map_tree_recursion(self,copyed_node,original_node):
        copyed_node.hn_g = original_node.hn_g
        if original_node.map_leaf == False:
            copyed_node.k = original_node.k
            for i in range(self.NUM_CHILDREN):
                copyed_node.children[i] = _LearnNode(copyed_node.depth+1,self.NUM_CHILDREN)
                self._copy_map_tree_recursion(copyed_node.children[i],original_node.children[i])
        else:
            copyed_node.sub_model = copy.deepcopy(original_node.sub_model)
            copyed_node.leaf = True

    def estimate_params(self,loss="0-1",visualize=True,filename=None,format=None):
        """Estimate the parameter of the stochastic data generative model under the given criterion.
        """

        if loss == "0-1":
            map_index = 0
            map_prob = 0.0
            for i,metatree in enumerate(self.hn_metatree_list):
                prob = self.hn_metatree_prob_vec[i] * self._map_recursion(metatree)
                if prob > map_prob:
                    map_index = i
                    map_prob = prob
            map_root = _LearnNode(0,self.NUM_CHILDREN)
            self._copy_map_tree_recursion(map_root,self.hn_metatree_list[map_index])
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
    
    def _visualize_model_recursion(self,tree_graph,node,node_id,parent_id,sibling_num,p_v):
        """Visualize the stochastic data generative model and generated samples.
        """
        tmp_id = node_id
        tmp_p_v = p_v
        
        # add node information
        label_string = f'k={node.k}\\lhn_g={node.hn_g:.2f}\\lp_v={tmp_p_v:.2f}\\lsub_params={{'
        if node.sub_model is not None:
            try:
                sub_params = node.sub_model.estimate_params(loss='0-1',output='dict')
            except:
                sub_params = node.sub_model.estimate_params(output='dict')
            
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
            tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        if node.leaf != True:
            for i in range(self.NUM_CHILDREN):
                node_id = self._visualize_model_recursion(tree_graph,node.children[i],node_id+1,tmp_id,i,tmp_p_v*node.hn_g)
        
        return node_id

    def visualize_posterior(self,filename=None,format=None):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import metatree
        >>> gen_model = metatree.GenModel()
        >>> x,y = gen_model.gen_sample(100)
        >>> learn_model = metatree.LearnModel()
        >>> learn_model.update_posterior(x,y)
        >>> learn_model.visualize_posterior()

        .. image:: ./images/metatree_posterior.png
        """
        MAP_index = np.argmax(self.hn_metatree_prob_vec)
        print(f'MAP probability of metatree:{self.hn_metatree_prob_vec[MAP_index]}')
        try:
            import graphviz
            tree_graph = graphviz.Digraph(filename=filename,format=format)
            tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
            self._visualize_model_recursion(tree_graph, self.hn_metatree_list[MAP_index], 0, None, None, 1.0)        
            # コンソール上で表示できるようにした方がいいかもしれない．
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
    
    def _calc_pred_dist_leaf(self,node,x):
            try:
                node.sub_model.calc_pred_dist(x)
            except:
                node.sub_model.calc_pred_dist()

    def _calc_pred_dist_recursion(self,node,x):
        if node.leaf == False:  # 内部ノード
            self._calc_pred_dist_recursion(node.children[x[node.k]],x)
        else:  # 葉ノード
            return self._calc_pred_dist_leaf(node,x)

    def calc_pred_dist(self,x):
        """Calculate the parameters of the predictive distribution."""
        self._tmp_x = np.copy(x)
        for root in self.hn_metatree_list:
            self._calc_pred_dist_recursion(root,self._tmp_x)

    def _make_prediction_recursion_squared(self,node):
            if node.leaf == False:  # 内部ノード
                return ((1 - node.hn_g) * node.sub_model.make_prediction(loss='squared')
                        + node.hn_g * self._make_prediction_recursion_squared(node.children[self._tmp_x[node.k]]))
            else:  # 葉ノード
                return node.sub_model.make_prediction(loss='squared')

    def _make_prediction_leaf_01(self,node):
        mode = node.sub_model.make_prediction(loss='0-1')
        pred_dist = node.sub_model.make_prediction(loss='KL')
        if type(pred_dist) is np.ndarray:
            mode_prob = pred_dist[mode]
        elif hasattr(pred_dist,'pdf'):
            mode_prob = pred_dist.pdf(mode)
        elif hasattr(pred_dist,'pmf'):
            mode_prob = pred_dist.pmf(mode)
        else:
            mode_prob = None
        return mode, mode_prob

    def _make_prediction_recursion_01(self,node):
        if node.leaf == False:  # 内部ノード
            mode1,mode_prob1 = self._make_prediction_leaf_01(node)
            mode2,mode_prob2 = self._make_prediction_recursion_01(node.children[self._tmp_x[node.k]])
            if (1 - node.hn_g) * mode_prob1 > node.hn_g * mode_prob2:
                return mode1,mode_prob1
            else:
                return mode2,mode_prob2
        else:  # 葉ノード
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
        Predicted_value : {float, numpy.ndarray}
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
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"0-1\".
            This function supports \"squared\", \"0-1\", and \"KL\".

        Returns
        -------
        Predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
        """
        _check.nonneg_int_vec(x,'x',DataFormatError)
        if x.shape[-1] != self.K:
            raise(DataFormatError(f"x.shape[-1] must equal to K:{self.K}"))
        if x.max() >= self.NUM_CHILDREN:
            raise(DataFormatError(f"x.max() must smaller than NUM_CHILDREN:{self.NUM_CHILDREN}"))
        self.calc_pred_dist(x)
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(x,y,alg_type='given_MT')
        return prediction
