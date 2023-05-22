# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Document Author
# Wenbin Yu <ywb827748728@163.com>
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("../bernoulli")
sys.path.append("..")

import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

import base
from _exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning
import _bernoulli

# print(os.getcwd())

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
    k : int
            a positive integer, by default None
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
        SubModel=_bernoulli.GenModel,
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
        Y = np.empty(sample_size)
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

if __name__ == '__main__':
    model = GenModel(h_g=0.75)
    model.gen_params()
    model.visualize_model()

# class LearnModel(base.Posterior,base.PredictiveMixin):
#     def __init__(self,h0_alpha=0.5,h0_beta=0.5):
#         # 例外処理は後日追記
#         self.h0_alpha = h0_alpha
#         self.h0_beta = h0_beta
#         self.hn_alpha = self.h0_alpha
#         self.hn_beta = self.h0_beta
#         self.p_alpha = self.hn_alpha
#         self.p_beta = self.hn_beta
    
#     def set_h0_params(self,h0_alpha,h0_beta):
#         # 例外処理は後日追記
#         self.h0_alpha = h0_alpha
#         self.h0_beta = h0_beta
#         self.hn_alpha = self.h0_alpha
#         self.hn_beta = self.h0_beta
#         self.p_alpha = self.hn_alpha
#         self.p_beta = self.hn_beta

#     def get_h0_params(self):
#         return {"h0_alpha":self.h0_alpha, "h0_beta":self.h0_beta}
    
#     def save_h0_params(self,filename):
#         # 例外処理は後日追記
#         np.savez_compressed(filename,h0_alpha=self.h0_alpha,h0_beta=self.h0_beta)

#     def load_h0_params(self,filename):
#         # 例外処理は後日追記
#         h0_params = np.load(filename)
#         self.set_h0_params(h0_params["h0_alpha"], h0_params["h0_beta"])

#     def get_hn_params(self):
#         return {"hn_alpha":self.hn_alpha, "hn_beta":self.hn_beta}
    
#     def save_hn_params(self,filename):
#         # 例外処理は後日追記
#         np.savez_compressed(filename,hn_alpha=self.hn_alpha,hn_beta=self.hn_beta)

#     def update_posterior(self,X):
#         # 例外処理は後日追記
#         self.hn_alpha += np.sum(X==1)
#         self.hn_beta += np.sum(X==0)

#     def estimate_params(self,loss="squared"):
#         # 例外処理は後日追記
#         if loss == "squared":
#             return self.hn_alpha / (self.hn_alpha + self.hn_beta)
#         elif loss == "0-1":
#             if self.hn_alpha > 1.0 and self.hn_beta > 1.0:
#                 return (self.hn_alpha - 1.0) / (self.hn_alpha + self.hn_beta - 2.0)
#             elif self.hn_alpha > 1.0:
#                 return 1.0
#             elif self.hn_beta > 1.0:
#                 return 0.0
#             else:
#                 print("MAP estimate doesn't exist.")
#                 return None
#         elif loss == "abs":
#             return ss_beta.median(self.hn_alpha,self.hn_beta)
#         elif loss == "KL":
#             return ss_beta(self.hn_alpha,self.hn_beta)
#         else:
#             print("Unsupported loss function!")
#             print("This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".")
#             return None
    
#     def estimate_interval(self,confidence=0.95):
#         # 例外処理は後日追記
#         return ss_beta.interval(confidence,self.hn_alpha,self.hn_beta)
    
#     def visualize_posterior(self):
#         p_range = np.linspace(0,1,100,endpoint=False)
#         fig, ax = plt.subplots()
#         ax.plot(p_range,self.estimate_params(loss="KL").pdf(p_range))
#         ax.set_xlabel("p")
#         plt.show()
    
#     def get_p_params(self):
#         return {"p_alpha":self.p_alpha, "p_beta":self.p_beta}
    
#     def save_p_params(self,filename):
#         # 例外処理は後日追記
#         np.savez_compressed(filename,p_alpha=self.p_alpha,p_beta=self.p_beta)
    
#     def calc_pred_dist(self):
#         self.p_alpha = self.hn_alpha
#         self.p_beta = self.hn_beta

#     def make_prediction(self,loss="squared"):
#         # 例外処理は後日追記
#         if loss == "squared":
#             return self.p_alpha / (self.p_alpha + self.p_beta)
#         elif loss == "0-1" or loss == "abs":
#             if self.p_alpha > self.p_beta:
#                 return 1
#             else:
#                 return 0
#         elif loss == "KL":
#             return np.array((self.p_beta / (self.p_alpha + self.p_beta),
#                              self.p_alpha / (self.p_alpha + self.p_beta)))
#             # return ss_betabinom(1,self.p_alpha,self.p_beta)
#         else:
#             print("Unsupported loss function!")
#             print("This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".")
#             return None    

#     def pred_and_update(self,x,loss="squared"):
#         # 例外処理は後日追記
#         self.calc_pred_dist()
#         prediction = self.make_prediction(loss=loss)
#         self.update_posterior(x)
#         return prediction

