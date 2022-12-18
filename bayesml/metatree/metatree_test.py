from bayesml import metatree
from bayesml import normal
from bayesml import poisson
from bayesml import bernoulli
import numpy as np
import copy

dim_continuous = 0
dim_categorical = 2

gen_model = metatree.GenModel(
    c_dim_continuous=dim_continuous,
    c_dim_categorical=dim_categorical,
    h_g=0.75,
    SubModel=normal,
)
    # sub_h_params={'h_alpha':0.1,'h_beta':0.1})
gen_model.gen_params(threshold_type='random')
gen_model.visualize_model(filename='tree.pdf')

x_continuous,x_categorical,y = gen_model.gen_sample(100)

learn_model = metatree.LearnModel(
    c_dim_continuous=dim_continuous,
    c_dim_categorical=dim_categorical,
    c_num_children_vec=2,
    SubModel=normal,
)
    # sub_h0_params={'h0_alpha':0.1,'h0_beta':0.1})
learn_model.update_posterior(x_continuous,x_categorical,y)
learn_model.calc_pred_dist(
    np.zeros(dim_continuous,dtype=float),
    np.zeros(dim_categorical,dtype=int))
learn_model.visualize_posterior(filename='tree2.pdf')
learn_model.estimate_params(filename='tree3.pdf')
