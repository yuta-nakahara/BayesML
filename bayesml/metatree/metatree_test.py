from bayesml import metatree
from bayesml import normal
from bayesml import poisson
from bayesml import bernoulli
import numpy as np
import copy

gen_model = metatree.GenModel(
    c_dim_continuous=2,
    c_dim_categorical=0,
    h_g=0.75,
    sub_h_params={'h_alpha':0.1,'h_beta':0.1})
gen_model.gen_params(threshold_type='random')
# gen_model.visualize_model(filename='tree.pdf')

x_continuous,x_categorical,y = gen_model.gen_sample(1000)

learn_model = metatree.LearnModel(
    c_dim_continuous=2,
    c_dim_categorical=0,
    c_num_children_vec=2,
    sub_h0_params={'h0_alpha':0.1,'h0_beta':0.1})
learn_model.update_posterior(x_continuous,x_categorical,y,n_estimators=1)
learn_model.visualize_posterior(filename='tree2.pdf')
