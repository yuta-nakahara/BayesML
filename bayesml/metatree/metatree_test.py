from bayesml import metatree
from bayesml import normal
from bayesml import poisson
from bayesml import bernoulli
import numpy as np
import copy

gen_model = metatree.GenModel(
    c_dim_continuous=1,
    c_dim_categorical=1,
    h_g=0.75)
gen_model.gen_params(threshold_type='random')
gen_model.visualize_model(filename='tree.pdf')
params = gen_model.get_params()

learn_model = metatree.LearnModel(
    c_dim_continuous=2,
    c_dim_categorical=0,
    c_num_children_vec=2)
print(learn_model._root_k_candidates)
print(learn_model.get_h0_params())
learn_model.set_h0_params(h0_k_weight_vec=np.random.dirichlet(np.ones(learn_model.c_dim_features)))
print(learn_model.get_h0_params())
learn_model.set_h0_params(h0_metatree_list=[params['root']])
learn_model.visualize_posterior(filename='tree2.pdf')
# print(learn_model.get_h0_params())
# learn_model.set_h0_params(h0_g=0.8)
# print(learn_model.get_h0_params())
# learn_model.set_h0_params(sub_h0_params={'h0_alpha':2})
# print(learn_model.get_h0_params())
# print(learn_model.get_hn_params())
