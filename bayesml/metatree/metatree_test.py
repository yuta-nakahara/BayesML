from bayesml import metatree
from bayesml import normal
from bayesml import poisson
from bayesml import bernoulli
from bayesml import exponential
import numpy as np
import time

dim_continuous = 2
dim_categorical = 0

gen_model = metatree.GenModel(
    c_dim_continuous=dim_continuous,
    c_dim_categorical=dim_categorical,
    h_g=1.0,
    SubModel=normal,
    sub_h_params={'h_kappa':0.1})
    # sub_h_params={'h_alpha':0.1,'h_beta':0.1})
gen_model.gen_params(threshold_type='even')
# gen_model.visualize_model(filename='tree.pdf')

x_continuous,x_categorical,y = gen_model.gen_sample(200)

learn_model = metatree.LearnModel(
    c_dim_continuous=dim_continuous,
    c_dim_categorical=dim_categorical,
    c_num_children_vec=2,
    SubModel=normal,
    sub_h0_params={'h0_kappa':0.1})
    # sub_h0_params={'h0_alpha':0.1,'h0_beta':0.1})

start = time.time()
learn_model.update_posterior(x_continuous,x_categorical,y)
end = time.time()

print(end-start)
