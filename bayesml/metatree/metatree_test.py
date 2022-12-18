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
    sub_h_params={'h_alpha':0.1,'h_beta':0.1})
gen_model.gen_params(threshold_type='even')
gen_model.visualize_model(filename='tree.pdf')

x_continuous,x_categorical,y = gen_model.gen_sample(200)
x_continuous_test,x_categorical_test,y_test = gen_model.gen_sample(10)

learn_model = metatree.LearnModel(
    c_dim_continuous=dim_continuous,
    c_dim_categorical=dim_categorical,
    c_num_children_vec=2,
    sub_h0_params={'h0_alpha':0.1,'h0_beta':0.1})
learn_model.update_posterior(x_continuous,x_categorical,y)
for i in range(10):
    print(learn_model.pred_and_update(x_continuous_test[i],x_categorical_test[i],y_test[i],loss='0-1'))