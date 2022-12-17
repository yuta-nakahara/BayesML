from bayesml.metatree import GenModel
from bayesml.metatree import LearnModel
from bayesml import normal
from bayesml import poisson
from bayesml import bernoulli
import numpy as np
import copy

gen_model = GenModel(1,1,2,h_g=0.75)
gen_model.gen_params(threshold_type='random')
gen_model.visualize_model('tree.pdf')

params = gen_model.get_params()
constants = gen_model.get_constants()

gen_model2 = GenModel(**constants)
gen_model2.set_params(**params)
gen_model.visualize_model('tree2.pdf')

# x_continuous,x_categorical,y = gen_model.gen_sample(3)
# print(x_continuous)
# print(x_categorical)
# print(y)
