from bayesml import metatree
from bayesml import normal
from bayesml import poisson
from bayesml import bernoulli
import numpy as np
import copy

model = metatree.GenModel(
    c_dim_continuous=1,
    c_dim_categorical=1)
model.gen_params(threshold_type='random')
model.visualize_model(format='png')

# params = gen_model.get_params()
# constants = gen_model.get_constants()

# gen_model2 = GenModel(**constants)
# gen_model2.set_params(**params)
# gen_model.visualize_model('tree2.pdf')

# x_continuous,x_categorical,y = gen_model.gen_sample(3)
# print(x_continuous)
# print(x_categorical)
# print(y)
