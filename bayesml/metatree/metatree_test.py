from bayesml.metatree import GenModel
from bayesml.metatree import LearnModel
from bayesml import poisson
from bayesml import bernoulli
import numpy as np
import copy

gen_model1 = GenModel(4,3,2,h_g=0.75,seed=0)
gen_model1.gen_params()
gen_model1.visualize_model('tree.pdf')
x1,y1 = gen_model1.gen_sample(10)

gen_model2 = GenModel(4,3,2,h_g=0.75,seed=0)
gen_model2.gen_params()
gen_model2.visualize_model('tree2.pdf')
x2,y2 = gen_model2.gen_sample(10)

print(x1-x2)
print(y1-y2)
