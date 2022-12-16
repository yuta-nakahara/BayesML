from bayesml import metatree
from bayesml import normal
import numpy as np
import copy

gen_model = metatree.GenModel(4,3,2,h_g=0.75,SubModel=normal)
gen_model.gen_params()
gen_model.visualize_model('tree.pdf')
x,y = gen_model.gen_sample(1000)

learn_model = metatree.LearnModel(4,3,2,h0_g=0.75,SubModel=normal)
learn_model.update_posterior(x,y)
learn_model.visualize_posterior('tree2.pdf')
params = learn_model.estimate_params(filename='tree3.pdf')

gen_model2 = metatree.GenModel(4,3,2,h_g=0.1,SubModel=normal)
gen_model2.visualize_model('tree4.pdf')
gen_model2.set_params(params)
# gen_model2.gen_params()
gen_model2.visualize_model('tree5.pdf')
