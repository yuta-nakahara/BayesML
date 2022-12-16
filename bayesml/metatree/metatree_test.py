from bayesml import metatree
from bayesml import normal
import numpy as np
import copy

gen_model = metatree.GenModel(4,3,2,h_g=0.75)
gen_model.gen_params()
gen_model.visualize_model('tree.pdf')
x,y = gen_model.gen_sample(1000)

learn_model = metatree.LearnModel(4,3,2,h0_g=0.75)
learn_model.update_posterior(x,y,n_estimators=1)
learn_model.visualize_posterior('tree2.pdf')

gen_model2 = metatree.GenModel(4,3,2,h_g=0.1)
gen_model2.visualize_model('tree3.pdf')
gen_model2.set_h_params(*learn_model.get_hn_params().values())
gen_model2.gen_params()
gen_model2.visualize_model('tree4.pdf')
