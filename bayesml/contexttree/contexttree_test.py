from bayesml.contexttree import GenModel
from bayesml.contexttree import LearnModel
import numpy as np

gen_model = GenModel(2,3,h_g=0.7,h_beta_vec=1.0)
gen_model.gen_params()
gen_model.visualize_model(filename='tree1.pdf')

x = gen_model.gen_sample(1000)

learn_model = LearnModel(2,4)
learn_model.visualize_posterior(filename='tree2.pdf')
learn_model.reset_hn_params()
learn_model.visualize_posterior(filename='tree3.pdf')
learn_model.update_posterior(x)
learn_model.visualize_posterior(filename='tree4.pdf')
learn_model.estimate_params()