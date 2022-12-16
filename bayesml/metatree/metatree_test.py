from bayesml.metatree import GenModel
from bayesml.metatree import LearnModel
from bayesml import poisson
from bayesml import bernoulli
import numpy as np
import copy

gen_model = GenModel(4,3,2,h_g=0.75)
gen_model.gen_params()
gen_model.visualize_model('tree.pdf')
x,y = gen_model.gen_sample(1000)

learn_model = LearnModel(4,3,2)
learn_model.update_posterior(x,y,n_estimators=1)
learn_model.visualize_posterior('tree2.pdf')

learn_model.calc_pred_dist(np.zeros(4,dtype=int))
print(learn_model.make_prediction(loss='squared'))

learn_model.calc_pred_dist(np.ones(4,dtype=int))
print(learn_model.make_prediction(loss='squared'))
