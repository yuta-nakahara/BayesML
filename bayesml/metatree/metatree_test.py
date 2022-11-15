from bayesml import metatree
from bayesml import bernoulli
from bayesml import normal
import numpy as np
import time

# GenModel
model = metatree.GenModel(h_g=0.75,SubModel=bernoulli.GenModel)
# model = metatree.GenModel(h_g=0.75,SubModel=normal.GenModel,sub_h_params={'h_kappa':0.001})
model.gen_params()
model.visualize_model()
x,y = model.gen_sample(sample_size=1000)

# LearnModel
# dummynode = metatree._LearnNode(depth=0,k=0,sub_model=bernoulli.LearnModel(),hn_g=1.0)
# dummynode.leaf = True
# learn_model = metatree.LearnModel(h0_g=0.5,SubModel=bernoulli.LearnModel,h0_metatree_list=[dummynode])
learn_model = metatree.LearnModel(c_k=3,h0_g=0.5,SubModel=bernoulli.LearnModel)
# learn_model = metatree.LearnModel(SubModel=normal.LearnModel,sub_h0_params={'h0_kappa':0.001})
learn_model.update_posterior(x,y)
# for i in range(10):
#     learn_model.pred_and_update(x[i],y[i])
learn_model.visualize_posterior(filename='posterior.gv')
# learn_model.estimate_params(filename='map_model.gv')

# learn_model.calc_pred_dist(np.array([0,1,1]))
# print(learn_model.make_prediction(loss='squared'))
# print(learn_model.make_prediction(loss='0-1'))