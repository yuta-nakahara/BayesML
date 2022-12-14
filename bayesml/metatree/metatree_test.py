from bayesml import metatree
from bayesml import normal
import numpy as np
import copy

gen_model = metatree.GenModel(3,3,h_g=0.7)
gen_model.gen_params()
gen_model.visualize_model(filename='tree.pdf')
params1 = copy.deepcopy(gen_model.get_params())
gen_model.gen_params(feature_fix=True)
gen_model.visualize_model(filename='tree2.pdf')
params2 = copy.deepcopy(gen_model.get_params())

# gen_model2 = metatree.GenModel(3,3)
# gen_model2.visualize_model(filename='tree3.pdf')
# gen_model2.set_h_params(h_metatree_list=[params1['root'],params2['root']])
# gen_model2.gen_params(feature_fix=True)
# gen_model2.visualize_model(filename='tree4.pdf')
