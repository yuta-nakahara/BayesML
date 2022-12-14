from bayesml import metatree
from bayesml import normal
import numpy as np
import copy

gen_model = metatree.GenModel(5,3,h_g=0.75,SubModel=normal)
gen_model.gen_params()
gen_model.visualize_model(filename='tree.pdf')
x,y = gen_model.gen_sample(sample_size=100)
print(x)
print(y)
# gen_model.gen_params(feature_fix=True,tree_fix=True)
# gen_model.visualize_model(filename='tree2.pdf')
# gen_model.gen_params(feature_fix=True,tree_fix=True)
# gen_model.visualize_model(filename='tree3.pdf')

# gen_model2 = metatree.GenModel(3,3,h_g=0.01)
# gen_model2.gen_params()
# gen_model2.visualize_model(filename='tree3.pdf')
# gen_model2.set_params(params1['root'])
# gen_model2.visualize_model(filename='tree4.pdf')
# gen_model2.gen_params(feature_fix=True)
# print(gen_model2.get_h_params())
# gen_model2.set_h_params(sub_h_params={'h_beta':100.0},h_g=0.99)
# print(gen_model2.get_h_params())
# gen_model2.gen_params()
# gen_model2.visualize_model(filename='tree3.pdf')

# gen_model2.set_h_params(h_g=0.99)
# gen_model2.gen_params()
# gen_model2.visualize_model(filename='tree4.pdf')

# gen_model2.gen_params(feature_fix=True)
# gen_model2.visualize_model(filename='tree4.pdf')
