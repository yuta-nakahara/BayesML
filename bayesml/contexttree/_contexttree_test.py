import numpy as np
import bayesml.contexttree as model

gen_model = model.GenModel(c_k=2,c_d_max=3,h_g=0.99)
# print(gen_model.get_params())
gen_model.gen_params()
gen_model.visualize_model()
# params = gen_model.get_params()

# gen_model = model.GenModel(2,c_d_max=3)
# gen_model.set_params(root=params['root'])
# gen_model.visualize_model(filename='tmp2.gv')
