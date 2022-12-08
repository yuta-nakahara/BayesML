from bayesml import contexttree
import numpy as np

gen_model = contexttree.GenModel(2,c_d_max=3)
gen_model.set_h_params(h_g=0.99,h_beta_vec=np.ones(1))
gen_model.visualize_model(filename='tree1.pdf')

gen_model.gen_params()
gen_model.visualize_model(filename='tree2.pdf')

params = gen_model.get_params()
# gen_model.set_params(root=params['root'])
# gen_model.visualize_model(filename='tree3.pdf')

gen_model2 = contexttree.GenModel(2,c_d_max=2)
gen_model2.set_h_params(h_root=params['root'])
gen_model2.gen_params()
gen_model2.visualize_model(filename='tree3.pdf')
