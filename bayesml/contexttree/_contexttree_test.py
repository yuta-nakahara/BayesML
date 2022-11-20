import numpy as np
import bayesml.contexttree as model

gen_model = model.GenModel(c_k=2,c_d_max=3,h_g=0.75)
# print(gen_model.get_params())
gen_model.gen_params()
gen_model.visualize_model(filename='tmp1.gv')
# params = gen_model.get_params()

x = gen_model.gen_sample(30)
print(x)

learn_model = model.LearnModel(2,3,h0_g=0.75)
print(learn_model.get_h0_params())
print(learn_model.get_hn_params())

learn_model.update_posterior(x)
learn_model.visualize_posterior(filename='tmp2.gv')
learn_model.estimate_params(filename='tmp3.gv')
print(learn_model.get_h0_params())
print(learn_model.get_hn_params())
